//! External path planning through the installed Python `cotengra` package.
//!
//! This module deliberately keeps the boundary narrow: symtn lowers
//! [`NetworkIR`] to `inputs/output/size_dict`, Python returns a recycled
//! active-pair path, and the normal Rust [`ContractionPlan`] validation and
//! executor do the rest.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::plancache::{
    CotengraMinimize, CotengraPythonConfig, CotengraPythonMethod, CotengraSlicingConfig,
};
use command_group::{CommandGroup, GroupChild};
use serde_json::{json, Value};

use crate::cost::DenseCostModel;
use crate::error::{ContractError, Result};
use crate::ir::NetworkIR;
use crate::optimizer::{ContractionStep, DenseContractionOptimizer};
use crate::plan::{
    dense_steps_from_active_pair_path, orient_unordered_active_pairs, ActivePair, ContractionPlan,
};
use crate::slice::{slice_plan_for_ordered, SliceKind, SlicedPlan};
use crate::TemporaryLabel;

const PYTHON_PLANNER: &str = r#"
import json
import os
import sys
import traceback

def main():
    spec = json.load(sys.stdin)
    import cotengra as ctg

    inputs = spec["inputs"]
    output = spec["output"]
    size_dict = spec["size_dict"]
    config = spec["config"]
    method = config["method"]
    minimize = config["minimize"]
    max_repeats = config["max_repeats"]
    seed = config["seed"]
    parallel = config["parallel"]

    if method == "auto":
        optimize = "auto"
    elif method == "auto-hq":
        optimize = "auto-hq"
    elif method == "greedy":
        optimize = ctg.GreedyOptimizer()
    elif method == "optimal":
        optimize = ctg.OptimalOptimizer(minimize=minimize)
    elif method == "random-greedy":
        optimize = ctg.RandomGreedyOptimizer(
            max_repeats=max_repeats,
            seed=seed,
            parallel=parallel,
        )
    elif method == "hyper":
        optimize = ctg.HyperOptimizer(
            minimize=minimize,
            max_repeats=max_repeats,
            parallel=parallel,
            progbar=False,
            on_trial_error="raise",
            simulated_annealing_opts=None,
            slicing_opts=None,
            slicing_reconf_opts=None,
            reconf_opts=None,
        )
    else:
        raise ValueError(f"unknown cotengra method: {method}")

    tree = ctg.array_contract_tree(
        inputs,
        output=output,
        size_dict=size_dict,
        optimize=optimize,
        canonicalize=False,
        sort_contraction_indices=False,
    )

    slicing = config["slicing"]
    kind = slicing["kind"]
    if kind == "none":
        pass
    elif kind == "slice":
        tree = tree.slice(
            target_size=slicing["target_size"],
            max_repeats=slicing["max_repeats"],
            allow_outer=slicing["allow_outer"],
            minimize=minimize,
            seed=seed,
        )
    elif kind == "reconfigure":
        reconf_opts = {"forested": slicing["forested"]}
        tree = tree.slice_and_reconfigure(
            target_size=slicing["target_size"],
            step_size=slicing["step_size"],
            max_repeats=slicing["max_repeats"],
            allow_outer=slicing["allow_outer"],
            minimize=minimize,
            reconf_opts=reconf_opts,
            progbar=False,
        )
    elif kind == "forest-reconfigure":
        tree = tree.slice_and_reconfigure_forest(
            target_size=slicing["target_size"],
            step_size=slicing["step_size"],
            num_trees=slicing["num_trees"],
            max_repeats=slicing["max_repeats"],
            allow_outer=slicing["allow_outer"],
            minimize=minimize,
            parallel=parallel,
            progbar=False,
        )
    else:
        raise ValueError(f"unknown cotengra slicing kind: {kind}")

    sliced = [
        {
            "label": info.ind,
            "inner": info.inner,
            "size": info.size,
            "project": info.project,
        }
        for info in tree.sliced_inds.values()
    ]
    json.dump({"path": tree.get_path(), "sliced": sliced}, sys.stdout)

try:
    main()
except Exception as exc:
    json.dump(
        {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
        sys.stdout,
    )
    sys.exit(1)
"#;

const SUPERVISOR_POLL_INTERVAL: Duration = Duration::from_millis(10);
const DIAGNOSTIC_SNIPPET_BYTES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CotengraPythonOptimizer {
    config: CotengraPythonConfig,
}

impl CotengraPythonOptimizer {
    pub fn new(config: CotengraPythonConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &CotengraPythonConfig {
        &self.config
    }

    /// Search a cotengra path and, if configured, cotengra slicing decision,
    /// then package both as a symtn [`SlicedPlan`]. This is planner-only: symtn's
    /// ordinary tensor executor does not execute the slices yet.
    pub fn optimize_sliced(
        &self,
        ir: &NetworkIR,
        cost_model: &DenseCostModel,
    ) -> Result<SlicedPlan> {
        if ir.tensors().len() < 2 {
            return Err(ContractError::NotEnoughTensors);
        }
        let spec = cotengra_spec(ir, cost_model, &self.config);
        let result = run_cotengra_python(&self.config, &spec)?;
        let pairs = path_to_active_pairs(&result.path, ir.tensors().len())?;
        let plan = ContractionPlan::from_steps(
            ir,
            crate::dense_steps_from_active_pair_path(ir, &pairs, cost_model)?,
        )?;
        let sliced = parse_sliced_labels(ir, cost_model, &result.sliced)?;
        let slice = slice_plan_for_ordered(ir, &plan, cost_model, &sliced);
        Ok(SlicedPlan::new(plan, slice))
    }
}

impl Default for CotengraPythonOptimizer {
    fn default() -> Self {
        Self::new(CotengraPythonConfig::default())
    }
}

impl DenseContractionOptimizer for CotengraPythonOptimizer {
    fn optimize(
        &self,
        ir: &NetworkIR,
        cost_model: &DenseCostModel,
    ) -> Result<Vec<ContractionStep>> {
        if ir.tensors().len() < 2 {
            return Err(ContractError::NotEnoughTensors);
        }

        let mut config = self.config.clone();
        config.slicing = CotengraSlicingConfig::None;
        let spec = cotengra_spec(ir, cost_model, &config);
        let result = run_cotengra_python(&config, &spec)?;
        let pairs = path_to_active_pairs(&result.path, ir.tensors().len())?;
        dense_steps_from_active_pair_path(ir, &pairs, cost_model)
    }
}

fn cotengra_spec(
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
    config: &CotengraPythonConfig,
) -> Value {
    let inputs = ir
        .tensors()
        .iter()
        .map(|tensor| {
            tensor
                .labels()
                .iter()
                .map(|label| label.as_str())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let output = ir
        .output_labels()
        .iter()
        .map(|label| label.as_str())
        .collect::<Vec<_>>();

    let mut size_dict = BTreeMap::new();
    for tensor in ir.tensors() {
        for label in tensor.labels() {
            size_dict.insert(label.as_str(), cost_model.dim(label).unwrap_or(1));
        }
    }
    for label in ir.output_labels() {
        size_dict.insert(label.as_str(), cost_model.dim(label).unwrap_or(1));
    }

    json!({
        "inputs": inputs,
        "output": output,
        "size_dict": size_dict,
        "config": {
            "method": method_name(&config.method),
            "minimize": minimize_name(&config.minimize),
            "max_repeats": config.max_repeats.max(1),
            "seed": config.seed,
            "parallel": config.parallel,
            "slicing": slicing_spec(&config.slicing),
        },
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CotengraPythonResult {
    path: Vec<Vec<usize>>,
    sliced: Vec<CotengraSlicedIndex>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CotengraSlicedIndex {
    label: String,
    inner: bool,
    project: Option<usize>,
}

fn run_cotengra_python(
    config: &CotengraPythonConfig,
    spec: &Value,
) -> Result<CotengraPythonResult> {
    let command = python_command(config);
    let command_text = command_text(&command);
    let bytes = serde_json::to_vec(spec).map_err(|err| {
        ContractError::InvalidContractionPlan(format!(
            "failed to serialize cotengra planner input: {err}; {}",
            diagnostic_context(config, &command_text, &[], &[])
        ))
    })?;

    let mut process = Command::new(&command.program);
    process
        .args(&command.args)
        .arg("-c")
        .arg(PYTHON_PLANNER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = spawn_group(&mut process).map_err(|err| {
        ContractError::InvalidContractionPlan(format!(
            "failed to start cotengra Python planner: {err}; {}",
            diagnostic_context(config, &command_text, &[], &[])
        ))
    })?;
    let mut child = ReapingChild::new(child);
    let stdin = child
        .inner_mut()
        .stdin
        .take()
        .ok_or_else(|| missing_pipe(config, &command_text, "stdin"))?;
    let stdout = child
        .inner_mut()
        .stdout
        .take()
        .ok_or_else(|| missing_pipe(config, &command_text, "stdout"))?;
    let stderr = child
        .inner_mut()
        .stderr
        .take()
        .ok_or_else(|| missing_pipe(config, &command_text, "stderr"))?;

    let (failure_tx, failure_rx) = mpsc::channel();
    let workers = IoWorkers {
        stdin: spawn_writer(stdin, bytes, failure_tx.clone()),
        stdout: spawn_reader(stdout, "stdout", failure_tx.clone()),
        stderr: spawn_reader(stderr, "stderr", failure_tx),
    };
    let mut clock = MonotonicClock::new();
    let supervision = supervise_child(&mut child, config.timeout, &failure_rx, &mut clock);
    if let Supervision::Failed(failure) = &supervision {
        if !failure.cleanup_errors.is_empty() {
            // An unexpected group-cleanup failure means pipe EOF is not guaranteed.
            // Retry through the guard, but detach readers instead of risking a join hang.
            drop(child);
            drop(workers);
            return Err(ContractError::InvalidContractionPlan(format!(
                "cotengra Python planner {}{}; {}",
                failure.reason,
                format_cleanup(&failure.cleanup_errors),
                diagnostic_context(config, &command_text, &[], &[]),
            )));
        }
    }
    // The group has exited or has been killed and reaped before any pipe is joined.
    let output = workers.join();
    let context = diagnostic_context(config, &command_text, &output.stdout, &output.stderr);

    if let Supervision::Failed(failure) = supervision {
        return Err(ContractError::InvalidContractionPlan(format!(
            "cotengra Python planner {}{}; {context}{}",
            failure.reason,
            format_cleanup(&failure.cleanup_errors),
            format_io_errors(&output.errors),
        )));
    }
    let Supervision::Exited(status) = supervision else {
        unreachable!()
    };
    if !output.errors.is_empty() {
        return Err(ContractError::InvalidContractionPlan(format!(
            "cotengra Python planner I/O failed; {context}{}",
            format_io_errors(&output.errors)
        )));
    }

    let value: Value = serde_json::from_slice(&output.stdout).map_err(|err| {
        ContractError::InvalidContractionPlan(format!(
            "cotengra Python planner returned non-JSON stdout: {err}; {context}"
        ))
    })?;

    if !status.success() {
        let message = value
            .get("traceback")
            .or_else(|| value.get("error"))
            .and_then(Value::as_str)
            .map(|message| bounded_snippet(message.as_bytes()))
            .unwrap_or_else(|| bounded_snippet(&output.stdout));
        return Err(ContractError::InvalidContractionPlan(format!(
            "cotengra Python planner failed: {message}; {context}"
        )));
    }

    parse_planner_output(&value)
}

fn spawn_group(command: &mut Command) -> io::Result<GroupChild> {
    #[cfg(windows)]
    {
        // command-group 5.0.1 can fail after spawning CREATE_SUSPENDED but
        // before returning GroupChild. symtn cannot clean up a handle it never
        // receives; supervision guarantees begin after this returns Ok.
        command.group().kill_on_drop(true).spawn()
    }
    #[cfg(not(windows))]
    {
        command.group_spawn()
    }
}

fn missing_pipe(config: &CotengraPythonConfig, command_text: &str, pipe: &str) -> ContractError {
    ContractError::InvalidContractionPlan(format!(
        "failed to open cotengra Python planner {pipe}; {}",
        diagnostic_context(config, command_text, &[], &[])
    ))
}

trait ChildGroup {
    type Status: Copy;

    fn try_wait(&mut self) -> io::Result<Option<Self::Status>>;
    fn kill(&mut self) -> io::Result<()>;
    fn wait(&mut self) -> io::Result<Self::Status>;
    fn cleanup_after_observed_exit(&mut self) -> io::Result<()>;
}

impl ChildGroup for GroupChild {
    type Status = ExitStatus;

    fn try_wait(&mut self) -> io::Result<Option<Self::Status>> {
        #[cfg(windows)]
        {
            // Do not populate GroupChild's leader-status cache: after killing
            // the Job, wait() must still wait for ACTIVE_PROCESS_ZERO.
            self.inner().try_wait()
        }
        #[cfg(unix)]
        {
            GroupChild::try_wait(self)
        }
    }

    fn kill(&mut self) -> io::Result<()> {
        GroupChild::kill(self)
    }

    fn wait(&mut self) -> io::Result<Self::Status> {
        GroupChild::wait(self)
    }

    fn cleanup_after_observed_exit(&mut self) -> io::Result<()> {
        #[cfg(windows)]
        {
            GroupChild::kill(self)
        }
        #[cfg(unix)]
        {
            match GroupChild::kill(self) {
                Err(err) if err.raw_os_error() == Some(libc::ESRCH) => Ok(()),
                result => result,
            }
        }
    }
}

struct ReapingChild<C: ChildGroup> {
    child: C,
    reaped: bool,
}

impl ReapingChild<GroupChild> {
    fn inner_mut(&mut self) -> &mut std::process::Child {
        self.child.inner()
    }
}

impl<C: ChildGroup> ReapingChild<C> {
    fn new(child: C) -> Self {
        Self {
            child,
            reaped: false,
        }
    }

    fn try_wait(&mut self) -> io::Result<Option<C::Status>> {
        self.child.try_wait()
    }

    fn kill(&mut self) -> io::Result<()> {
        self.child.kill()
    }

    fn wait(&mut self) -> io::Result<C::Status> {
        self.child.wait()
    }

    fn cleanup_after_observed_exit(&mut self) -> io::Result<()> {
        self.child.cleanup_after_observed_exit()
    }

    fn mark_reaped(&mut self) {
        self.reaped = true;
    }
}

impl<C: ChildGroup> Drop for ReapingChild<C> {
    fn drop(&mut self) {
        if !self.reaped {
            if self.child.kill().is_ok() {
                let _ = self.child.wait();
            } else {
                // Never turn cleanup failure or unwinding into an unbounded wait.
                let _ = self.child.try_wait();
            }
        }
    }
}

trait Clock {
    fn now(&self) -> Duration;
    fn sleep(&mut self, duration: Duration);
}

struct MonotonicClock(Instant);

impl MonotonicClock {
    fn new() -> Self {
        Self(Instant::now())
    }
}

impl Clock for MonotonicClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }

    fn sleep(&mut self, duration: Duration) {
        thread::sleep(duration);
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Supervision<S> {
    Exited(S),
    Failed(SupervisionFailure),
}

#[derive(Debug, PartialEq, Eq)]
struct SupervisionFailure {
    reason: String,
    cleanup_errors: Vec<String>,
}

fn supervise_child<C: ChildGroup, T: Clock>(
    child: &mut ReapingChild<C>,
    timeout: Option<Duration>,
    failures: &Receiver<String>,
    clock: &mut T,
) -> Supervision<C::Status> {
    let deadline = timeout.map(|timeout| clock.now().checked_add(timeout).unwrap_or(Duration::MAX));

    loop {
        if let Ok(failure) = failures.try_recv() {
            return failed_after_cleanup(child, failure);
        }

        let now = clock.now();
        if deadline.is_some_and(|deadline| now >= deadline) {
            // Close the race where the group exits at the deadline.
            return match child.try_wait() {
                Ok(Some(_)) => finish_observed_exit(child),
                Ok(None) => Supervision::Failed(SupervisionFailure {
                    reason: "timed out".to_string(),
                    cleanup_errors: kill_and_reap(child),
                }),
                Err(err) => {
                    failed_after_cleanup(child, format!("status polling failed at deadline: {err}"))
                }
            };
        }

        match child.try_wait() {
            Ok(Some(_)) => return finish_observed_exit(child),
            Ok(None) => {}
            Err(err) => {
                return failed_after_cleanup(child, format!("status polling failed: {err}"))
            }
        }

        let sleep = deadline
            .map(|deadline| deadline.saturating_sub(clock.now()))
            .unwrap_or(SUPERVISOR_POLL_INTERVAL)
            .min(SUPERVISOR_POLL_INTERVAL);
        clock.sleep(sleep);
    }
}

fn finish_observed_exit<C: ChildGroup>(child: &mut ReapingChild<C>) -> Supervision<C::Status> {
    // A launcher leader can exit before non-child group members that still own
    // pipes. GroupChild caches the leader status, so terminate descendants first.
    if let Err(err) = child.cleanup_after_observed_exit() {
        return Supervision::Failed(SupervisionFailure {
            reason: "could not ensure the exited planner group was empty".to_string(),
            cleanup_errors: vec![format!("remaining-group cleanup failed: {err}")],
        });
    }
    let mut errors = Vec::new();
    match child.wait() {
        Ok(status) if errors.is_empty() => {
            child.mark_reaped();
            Supervision::Exited(status)
        }
        Ok(_) => Supervision::Failed(SupervisionFailure {
            reason: "could not ensure the exited planner group was empty".to_string(),
            cleanup_errors: errors,
        }),
        Err(err) => {
            errors.push(format!("reap failed: {err}"));
            Supervision::Failed(SupervisionFailure {
                reason: "could not reap exited process group".to_string(),
                cleanup_errors: errors,
            })
        }
    }
}

fn failed_after_cleanup<C: ChildGroup>(
    child: &mut ReapingChild<C>,
    reason: String,
) -> Supervision<C::Status> {
    Supervision::Failed(SupervisionFailure {
        reason,
        cleanup_errors: cleanup_group(child),
    })
}

fn cleanup_group<C: ChildGroup>(child: &mut ReapingChild<C>) -> Vec<String> {
    match child.try_wait() {
        Ok(Some(_)) | Ok(None) => kill_and_reap(child),
        Err(err) => {
            let mut errors = vec![format!("final status check failed: {err}")];
            errors.extend(kill_and_reap(child));
            errors
        }
    }
}

fn kill_and_reap<C: ChildGroup>(child: &mut ReapingChild<C>) -> Vec<String> {
    if let Err(error) = kill_group(child) {
        return vec![error];
    }
    let mut errors = Vec::new();
    match child.wait() {
        Ok(_) if errors.is_empty() => child.mark_reaped(),
        Ok(_) => {}
        Err(err) => errors.push(format!("reap failed: {err}")),
    }
    errors
}

fn kill_group<C: ChildGroup>(child: &mut ReapingChild<C>) -> std::result::Result<(), String> {
    match child.kill() {
        Ok(()) => Ok(()),
        #[cfg(unix)]
        Err(err) if err.raw_os_error() == Some(libc::ESRCH) => Ok(()),
        Err(err) => Err(format!("group kill failed: {err}")),
    }
}

struct IoWorkers {
    stdin: JoinHandle<io::Result<()>>,
    stdout: JoinHandle<io::Result<Vec<u8>>>,
    stderr: JoinHandle<io::Result<Vec<u8>>>,
}

struct IoOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    errors: Vec<String>,
}

impl IoWorkers {
    fn join(self) -> IoOutput {
        let mut errors = Vec::new();
        match self.stdin.join() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => errors.push(format!("stdin write failed: {err}")),
            Err(_) => errors.push("stdin writer panicked".to_string()),
        }
        let stdout = join_reader(self.stdout, "stdout", &mut errors);
        let stderr = join_reader(self.stderr, "stderr", &mut errors);
        IoOutput {
            stdout,
            stderr,
            errors,
        }
    }
}

fn spawn_writer(
    mut stdin: std::process::ChildStdin,
    bytes: Vec<u8>,
    failures: mpsc::Sender<String>,
) -> JoinHandle<io::Result<()>> {
    thread::spawn(move || {
        let result = stdin.write_all(&bytes);
        if let Err(err) = &result {
            let _ = failures.send(format!("stdin write failed: {err}"));
        }
        result
    })
}

fn spawn_reader<R: Read + Send + 'static>(
    mut reader: R,
    name: &'static str,
    failures: mpsc::Sender<String>,
) -> JoinHandle<io::Result<Vec<u8>>> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = reader.read_to_end(&mut bytes).map(|_| bytes);
        if let Err(err) = &result {
            let _ = failures.send(format!("{name} read failed: {err}"));
        }
        result
    })
}

fn join_reader(
    reader: JoinHandle<io::Result<Vec<u8>>>,
    name: &str,
    errors: &mut Vec<String>,
) -> Vec<u8> {
    match reader.join() {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(err)) => {
            errors.push(format!("{name} read failed: {err}"));
            Vec::new()
        }
        Err(_) => {
            errors.push(format!("{name} reader panicked"));
            Vec::new()
        }
    }
}

fn diagnostic_context(
    config: &CotengraPythonConfig,
    command_text: &str,
    stdout: &[u8],
    stderr: &[u8],
) -> String {
    format!(
        "command={:?}, timeout={}, method={}, minimize={}, slicing={}, stdout={:?}, stderr={:?}",
        bounded_snippet(command_text.as_bytes()),
        config
            .timeout
            .map(|timeout| format!("{timeout:?}"))
            .unwrap_or_else(|| "unbounded".to_string()),
        method_name(&config.method),
        minimize_name(&config.minimize),
        slicing_name(&config.slicing),
        bounded_snippet(stdout),
        bounded_snippet(stderr),
    )
}

fn bounded_snippet(bytes: &[u8]) -> String {
    let truncated = bytes.len() > DIAGNOSTIC_SNIPPET_BYTES;
    let mut snippet =
        String::from_utf8_lossy(&bytes[..bytes.len().min(DIAGNOSTIC_SNIPPET_BYTES)]).into_owned();
    if truncated {
        snippet.push_str("...[truncated]");
    }
    snippet
}

fn format_cleanup(errors: &[String]) -> String {
    if errors.is_empty() {
        String::new()
    } else {
        format!(" (cleanup: {})", errors.join(", "))
    }
}

fn format_io_errors(errors: &[String]) -> String {
    if errors.is_empty() {
        String::new()
    } else {
        format!("; I/O errors: {}", errors.join(", "))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PythonCommand {
    program: String,
    args: Vec<String>,
}

fn python_command(config: &CotengraPythonConfig) -> PythonCommand {
    PythonCommand {
        program: config.python.clone(),
        args: config.python_args.clone(),
    }
}

fn command_text(command: &PythonCommand) -> String {
    std::iter::once(command.program.as_str())
        .chain(command.args.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_planner_output(value: &Value) -> Result<CotengraPythonResult> {
    Ok(CotengraPythonResult {
        path: parse_path(value)?,
        sliced: parse_sliced(value)?,
    })
}

fn parse_path(value: &Value) -> Result<Vec<Vec<usize>>> {
    let path = value.get("path").ok_or_else(|| {
        ContractError::InvalidContractionPlan(
            "cotengra Python planner output is missing `path`".to_string(),
        )
    })?;
    let path = path.as_array().ok_or_else(|| {
        ContractError::InvalidContractionPlan(
            "cotengra Python planner `path` is not an array".to_string(),
        )
    })?;
    path.iter()
        .map(|step| {
            let step = step.as_array().ok_or_else(|| {
                ContractError::InvalidContractionPlan(
                    "cotengra Python planner path step is not an array".to_string(),
                )
            })?;
            step.iter()
                .map(|index| {
                    index.as_u64().map(|value| value as usize).ok_or_else(|| {
                        ContractError::InvalidContractionPlan(
                            "cotengra Python planner path index is not an unsigned integer"
                                .to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect()
}

fn parse_sliced(value: &Value) -> Result<Vec<CotengraSlicedIndex>> {
    let Some(sliced) = value.get("sliced") else {
        return Ok(Vec::new());
    };
    let sliced = sliced.as_array().ok_or_else(|| {
        ContractError::InvalidContractionPlan(
            "cotengra Python planner `sliced` is not an array".to_string(),
        )
    })?;
    sliced
        .iter()
        .map(|entry| {
            let label = entry
                .get("label")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ContractError::InvalidContractionPlan(
                        "cotengra Python planner sliced entry is missing string `label`"
                            .to_string(),
                    )
                })?
                .to_string();
            let inner = entry.get("inner").and_then(Value::as_bool).ok_or_else(|| {
                ContractError::InvalidContractionPlan(
                    "cotengra Python planner sliced entry is missing bool `inner`".to_string(),
                )
            })?;
            let project = match entry.get("project") {
                None | Some(Value::Null) => None,
                Some(value) => Some(value.as_u64().ok_or_else(|| {
                    ContractError::InvalidContractionPlan(
                        "cotengra Python planner sliced `project` is not an unsigned integer"
                            .to_string(),
                    )
                })? as usize),
            };
            Ok(CotengraSlicedIndex {
                label,
                inner,
                project,
            })
        })
        .collect()
}

fn parse_sliced_labels(
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
    sliced: &[CotengraSlicedIndex],
) -> Result<Vec<TemporaryLabel>> {
    let mut labels = Vec::with_capacity(sliced.len());
    for index in sliced {
        if index.project.is_some() {
            return Err(ContractError::UnsupportedPlannerProjection {
                label: index.label.clone(),
            });
        }
        let label = TemporaryLabel::from(index.label.as_str());
        if cost_model.dim(&label).is_none() {
            return Err(ContractError::UnknownPlannerSliceLabel {
                label: index.label.clone(),
            });
        }
        let expected = if ir.output_labels().contains(&label) {
            SliceKind::Output
        } else {
            SliceKind::Internal
        };
        let actual = if index.inner {
            SliceKind::Internal
        } else {
            SliceKind::Output
        };
        if actual != expected {
            return Err(ContractError::PlannerSliceKindMismatch {
                label: index.label.clone(),
                expected,
                actual,
            });
        }
        labels.push(label);
    }
    Ok(labels)
}

fn path_to_active_pairs(path: &[Vec<usize>], tensor_count: usize) -> Result<Vec<ActivePair>> {
    let pairs = path
        .iter()
        .map(|step| match step.as_slice() {
            [lhs, rhs] => Ok(ActivePair::new(*lhs, *rhs)),
            other => Err(ContractError::InvalidContractionPlan(format!(
                "cotengra returned a non-pairwise step with {} operands; symtn plans are strictly pairwise",
                other.len()
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    orient_unordered_active_pairs(&pairs, tensor_count)
}

fn method_name(method: &CotengraPythonMethod) -> &'static str {
    match method {
        CotengraPythonMethod::Auto => "auto",
        CotengraPythonMethod::AutoHq => "auto-hq",
        CotengraPythonMethod::Greedy => "greedy",
        CotengraPythonMethod::Optimal => "optimal",
        CotengraPythonMethod::RandomGreedy => "random-greedy",
        CotengraPythonMethod::Hyper => "hyper",
    }
}

fn minimize_name(minimize: &CotengraMinimize) -> &str {
    match minimize {
        CotengraMinimize::Flops => "flops",
        CotengraMinimize::Size => "size",
        CotengraMinimize::Write => "write",
        CotengraMinimize::Combo => "combo",
        CotengraMinimize::Limit => "limit",
        CotengraMinimize::Custom(value) => value.as_str(),
    }
}

fn slicing_name(slicing: &CotengraSlicingConfig) -> &'static str {
    match slicing {
        CotengraSlicingConfig::None => "none",
        CotengraSlicingConfig::Slice { .. } => "slice",
        CotengraSlicingConfig::Reconfigure { .. } => "reconfigure",
        CotengraSlicingConfig::ForestReconfigure { .. } => "forest-reconfigure",
    }
}

fn slicing_spec(slicing: &CotengraSlicingConfig) -> Value {
    match slicing {
        CotengraSlicingConfig::None => json!({"kind": "none"}),
        CotengraSlicingConfig::Slice {
            target_size,
            max_repeats,
            allow_outer,
        } => json!({
            "kind": "slice",
            "target_size": target_size,
            "max_repeats": max_repeats.max(&1),
            "allow_outer": allow_outer,
        }),
        CotengraSlicingConfig::Reconfigure {
            target_size,
            step_size,
            max_repeats,
            allow_outer,
            forested,
        } => json!({
            "kind": "reconfigure",
            "target_size": target_size,
            "step_size": step_size.max(&1),
            "max_repeats": max_repeats.max(&1),
            "allow_outer": allow_outer,
            "forested": forested,
        }),
        CotengraSlicingConfig::ForestReconfigure {
            target_size,
            step_size,
            num_trees,
            max_repeats,
            allow_outer,
        } => json!({
            "kind": "forest-reconfigure",
            "target_size": target_size,
            "step_size": step_size.max(&1),
            "num_trees": num_trees.max(&1),
            "max_repeats": max_repeats.max(&1),
            "allow_outer": allow_outer,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::test_ir;
    use crate::DenseTensorInfo;

    #[test]
    fn spec_preserves_labels_and_dims() {
        let ir = test_ir(&[&["a", "b"], &["b", "c"]], &["a", "c"]);
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![3, 4]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let spec = cotengra_spec(&ir, &cost, &CotengraPythonConfig::default());
        assert_eq!(spec["inputs"], json!([["a", "b"], ["b", "c"]]));
        assert_eq!(spec["output"], json!(["a", "c"]));
        assert_eq!(spec["size_dict"], json!({"a": 2, "b": 3, "c": 4}));
        assert_eq!(spec["config"]["method"], json!("auto-hq"));
        assert_eq!(spec["config"]["slicing"], json!({"kind": "none"}));
        assert!(spec["config"].get("timeout").is_none());
    }

    #[test]
    fn spec_encodes_reconfigure_slicing() {
        let ir = test_ir(&[&["a", "b"], &["b", "c"]], &["a", "c"]);
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![3, 4]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let config = CotengraPythonConfig {
            slicing: CotengraSlicingConfig::Reconfigure {
                target_size: 8,
                step_size: 2,
                max_repeats: 7,
                allow_outer: false,
                forested: true,
            },
            ..Default::default()
        };
        let spec = cotengra_spec(&ir, &cost, &config);
        assert_eq!(
            spec["config"]["slicing"],
            json!({
                "kind": "reconfigure",
                "target_size": 8,
                "step_size": 2,
                "max_repeats": 7,
                "allow_outer": false,
                "forested": true,
            })
        );
    }

    #[test]
    fn parses_path_and_sliced_indices() {
        let value = json!({
            "path": [[0, 1], [0, 1]],
            "sliced": [
                {"label": "a", "inner": false, "size": 2, "project": null},
                {"label": "b", "inner": true, "size": 3, "project": null},
            ],
        });
        let parsed = parse_planner_output(&value).unwrap();
        assert_eq!(parsed.path, vec![vec![0, 1], vec![0, 1]]);
        assert_eq!(
            parsed.sliced,
            vec![
                CotengraSlicedIndex {
                    label: "a".to_string(),
                    inner: false,
                    project: None,
                },
                CotengraSlicedIndex {
                    label: "b".to_string(),
                    inner: true,
                    project: None,
                },
            ]
        );
    }

    #[test]
    fn unsupported_slice_outputs_are_typed() {
        let ir = test_ir(&[&["a", "b"], &["b", "c"]], &["a", "c"]);
        let cost = DenseCostModel::from_network(
            &ir,
            &[
                crate::DenseTensorInfo::new(vec![2, 3]),
                crate::DenseTensorInfo::new(vec![3, 4]),
            ],
        )
        .unwrap();
        let projected = [CotengraSlicedIndex {
            label: "b".to_string(),
            inner: true,
            project: Some(0),
        }];
        assert!(matches!(
            parse_sliced_labels(&ir, &cost, &projected),
            Err(ContractError::UnsupportedPlannerProjection { .. })
        ));
        let wrong_kind = [CotengraSlicedIndex {
            label: "b".to_string(),
            inner: false,
            project: None,
        }];
        assert!(matches!(
            parse_sliced_labels(&ir, &cost, &wrong_kind),
            Err(ContractError::PlannerSliceKindMismatch { .. })
        ));
        let unknown = [CotengraSlicedIndex {
            label: "missing".to_string(),
            inner: true,
            project: None,
        }];
        assert!(matches!(
            parse_sliced_labels(&ir, &cost, &unknown),
            Err(ContractError::UnknownPlannerSliceLabel { .. })
        ));
    }

    #[test]
    fn rejects_non_pairwise_path_steps() {
        let err = path_to_active_pairs(&[vec![0], vec![0, 1, 2]], 3).unwrap_err();
        assert!(err.to_string().contains("non-pairwise"));
    }

    #[test]
    fn unordered_path_pairs_are_oriented_by_written_subtree() {
        assert_eq!(
            path_to_active_pairs(&[vec![0, 1], vec![0, 1]], 3).unwrap(),
            vec![ActivePair::new(0, 1), ActivePair::new(1, 0)]
        );
    }

    #[test]
    fn uv_project_config_builds_python_command() {
        let expected_project = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/cotengra-python");
        let config = CotengraPythonConfig::with_uv_project(expected_project);
        assert_eq!(
            python_command(&config),
            PythonCommand {
                program: "uv".to_string(),
                args: vec![
                    "run".to_string(),
                    "--project".to_string(),
                    expected_project.to_string(),
                    "python".to_string(),
                ],
            }
        );
    }

    #[derive(Default)]
    struct FakeState {
        polls: usize,
        kills: usize,
        leader_status_observed: bool,
        group_waits: usize,
    }

    struct FakeChild {
        state: std::rc::Rc<std::cell::RefCell<FakeState>>,
        exit_on_poll: Option<usize>,
        kill_error: Option<io::ErrorKind>,
    }

    impl ChildGroup for FakeChild {
        type Status = u8;

        fn try_wait(&mut self) -> io::Result<Option<Self::Status>> {
            let mut state = self.state.borrow_mut();
            state.polls += 1;
            let status = (Some(state.polls) == self.exit_on_poll).then_some(0);
            state.leader_status_observed |= status.is_some();
            Ok(status)
        }

        fn kill(&mut self) -> io::Result<()> {
            self.state.borrow_mut().kills += 1;
            self.kill_error.map(io::Error::from).map_or(Ok(()), Err)
        }

        fn wait(&mut self) -> io::Result<Self::Status> {
            self.state.borrow_mut().group_waits += 1;
            Ok(0)
        }

        fn cleanup_after_observed_exit(&mut self) -> io::Result<()> {
            // Model Windows, where a leader exit may leave Job descendants.
            self.kill()
        }
    }

    #[derive(Default)]
    struct FakeClock {
        now: Duration,
        sleeps: Vec<Duration>,
    }

    impl Clock for FakeClock {
        fn now(&self) -> Duration {
            self.now
        }

        fn sleep(&mut self, duration: Duration) {
            self.sleeps.push(duration);
            self.now += duration;
        }
    }

    fn fake_child(
        exit_on_poll: Option<usize>,
    ) -> (
        ReapingChild<FakeChild>,
        std::rc::Rc<std::cell::RefCell<FakeState>>,
    ) {
        let state = std::rc::Rc::new(std::cell::RefCell::new(FakeState::default()));
        (
            ReapingChild::new(FakeChild {
                state: state.clone(),
                exit_on_poll,
                kill_error: None,
            }),
            state,
        )
    }

    #[test]
    fn supervisor_timeout_kills_and_reaps() {
        let (mut child, state) = fake_child(None);
        let mut clock = FakeClock::default();
        let (_sender, failures) = mpsc::channel();

        assert_eq!(
            supervise_child(
                &mut child,
                Some(Duration::from_millis(25)),
                &failures,
                &mut clock,
            ),
            Supervision::Failed(SupervisionFailure {
                reason: "timed out".to_string(),
                cleanup_errors: Vec::new(),
            })
        );
        assert_eq!(clock.now, Duration::from_millis(25));
        assert_eq!(state.borrow().kills, 1);
        assert_eq!(state.borrow().group_waits, 1);
    }

    #[test]
    fn supervisor_final_poll_closes_deadline_exit_race() {
        let (mut child, state) = fake_child(Some(4));
        let mut clock = FakeClock::default();
        let (_sender, failures) = mpsc::channel();

        assert_eq!(
            supervise_child(
                &mut child,
                Some(Duration::from_millis(25)),
                &failures,
                &mut clock,
            ),
            Supervision::Exited(0)
        );
        assert_eq!(state.borrow().polls, 4);
        assert!(state.borrow().leader_status_observed);
        assert_eq!(state.borrow().kills, 1);
        assert_eq!(state.borrow().group_waits, 1);
    }

    #[test]
    fn observed_leader_status_still_kills_and_waits_for_group() {
        let (mut child, state) = fake_child(Some(1));
        let mut clock = FakeClock::default();
        let (_sender, failures) = mpsc::channel();

        assert_eq!(
            supervise_child(&mut child, None, &failures, &mut clock),
            Supervision::Exited(0)
        );
        let state = state.borrow();
        assert!(state.leader_status_observed);
        assert_eq!(state.kills, 1);
        assert_eq!(state.group_waits, 1);
    }

    #[test]
    fn supervisor_io_failure_kills_and_reaps() {
        let (mut child, state) = fake_child(None);
        let mut clock = FakeClock::default();
        let (sender, failures) = mpsc::channel();
        sender.send("stdin write failed".to_string()).unwrap();

        assert!(matches!(
            supervise_child(&mut child, None, &failures, &mut clock),
            Supervision::Failed(SupervisionFailure { reason, .. })
                if reason == "stdin write failed"
        ));
        assert_eq!(state.borrow().kills, 1);
        assert_eq!(state.borrow().group_waits, 1);
    }

    #[test]
    fn kill_failure_never_enters_an_unbounded_wait() {
        let (mut child, state) = fake_child(None);
        child.child.kill_error = Some(io::ErrorKind::PermissionDenied);
        let mut clock = FakeClock::default();
        let (_sender, failures) = mpsc::channel();

        let result = supervise_child(&mut child, Some(Duration::ZERO), &failures, &mut clock);
        assert!(matches!(
            result,
            Supervision::Failed(SupervisionFailure {
                ref cleanup_errors,
                ..
            }) if cleanup_errors.len() == 1
        ));
        assert_eq!(state.borrow().kills, 1);
        assert_eq!(state.borrow().group_waits, 0);

        drop(child);
        assert_eq!(state.borrow().kills, 2);
        assert_eq!(state.borrow().group_waits, 0);
    }

    #[test]
    fn reaping_guard_cleans_up_during_unwind() {
        let (child, state) = fake_child(None);

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _child = child;
            panic!("test unwind");
        }));

        assert_eq!(state.borrow().kills, 1);
        assert_eq!(state.borrow().group_waits, 1);
    }

    #[test]
    fn diagnostic_context_bounds_stream_snippets() {
        let bytes = vec![b'x'; DIAGNOSTIC_SNIPPET_BYTES + 100];
        let snippet = bounded_snippet(&bytes);
        assert!(snippet.ends_with("...[truncated]"));
        assert_eq!(
            snippet.len(),
            DIAGNOSTIC_SNIPPET_BYTES + "...[truncated]".len()
        );

        let context =
            diagnostic_context(&CotengraPythonConfig::default(), "python3", &bytes, &bytes);
        assert!(context.contains("timeout=300s"));
        assert!(context.contains("method=auto-hq"));
        assert!(context.contains("minimize=flops"));
        assert!(context.contains("slicing=none"));
        assert!(context.len() < 2 * (DIAGNOSTIC_SNIPPET_BYTES + 100));
    }

    #[cfg(unix)]
    fn supervise_shell(script: &str, timeout: Duration) -> (Supervision<ExitStatus>, IoOutput) {
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = ReapingChild::new(spawn_group(&mut command).unwrap());
        let stdin = child.inner_mut().stdin.take().unwrap();
        let stdout = child.inner_mut().stdout.take().unwrap();
        let stderr = child.inner_mut().stderr.take().unwrap();
        let (failure_tx, failure_rx) = mpsc::channel();
        let workers = IoWorkers {
            stdin: spawn_writer(stdin, Vec::new(), failure_tx.clone()),
            stdout: spawn_reader(stdout, "stdout", failure_tx.clone()),
            stderr: spawn_reader(stderr, "stderr", failure_tx),
        };

        let result = supervise_child(
            &mut child,
            Some(timeout),
            &failure_rx,
            &mut MonotonicClock::new(),
        );
        let output = workers.join();
        (result, output)
    }

    #[cfg(unix)]
    #[test]
    fn unix_normal_exit_tolerates_an_empty_group() {
        let (result, output) =
            supervise_shell("printf stdout; printf stderr >&2", Duration::from_secs(2));

        assert!(matches!(result, Supervision::Exited(status) if status.success()));
        assert!(output.errors.is_empty(), "{:?}", output.errors);
        assert_eq!(output.stdout, b"stdout");
        assert_eq!(output.stderr, b"stderr");
    }

    #[cfg(unix)]
    #[test]
    fn unix_leader_exit_kills_pipe_holding_descendant() {
        // A surviving descendant keeps the pipes open, so `workers.join()`
        // never returns and the test hangs instead of passing slowly.
        let (result, output) = supervise_shell("sleep 600 & exit 0", Duration::from_secs(2));

        assert!(matches!(result, Supervision::Exited(status) if status.success()));
        assert!(output.errors.is_empty(), "{:?}", output.errors);
    }

    #[cfg(unix)]
    #[test]
    fn unix_timeout_kills_descendants_after_draining_both_pipes() {
        let (result, output) = supervise_shell(
            "(dd if=/dev/zero bs=131072 count=1 2>/dev/null) & \
                 (dd if=/dev/zero bs=131072 count=1 1>&2 2>/dev/null) & \
                 wait; sleep 30",
            Duration::from_secs(1),
        );

        assert!(matches!(
            result,
            Supervision::Failed(SupervisionFailure { ref reason, .. })
                if reason == "timed out"
        ));
        assert!(output.errors.is_empty(), "{:?}", output.errors);
        assert_eq!(output.stdout.len(), 131072);
        assert_eq!(output.stderr.len(), 131072);
    }

    #[test]
    #[ignore = "requires SYMTN_RUN_COTENGRA_PYTHON_TEST and an installed cotengra environment"]
    fn runs_installed_cotengra_when_requested() {
        if std::env::var_os("SYMTN_RUN_COTENGRA_PYTHON_TEST").is_none() {
            return;
        }

        let ir = test_ir(&[&["a", "b"], &["b", "c"]], &["a", "c"]);
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![3, 4]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let config = std::env::var("SYMTN_COTENGRA_UV_PROJECT")
            .map(CotengraPythonConfig::with_uv_project)
            .unwrap_or_else(|_| {
                CotengraPythonConfig::with_uv_project(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tools/cotengra-python"
                ))
            })
            .timeout(Duration::from_secs(30));

        let steps = CotengraPythonOptimizer::new(config)
            .optimize(&ir, &cost)
            .unwrap();
        assert_eq!(steps.len(), 1);
    }
}
