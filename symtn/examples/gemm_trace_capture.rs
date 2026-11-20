//! Captures the grouped GEMM calls of two-site effective-Hamiltonian
//! applications (U(1) and SU(2) spin-1/2 Heisenberg, f64 and c64, three bond
//! dimensions) as JSON lines: one provenance record, then one record per
//! distinct call of each contraction step with its multiplicity.
//!
//! ```text
//! cargo run --release -p symtn --features gemm-trace --example gemm_trace_capture -- \
//!     OUT.jsonl BRANCH_SHA BASE_SHA THREADS
//! ```
//!
//! Tensors are random: only their block structure decides the job lists.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use symtn::expert::gemm_trace::{self, Call};
use symtn::expert::{DenseScalar, MatrixOp};
use symtn::sector::U1Irrep;
use symtn::sector::{GroupId, LieGroupFusionRule, MultiplicityFreeAdmissionMode, U1FusionRule};
use symtn::typed::{Complex64, ContractSpec, CpuEngine, GradedSpace, Pool, Runtime, TensorMap};

const MAX_JOBS: usize = 1 << 24;

/// Degeneracies of `sectors` labels, Gaussian in the label, scaled so the
/// state count `sum(deg * dim)` is about `states`.
fn degeneracies(labels: &[i64], dim: impl Fn(i64) -> usize, states: usize) -> Vec<usize> {
    let width = labels.iter().map(|l| l.abs()).max().unwrap().max(1) as f64 / 1.5;
    let weight: Vec<f64> = labels
        .iter()
        .map(|&l| (-(l as f64 / width).powi(2) / 2.0).exp())
        .collect();
    let total: f64 = labels
        .iter()
        .zip(&weight)
        .map(|(&l, w)| w * dim(l) as f64)
        .sum();
    weight
        .iter()
        .map(|w| ((w * states as f64 / total).round() as usize).max(1))
        .collect()
}

fn op_name(op: MatrixOp) -> &'static str {
    match op {
        MatrixOp::Identity => "N",
        MatrixOp::Transpose => "T",
        MatrixOp::Adjoint => "C",
    }
}

fn scalar(value: &DenseScalar) -> String {
    match *value {
        DenseScalar::F32(x) => format!("[{x:?},0.0]"),
        DenseScalar::F64(x) => format!("[{x:?},0.0]"),
        DenseScalar::C32(z) => format!("[{:?},{:?}]", z.re, z.im),
        DenseScalar::C64(z) => format!("[{:?},{:?}]", z.re, z.im),
    }
}

/// The call fields of one record, without provenance.
fn call_json(call: &Call) -> String {
    let mut s = format!(
        "\"dtype\":\"{:?}\",\"ops\":[\"{}\",\"{}\"],\"alpha\":{},\"beta\":{},\"threads\":{},\"runs\":{:?},\"jobs\":[",
        call.dtype,
        op_name(call.ops[0]),
        op_name(call.ops[1]),
        scalar(&call.alpha),
        call.beta.as_ref().map_or("null".into(), scalar),
        call.threads,
        call.runs,
    );
    for (i, j) in call.jobs.iter().enumerate() {
        let sep = if i == 0 { "" } else { "," };
        write!(
            s,
            "{sep}[{},{},{},{},{},{},{},{},{}]",
            j.a_offset, j.b_offset, j.c_offset, j.m, j.n, j.k, j.lda, j.ldb, j.ldc
        )
        .unwrap();
    }
    s.push(']');
    s
}

struct Sink {
    out: String,
    calls: usize,
    records: usize,
    jobs: usize,
}

impl Sink {
    /// Appends one step's trace, merging identical calls into a multiplicity.
    fn step(&mut self, model: &str, states: usize, multiplets: usize, scalar: &str, step: &str) {
        let trace = gemm_trace::take();
        assert_eq!(trace.dropped_calls, 0, "raise MAX_JOBS");
        let mut order = Vec::new();
        let mut count: HashMap<String, usize> = HashMap::new();
        for call in &trace.calls {
            self.calls += 1;
            self.jobs += call.jobs.len();
            let body = call_json(call);
            *count.entry(body.clone()).or_insert_with(|| {
                order.push(body);
                0
            }) += 1;
        }
        for body in order {
            self.records += 1;
            writeln!(
                self.out,
                "{{\"kind\":\"call\",\"model\":\"{model}\",\"chi_states\":{states},\"chi_multiplets\":{multiplets},\"scalar\":\"{scalar}\",\"step\":\"{step}\",\"multiplicity\":{},{body}}}",
                count[&body]
            )
            .unwrap();
        }
    }
}

/// `H_eff * theta` for theta: `v (x) p (x) p <- v`, environments
/// L: `v (x) w <- v`, R: `v <- w (x) v`, MPO W: `p (x) w <- w (x) p`.
macro_rules! heff {
    ($sink:expr, $engine:expr, $rt:expr, $model:expr, $states:expr, $mult:expr, $d:ty, $v:expr, $p:expr, $w:expr) => {{
        let (engine, rt, v, p, w) = (&mut *$engine, $rt, $v, $p, $w);
        let l = TensorMap::<_, $d>::rand_with_seed(engine, rt, [v, w], [v], 1).unwrap();
        let theta = TensorMap::<_, $d>::rand_with_seed(engine, rt, [v, p, p], [v], 2).unwrap();
        let mpo = TensorMap::<_, $d>::rand_with_seed(engine, rt, [p, w], [w, p], 3).unwrap();
        let r = TensorMap::<_, $d>::rand_with_seed(engine, rt, [v], [w, v], 4).unwrap();
        let s1 = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 1, 2, 3],
            domain: &[4],
        };
        let s2 = ContractSpec {
            lhs: &[1, 2],
            rhs: &[2, 3],
            codomain: &[0, 3, 4, 1],
            domain: &[2],
        };
        let s3 = ContractSpec {
            lhs: &[2, 3],
            rhs: &[2, 3],
            codomain: &[0, 1, 3, 4],
            domain: &[2],
        };
        let s4 = ContractSpec {
            lhs: &[3, 4],
            rhs: &[1, 0],
            codomain: &[0, 1, 2],
            domain: &[3],
        };
        let scalar = stringify!($d);
        // Warm structure caches first; job lists do not depend on them.
        for traced in [false, true] {
            let mut step = |name: &str, f: &mut dyn FnMut() -> TensorMap<_, $d>| {
                if traced {
                    gemm_trace::start(MAX_JOBS);
                }
                let x = f();
                if traced {
                    $sink.step($model, $states, $mult, scalar, name);
                }
                x
            };
            let x1 = step("heff.L_theta", &mut || {
                l.contract(engine, &theta, &s1).unwrap()
            });
            let x2 = step("heff.W1", &mut || x1.contract(engine, &mpo, &s2).unwrap());
            let x3 = step("heff.W2", &mut || x2.contract(engine, &mpo, &s3).unwrap());
            step("heff.R", &mut || x3.contract(engine, &r, &s4).unwrap());
        }
    }};
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, out_path, branch, base, threads] = &args[..] else {
        panic!("usage: gemm_trace_capture OUT.jsonl BRANCH_SHA BASE_SHA THREADS");
    };
    let threads: usize = threads.parse().unwrap();
    let engine = &mut if threads == 1 {
        CpuEngine::serial()
    } else {
        CpuEngine::new(Pool::builder().threads(threads).build().unwrap())
    };
    let rt = &Runtime::builder().build().unwrap();
    let provider = if cfg!(feature = "cpu-blas") {
        "cpu-blas"
    } else {
        "cpu-faer"
    };
    let mut sink = Sink {
        out: String::new(),
        calls: 0,
        records: 0,
        jobs: 0,
    };
    writeln!(
        sink.out,
        "{{\"kind\":\"provenance\",\"format\":\"symtn-gemm-trace/1\",\"symtn_commit\":\"{branch}\",\"symtn_base\":\"{base}\",\"provider\":\"{provider}\",\"engine_threads\":{threads},\"job_fields\":[\"a_offset\",\"b_offset\",\"c_offset\",\"m\",\"n\",\"k\",\"lda\",\"ldb\",\"ldc\"],\"layout\":\"column-major\"}}"
    )
    .unwrap();

    // U(1): twice Sz. Bonds carry even charges, the MPO leg 0 (x3), +-2.
    let u1 = Arc::new(U1FusionRule);
    let u1_space = |pairs: Vec<(i64, usize)>| {
        GradedSpace::try_new(
            Arc::clone(&u1),
            pairs
                .into_iter()
                .map(|(q, d)| (U1Irrep::new(q as i32).unwrap(), d)),
        )
        .unwrap()
    };
    let p = u1_space(vec![(1, 1), (-1, 1)]);
    let w = u1_space(vec![(-2, 1), (0, 3), (2, 1)]);
    for (states, radius) in [(64, 3), (256, 4), (1024, 6)] {
        let labels: Vec<i64> = (-radius..=radius).map(|q| 2 * q).collect();
        let degs = degeneracies(&labels, |_| 1, states);
        let total = degs.iter().sum();
        let v = u1_space(labels.into_iter().zip(degs).collect());
        heff!(
            sink,
            engine,
            rt,
            "u1_heisenberg",
            total,
            total,
            f64,
            &v,
            &p,
            &w
        );
        heff!(
            sink,
            engine,
            rt,
            "u1_heisenberg",
            total,
            total,
            Complex64,
            &v,
            &p,
            &w
        );
    }

    // SU(2): twice spin. Bonds carry integer spins, the MPO leg 0 (x2), 1.
    let su2 = Arc::new(
        LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(GroupId::su(2).unwrap()).unwrap(),
    );
    let su2_space = |pairs: Vec<(i64, usize)>| {
        let irreps: Vec<_> = pairs
            .into_iter()
            .map(|(j, d)| (su2.category().irrep(&[j]).unwrap(), d))
            .collect();
        GradedSpace::try_new(Arc::clone(&su2), irreps).unwrap()
    };
    let p = su2_space(vec![(1, 1)]);
    let w = su2_space(vec![(0, 2), (2, 1)]);
    for (states, top) in [(64, 4), (256, 6), (1024, 8)] {
        let labels: Vec<i64> = (0..=top).step_by(2).collect();
        let degs = degeneracies(&labels, |j| j as usize + 1, states);
        let multiplets = degs.iter().sum();
        let total = labels
            .iter()
            .zip(&degs)
            .map(|(&j, d)| d * (j as usize + 1))
            .sum();
        let v = su2_space(labels.into_iter().zip(degs).collect());
        heff!(
            sink,
            engine,
            rt,
            "su2_heisenberg",
            total,
            multiplets,
            f64,
            &v,
            &p,
            &w
        );
        heff!(
            sink,
            engine,
            rt,
            "su2_heisenberg",
            total,
            multiplets,
            Complex64,
            &v,
            &p,
            &w
        );
    }

    std::fs::write(out_path, &sink.out).unwrap();
    eprintln!(
        "gemm_trace_capture: {} calls, {} jobs, {} distinct records -> {out_path}",
        sink.calls, sink.jobs, sink.records
    );
}
