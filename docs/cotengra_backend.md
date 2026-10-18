# cotengra Python contraction-path backend

An **optional**, feature-gated path planner that shells out to the installed
Python [`cotengra`](https://github.com/jcmgray/cotengra) package for
contraction-order search (greedy / optimal / auto-hq / **hyper**) and slicing
decisions (slice / reconfigure / forest-reconfigure). Built-in greedy is the
default; the pure-Rust `opt-einsum-path` optimizers are optional. Cotengra adds the
hyper-optimization tier (KaHyPar hypergraph partitioning + Bayesian
hyper-parameter search) that has no pure-Rust equivalent.

## Why a Python subprocess

Path planning is a coarse-grained, one-shot operation — input is an einsum
spec plus leg dimensions, output is a pairwise path (and an optional slicing).
It is off the hot per-contraction execution path and is cached per topology
(`ReplanPolicy::BakeOnce`), so the subprocess boundary costs little (see
[Latency](#latency)) while avoiding a reimplementation of cotengra's
hyper-optimizer. The bridge is JSON over stdin/stdout; nothing is linked into
the Rust process. `cotengrust` (a Rust cotengra port) is **AGPL**, so it is not
vendored — the Python package is called instead.

## Setup

The reproducible Python environment is the [uv](https://docs.astral.sh/uv/)
project under `tools/cotengra-python`:

```sh
uv sync --project tools/cotengra-python
```

Dependencies (`tools/cotengra-python/pyproject.toml`):

| package | role | without it |
|---------|------|-----------|
| `cotengra` | the path optimizer | — |
| `kahypar` | recommended hypergraph partitioner | `hyper` silently falls back to the basic `labels` partitioner |
| `optuna` | Bayesian hyper-parameter sampling | `hyper` silently falls back to purely random sampling |

`kahypar` and `optuna` are what make the `hyper` method actually
hyper-optimize; both ship as wheels (no C++ build on macOS/Linux). Verify:

```sh
uv run --project tools/cotengra-python python -c \
  'import cotengra, kahypar, optuna; print(cotengra.__version__)'
```

## Enabling it in a crate

Turn on the `cotengra-python` feature on `symtn-network`:

```toml
symtn-network = { path = "...", features = ["cotengra-python"] }
```

## Using it from Rust

```rust
use symtn_network::{configure_plan_cache, CotengraPythonConfig, Optimizer, PlanCacheConfig};
use std::time::Duration;

let optimizer = Optimizer::CotengraPython(
    CotengraPythonConfig::with_uv_project("/path/to/symtn-rs/tools/cotengra-python")
        .timeout(Duration::from_secs(120)),
);
let rt = Runtime::builder().build()?;
configure_plan_cache(&rt, PlanCacheConfig { optimizer, ..Default::default() });
```

The project path is stored verbatim and resolved by `uv` against the process
working directory at launch; symtn neither probes the CWD nor bakes in a
source-tree location, so a downstream crate passes an absolute path (or one
built from its own `CARGO_MANIFEST_DIR`). The interpreter is a config field;
symtn reads no environment variable to choose it:

- `CotengraPythonConfig::python(program)` — run a specific interpreter directly.
- `CotengraPythonConfig::uv_project(path)` — use `uv run --project <path> python`.
- the default is `python3` on `PATH` (must have cotengra importable).

`CotengraPythonConfig` selects the `method` (`Greedy` / `Optimal` / `AutoHq` /
`Hyper`), `minimize` (flops / size / …), `max_repeats`, `seed`, and a
`slicing` config (none / slice / reconfigure / forest-reconfigure with
`target_size` / `step_size` / `max_repeats` / `allow_outer`).

Request transfer and planner search share a five-minute deadline by default. On
expiry symtn terminates the whole Python process group (including `uv`
descendants) and reaps it. Use `.timeout(duration)` to choose another finite
deadline. `.without_timeout()` is the explicit unbounded opt-out.

## Latency

Measured on a 16-tensor periodic grid (warm uv cache), per planner call:

| step | time |
|------|------|
| subprocess + `uv run` + `import cotengra` | ~50 ms |
| greedy / auto search | sub-ms |
| auto-hq search | ~45 ms |
| hyper search (full kahypar + optuna, `max_repeats`≈64) | ~1 s |

So a full call is ~50 ms for greedy, ~100 ms for auto-hq, ~1 s for hyper. Every
call is amortized to **once per topology** by `ReplanPolicy::BakeOnce`. The
`hyper` cost is the real hyper-optimization price and pays off only on large /
hard networks where greedy is suboptimal; for the small local contractions of a
boundary method, greedy / auto-hq (or the pure-Rust `opt-einsum-path` drivers)
are the right choice. A persistent Python worker to remove the ~50 ms spawn is
**not** worth it at this latency.

## Limitations

- **Deployment**: opting in requires a Python environment (uv + cotengra)
  alongside the Rust binary. Built-in greedy has no such requirement.
- **Sliced execution is explicit**: a cotengra sliced plan is lowered and
  executed through the Host dense `Network` APIs. `Network::contract` does
  not automatically lower or execute slices.
- **Windows process-group creation errors**: `command-group` 5.0.1 can fail
  after spawning a suspended child but before returning its handle if Job
  assignment or thread resumption fails. symtn's timeout and unwind guarantees
  begin after successful group creation; the dependency does not expose the
  child needed to clean up this rare spawn-error path. This dependency boundary
  is tracked in [issue #1048](https://github.com/Ryo-wtnb11/symtn-rs/issues/1048).
