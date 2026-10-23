# Optional cotengra path planner

Feature `cotengra-python` on `symtn-network` enables an external Python
[cotengra](https://github.com/jcmgray/cotengra) planner. The default is built-in
greedy; feature `opt-path` provides Rust alternatives. Python searches the
pairwise order; normal Rust plan validation and execution still apply.

The subprocess exchanges an einsum specification and path as JSON over
stdin/stdout. Search runs on a plan-cache miss, not inside each dense
contraction. `ReplanPolicy::BakeOnce` reuses orders by topology; cache eviction,
clearing or a changed configuration can require another search.

## Configuration

The Python environment is [`tools/cotengra-python`](../tools/cotengra-python/).
Its manifest includes cotengra, KaHyPar and Optuna for hyper-optimization.
Install it with `uv sync --project tools/cotengra-python`, then configure:

```rust
use symtn::typed::Runtime;
use symtn_network::{configure_plan_cache, CotengraPythonConfig, Optimizer, PlanCacheConfig};

let rt = Runtime::builder().build()?;
let optimizer = Optimizer::CotengraPython(
    CotengraPythonConfig::with_uv_project("/absolute/path/to/symtn-rs/tools/cotengra-python"),
);
configure_plan_cache(&rt, PlanCacheConfig { optimizer, ..Default::default() });
```

The default executable is `python3` on `PATH`; `.python(program)` selects an
interpreter and `.uv_project(path)` selects a uv project. Project paths are
resolved by uv against the launch working directory, so downstream users
should pass an absolute path. symtn reads no interpreter-selection environment
variable. Search method, objective, trials, seed and slicing are config fields;
see [`plancache/config.rs`](../symtn-network/src/plancache/config.rs).

## Boundaries

- Request transfer and search share a five-minute deadline by default.
  `.timeout(duration)` changes it; `.without_timeout()` explicitly disables
  it. On expiry, symtn terminates and reaps the process group.
- Slicing requires explicit sliced-plan search, lowering and Host dense
  execution. `Network::contract` uses path-only planning and does not
  automatically execute slices.
- On Windows, `command-group` can fail after creating a suspended child but
  before returning its handle. Cleanup guarantees start after successful
  group creation; the dependency exposes no child handle for that error path.

Bridge implementation and subprocess tests are in
[`cotengra_python.rs`](../symtn-network/src/cotengra_python.rs). Algorithm
details belong in cotengra's own documentation.
