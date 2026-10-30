# Documentation

The [crate guide](../symtn/README.md#primitive-examples) and generated rustdoc
contain operation examples and method-level bounds.

| Read for | Document |
| --- | --- |
| Tensor representation and execution path | [Design](design.md) |
| Crate, symbol and cache ownership | [Layering](layering.md) |
| Intentional differences from TensorKit | [Differences](tensorkit-differences.md) |
| Adding a category and provider contracts | [Provider interface](provider_interface.md) |
| Sector labels, ordering and compatibility | [Sector IDs](sector_id_compatibility.md) |
| Host/CUDA selection, resources and device checks | [Backend policy](backend_policy.md) |
| Structured FLOP and storage obligations | [Complexity policy](complexity_parity_policy.md) |
| Oracles, exactness and tolerances | [Numerical testing](testing_numerics.md) |
| Optional Python path planner | [cotengra bridge](cotengra_backend.md) |
| Optional Host tensor solvers | [Krylov integration](krylov.md) |

For theory, see [TensorKit's manual](https://docs.juliahub.com/General/TensorKit/stable/)
and the [papers and source references](../symtn/references.md).

Contributors start with [AGENTS.md](../AGENTS.md) and the
[repository rules](../REPOSITORY_RULES.md). Describe current code in plain
English; name the API, provider, scalar, storage and placement being claimed.
Use source and focused tests as evidence. Keep revision-specific audits and
measurements in review records or [benchmarks](../benchmarks/README.md), with
revision, provider, build, thread count, workload and raw results.

[CI](../.github/workflows/ci.yml) runs lint, feature/backend contracts, the Host
pool audit, Linux coverage tests, doctests, examples and rustdoc on ready PRs
and main pushes. The lint job also runs `cargo run -p symtn-production-contract`,
a [normal-dependency consumer](../tests/production-contract/src/main.rs) that
checks category A/B forwarding and query costs with `testing` disabled.
The full default-provider macOS suite runs on main nightly at
15:17 UTC and by manual dispatch in [macOS nightly](../.github/workflows/nightly.yml);
its runtime checks detect platform regressions later than the push/PR gates.
The scoped Accelerate backend job stays in CI. Ordinary mixed check/test jobs
use zero dev/test debug information before restoring their per-job caches;
debug assertions and overflow checks stay enabled. Coverage keeps its own
profile, disables target-artifact caching to avoid disk exhaustion, and retains
all per-crate floors.

Local dev and test builds use line-table debug information to retain file:line
backtraces while reducing linked artifact size. The tensor crates group
integration tests into explicit `integration` and `serial` targets per crate.
`integration` members neither observe nor reset process-global state and run
concurrently without serial guards. Whole files that install the counting
allocator, assert on process-global counters or cache state, or reset that
observed state belong to `serial`. Its root installs the counting allocator
once, and every test takes [the shared serial guard](../tests/support/counting_alloc.rs)
first. Cold-cache tests establish their preconditions through public clear/reset
APIs. Runtime-local caches and thread-local CUDA transfer counters do not require
serialization. Tests requiring set-once process state keep separate targets:
`intermediate_allocations`. Run a member with a module filter, for example
`cargo test -p symtn --no-default-features --features krylov,blas-openblas --test serial krylov_vector::`.

Coverage and macOS nightly also serialize the `symtn-tensors` lib binary: its
[cache-test support](../symtn-tensors/src/test_support.rs) explains why unrelated
constructors can disturb process-global snapshots. A separate `--test '*'`
selects integration targets only, so the lib runs once.
