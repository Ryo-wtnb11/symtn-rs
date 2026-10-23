# Documentation map

The [root README](../README.md) is the entry point for users. The `symtn`
crate's [primitive examples](../symtn/README.md#primitive-examples) and generated rustdoc contain
operation examples and function-level contracts.

| Read for | Document |
| --- | --- |
| Layer ownership (symbols, tensors, dense backend) | [Layering and ownership](layering.md) |
| Architecture and ownership | [symtn design](design.md) |
| Mathematical conventions | [Tensor-map mathematics](../symtn/src/mathematics.md) |
| Intentional differences from TensorKit | [Differences from TensorKit](tensorkit-differences.md) |
| Implementing a symmetry provider | [Provider interface](provider_interface.md) |
| Selecting dense and device backends | [Backend policy](backend_policy.md) |
| Structural complexity obligations | [Complexity policy](complexity_parity_policy.md) |
| Stable sector-label and storage expectations | [Sector ID compatibility](sector_id_compatibility.md) |
| Primitive tensor operations | [Runnable examples](../symtn/README.md#primitive-examples) |
| Labeled contraction | [contract example](../symtn-network/examples/contract.rs) |
| Optional path planning | [cotengra bridge](cotengra_backend.md) |
| Krylov methods on tensors (feature `krylov`) | [krylov correspondence](krylov.md) |
| Revision-specific performance results | [Benchmark records](../benchmarks/README.md) |

For contributors and coding agents, start with the repository's
[AGENTS.md](../AGENTS.md) and [repository rules](../REPOSITORY_RULES.md).
[Numerical testing](testing_numerics.md), [writing style](writing_style.md),
and [release steps](releasing.md) describe their specific gates.

[CI](../.github/workflows/ci.yml) runs lint, feature/backend contracts, the Host
pool audit, Linux coverage tests, doctests, examples and rustdoc on ready PRs
and main pushes. The full default-provider macOS suite runs on main nightly at
15:17 UTC and by manual dispatch in [macOS nightly](../.github/workflows/nightly.yml);
its runtime checks detect platform regressions later than the push/PR gates.
The scoped Accelerate backend job stays in CI. Ordinary mixed check/test jobs
use zero dev/test debug information before restoring their per-job caches;
debug assertions and overflow checks stay enabled. Coverage keeps its own
profile, disables target-artifact caching to avoid disk exhaustion, and retains
all per-crate floors.

Coverage and macOS nightly serialize only the `symtn-tensors` lib binary: its
[cache-test support](../symtn-tensors/src/test_support.rs) explains why unrelated
constructors can disturb process-global snapshots. The following
`--test '*'` command selects the integration targets only, so every test runs
exactly once.
Of the 14 default-feature integration targets, six allocation/cache targets and
`warm_trace_compile_allocations` have one active test each;
`lowered_cold_contract_plan_allocations`,
`tree_transform_compile_allocations` and `warm_contract_compile_allocations`
guard shared-state tests with [the counting allocator's mutex](../tests/support/counting_alloc.rs).
`rule_identity_boundary`, `tensors_exports` and `tree_transform_categorical_recoupling`
do not assert/reset global snapshots; `cuda_tree_transform_categorical` is
CUDA-gated. Recheck this selection and isolation when adding tensor tests.

The [revision-pinned audit index](audit/README.md) preserves evidence from
earlier code. Superseded design drafts remain in Git history, not the current
documentation tree. An old audit is not current API or performance authority;
verify a claim against present source and tests before using it.
