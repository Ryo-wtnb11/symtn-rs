# Documentation map

The [root README](../README.md) is the entry point for users. The `symtn`
crate's [tutorial](../symtn/src/tutorial.md) and generated rustdoc contain
operation examples and function-level contracts.

| Read for | Document |
| --- | --- |
| Layer ownership (symbols, tensors, dense backend) | [Layering and ownership](layering.md) |
| Architecture and ownership | [symtn design](design.md) |
| Mathematical conventions | [Tensor-map mathematics](../symtn/src/mathematics.md) |
| Implementing a symmetry provider | [Provider interface](provider_interface.md) |
| Selecting dense and device backends | [Backend policy](backend_policy.md) |
| Structural complexity obligations | [Complexity policy](complexity_parity_policy.md) |
| Stable sector-label and storage expectations | [Sector ID compatibility](sector_id_compatibility.md) |
| Building a complete example | [U(1) iTEBD guide](itebd_heisenberg.md) |
| Optional path planning | [cotengra bridge](cotengra_backend.md) |
| Krylov methods on tensors (feature `krylov`) | [krylov correspondence](krylov.md) |
| Revision-specific performance results | [Benchmark records](../benchmarks/README.md) |

For contributors and coding agents, start with the repository's
[AGENTS.md](../AGENTS.md) and [repository rules](../REPOSITORY_RULES.md).
[Numerical testing](testing_numerics.md), [writing style](writing_style.md),
and [release steps](releasing.md) describe their specific gates.

The [revision-pinned audit index](audit/README.md) preserves evidence from
earlier code. Superseded design drafts remain in Git history, not the current
documentation tree. An old audit is not current API or performance authority;
verify a claim against present source and tests before using it.
