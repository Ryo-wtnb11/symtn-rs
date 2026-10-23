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
