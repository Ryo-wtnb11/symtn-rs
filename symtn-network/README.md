# symtn-network

The user-facing home of labeled tensor networks: build a `Network` from
per-operand labels and contract it with `Network::contract`, which plans
through the Runtime's cache. `contract!([i; k] := a[i; j] * b[j; k])` writes
the same network in index notation. `NetworkIR`, `PlannedNetwork`,
`ContractionPlan`, `Network::plan`,
optimizer traits, and slicing types let expert users inspect or reuse structural plans;
symtn executes the resulting plan locally.

The default host backend is `cpu-faer`; enable at least one host backend.
`opt-path` enables opt-einsum-path search and `cotengra-python` calls Python
cotengra for path search. `cuda` still requires a host backend, while
`cgc-gen` forwards `LieGroupFusionRule` support; see `Cargo.toml` for the
other provider and BLAS flags.

Run the [contraction example](examples/contract.rs) with:

```sh
cargo run -p symtn-network --example contract
```
