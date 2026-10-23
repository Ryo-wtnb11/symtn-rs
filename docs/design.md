# symtn design

symtn supplies symmetric tensor primitives. Applications own their algorithm
and operation schedule; symtn owns categorical semantics and reduced-block
structure. [Layering](layering.md) records the crate and dependency boundaries.

## Public model

`TensorMap<R, D, S>` is a map `codomain <- domain`: `R` is the provider,
`D` the payload scalar, and `S` the storage (Host by default). Rank, sectors
and degeneracy dimensions are runtime values. A `GradedSpace<R>` describes
one leg by sector labels, degeneracies and duality.

Only symmetry-allowed reduced blocks are stored. Physical dimensions include
quantum-dimension weights; degeneracies count stored degrees of freedom.
Axes are zero-based, codomain first and domain second. Moving a leg across
that split dualizes it. See [TensorKit's manual](https://docs.juliahub.com/General/TensorKit/stable/)
for the tensor-map and fusion-category theory.

A tensor carries a shared `Runtime` handle. Operations reuse its execution
resources; cloning the runtime shares those resources. Storage selects Host
or CUDA kernels through sealed operation-family traits, independently of the
provider's multiplicity-free or checked Generic admission mode. See
[`symtn::typed`](../symtn/src/typed.rs) for dispatch and per-operation bounds.

## One operation's path

```text
TensorMap method and argument validation
  -> provider admission, fusion trees and reduced-block layout
  -> structural transforms and dense jobs
  -> symtn-dense -> Tenferro, or structural movement through Strided
  -> result tensor or validated destination
```

`FusionTreeKey` identifies a basis state by uncoupled sectors, coupled sector,
dual flags, inner lines and vertex multiplicities. Equality checks all five;
a cached hash does not replace identity. Keys retain their immutable tree
arena. `BlockStructure` combines tree identities with degeneracy shapes,
strides and offsets. Equal element counts alone do not imply equal layouts.

`TreeTransformStructure` contains layout moves and recoupling coefficients,
without tensor payload. Replay performs direct moves or pack/GEMM/scatter.
Contraction may need operand and output transforms around its block product.
These steps stay inside one public operation.

## Factorization

Dense factorization groups the tensor into coupled-sector matrices and
delegates the dense problems to Tenferro. The multiplicity-free Host
direct-region and CUDA routes for dense compact QR/SVD and full EIGH use
`CompactFactorPlan` as the per-call, placement-neutral authority for bond
spaces, factor layouts and sector routes. For sector matrix shape `(m, n)`,
the compact bond has `min(m, n)` degeneracy; zero-rank routes are skipped.
Host direct output requires matching tree order; other layouts scatter by
tree identity. Checked Generic QR builds factors through its own checked
staging path.

This is the output-initialization/per-block split of TensorKit
`src/factorizations/matrixalgebrakit.jl:initialize_output` and `foreachblock`;
QSpace's `Source/mpsortho.cc:SVD_Data::blockSVD` and
`Source/QSpace.cc:EigenSymmetric` provide the grouped SVD/EIGH counterparts.
QSpace has no symmetric-tensor QR counterpart. See
[references](../symtn/references.md) and the
[cross-storage factor-space test](../symtn/tests/typed_cuda_single_precision_factorizations.rs).

An owned Host compact diagonal on `V <- V` bypasses the plan and dense
executor for QR, returning both factors in compact storage with `W = V`,
including dual orientation, for multiplicity-free and checked Generic
providers; see the [`qr_compact` rustdoc](../symtn/src/typed/factorize/api.rs#L184-L198).

Truncation is a separate composition: factorize, inspect the spectrum with
`diagview`, select with `GradedSpace::find_truncated`, then `restrict_leg`.
The application chooses the error budget and retained space.

## Reuse and placement

Ordinary eager methods are the primary API. A compact diagonal stores its
spectrum; a lazy adjoint retains its parent. Methods either exploit that
representation, take their documented materialization route, or reject it.
`dense_data` does not silently expand a compact diagonal: use `materialize`.
See the [complexity policy](complexity_parity_policy.md).

Destination methods expose output reuse and validate layout, aliasing and
ownership. They do not provide general transactional rollback after execution
starts; each method documents its error guarantees.

Prepared APIs are operation-specific. `ContractPlan` and `ComposePlan` share
`ContractWorkspace`; `EighFullPlan` has `EighFullWorkspace`. Immutable plans
hold structural assumptions; workspaces hold execution scratch and output
buffers where documented. Bindings are checked per call. `StackedTensorMap`
is an expert fixed-stride representation of homogeneous members, not a
required container for ordinary operations. There is no general public work
graph or symtn-owned second CPU scheduler.

Host/CUDA transfers are explicit `to_cuda` / `to_host` calls. Unsupported
combinations fail through trait bounds or typed errors. Placement and
resource costs are in the [backend policy](backend_policy.md).

## Source map

| Concern | Source |
| --- | --- |
| Public operations and dispatch | [`symtn/src/typed.rs`](../symtn/src/typed.rs) and its `typed/` modules |
| Execution resources | [`symtn/src/runtime.rs`](../symtn/src/runtime.rs) |
| Providers | [`symtn-sectors`](../symtn-sectors/) |
| Spaces, trees and block layouts | [`symtn-core`](../symtn-core/) |
| Transform and contraction execution | [`symtn-operations`](../symtn-operations/), [`symtn-tensors`](../symtn-tensors/) |
| Factor-space authority | [`symtn-matrixalgebra`](../symtn-matrixalgebra/) |
| Dense adapter | [`symtn-dense`](../symtn-dense/) |
| Network planning and execution | [`symtn-network`](../symtn-network/) |
