# symtn design

symtn supplies symmetric tensor primitives from which applications build their
own algorithms. Its public API exposes tensor maps, spaces, categorical
providers, and mathematical operations. It does not prescribe an MPS, PEPS,
VMC, or device-transfer schedule. This makes the same small operations usable
in code written by a person or a coding agent.

## Public model

A `TensorMap<R, D, S>` is a map `codomain <- domain`: `R` supplies the fusion
rule, `D` is the payload scalar, and `S` is the storage type. Tensor rank,
sector content, and leg dimensions are runtime values. `GradedSpace<R>`
describes a leg. `Runtime` owns the resources needed to execute operations.
The ordinary eager methods are the first API to use; prepared state and
destination methods are explicit options for repeated work.

TensorKit supplies the reference for mathematical conventions and public
behavior. QSpace supplies applicable reduced-block, runtime-rank, and
contiguous-dataflow techniques. symtn implements the selected semantics with
Rust ownership and checked errors. See [the mathematics](../symtn/src/mathematics.md)
and [provider interface](provider_interface.md) for the precise conventions
and trait bounds.

## One operation's path

```text
TensorMap operation and checked arguments
  -> provider data, fusion trees, sectors, and reduced-block structure
  -> exact block layouts, transforms, and dense jobs
  -> Tenferro/Strided kernels on the selected storage
  -> checked result or caller-provided destination
```

The fusion-rule provider owns categorical labels and coefficients. symtn owns
fusion-tree semantics, recoupling, signs, block layouts, structural pack and
scatter decisions, and result validation. fusion-symbols generates SU(2) coefficient
data; symtn does not mirror its tables. Tenferro owns dense kernels,
factorizations, provider execution resources, and backend scheduling. Strided
owns applicable strided traversal and movement primitives.

`FusionTreeKey` identifies a basis state by uncoupled sectors, coupled sector,
dual flags, inner lines, and vertex multiplicities. All five fields participate
in equality and ordering; a cached hash only avoids recomputing the hash during
lookups. A key is a handle into an immutable tree arena owned by its holder: a
sector structure or a composed-coefficient cache entry owns one arena for all
of its trees, so a held key keeps that arena alive. `BlockStructure` keeps sector/tree identity alongside degeneracy
shapes, strides, and offsets. An owned dense tensor's reduced payload uses
that checked layout; compact diagonal and lazy-adjoint representations retain
their own structure instead of pretending to be dense storage.

Tree transforms compile source and destination identities, layout moves, and
recoupling coefficients into `TreeTransformStructure`. The structure contains
no tensor payload. Replay applies its Single moves, inactive destination fills,
and Multi pack/GEMM/scatter work to bound storage. General contraction can
also require source transforms and a destination transform around its core
block contraction. These are semantic steps, not extra user-visible operations.

This boundary lets symtn group small dense jobs *within* an operation. The
application chooses the sequence of operations and may design an
algorithm-specific fusion or batch. symtn does not run a second CPU scheduler
around concurrent Tenferro calls. A new fused public operation needs a
concrete measured cost and an explicit semantic contract.

## Reuse and placement

Where a split plan/workspace API exists, the immutable plan contains reusable
mathematical and layout structure. The mutable workspace owns execution
scratch and, where documented, an output buffer. Bindings and destinations
are checked per call. Batch size may affect workspace capacity or backend
artifacts; it does not define a structural plan's semantic identity.
`ContractPlan` and `ComposePlan` implement this split for stacked contraction
and composition; both execute with a `ContractWorkspace`, the executor state
the two operations share.
`EighFullPlan` / `EighFullWorkspace` implement it for stacked real Hermitian
eigendecomposition.

`StackedTensorMap` is an expert fixed-stride representation for homogeneous
members. It is not a required batch type for ordinary tensor operations.
General member-batched contract is not public API
([#1506](https://github.com/Ryo-wtnb11/symtn-rs/issues/1506)). Public prepared
APIs remain operation-specific; there is no general public work graph or
mandatory logical batch type.

Host is the broadest execution path. CUDA is explicit: a runtime attaches a
device, and `to_cuda()` / `to_host()` transfer tensor payloads. A device
operation either follows its supported path or reports an unsupported
capability; it does not silently execute through Host storage. The caller
chooses transfers between operations. See [backend policy](backend_policy.md)
for the current selection and resource contract.

## Source map

| Concern | Current source |
| --- | --- |
| Public tensor API and operation dispatch | [`symtn/src/typed.rs`](../symtn/src/typed.rs) and its `typed/` modules |
| Symmetry providers | [`symtn-sectors`](../symtn-sectors/) |
| Fusion-tree spaces and block structure | [`symtn-core`](../symtn-core/) |
| Structural operations and replay | [`symtn-operations`](../symtn-operations/) and [`symtn-tensors`](../symtn-tensors/) |
| Dense backend boundary | [`symtn-dense`](../symtn-dense/) |
| Network planning and contraction | [`symtn-network`](../symtn-network/) |

This page describes the ownership boundary, not a claim that every operation
works for every provider, scalar, or placement. The current API, method docs,
and tests decide supported combinations. [Documentation index](README.md)
separates these standing design rules from revision-pinned evidence.
