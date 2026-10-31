# Layering and ownership

Each layer owns one kind of knowledge and reaches the next through a narrow
interface. This document defines the ownership implemented in this stack.

| Concern | TensorKit ecosystem | This stack |
| --- | --- | --- |
| Elementary categorical data | TensorKitSectors, WignerSymbols, SUNRepresentations, CategoryData | [fusion-symbols](https://github.com/Ryo-wtnb11/fusion-symbols) |
| Symmetric tensors and fusion trees | TensorKit | symtn |
| Dense kernels and devices | LinearAlgebra, TensorOperations | Tenferro |
| Strided movement | Strided | strided-rs (`strided-kernel`) |

## Layers

```text
Application -> symtn facade / symtn-network
  -> symtn tensor structure and operations
     -> symtn-sectors -> fusion-symbols
     -> symtn-dense -> Tenferro
     -> Strided (structural data movement)
```

No layer depends upward. Applications own MPS, PEPS, VMC and transfer
schedules; symtn does not prescribe them.

### 1. Symbol layer (`fusion-symbols` crate)

fusion-symbols owns elementary fusion, multiplicities, dimensions, duals,
Frobenius–Schur data, twists, F/R/A/B/CGC data, gauges and coefficient caches.
It owns no symtn tensor, fusion tree, block layout or `SectorId` encoding.
symtn disables its default features; `cgc-gen` opts into `lie-generated`.

| Provider data | Boundary |
| --- | --- |
| Z2, Z_N, U(1), fermion parity, CU(1) | Built-in rules use the category adapter for every symbol, including A/B and U(1)'s CGC. symtn keeps labels, codecs, identities and checked errors. |
| Compact SU(2), general Lie groups | `SU2FusionRule` and `LieGroupFusionRule` use the adapter over `LieGroup` for fusion, N, F/R/A/B, dimensions, duals, twists and Frobenius–Schur phases. symtn keeps codecs, identities, ordering and checked admission. |
| AnyonWiki categories | `TableRule` uses the category adapter over pinned tables in their own gauge. |
| An application category | `CategoryRule<C, K>` adapts `FusionCategory`; the codec owns labels and admission, not symbol formulas. |
| Products | `ProductFusionRule` uses the category adapter over fusion-symbols' `Product` for all categorical data. symtn keeps factor admission, labels, codecs, identities and TensorKit sector order. |

Built-in rules retain their own types over the shared adapter; they are not
`CategoryRule` aliases. SU2 retains its compact doubled-spin codec and checked
scalar CGC entries in descending magnetic order. `LieGroupFusionRule` retains
its graded codec over Dynkin labels then zigzag torus charges and its bounded irrep interner, constructs its `LieGroup`
eagerly, validates complete `GroupId` admission and uses checked `Generic`
admission at every rank. Its identity is the `GroupId` value, held weakly as
for `CategoryRule`; equal fingerprints do not merge groups. Warm interner hits clone allocation-free
provider handles; returned blocks and channels have separate allocation costs.
See the [provider interface](provider_interface.md) for admission and error bounds.

The sectors adapter is the production boundary for validating returned
F/R/A/B storage, for both multiplicity-free and checked Generic providers.
It obtains logical axes from N, checks storage length and live-block axes,
and normalizes an absent vertex's bare zero. Malformed data follows the
codec's provider-error mapping. Core consumes the category authority's A/B
blocks; its categorical shape rechecks and F-derived A/B oracles are limited
to unit tests and `testing`. Tree bend/fold normalization and production
tree/index guards remain core responsibilities.

Elementary-symbol caches belong to fusion-symbols. Only the application may
reset or budget them. A fixed gauge constrains numerical values, not their
bits across backends or thread counts.

### 2. symtn (this repository)

symtn owns tensor maps, graded spaces, fusion trees, bends and recoupling,
sector/degeneracy structure, block layouts, contraction routing, structural
pack/scatter decisions and their structure caches. It delegates dense kernels,
factorizations, provider resources and backend scheduling.

| Crate | Owns |
| --- | --- |
| `symtn-sectors` | Providers, codecs and symbol adapters |
| `symtn-core` | Spaces, trees, checked fusion algebra, block structure and HomSpace-level caches |
| `symtn-operations` | Tree-transform structures, replay and dense-job descriptions |
| `symtn-tensors` | Contraction, trace, transforms and tree-transform caches |
| `symtn-matrixalgebra` | Per-sector factorization and factor-space authority |
| `symtn-dense` | Tenferro adapter, dense executors, sessions and placement |
| `symtn` | Typed facade, eager/prepared operations and `Runtime` |
| `symtn-network` | Labeled networks, planning, execution and plan cache |
| `symtn-macros` | `contract!` syntax and compile-time label validation |

### 3. tenferro and strided-rs (dense backend)

Tenferro owns dense tensors, GEMM and grouped GEMM, factorizations, CPU/GPU
providers, thread control and provider workspace. Strided owns strided
movement kernels. symtn submits grouped dense jobs or per-sector
factorizations through these boundaries; it does not embed replacement dense
kernels in operator code. Report backend gaps to the owning dependency.

## Inside symtn

Dependencies point from facade to execution to structure to providers.
`symtn-core` does not depend on dense or device execution; `symtn-dense`
depends on `symtn-core` only for the structure-cache wrapper and registry
that hold its GEMM plans. Normal dependency
edges are recorded in the crates' `Cargo.toml` files.

```text
contract (symtn)
  -> admit spaces and legs (symtn / symtn-core)
  -> contraction route (symtn-tensors)
  -> operand/output tree transforms (symtn-operations)
  -> grouped block GEMM (symtn-dense -> Tenferro)

svd / qr / eigh (symtn)
  -> admit leg roles and source
  -> factor spaces and per-sector stages (symtn-matrixalgebra)
  -> dense factorization (symtn-dense -> Tenferro; native compact SVD/QR:
     symtn-dense block kernel -> tlinalg on the engine's guarded linalg entry)
  -> publish factors in the validated spaces
```

The facade selects provider admission mode and storage kernels independently;
see [design](design.md) and [`symtn::typed`](../symtn/src/typed.rs).

**Who owns which cache**

Five process-global, byte-bounded structure caches share the wrapper and
registry in [`symtn-core/src/cache.rs`](../symtn-core/src/cache.rs).
Only the application calls `symtn::cache::{stats, clear, configure_budgets}`
to observe, clear or size them.

| Cache | Key distinctions | Owner |
| --- | --- | --- |
| Sector structure | Rule, mode, sectors and duality; no degeneracies | `symtn-core` |
| Complete HomSpace / block structure | Rule, mode and full HomSpace, including degeneracies | `symtn-core` |
| Completed tree transformer | Rule, mode, coefficient type, operation/scope, orientation, basis/storage conventions and source/destination content identities | `symtn-tensors` |
| Composed transformation coefficients | Rule, mode, coefficient type, operation/scope, orientation and ordered source tree group; no degeneracies or content identities | `symtn-tensors` |
| Dense GEMM plans | Scalar type, block extents, effective A/B/D strides and conjugation; no offsets, values, coefficients or engine settings | `symtn-dense` (native provider) |

A degeneracy-only change can reuse sector structure and composed coefficients.
Unique fusion bypasses the composed-coefficient cache. Checked Generic
transforms publish staged coefficient groups only after destination commit.
Complete HomSpace entries own canonical block structures; completed-transformer
entries retain replay cores without retaining those structures.

Network contraction-order caches, per-call CUDA copies of prepared structures
and the engine's cuTENSOR plans are execution resources, separate from these
five.
Elementary-symbol caches remain in fusion-symbols. symtn writes no
persistent tree-transform plans; network-order persistence is explicit.

## Interfaces between layers

| From → to | Through | Data |
| --- | --- | --- |
| symtn → symbols | `symtn-sectors` | Categorical queries translated to symtn labels, IDs, errors and scalars |
| symtn → Tenferro | `symtn-dense` | Dense jobs and per-sector factorization requests |
| symtn → Strided | `symtn-operations`, `symtn-tensors` | Structural permutation, pack and scatter |

Only `symtn-sectors` imports fusion-symbols as a normal dependency, and only
`symtn-dense` imports Tenferro. Test-only dependencies may exercise symbol
cache controls. Application code owns cache budgets and thread policy;
dependency changes require explicit approval under [AGENTS.md](../AGENTS.md).

## Current deviations

- `symtn-dense` and `symtn::typed` re-export Tenferro's `Engine`, `CpuEngine`
  and `Pool`, the caller-owned execution resources every numerical
  operation takes.
- `symtn-dense` and `symtn::typed` re-export Tenferro's `CudaEngine` and
  `CudaDeviceId` (feature `cuda`), the caller-owned device engine a device
  operation borrows; see [backend policy](backend_policy.md#parallel-execution).
