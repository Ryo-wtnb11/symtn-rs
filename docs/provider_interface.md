# Symmetry providers

A symtn *provider* is a Rust value that supplies categorical data: fusion
channels, duals, F/R/A/B symbols, and quantum dimensions. Tensor operations
select the provider traits they need, and every provider, built-in or added by
an application, uses the ordinary `GradedSpace<R>` / `TensorMap<R, D, S>` API
under the same public bounds. Through the `symtn` facade, every trait and type
named here is imported from `symtn::sector`.

This document is the contract. It says how a new symmetry enters symtn, what
each provider trait owns, which laws the engine assumes without checking, and
the current typed API boundaries.

Read [`symtn/src/mathematics.md`](../symtn/src/mathematics.md) first if you need
the tensor-map convention itself (`codomain <- domain`, duality, block layout).
That document defines the semantics; this one defines the provider interface.

## Adding a symmetry

A symmetry is a `fusion_symbols::category::FusionCategory`: fusion-symbols owns
its mathematics (fusion, N, F, R, A, B, dimensions, twists, gauge
certificates). `CategoryRule<C, K>` turns a category `C` into a symtn provider
under a codec `K: CategoryCodec<C>`, which owns what symtn needs beyond the
mathematics: public labels, the `SectorId` numbering, `RuleIdentity`, sector
order, operation-specific representability, error translation, and the
admission mode. A codec contains no symbol formula.

For a category that provides object enumeration (`object_count`,
`object_at`, `object_index`), `EnumeratedCodec` is the codec:

```rust
use fusion_symbols::finite::ZN;
use symtn::sector::{CategoryRule, EnumeratedCodec, MultiplicityFreeAdmissionMode};

let z3 = CategoryRule::try_new(
    ZN::new(3)?,
    EnumeratedCodec::<MultiplicityFreeAdmissionMode>::new(),
)?;
```

Its labels are the category's objects, an id is the object's enumeration
index, fusion channels keep the category's order, and the sector order is the
enumeration order. Equal category values give equal rule identities, so two
independently constructed rules that are alive together share caches and mix
in one tensor. The identity refers to the rule's category weakly: a cache key
never keeps a dropped rule's category alive, and once every rule over a
category value is dropped its identity equals no other, so its cache entries
are only evicted, never reused. The
fusion-symbols authority fingerprint is never used as identity, because it is
shared across parameters such as the ZN modulus. Its error is
`FusionAlgebraError`: an id that names no object is `InvalidSector`, a foreign
label or one whose index does not fit an id is `UnrepresentableSectorLabel`, a
fusion outcome that does not fit an id is `FusionNotRepresentable`, and a
category failure or a category without enumeration is `CategoryFailure`,
which keeps the source error. A Deligne product
`fusion_symbols::product::Product<(A, B)>` is itself an enumerable category.

Implement `CategoryCodec` only for labels, numbering, or validation order of
your own. Its `admit` hook admits an operation's inputs (`CategoryOperation`)
and may check representability in any order; `admit_r` may admit an R block
known to be zero without converting every input; `rule_identity` returns a
complete canonical identity or `None` for a fresh identity per constructed
rule (shared by its clones).

The codec's type parameters select the admission:

| codec selector | provider traits of the rule |
| --- | --- |
| `Mode = MultiplicityFreeAdmissionMode`, `Error = FusionAlgebraError` | `FusionRule`, `CheckedFusionAlgebra`, `SectorCodec`, `MultiplicityFreeFusionRule`, `MultiplicityFreeFusionSymbols`, `MultiplicityFreeRigidSymbols` |
| `Mode = CheckedGenericAdmissionMode` | `CheckedGenericFusion` (always reporting `Generic` fusion), `CheckedGenericRigidSymbols`, `CheckedGenericPivotal` |
| `Unit = CanonicalUnitAdmission` | additionally `CanonicalUnitFusionRule` or `CheckedCanonicalUnitFusionRule` |

`EnumeratedCodec<M, U = CanonicalUnitAdmission>` takes the mode and the unit
selector as `M` and `U`. Every rule implements `TypedSectorAdmission` with its
codec's labels, error, and mode. `CategoryRule::try_new` refuses
multiplicity-free admission of a category whose fusion has multiplicity,
canonical-unit admission of a category without the `has_canonical_unit`
certificate (`UncertifiedUnitAdmission` admits it without the markers), and a
unit the codec cannot encode or round-trip; an invalid mode, error, or marker
combination does not compile.

The provider traits are sealed: they are the bounds generic code names, not
an extension point. Z2, Z_N, U(1), fermion parity, CU(1) and `TableRule`
implement them through the same category adapter as `CategoryRule`, with
private codecs that keep their labels, ids, identities and checked errors.
Native implementations exist only for `SU2FusionRule`, `SUNFusionRule` and
`ProductFusionRule` and, under the `testing` feature, for test fixtures that implement
`symtn_sectors::testing::Provider` (re-exported as
`symtn_core::testing::Provider`) to inject failures or instrument queries.

## The provider traits

The typed API selects one of two admission modes through
`TypedSectorAdmission::Mode`:

| mode | required provider capabilities | scope |
| --- | --- | --- |
| multiplicity-free | `MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec` and `TypedSectorAdmission<Sector = <R as SectorCodec>::Sector, Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>`, plus the coefficient and payload bounds of the operation | ordinary typed construction and multiplicity-free operations |
| checked Generic | `TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>` plus the checked Generic fusion, rigidity, pivotal, and symbol traits required by each operation | outer multiplicity with fallible provider queries; Host-only |

For the multiplicity-free mode, `FusionRule` supplies the infallible algebra,
`CheckedFusionAlgebra` validates the representable domain, and `SectorCodec`
maps public labels to engine sector ids. The multiplicity-free symbol and
rigidity traits supply F/R coefficients, dimensions, twists, and bends. A
provider with outer multiplicity (`N_ab^c > 1`) uses the checked Generic traits;
see [Outer multiplicity](#outer-multiplicity).

The built-in rules keep their own types, labels, and identities:
`symtn-sectors/src/table_rule.rs` is one provider over every pinned AnyonWiki
table of `fusion_symbols::table::TableCategory` (Fibonacci, Ising, and
fusion-only categories such as Rep(S₃)), whose `RuleIdentity` is built from
the table's authority fingerprint; `symtn-sectors/src/abelian.rs` has the
abelian ones; `symtn-sectors/src/su2.rs` delegates its coefficients to
fusion-symbols. fusion-symbols 0.7.0 supplies mathematics through
`category::FusionCategory`; symtn validates sector ids before constructing
objects in their owning category and uses `frobenius_schur_phase`.
`SUNFusionRule` accepts shaped empty F/R blocks for absent channels. Its SU(2)
Dynkin label is restricted to `0..=u32::MAX` and uses the provider's exact A1
authority; `SU2FusionRule` retains its compact `0..=254` codec and descending
magnetic basis. SU(N) codecs and identities for N >= 3 are unchanged.

## 1. `FusionRule`

`FusionRule` in `symtn-sectors/src/algebra.rs` is the engine's infallible
algebra interface.

```rust
fn rule_identity(&self) -> RuleIdentity;
fn fusion_style(&self) -> FusionStyleKind;      // Unique | Simple | Generic
fn braiding_style(&self) -> BraidingStyleKind;  // NoBraiding | Bosonic | Fermionic | Anyonic
fn vacuum(&self) -> SectorId;
fn dual(&self, sector: SectorId) -> SectorId;   // defaults to identity
fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec;
```

**`SectorId` is opaque and provider-local.** It is an index your codec assigns,
not a physical charge and not a portable label. Two providers may use the same
numeric value for unrelated sectors; nothing in the type system prevents mixing
them, so the engine relies on `RuleIdentity` instead (below).

**The infallible methods carry a domain precondition.** They may panic on an id
your provider cannot represent. The engine passes ids encoded or generated by
the provider after checked construction has accepted them. The checked
interface is where the real domain check lives.

**Styles are claims the engine acts on.** `Unique` means every product has
exactly one channel; `Simple` means several channels but no multiplicity;
`Generic` means multiplicity. Declaring `Unique` when a product has two channels
selects a lowering that silently drops terms. `Bosonic`/`Fermionic` assert
symmetric braiding (`R` squares to the identity up to signs); `Anyonic` and
`NoBraiding` do not, so ordinary `contract` (and network contraction) rejects
them, as TensorKit `blas_contract!` does, while `compose` admits every style.

**`fusion_channels` order is part of your convention.** Keep it stable and match
your reference. `TableRule` in `symtn-sectors/src/table_rule.rs`, for
example, lists channels in the table's object order, which for Fibonacci is
TensorKitSectors' iteration order.

### `RuleIdentity`

`RuleIdentity` is defined in `symtn-sectors/src/rule_identity.rs`. Equal
identities promise that all categorical data that can affect layout or
coefficients agrees: styles, vacuum, channels, duals, F, R, dimensions, twists,
bends, and the codec. It is not, by itself, the complete key for every retained
network plan.

A `CategoryCodec` that returns its own identity picks one of three
constructors by how the rule carries its data:

- `RuleIdentity::of_type::<Self>()` — the provider is a unit struct with no
  data that can change an answer.
- `RuleIdentity::from_canonical_bytes::<Self>(prehash, bytes)` — the provider
  carries parameters or tables. Serialize everything that can change an
  answer. Shipped ZN and SU2 providers use canonical identities.
- `RuleIdentity::new_unique::<Self>()` — you cannot canonicalize the data. Each
  allocation becomes its own identity, so nothing is shared between two
  instances that are in fact equal. Correct, and slower.

Do not use `TypeId` or the provider's address directly: one Rust type can hold
different tables, and two allocations of the same immutable rule must be able to
share cache entries.

## 2. `CheckedFusionAlgebra`

`CheckedFusionAlgebra` in `symtn-sectors/src/algebra.rs` provides the same core
algebra queries as fallible operations. This is where a provider validates that
an incoming `SectorId` is in its domain:

```rust
fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError>;
fn try_fusion_channels(&self, l: SectorId, r: SectorId) -> Result<SectorVec, FusionAlgebraError>;
fn try_nsymbol(&self, l: SectorId, r: SectorId, c: SectorId) -> Result<usize, FusionAlgebraError>;
```

Checked construction runs through these methods. A provider may retain
successful catalog entries created before a later failure. The failed operation
still returns no symtn tensor or partial factor tuple.

Both interfaces must agree. A `CategoryRule` answers both from one admitted
category query; a native test provider implements the checked logic once and
forwards the infallible methods to the `dual_or_panic`,
`fusion_channels_or_panic`, and `nsymbol_or_panic` helpers.

## 3. `SectorCodec`

`SectorCodec` in `symtn-sectors/src/algebra.rs` defines the public label type
and its translation to `SectorId`. Without it the provider can still work at
the expert layer but is not reachable through `GradedSpace`/`TensorMap`,
because those return labels, never ids. `TableRule` implements this boundary
with `TableSector`, the table's one-based object label, rejecting labels outside
the rule's rank; its separate typed-tensor restriction is the complex
coefficient scalar described below.

Four laws:

1. `decode(encode(x)) == x` for every representable label;
2. `encode(decode(id)) == id` for every decodable id;
3. distinct labels never encode to the same id;
4. **decode is total over the reachable algebra**: `vacuum`, and every id your
   own `dual` and `fusion_channels` produce from decodable inputs, must decode.

Law 4 is the one that bites. A decode domain narrower than that closure is a
codec bug, not a user error: the engine hands you back exactly the ids your own
fusion produced, so a facade inspecting a legitimately built tensor would fail
on your own output. Encoding may still reject labels outside your representable
range — that value came from the user.

## Multiplicity-free coefficients

`MultiplicityFreeFusionRule` in `symtn-sectors/src/algebra.rs` is an empty
marker asserting
`N_ab^c ∈ {0, 1}`.

`MultiplicityFreeFusionSymbols` declares the coefficient
scalar and the two symbols:

```rust
type Scalar: CategoricalScalar + Send + Sync;
fn f_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId,
                   d: SectorId, e: SectorId, f: SectorId) -> Self::Scalar;
fn r_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> Self::Scalar;
```

`CategoricalScalar` (shared with `CheckedGenericRigidSymbols::Scalar`) needs
`+`, `*` and `/`: the trace channel factor divides quantum dimensions,
`dim(c) / dim(first)`, as TensorKit does.

Argument order is TensorKitSectors' `Fsymbol(a,b,c,d,e,f)` / `Rsymbol(a,b,c)`
verbatim: `e` is the `(a,b)` channel, `f` is the `(b,c)` channel, `d` is the
total.

Both symbols are defined only for admissible labels: every `N`-symbol of the
configuration is nonzero (for `R`, `N(a,b,c)`; for `F`, `N(a,b,e)`,
`N(e,c,d)`, `N(b,c,f)`, `N(a,f,d)`), i.e. the labels come from a valid fusion
tree. Off-shell values are unspecified — a provider may return any scalar, `0`,
or panic — and callers gate with `nsymbol`, as the engine does. Do not rely on
a zero. TensorKitSectors multiplies by or gates on the `Nsymbol`s, so its
off-shell value is `0`, and so is that of every shipped symtn provider (the
Abelian, CU(1) and SU(2) rules through fusion-symbols, `TableRule` from its
table); a custom provider need not follow. `ProductFusionRule` multiplies its components' values, so it inherits each
component's off-shell value and panics. `a_symbol_scalar`, `b_symbol_scalar`
inherit the contract: they are defined for `N(left,right,coupled) != 0`.

`MultiplicityFreeRigidSymbols` adds `dim_scalar`, `inv_dim_scalar`,
`sqrt_dim_scalar`, `inv_sqrt_dim_scalar`, `twist_scalar`, and
`frobenius_schur_phase_scalar`. `a_symbol_scalar`
and `b_symbol_scalar` have correct default bodies derived from F; a
`CategoryRule`, and every built-in rule backed by the category adapter,
forwards the category's A and B instead.

One capability flag, defaulting to the conservative answer (a `CategoryRule`
forwards the category's certificate; `TableRule` withholds it, as it reports
every braided table `Anyonic`):

- `has_trivial_associator_gauge()` — return `true` only if
  *every* allowed F coefficient in your gauge is exactly the scalar unit. It
  selects a direct permutation lowering. Symmetric braiding alone is not enough.

`CanonicalUnitFusionRule` asserts the vacuum is self-dual,
that `1 ⊗ a -> a` and `a ⊗ 1 -> a` are the only channels involved, and that
unitors and associators act as the identity there. `TableRule` shows `Simple`
providers that still qualify (`symtn-sectors/src/table_rule.rs`).

## Outer multiplicity

If `N_ab^c` can exceed 1, the provider implements `CheckedGenericFusion`,
`CheckedGenericRigidSymbols`, the checked pivotal/symbol capabilities needed by
the operations, and `TypedSectorAdmission` with
`Mode = CheckedGenericAdmissionMode`; a `CategoryRule` does so when its codec
selects that mode. These traits are deliberately independent of `FusionRule`
so that a provider with a finite generated catalog can use the checked path
while reporting unavailable data as an error. F, R, A, and B become dense
blocks (`GenericFArray`, `GenericRMatrix`) indexed by one-based vertex labels;
a `CategoryRule` takes each block's shape from the checked N symbols, so the
single zero fusion-symbols stores for an absent vertex becomes an empty block,
and a nonzero entry where N admits none is an error. The tensor engine
derives its A/B moves from the validated F and rigidity data;
`try_a_symbol_generic` and `try_b_symbol_generic` expose the authority's A
and B blocks.
`CheckedGenericRigidSymbols::try_dim_scalar` defaults to
`try_sqrt_dim_scalar²`; override it when `dim` is known exactly (SU(N) does),
because truncation budgets and quantum-dimension-weighted norms compare
`dim(c)` exactly, as TensorKit's integer dimensions do.

`InfallibleGeneric<'a, R>` adapts an existing infallible
provider to the checked interface; see `symtn-sectors/src/sun.rs` for a
catalog-bounded provider. Its `try_dim_scalar` is the testing-tier
`GenericRigidSymbols::dim_scalar`, which likewise defaults to
`sqrt_dim_scalar²`: override it for any integer dimension that is not a
perfect square.

Current checked Generic coverage is defined by the public bounds and executable
tests at this revision. It is a Host path; CUDA typed execution is
limited to supported multiplicity-free paths over `f32`, `f64`, `Complex32`
and `Complex64`. The
[operation matrix](audit/operation-matrix.md) is retained only as
revision-pinned historical evidence.

## Products

`ProductFusionRule<Left, Right, Codec>` is itself a provider, so products nest.
The product's coefficient scalar is the promotion of its components'
(`PromoteCoefficientScalar`), following TensorKitSectors'
`fusionscalartype(ProductSector)`: components need not agree, and a complex
component widens the product. `TensorKitProductCodec`, the Cantor-pairing codec,
is the default type parameter. Use `PackedProductCodec` explicitly when you
need its fixed-width layout. Component order and association
are part of the Rust type and of the `ProductSector` label; `U(1) ⊠ fZ2` and
`fZ2 ⊠ U(1)` are different providers, never automatically equivalent.

## Current typed API boundaries

- The multiplicity-free root uses operation-specific coefficient and payload
  bounds; there is no blanket `Scalar = f64` requirement.
- `TableRule` has `Complex64` categorical coefficients. Its tested typed
  scope is construction, transformations including explicit planar braids,
  tensor products, composition, `scale`, `scale_assign`, `axpby`,
  `axpby_into`, the adjoint (eager, lazy, and as a `compose` operand; a
  transform of a lazy adjoint reads the parent through the dagger
  operation, as for real coefficients), the quantum-dimension-weighted
  reductions `scalar`, `norm`, `inner`, `tr` and `GradedSpace::dim`, unit
  legs, `restrict_leg` (also batched), `is_diagonal`, and the structural
  degeneracy operations behind the same bounds: compact
  `TensorMap::diagonal` and `expert::diagonal_spectrum`, `embed_leg`, `cat`
  along either side, and serialization decode (`from_bytes_with`), and the
  blockwise factorizations: `svd_vals`/`svd_compact`/`svd_full`, QR and LQ
  (compact and full), left and right polar, `eigh_vals`/`eigh_full`,
  `eig_vals`/`eig_full`, `left_null`/`right_null`, `exp`, `inv`, `pinv`,
  `solve` and `map_diagonal`, together with truncation
  (`GradedSpace::find_truncated`, then `restrict_leg`), and on the Host
  `twist` and `flip` (also of a lazy adjoint and, for `twist`, of a compact
  diagonal), with θ, χ and their inverses θ̄, χ̄ in `Complex64` as in
  TensorKit. Those weights,
  truncation included, read the real
  quantum dimension: a `Complex64` dimension with a nonzero imaginary part
  is refused with `InvalidArgument`, since no unitary category has one.
  `TableRule` payloads are `Complex64` only, so the prepared batched
  `EighFullPlan`, which admits real payloads only, refuses them with its
  typed error. Ordinary arbitrary-axis contraction, `trace_pairs`, prepared
  contraction and composition plans, and general network
  execution are not supported or claimed; a planar contraction is the
  crossing-free `transpose` of each operand followed by `compose`, as in
  TensorKit's `planarcontract!`. A complex-symbol tensor's payload can be
  uploaded with `to_cuda` (storage only), but every device operation,
  `twist` included, needs `Scalar = f64` symbols, so device use is a
  compile-time boundary.
  A fusion-only table reports `NoBraiding`: every operation that needs an
  R-symbol or a twist is rejected with its typed error before any provider
  call, and `r_symbol_scalar`/`twist_scalar` are defined only for a braided
  rule.
- Checked Generic typed execution is Host-only. Representative tested scope
  includes construction, transforms, tensor products, contraction and
  composition (a lazy adjoint operand read through its parent's payload, as
  in the multiplicity-free mode), trace, static N-ary networks, and explicit
  sliced execution. Operations
  require the exact checked fusion, rigidity, pivotal, and symbol traits they
  use; borrowed or compact routes are only admitted where their direct owned
  contract says so. CUDA does not support Checked Generic providers.
- Multiplicity-free providers still implement overlapping infallible
  `FusionRule` and fallible `CheckedFusionAlgebra` methods.

Shipped providers do not use a private operation capability that a
`CategoryRule` lacks. The same public trait bounds apply to both.

## Validating a new category

symtn validates representability and structural shapes but does not prove
categorical coherence laws such as the pentagon and hexagon identities;
fusion-symbols provides `pentagon_residual` and `hexagon_residual` for that.

- validate F/R/dimension/twist coefficients against an independent reference,
  and record the source, gauge, and convention used;
- pin exact values where the reference is exact (the Fibonacci golden
  compares bit patterns, not tolerances);
- cover the boundary: an unrepresentable id must produce
  `FusionAlgebraError::InvalidSector` from every checked entry point;
- cover the unit laws if the category claims `has_canonical_unit`, and the
  associator gauge if it claims `has_trivial_associator_gauge`;
- then run one end-to-end tensor identity — a permute round trip and a
  contraction — because a wrong `fusion_channels` order or a wrong style claim
  survives every per-symbol test.
