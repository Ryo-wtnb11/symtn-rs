# Symmetry providers

A provider supplies categorical data for `GradedSpace<R>` and
`TensorMap<R, D, S>`. Built-in and application categories use the same
operation bounds. Facade imports are under `symtn::sector`.

This page defines the adapter, identity, codec and capability contracts. See
[design](design.md) for tensors and
[TensorKit's manual](https://docs.juliahub.com/General/TensorKit/stable/) for theory.

## Adding a symmetry

A symmetry is a `fusion_symbols::category::FusionCategory`: fusion-symbols owns
its mathematics (fusion, N, F, R, A, B, dimensions, twists, gauge
certificates). `CategoryRule<C, K>` turns a category `C` into a symtn provider
under a codec `K: CategoryCodec<C>`, which owns what symtn needs beyond the
mathematics: public labels, the `SectorId` numbering, `RuleIdentity`, sector
order, operation-specific representability, error translation, and the
admission mode. A codec contains no symbol formula.

For an enumerable category, use `EnumeratedCodec`:

```rust
use fusion_symbols::finite::ZN;
use symtn::sector::{CategoryRule, EnumeratedCodec, MultiplicityFreeAdmissionMode};

let z3 = CategoryRule::try_new(
    ZN::new(3)?,
    EnumeratedCodec::<MultiplicityFreeAdmissionMode>::new(),
)?;
```

Labels are category objects; IDs and sector order follow enumeration, and
fusion channels retain category order. Equal live category values share a
rule identity. It is weakly held: after all rules over that value are dropped,
old cache identities are not reused. The authority fingerprint alone is not
identity because it can be shared across category parameters.
`FusionAlgebraError` distinguishes invalid IDs, unrepresentable labels or
fusion outcomes, and category failures with their source errors.

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
an extension point. Native implementations exist only for the built-in rules
and, under the `testing` feature, for test fixtures that implement
`symtn_sectors::testing::Provider` (re-exported as
`symtn_core::testing::Provider`) to inject failures or instrument queries.

## The provider traits

`TypedSectorAdmission::Mode` selects one of two paths:

| Mode | Capabilities |
| --- | --- |
| `MultiplicityFreeAdmissionMode` | Infallible `FusionRule`, fallible `CheckedFusionAlgebra`, `SectorCodec` and multiplicity-free symbol/rigidity traits; label and error types must agree |
| `CheckedGenericAdmissionMode` | Checked fusion, rigidity, pivotal and symbol traits required by the operation; Host-only |

Each operation adds its coefficient and payload bounds. Outer multiplicity
(`N_ab^c > 1`) requires the checked Generic path.

Built-in rules keep their own types, labels and identities while delegating
elementary symbols to fusion-symbols. They are native implementations, not
`CategoryRule` aliases. `ProductFusionRule` composes its factors' symbols in
symtn. See [layering](layering.md#1-symbol-layer-fusion-symbols-crate).

## 1. `FusionRule`

[`FusionRule`](../symtn-sectors/src/algebra.rs) supplies identity, styles,
vacuum, duals and ordered fusion channels. IDs are provider-local, not
physical labels: the engine uses `RuleIdentity` to prevent incompatible rules
from mixing. Infallible queries may panic outside their domain; checked
construction must establish representability first.

**Styles are claims the engine acts on.** `Unique` means every product has
exactly one channel; `Simple` means several channels but no multiplicity;
`Generic` means multiplicity. Declaring `Unique` when a product has two channels
selects a lowering that silently drops terms. `Bosonic`/`Fermionic` assert
symmetric braiding; `Anyonic` and
`NoBraiding` do not, so ordinary `contract` (and network contraction) rejects
them, as TensorKit `blas_contract!` does, while `compose` admits every style.

`fusion_channels` order is part of the convention. Keep it stable; table
rules use table object order.

### `RuleIdentity`

`RuleIdentity` is defined in
[`rule_identity.rs`](../symtn-sectors/src/rule_identity.rs). Equal
identities promise that all categorical data that can affect layout or
coefficients agrees: styles, vacuum, channels, duals, F, R, dimensions, twists,
bends, and the codec. It is not, by itself, the complete key for every retained
network plan.

For a custom codec identity, choose by how the rule carries data:

| Constructor | Use |
| --- | --- |
| `RuleIdentity::of_type::<Self>()` | Unit provider with no answer-changing data |
| `RuleIdentity::from_canonical_bytes::<Self>(prehash, bytes)` | Canonical encoding of every answer-changing parameter/table |
| `RuleIdentity::new_unique::<Self>()` | Fresh identity when canonicalization is unavailable; correct but prevents reuse across equal instances |

A bare `TypeId` misses parameters; an address distinguishes equal allocations
and can be reused after a drop. Neither is a sufficient identity.

## 2. `CheckedFusionAlgebra`

`CheckedFusionAlgebra` supplies fallible dual, fusion-channel and N-symbol
queries over `SectorId`. Checked construction uses these domain checks.
Both algebra interfaces must agree; `CategoryRule` forwards both to the same
admitted query. Provider failure returns no tensor or partial factor tuple,
but successful catalog entries from earlier queries may remain.

## 3. `SectorCodec`

`SectorCodec` maps public labels to `SectorId` and back. Typed spaces and
tensors expose those labels; raw IDs remain an expert interface. For example,
`TableSector` is a one-based object label checked against its table rank.

Four laws:

1. `decode(encode(x)) == x` for every representable label;
2. `encode(decode(id)) == id` for every decodable id;
3. distinct labels never encode to the same id;
4. **decode is total over the reachable algebra**: `vacuum`, and every id your
   own `dual` and `fusion_channels` produce from decodable inputs, must decode.

A decode domain narrower than this closure is a codec defect: it rejects
IDs produced by the provider's own valid algebra. Encoding may reject user
labels outside the representable range.

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

`CategoricalScalar` is also the checked Generic coefficient bound. It
includes division for quantum-dimension factors in traces.

Argument order is TensorKitSectors' `Fsymbol(a,b,c,d,e,f)` / `Rsymbol(a,b,c)`
verbatim: `e` is the `(a,b)` channel, `f` is the `(b,c)` channel, `d` is the
total.

F and R are defined only on admissible fusion configurations. Callers must
gate with N symbols: off-shell behavior may be zero, another scalar or a
panic. The same precondition applies to A/B symbols. Products inherit their
components' preconditions.

`MultiplicityFreeRigidSymbols` adds `dim_scalar`, `inv_dim_scalar`,
`sqrt_dim_scalar`, `inv_sqrt_dim_scalar`, `twist_scalar`, and
`frobenius_schur_phase_scalar`. `a_symbol_scalar`
and `b_symbol_scalar` have correct default bodies derived from F; a
`CategoryRule` forwards the category's A and B instead.

`has_trivial_associator_gauge()` selects direct permutation only when every
allowed F coefficient is exactly the scalar unit. It defaults to false;
`CategoryRule` forwards the category certificate. Symmetric braiding alone
does not prove this property.

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

A catalog-bounded provider must reject any coupled sector whose complete
fusion-tree set leaves the catalog; silently keeping only the available trees
would change the tensor space. Checked Generic execution is Host-only; see
[backend policy](backend_policy.md) for device bounds.

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

Bounds are operation-specific; there is no blanket `Scalar = f64` requirement
on Host. Built-in and `CategoryRule` providers use the same public bounds.

- Multiplicity-free real-symbol rules admit supported `f32`, `f64`,
  `Complex32` and `Complex64` payloads. Checked Generic execution remains
  Host-only and uses the checked fusion, rigid, pivotal and symbol traits
  named by each operation.
- `TableRule` uses `Complex64` coefficients and payloads. Its Host scope
  includes transformations, composition, planar contraction, reductions,
  structural degeneracy operations and blockwise factorizations; see the
  [planar](../symtn/tests/table_planar_contract.rs) and
  [factorization](../symtn/tests/table_factorizations.rs) tests. Weighted
  reductions and truncation reject non-real quantum dimensions.
- Ordinary arbitrary-axis `contract` requires symmetric braiding. For
  anyonic rules use `planar_contract` with a crossing-free specification,
  or explicit `transpose` followed by `compose`. General table-rule network
  execution and prepared contraction/composition plans are not supported.
- A fusion-only table reports `NoBraiding`. Operations needing R or twist
  return a typed error before querying the provider.
- Uploading a complex-symbol payload with `to_cuda` provides storage only.
  Device operation bounds require `Scalar = f64`, excluding `TableRule` and
  checked Generic execution. Prepared batched `EighFullPlan` admits real
  payloads only.

## Validating a new category

symtn validates representability and structural shapes; it does not prove
the pentagon and hexagon coherence laws. Use fusion-symbols’
`pentagon_residual` and `hexagon_residual` for those checks.

- validate F/R/dimension/twist coefficients against an independent reference,
  and record the source, gauge, and convention used;
- use exact comparisons only under the [numerical policy](testing_numerics.md);
- cover invalid IDs at every checked entry point using the provider’s typed
  error (`FusionAlgebraError::InvalidSector` on the multiplicity-free path);
- cover the unit laws if the category claims `has_canonical_unit`, and the
  associator gauge if it claims `has_trivial_associator_gauge`;
- test an admitted transform round trip and a supported composition or
  contraction: wrong channel order or style can survive per-symbol tests.

See the [adapter tests](../symtn-sectors/tests/category_rule.rs) for
constructor, identity, codec and failure cases.
