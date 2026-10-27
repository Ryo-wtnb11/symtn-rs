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
multiplicity-free admission of a category whose fusion has multiplicity, a
fusion or braiding style symtn does not know (`IncompatibleFusionStyle`,
`UnsupportedBraidingStyle`), canonical-unit admission of a category without the `has_canonical_unit`
certificate (`UncertifiedUnitAdmission` admits it without the markers), and a
unit the codec cannot encode or round-trip; an invalid mode, error, or marker
combination does not compile.

The provider traits are sealed: they are the bounds generic code names, not
an extension point. Z2, Z_N, U(1), fermion parity, CU(1), `TableRule`,
`SU2FusionRule` and `LieGroupFusionRule` implement them through the same category
adapter as `CategoryRule`, with private codecs that keep their labels, ids,
identities and checked errors. `ProductFusionRule` uses the same adapter over
fusion-symbols' `Product`. Native implementations remain under `testing` for
fixtures that implement
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

Built-in rules keep their own types over the shared adapter, rather than being
`CategoryRule` aliases. [`TableRule`](../symtn-sectors/src/table_rule.rs)
serves every pinned AnyonWiki `TableCategory`, including Fibonacci, Ising and
fusion-only Rep(S₃), with identity built from the table's authority fingerprint.
The [abelian](../symtn-sectors/src/abelian.rs) and
[CU(1)](../symtn-sectors/src/cu1.rs) rules forward every symbol, including A/B;
U(1)'s physical entries also use the category's CGC.
The [SU2](../symtn-sectors/src/su2.rs) and
[Lie-group](../symtn-sectors/src/lie.rs) wrappers use `LieGroup` through the adapter.
symtn validates IDs before constructing category objects and queries the
category's `frobenius_schur_phase`.

| Lie wrapper | symtn codec and admission | mathematical authority |
| --- | --- | --- |
| `SU2FusionRule` | doubled spin `0..=254`, ascending channels, checked fusion-closure rejection above 254, multiplicity-free `Simple`; physical indices descend from `m=j` to `m=-j` | exact A1 `LieGroup` category; physical entries use the checked scalar CGC evaluator |
| `LieGroupFusionRule` | graded-lexicographic ids over Dynkin labels then zigzag torus charges, TensorKit `ProductSector` order keys per factor, channels sorted by encoded id, checked `Generic` at every rank, labels as representable as fusion-symbols makes them; no physical expansion | every `GroupId` that `LieGroup::new` executes (A/B/C/D factors, semisimple products, tori, U(n) and central quotients), with full character-lattice admission; exact A1/B1/C1 symbols |

Both keep their identities, sector order, typed errors and validation order.
`LieGroupFusionRule` keeps shaped empty F/R blocks for absent channels and rejects
nonfinite dimensions. It constructs the category eagerly, so unsupported
groups fail at construction, and reuses its bounded irrep interner: a warm hit clones
the provider's allocation-free handle, without a Dynkin-label `Vec`. The adapter
has no separate object cache. Encoding interns the irrep on a miss, so fusion-symbols checks
representability once per interned sector; while the sector is interned, a
warm encode is arithmetic and one lookup, without allocation; admission runs
in fusion-symbols on a miss. The interner holds at most `IRREP_TABLE_CAPACITY` (65,536)
sectors; past it, each encode or decode miss builds a fusion-symbols irrep
again. Provider generation and returned symbol/channel
storage have their own allocation costs; Generic blocks copy provider data.
Labels that fusion-symbols cannot represent (for example a rank-one doubled
spin above `u32::MAX` or a torus charge beyond `±i32::MAX`), and labels whose id
overflows `usize`, return `LieGroupFusionRuleError::UnrepresentableLabel`, both
on encoding and when a fusion produces them; a fusion overflow names the
coordinate-wise sum of its inputs, the product's highest weight.

Construct it with `LieGroupFusionRule::new(group)`, where `group` is a
`fusion_symbols::lie::GroupId`; `group_id()` borrows that complete metadata.
Supported families are SU(n), Sp(m), Spin(n), SO(n), their adjoint forms and
fs's supported central quotients, including half-spin forms. Sp's argument is
the even matrix dimension. D2 forms (Spin(4), SO(4), PSO(4)), semisimple
products, tori, U(n) and semisimple×torus quotients are admitted wherever
fusion-symbols executes them; groups it does not execute, such as exceptional
factors, return `LieGroupFusionRuleError::Provider(LieError::Unsupported { .. })`
at construction.

`encode_irrep(dynkin, charges)` and `decode_irrep(id)` take the label as
`LieGroup::irrep` does; the typed label is the single vector of Dynkin labels
followed by charges. The codec maps each charge by zigzag (`0, 1, -1, 2, -2, …`
to `0, 1, 2, 3, 4, …`) and ranks the whole coordinate vector by total grade
then lexicographically, so a single simple factor keeps its Dynkin-only ids.
`sector_order_key` does not use that rank: it lists one TensorKit position per
`GroupId` factor (a simple factor's own Dynkin rank, two SU(2) labels for D2,
and each charge's `U1Irrep` `findindex - 1`), so the rule orders sectors as
TensorKit orders the matching `ProductSector` and as a `ProductFusionRule` of
the same factors does. A warm key reads the interned irrep and allocates
nothing up to four positions, the key's inline capacity.
Global-form restrictions, including torus congruences such as U(n)'s, leave
holes in that numbering: encoding or decoding an excluded label returns
`LieGroupFusionRuleError::Provider(LieError::NotAdmissible)`, unless the
label's id already overflows, which returns
`LieGroupFusionRuleError::UnrepresentableLabel`. Object conversion
checks the complete category membership before using an interned label.
The identity is the `GroupId` value under the rule type: codec, admission and
scalar conventions are fixed by the type, and the category fingerprint by the
group and the build. Construction hashes the sparse canonical `GroupId` once
and comparison uses its equality, without building the character lattice.
Equal lattices share one identity whatever constructor or basis built them;
full `GroupId` equality distinguishes isomorphic A1/B1/C1 groups even when
their provider fingerprints agree. As for `CategoryRule`, the group is held
weakly: after all rules over it are dropped, old cache identities are not
reused. Persist semantic labels and rebuild caches when provider identity or
codec conventions change.

symtn retains only the bounded codec interner (65,536 label/object pairs and
its dual lookup), with uncached conversion after capacity is reached. It
computes no symbols and stores no symbol cache. fusion-symbols owns generated
coefficients and its B/C/D catalog's default 256 MiB retention budget; that
budget also charges the trivial and defining seeds, so construction fails for
B/C/D ranks whose defining generators exceed it (roughly rank 200), and it
does not bound temporary generation memory. The categorical scalar is
`f64`, and tensor execution uses the ordinary supported real/complex Host
payloads; physical expansion and CUDA execution are not provided by this rule.
Sp fundamental representations have FS indicator -1 and twist +1; this is
bosonic braiding, not fermionic parity.

## 1. `FusionRule`

[`FusionRule`](../symtn-sectors/src/algebra.rs) supplies identity, styles,
vacuum, duals and ordered fusion channels. IDs are provider-local, not
physical labels: the engine uses `RuleIdentity` to prevent incompatible rules
from mixing. Infallible queries may panic outside their domain; checked
construction must establish representability first.

**Styles are claims the engine acts on.** `Unique` means every product has
exactly one channel; `Simple` means several channels but no multiplicity;
`Generic` admits multiplicity and may also be selected conservatively, as
a `LieGroupFusionRule` over `GroupId::su(2)` does. Declaring `Unique` when a product has two channels
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
`frobenius_schur_phase_scalar`. Adapter-backed rules forward `a_symbol_scalar`
and `b_symbol_scalar` to the category's A and B. F-derived A/B defaults, both
scalar and Generic, exist only for fixtures in unit tests or under `testing`.

`has_trivial_associator_gauge()` selects direct permutation only when every
allowed F coefficient is exactly the scalar unit. It defaults to false;
`CategoryRule`, `TableRule` and products forward the category certificate.
`TableRule` also forwards the category's braiding style. Symmetric braiding
alone does not prove a trivial associator.

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
blocks (`GenericFArray`, `GenericRMatrix`) indexed by one-based vertex labels.
The sectors adapter validates returned F/R/A/B storage against logical axes
from checked N symbols on both admission paths. Wrong lengths produce
`CategoryAdapterError::MalformedSymbolData`; equal-volume wrong axes produce
`CategoryAdapterError::MalformedSymbolAxes { expected, actual }`. The codec
maps these to the provider's error type; `LieGroupFusionRule` exposes the corresponding
`LieGroupFusionRuleError` variants. An absent vertex may use empty storage or a
single zero, which becomes scalar zero or an empty Generic block. Empty
storage for a live vertex and nonzero storage for an absent vertex are errors.
Successful multiplicity-free validation borrows the block without allocating.

`LieGroupFusionRule` preserves its absent-channel admission before requiring a
finite dimension or an encoded derived dual. The tensor engine consumes the
category authority's A/B blocks through `try_a_symbol_generic` and
`try_b_symbol_generic`; it retains the dimension and Frobenius–Schur factors
of tree bends and folds. Production A/B accessors delegate directly to the
provider, whose admission owns error precedence and whose errors propagate
without core N, dual, F or rigidity queries. `LieGroupFusionRule` reuses its interned dual for
the column N; all-Lie-group product admission retains the factors' derived dual
objects for that query. A custom `CategoryCodec::admit` override still runs unchanged;
its returned inputs require a separate dual calculation. Adapters own
production symbol validation. Only in unit tests or under `testing`, core
queries the row N, dual, and column N before the provider and repeats the
categorical shape check, so failures in that prefix take precedence for
native synthetic providers that can bypass adapters. Public tree and index
guards always apply.
`CheckedGenericRigidSymbols::try_dim_scalar` defaults to
`try_sqrt_dim_scalar²`; override it when `dim` is known exactly (`LieGroupFusionRule` does),
because truncation budgets and quantum-dimension-weighted norms compare
`dim(c)` exactly, as TensorKit's integer dimensions do.

A catalog-bounded provider must reject any coupled sector whose complete
fusion-tree set leaves the catalog; silently keeping only the available trees
would change the tensor space.

`InfallibleGeneric<'a, R>` adapts infallible structural `FusionRule` queries
to the checked interface. Under `testing`, its rigid-symbol implementation
uses `GenericRigidSymbols::dim_scalar` for `try_dim_scalar`, which likewise
defaults to `sqrt_dim_scalar²`: override it for any integer dimension that is
not a perfect square. `LieGroupFusionRule` instead implements the checked Generic
traits directly; its bounded interner falls back to per-call conversion when
full.

Checked Generic execution is Host-only; CUDA typed execution is limited to
supported multiplicity-free paths over `f32`, `f64`, `Complex32` and `Complex64`.
See [backend policy](backend_policy.md) for device bounds.

## Products

`ProductFusionRule<Left, Right, Codec>` adapts
`fusion_symbols::product::Product<(Left::Category, Right::Category)>`.
Construct it with `ProductFusionRule::try_new(left, right)`; construction
requires the category's canonical-unit certificate and fusion and braiding
styles symtn knows. Every built-in rule,
`CategoryRule` and product implements the sealed `CategoryProvider` interface,
so products nest. Factor access and component-id packing remain in symtn.

One codec handles all four admission-mode pairs. `ProductAdmission` selects
multiplicity-free admission and `FusionAlgebraError` for two multiplicity-free
factors; any checked-Generic factor selects checked-Generic admission and
`ProductFusionRuleError`. Outer ids are decoded before the left factor's
operation-specific admission, then the right factor's. Generic errors retain
left/right factor errors and their sources, codec errors and unrepresentable
dimensions. Product category failures retain the original fusion-symbols
error as a `CategoryFailure` source, including multiplicity overflow and
malformed symbol data.

All product mathematics, including A/B, dimensions and scalar promotion,
comes from fusion-symbols. Fusion channels vary the first factor fastest in
the factors' native category order, including generated Lie categories; they
are never sorted by packed ids. Each F/R/A/B multiplicity axis varies the last
factor fastest, with row-major storage. No cross-factor fermionic sign is
added. Tensor operations use the factors' `SectorOrderKey`s for TensorKit
sector order independently of channel enumeration.

`TensorKitProductCodec` (Cantor pairing) is the default; use
`PackedProductCodec` for a fixed-width layout. Component order and association
are part of the Rust type, `ProductSector` label and `RuleIdentity`;
`U(1) ⊠ fZ2` and `fZ2 ⊠ U(1)` are different providers. Unlike TensorKit's flat
products, symtn retains explicit nesting.

Product construction clones factor category values; queries reconstruct objects
without a product-object cache. Generic F/R/A/B materialization allocates
product storage in fusion-symbols and copies its entries into symtn's owned
checked blocks. Scalar promotion does not widen tensor backend support:
checked-Generic tensor execution remains Host-only with `f64` coefficients.
Products gain no physical-dense expansion capability.

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

- validate F/R/A/B/dimension/twist coefficients against an independent reference,
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
