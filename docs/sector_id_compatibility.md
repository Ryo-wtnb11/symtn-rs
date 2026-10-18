# Sector ID and storage compatibility

`SectorId` is an opaque, rule-local identifier. Its numeric `usize` value is
not a semantic irrep label, serialization format, or cross-version API.
Persist semantic labels such as `(charge, parity)` or
`(parity, charge, twice_spin)` instead.

## Product-sector codecs

The canonical `ProductFusionRule` is generic over its codec and defaults to
`TensorKitProductCodec`. `PackedProductCodec` remains available when an
application explicitly chooses a fixed-width layout. A provider differing only
in its codec labels the same category with different numeric IDs and a
different `RuleIdentity` — which is exactly why those IDs are not a wire format.

Changing the codec preserves:

- sector labels and degeneracies;
- fusion, duality, F, R, and pivotal semantics;
- tensor results when inputs are matched by semantic fusion-tree labels.

Changing the codec can change:

- numeric `SectorId::id()` values;
- the order returned by `GradedSpace::sectors()`;
- block and reduced-data storage order;
- serialized raw IDs or raw storage buffers;
- the tensor produced by `TensorMap::rand_with_seed` for the same seed.

`rand_with_seed` fills internal storage order. It is reproducible within a
fixed symtn version and layout, not a cross-version semantic fixture. Use
`TensorMap::from_subblock_fn` and inspect provider-labelled fusion trees plus local indices when an input
must remain identical across codec or layout changes.

`GradedSpace::sectors()` is likewise ordered by current internal IDs. Consumers
that need stable behavior must key by the provider's semantic sector type or
apply their own ordering rather than relying on the returned `Vec` order.

## Sector order where TensorKit's order is observable

symtn sorts by `SectorId` for storage, but a result that depends on sector
order must use TensorKit's. Today one result does: which sector a truncation
keeps at an exact cross-sector tie. TensorKit 0.17.1
`findtruncated(::SectorVector, ...)` (`src/factorizations/truncation.jl`)
sorts the flat values of a `SectorVector` with `sortperm` or
`partialsortperm`, whose ties go to the lower flat index. The flat order is
the block order, sorted by TensorKitSectors `isless`. So `truncrank` keeps the
`isless`-earlier sector first, and `truncerror` discards it first.
`truncspace` and `trunctol` decide each sector on its own, so they have no
cross-sector tie. `select_truncation` takes its decision in the order of
`FusionRule::sector_order_key` (or `CheckedGenericFusion::sector_order_key`),
which is TensorKitSectors `findindex - 1` per factor, ordered by degree and
then lexicographically, as TensorKitSectors 0.3.9 `isless` does for products
(`src/product.jl`).

| Provider | TensorKit order (TensorKitSectors 0.3.9) | Key | `SectorId` monotone? |
| --- | --- | --- | --- |
| `U1FusionRule` | `isless(::U1Irrep)`: 0, +1, -1, +2, -2, ... (`src/irreps/u1irrep.jl`) | `4q - 1` for `q > 0`, `4\|q\|` otherwise (TensorKit also enumerates half-integers) | no: zigzag ids 0, -1, +1, ... |
| `ZNFusionRule`, `Z2FusionRule` | `isless(c1.n, c2.n)` (`src/irreps/znirrep.jl`) | id = `n` | yes |
| `FermionParityFusionRule` | even < odd (`src/fermions.jl`) | id = parity | yes |
| `SU2FusionRule` | `isless(s1.j, s2.j)` (`src/irreps/su2irrep.jl`) | id = `2j` | yes |
| `CU1FusionRule` | `j`, then `s` at `j = 0`: `0+ < 0- < 1/2 < 1 < ...` (`src/irreps/cu1irrep.jl`) | id = `findindex - 1` | yes |
| `TableRule` | Fibonacci: unit < tau (`src/anyons.jl`). Other entries: AnyonWiki object order, which for Ising is `1, ψ, σ` against TensorKit's `I, σ, ψ` | id | yes |
| `ProductFusionRule` (any codec, any nesting) | `isless(::ProductSector)`: degree `sum(findindex) - n`, then lexicographic over the flattened factors (`src/product.jl`) | factor keys concatenated | no in general |
| `SUNFusionRule` (checked Generic) | SUNRepresentations 0.4.0 `isless(::SUNIrrep)`: Dynkin-label sum, then lexicographic (`src/sunirrep.jl`) | id (encoded in that order) | yes |

A provider without a TensorKit counterpart keeps the default key, its
`SectorId`, and its ties follow id order. A provider whose ids do not ascend in
TensorKit order must override the hook.

## Caches

Codec types participate in fusion-rule identity, so in-process symtn caches do
not reuse Cantor plans as packed plans. Complete tree-transform plans and
structures are execution-context-local; source-column recoupling rows are
compile-local and are not retained across plan misses; symtn persists no
tree-transform plan files. Raw-ID-keyed application caches
must likewise be rebuilt from semantic labels.

This does not change `symtn-network::{save_plan_cache, load_plan_cache}`.
Network contraction-order persistence remains an explicit application opt-in
and stores topology-derived optimizer output, not lowered tree-transform
execution plans.

## Explicit codec choice

`TensorKitProductCodec` is the default codec type parameter of
`ProductFusionRule`. Code that requires a packed numeric layout can name
`PackedProductCodec` explicitly. Neither numeric representation is a semantic
wire format.

Do not mix sectors, spaces, plans, or caches produced by the two codecs. Their
rule identities are distinct even when decoded semantic labels agree.

## Target width

The packed layouts reserve:

- 33 bits for `U(1) x fZ2`;
- 41 bits for `fZ2 x U(1) x SU(2)`.

A 64-bit target is therefore required for these packed product spaces and
for representing the complete encoded `i32` U(1) label set. On a narrower
target the checked codec and space constructors return a width error rather
than truncating, wrapping, or overlapping component bits.

The representable U(1) charge range is `i32::MIN + 1 ..= i32::MAX`.
`U1Irrep::try_new(i32::MIN)` returns `None`, and `U1Irrep::new(i32::MIN)`
panics because that charge has no representable dual. Addition outside that
range, including a valid sum equal to `i32::MIN`, returns
`FusionAlgebraError::U1FusionOverflow`. symtn does not wrap, saturate, or
reinterpret these charges.
