# symtn-sectors

Provider vocabulary for symtn's symmetry and category layers. A new symmetry
is a `fusion_symbols` category in `CategoryRule`, with `EnumeratedCodec` or a
`CategoryCodec` of its own; the provider traits (`FusionRule`, `SectorCodec`,
`CheckedFusionAlgebra`, and the capability traits) are sealed bounds that
generic code names. Built-in providers include U(1), Z/N, fZ2, SU(2), and
Fibonacci. `ProductFusionRule::try_new` builds ordered, nested products over
fusion-symbols' `Product`: factor order and association remain part of the
provider and sector type. Multiplicity-free factors select multiplicity-free
admission; a checked-Generic factor selects checked-Generic admission.

The `cgc-gen` feature enables `LieGroupFusionRule::new(GroupId)` over
every group fusion-symbols' `LieGroup` executes: A/B/C/D factors, semisimple
products, tori, U(n) and their central quotients. It validates full
global-form admission, ranks Dynkin labels then zigzag-mapped torus charges
by graded lexicographic order, uses checked Generic Host execution, and
forwards all symbols to fusion-symbols. Groups it does not execute, such as
exceptional factors, fail at construction with `LieGroupFusionRuleError::Provider`. The compact
`SU2FusionRule` keeps its multiplicity-free and physical-basis capabilities.
Raw `SectorId` values remain rule-local rather than stable wire data.
