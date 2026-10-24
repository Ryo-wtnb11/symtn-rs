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
fusion-symbols A/B/C/D categories. It validates full global-form admission,
uses graded Dynkin-vector labels and checked Generic Host execution, and
forwards all symbols to fusion-symbols. Unsupported groups, including D2,
fail at construction with `LieGroupFusionRuleError::Provider`. The compact
`SU2FusionRule` keeps its multiplicity-free and physical-basis capabilities.
Raw `SectorId` values remain rule-local rather than stable wire data.
