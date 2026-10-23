# symtn-sectors

Provider vocabulary for symtn's symmetry and category layers. A new symmetry
is a `fusion_symbols` category in `CategoryRule`, with `EnumeratedCodec` or a
`CategoryCodec` of its own; the provider traits (`FusionRule`, `SectorCodec`,
`CheckedFusionAlgebra`, and the capability traits) are sealed bounds that
generic code names. Built-in providers include U(1), Z/N, fZ2, SU(2), and
Fibonacci. `ProductFusionRuleExt` builds ordered, nested products: factor
order and association remain part of the provider and sector type.

The `cgc-gen` feature enables the generated SUN provider. It does not
turn arbitrary product labels or raw `SectorId` values into stable wire data.
