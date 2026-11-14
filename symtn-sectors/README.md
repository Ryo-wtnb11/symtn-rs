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

`LieGroupFusionRule::<M>::new(GroupId)` builds a Lie-group rule in admission
mode `M`. Checked Generic mode (`CheckedGenericAdmissionMode`, the default)
builds every group fusion-symbols' `LieGroup` executes: factors of every simple type, A–G, semisimple products,
tori, U(n) and their central quotients. Without the `cgc-gen` feature,
fusion-symbols executes only A1, B1, C1 and D2 factors (with tori), and other
groups return `Provider(LieError::Unsupported { requires: "lie-generated" })`;
`cgc-gen` enables fusion-symbols' `lie-generated` for the rest. It validates full
global-form admission, ranks Dynkin labels then zigzag-mapped torus charges
by graded lexicographic order, and forwards all symbols to fusion-symbols;
its typed label is fusion-symbols' `LieIrrep`. Checked Generic mode runs every
rank on the Host multiplicity path. Multiplicity-free mode
(`MultiplicityFreeAdmissionMode`) admits a group whose fusion-symbols style is
`Unique` or `Simple` (tori, rank-one groups, D2, U(2) and their products) and
gives the multiplicity-free provider traits, so the group takes the
multiplicity-free tensor pipeline, CUDA and multiplicity-free products; it
refuses any other group with `LieGroupFusionRuleError::IncompatibleFusionStyle`.
A group fusion-symbols rejects fails at construction with
`LieGroupFusionRuleError::Provider`. The compact
`SU2FusionRule` keeps its multiplicity-free and physical-basis capabilities.
Raw `SectorId` values remain rule-local rather than stable wire data.
