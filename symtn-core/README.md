# symtn-core

Expert structural layer below the `symtn` facade: spaces, fusion trees and
block layouts, with no tensor type of its own. Entry points are
`SectorLeg`, `FusionProductSpace`, `FusionTreeHomSpace`, fusion-tree and block
keys, `BlockStructure`, block views, and storage traits. It validates spaces and layouts but
does not provide high-level execution.

Enable `cgc-gen` to pass `LieGroupFusionRule` support through to sectors.
