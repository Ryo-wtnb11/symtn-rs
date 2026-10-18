# Provenance and references

Upstream source coordinates for the semantic claims made by `symtn`'s public
rustdoc. The rustdoc keeps the semantic anchor (the TensorKit /
MatrixAlgebraKit API name and the behavioral claim); the exact `file:line`
provenance lives here, against the pinned revisions below, so the docs never
carry coordinates that can silently rot.

Scope: public items of the typed facade (`symtn::typed`, `src/typed.rs`) whose
rustdoc carries an upstream semantic claim. Private and `pub(crate)` rustdoc
keeps its inline coordinates (implementation-facing, reviewed with the code).

## Pinned upstream revisions

Coordinates whose revision column names TensorKit 0.17.0, TensorKitSectors
0.3.4, or MatrixAlgebraKit 0.6.9 are line numbers in these exact trees
(immutable Julia package installs; the path slug is derived from the
`git-tree-sha1`, so the on-disk tree cannot drift from the pin). The physical
conversion rows use the separate current-source audit pins recorded in their
subsection below.

| Upstream | Version | git-tree-sha1 | Local tree |
| --- | --- | --- | --- |
| TensorKit.jl | 0.17.0 | `025081058ed953b53aeea9bb8bdbdf241a1fa54f` | `~/.julia/packages/TensorKit/jCjQQ` |
| TensorKitSectors.jl | 0.3.4 | `bb54ef851826493b1ec37fd6a9a9ec573ccd5373` | `~/.julia/packages/TensorKitSectors/JS0fj` |
| MatrixAlgebraKit.jl | 0.6.9 | `0f958855fc2efdc59f1b6a5a066ee3026e87edb3` | `~/.julia/packages/MatrixAlgebraKit/AbU5a` |

Version/tree-sha1 pairs resolved from the local package installs'
`Project.toml` plus the General registry's `Versions.toml` (slug recomputed
with `Base.version_slug` to confirm the correspondence).

MatrixAlgebraKit is pinned for the factorization *names* the rustdoc anchors
to (`svd_compact`, `qr_compact`, `DiagonalAlgorithm`, ...); every `source`
path below that mentions MatrixAlgebraKit behavior
(`factorizations/matrixalgebrakit.jl`, `factorizations/diagonal.jl`) is a
TensorKit file wrapping it. `tensors/tensoroperations.jl` and
`tensors/vectorinterface.jl` are likewise TensorKit's implementations of the
TensorOperations / VectorInterface interfaces.

### Physical-dense executable oracle

The physical-dense SU(2) fixtures were executed with Julia 1.11.6,
TensorKit 0.16.3 (`git-tree-sha1`
`74461ab2eee572c2b4b2ea486a716d118825f642`, local package tree
`TensorKit/Lo4KY`) and TensorKitSectors 0.3.6 (`git-tree-sha1`
`334a0ed5a0a0088a2b6fe7a39f78dda928038d85`, local package tree
`TensorKitSectors/LXgpR`). The reduced fusion-tree scalar was Julia `Float64`
and `convert(Array, tree)` returned dense `Float64`. The executable SU(2)
carrier order was descending `m = j, j - 1, ..., -j`.

That executed environment is separate from the immutable current-source audit
pins:

- TensorKit 0.17.1, commit
  `64430c522c7da3c13c8376fc9734a8ecc054a324`, tree
  `913daeb8110047c283cf08423f78655df8167c96`;
- TensorKitSectors 0.3.9, commit
  `67d342a130bd9e31caf3eccbf9ca46ec8b088556`, tree
  `9f536263310aae90df175337d586a1655104aae3`.

The former defines expansion/projection in
`src/tensors/abstracttensor.jl:730-790` and recursive fusion-tree arrays in
`src/fusiontrees/fusiontrees.jl:272-339`; the latter defines the descending
SU(2) basis in `src/irreps/su2irrep.jl:72-86`. No fixture is claimed to have
been executed under these newer versions.

## Coordinates

`source` paths are relative to the upstream package's `src/`. Rows flagged
**divergence** in the note provenance a documented deliberate difference —
the full semantic statement stays in the rustdoc; only the coordinate lives
here.

A struct's row also covers its documented public fields. A source with no line
range is a deliberate name-only row: the claim has no single defensible
upstream location (the note says why).

| Item | Upstream component | Revision | Source | Note |
| --- | --- | --- | --- | --- |
| `typed::TensorMap::axpby` | TensorKit | 0.17.0 | `tensors/vectorinterface.jl:67-99` | divergence: TK computes `β*ty + α*tx` (first coefficient on the *second* argument); symtn's `alpha` belongs to `self`; named `axpby`, not `add`, so a ported call fails to compile (#1542) |
| `typed::TensorMap::adjoint` | TensorKit | 0.17.0 | `tensors/adjoint.jl:19` | `Base.adjoint` returns the lazy `AdjointTensorMap` wrapper (struct at 9-12) |
| `typed::TensorMap::braid` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:331-342` | typed facade |
| `typed::TensorMap::compose` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:388-420` | fermionic supertrace twist applied only in `blas_contract!`, never in `mul!` |
| `typed::TensorMap::compose` | TensorKit | 0.17.0 | `tensors/linalg.jl:38-42` | categorical composition `compose` behind `A * B` |
| `typed::TensorMap::contract` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:119-146` | `TO.tensorcontract!` and its explicit `pAB` output permutation |
| `typed::TensorMap::dense_data` | TensorKit | 0.17.0 | `tensors/tensor.jl:10-35` | `TensorMap`'s flat `data` vector — the coupled-sector matrix layout symtn's storage claims equivalence with |
| `typed::TensorMap::inner` | TensorKit | 0.17.0 | `tensors/linalg.jl:255` | TensorKit exposes the same inner product as `LinearAlgebra.dot` |
| `typed::TensorMap::eig_full` | MatrixAlgebraKit | 0.6.9 | `interface/eig.jl:12-33` | typed facade |
| `typed::TensorMap::eig_vals` | MatrixAlgebraKit | 0.6.9 | `interface/eig.jl:140-158` | typed facade |
| `typed::TensorMap::eigh_full` | MatrixAlgebraKit | 0.6.9 | `interface/eigh.jl:14-37` | typed facade |
| `typed::TensorMap::eigh_vals` | MatrixAlgebraKit | 0.6.9 | `interface/eigh.jl:144-162` | typed facade |
| `typed::TensorMap::exp` | TensorKit | 0.17.0 | `tensors/linalg.jl:44` | `exp` copies, then calls `exp!` |
| `typed::TensorMap::exp` | TensorKit | 0.17.0 | `tensors/linalg.jl:420-427` | `exp!`: `domain == codomain` check, per-block dense exponential |
| `typed::TensorMap::exp` | TensorKit | 0.17.0 | `tensors/diagonal.jl:383-390` | compact arm: `exp(::DiagonalTensorMap)` is unconditionally elementwise |
| `typed::TensorMap::flip` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:8-29` | |
| `typed::TensorMap::flip` | TensorKit | 0.17.0 | `fusiontrees/braiding_manipulations.jl:384-413` | per-leg Z-isomorphism phase (χ, θ) of the fusion-tree `flip` |
| `typed::TensorMap::from_subblock_fn` | TensorKit | 0.17.0 | `tensors/tensor.jl` | divergence: no TK counterpart (name-only row — no single defensible range); TK's constructor surface is `undef`/`zeros`/`ones` (283-318) and the rand family (320-408) |
| `typed::TensorMap::inner` | TensorKit | 0.17.0 | `tensors/vectorinterface.jl:114-123` | `VectorInterface.inner`, quantum-dimension weighted; TK `dot` alias at `tensors/linalg.jl:255` |
| `typed::TensorMap::inv` | TensorKit | 0.17.0 | `tensors/linalg.jl:375-387` | typed facade |
| `typed::TensorMap::inv` | TensorKit | 0.17.0 | `tensors/diagonal.jl:357` | compact arm, every fusion mode: `inv(::DiagonalTensorMap)` is `inv.(d.data)` on the same bond (#1735) |
| `typed::TensorMap::solve` | TensorKit | 0.17.0 | `tensors/diagonal.jl:358-361`, `tensors/linalg.jl:397-407` | compact divisor, every fusion mode: `D1 \ D2` stays diagonal; `D \ t` is blockwise `Diagonal \ Matrix` (bond-axis scaling) (#1735) |
| `typed::TensorMap::isometry` | TensorKit | 0.17.0 | `tensors/linalg.jl:149-158` | |
| `typed::TensorMap::isomorphism` | TensorKit | 0.17.0 | `tensors/linalg.jl:102-109` | also TK `id(V)` (75-82) as `isomorphism(V, V)` and `unitary` (129-132), whose extra Euclidean check every symtn provider satisfies |
| `typed::TensorMap::left_null` | MatrixAlgebraKit | 0.6.9 | `interface/orthnull.jl:167-244` | typed facade |
| `typed::TensorMap::left_polar` | MatrixAlgebraKit | 0.6.9 | `interface/polar.jl:3-20` | typed facade; TK factor-space glue on the typed `left_polar` rows |
| `typed::TensorMap::lq_compact` | MatrixAlgebraKit | 0.6.9 | `interface/lq.jl:22-43` | typed facade; TK diagonal fast-path divergence on the typed `lq_compact` row |
| `typed::TensorMap::lq_full` | MatrixAlgebraKit | 0.6.9 | `interface/lq.jl:3-20` | typed facade |
| `typed::TensorMap::map_diagonal` | TensorKit | 0.17.0 | `tensors/diagonal.jl:384-390` | `DiagonalTensorMap($f.(d.data), d.domain)` with the caller's `f`; compact diagonals only |
| `typed::TensorMap::norm` | TensorKit | 0.17.0 | `tensors/linalg.jl:257-275` | `norm(t, p)` and the `_norm` block reduction: `p = 2` is the quantum-dimension-weighted Frobenius norm, `p == Inf` (`linalg.jl:262-265`) the unweighted maximum absolute stored entry |
| `typed::TensorMap::permute` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:242-259` | typed facade |
| `typed::TensorMap::pinv` | TensorKit | 0.17.0 | `tensors/linalg.jl:388-396` | typed facade |
| `typed::TensorMap::qr_compact` | MatrixAlgebraKit | 0.6.9 | `interface/qr.jl:22-44` | typed facade; TK diagonal fast-path divergence on the typed `qr_compact` row |
| `typed::TensorMap::qr_full` | MatrixAlgebraKit | 0.6.9 | `interface/qr.jl:3-20` | typed facade; see also the typed `qr_full` row |
| `typed::TensorMap::repartition` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:464-474` | typed facade |
| `typed::TensorMap::right_null` | MatrixAlgebraKit | 0.6.9 | `interface/orthnull.jl:246-323` | typed facade |
| `typed::TensorMap::right_polar` | MatrixAlgebraKit | 0.6.9 | `interface/polar.jl:22-40` | typed facade; TK factor-space glue on the typed `right_polar` row |
| `typed::TensorMap::scale` | TensorKit | 0.17.0 | `tensors/vectorinterface.jl:24-27` | `VectorInterface.scale`, behind `α * t` |
| `typed::TensorMap::space` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:196-201` | `space(t, i)` flat-leg convention (domain legs dualized) |
| `typed::TensorMap::svd_compact` | MatrixAlgebraKit | 0.6.9 | `interface/svd.jl:23-42` | typed facade |
| `typed::TensorMap::svd_full` | MatrixAlgebraKit | 0.6.9 | `interface/svd.jl:3-21` | typed facade |
| `typed::TensorMap::svd_vals` | MatrixAlgebraKit | 0.6.9 | `interface/svd.jl:144-156` | typed facade |
| `typed::TensorMap::tr` | TensorKit | 0.17.0 | `tensors/linalg.jl:319-327` | typed facade |
| `typed::TensorMap::trace_pairs` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:72-87` | `TO.tensortrace!` |
| `typed::TensorMap::transpose` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:401-411` | TK `transpose` with its cyclic `Index2Tuple` always explicit; TK's argument-free default is the full rotation |
| `typed::TensorMap::twist` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:62-97` | `twist!` (62-78) and `twist` (90-97) |
| `typed::TensorMap::zeros` | TensorKit | 0.17.0 | `tensors/tensor.jl:283-318` | the generated `zeros`/`ones` constructor pair |
| `typed` (module doc) | TensorKitSectors | 0.3.4 | `product.jl:245-294` | `ProductSector` / Deligne product `⊠` |
| `typed::BlockFusionTrees` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:348-352` | named after `fusiontrees(t)`; its accessors mirror the `f₁.uncoupled`/`f₂.uncoupled`/coupled fields of TK fusion-tree pairs, and `blocksectors` (331-335) is the coarser surface it deliberately is not |
| `typed::GradedSpace` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:2-29` | the struct this leg type mirrors (docstring 2-25, definition 26-29) |
| `typed::GradedSpace::degeneracies` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:96-101` | `dim(V, c)` |
| `typed::GradedSpace::is_dual` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:114` | `isdual(V) = V.dual` — the orientation flag TK's `V'` toggles |
| `typed::GradedSpace::sectors` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:180-187` | divergence: symtn returns stored labels as-is, TK dualizes stored keys on read when `isdual(V)` |
| `typed::GradedSpace::truncspace` | TensorKit | 0.17.0 | `factorizations/truncation.jl:261-270` | `findtruncated(_svd)` for `TruncationSpace` — the semantics of the strategy; `truncspace` itself is `factorizations/truncation.jl:23-26` |
| `typed::GradedSpace::find_truncated` | MatrixAlgebraKit | 0.6.9 | `implementations/truncation.jl:6-15`, `implementations/svd.jl:231-236` | `svd_trunc!` is `svd_compact!` + `truncate` (`findtruncated` on `diagview(S)`, then indexing) + `truncation_error!`, and `eigh_trunc!`/`eig_trunc!` are the same on `(D, V)`; symtn exposes only the steps (`diagview`, `find_truncated`, `restrict_leg`), #1534; `restrict_leg` on both legs of a compact `s`/`d` is TensorKit `truncate_diagonal!` (`factorizations/truncation.jl:59-66`), on one bond leg `truncate_domain!`/`truncate_codomain!` (41-58), #1561 |
| `typed::GradedSpace::extend_spectrum` | TensorKit | cfaa073 | `src/factorizations/truncation.jl` `MAK.truncate(::typeof(left_null!), (U, S), strategy)` | `extended_S`: `diagview(S)` copied into a zero `SectorVector` over `fuse(codomain(U))`; symtn exposes the padding as a step so `find_truncated` keeps its complete-spectrum check, #1984 |
| `typed::Truncation::below`, `typed::Truncation::rank_smallest` | MatrixAlgebraKit | 0.6.8 | `interface/truncation.jl` `null_truncation_strategy`, `trunctol(; keep_below = true)`, `truncrank(; rev = false)`; `implementations/truncation.jl` `findtruncated(::TruncationByValue)` | the rank-revealing null space `left_null(t; alg = :svd, trunc)` is `svd_full` + `extend_spectrum` + `find_truncated` + `restrict_leg`, #1984 |
| `typed::GradedSpace::try_dual` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:112` | divergence: TK flips the flag and dualizes lazily on read; symtn rewrites the stored sector table eagerly |
| `typed::GradedSpace::try_dual` | TensorKit | 0.17.0 | `spaces/vectorspaces.jl:69-73` | the `dual(dual(V)) == V` contract |
| `typed::GradedSpace::try_new` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:70-85` | `GradedSpace` / `Vect[I]` constructor family |
| `typed::TensorMap` | TensorKit | 0.17.0 | `tensors/tensor.jl:10-35` | convention: payload scalar `T` independent of the sector type, as TK's `TensorMap{T, S, ...}` parameters separate them |
| `typed::TensorMap::absorb` | TensorKit | 0.17.0 | `tensors/linalg.jl:531-545` | `absorb` (531), `absorb!` (532-545); rank-check `DimensionError` at 533-534; shared-block `min` region copy at 538-543 |
| `typed::TensorMap::adjoint` | TensorKit | 0.17.0 | `tensors/adjoint.jl:9-19` | dense storage returns a lazy parent-backed view; compact diagonal storage keeps its direct owned conjugation path |
| `typed::TensorMap::block` | TensorKit | 0.17.0 | `tensors/tensor.jl:447,462-476`, `tensors/adjoint.jl:28`, `tensors/diagonal.jl:153-168` | zero-copy view of the coupled-sector matrix; lazy adjoint `block(parent, c)'`, compact diagonal `Diagonal(view)`. an absent sector is the empty `blockdim(codomain, c) × blockdim(domain, c)` view. Divergence: rows/columns follow symtn's storage tree order, reported by `CoupledBlock::row_trees`/`col_trees` |
| `typed::TensorMap::blocks` | TensorKit | 0.17.0 | `tensors/tensor.jl:449,455-460`, `tensors/abstracttensor.jl:380-385` | `c => block(t, c)` pairs, in storage sector order |
| `typed::TensorMap::subblock_count` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:346-352` | `length(fusiontrees(t))`, the subblock count, not `blocksectors` |
| `typed::TensorMap::subblocks` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:415`, `tensors/tensor.jl:480-491` | `subblocks(t)` / `subblock(t, (f₁, f₂))` |
| `typed::TensorMap::cat` | TensorKit | 0.17.0 | `tensors/linalg.jl:498-514` | `catcodomain` (`Side::Codomain`): domain match 499-500; codomain duality 503-504; direct sum `V = V1 ⊕ V2` 506; per-sector row slabs, `t1` first, 509-512 |
| `typed::TensorMap::cat` | TensorKit | 0.17.0 | `tensors/linalg.jl:479-497` | `catdomain` (`Side::Domain`): codomain match 480-483; domain duality 486-487; direct sum `V = V1 ⊕ V2` 489; per-sector column slabs, `t1` first, 492-495 |
| `typed::TensorMap::codomain_rank` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:239-241` | `numout` |
| `typed::TensorMap::codomain` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:204-214` | `codomain(t)` |
| `typed::TensorMap::contract` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:159-167` | destination structure `permute(compose(sA, sB), pAB)` |
| `typed::TensorMap::domain_rank` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:253-255` | `numin` |
| `typed::TensorMap::domain` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:217-226` | `domain(t)` |
| `typed::TensorMap::exp` | TensorKit | 0.17.0 | `tensors/linalg.jl:44` | `exp` copies, then calls `exp!` |
| `typed::TensorMap::exp` | TensorKit | 0.17.0 | `tensors/linalg.jl:420-427` | `exp!`: `domain == codomain` check, per-block dense exponential |
| `typed::TensorMap::flip` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:21-29` | |
| `typed::TensorMap::flip` | TensorKit | 0.17.0 | `fusiontrees/braiding_manipulations.jl:384-413` | per-leg Z-isomorphism phase (χ, θ) of the fusion-tree `flip` |
| `typed::TensorMap::im` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:718-728` | `Base.imag` |
| `typed::TensorMap::insert_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:124-138` | `insertleftunit` (`seam = Side::Domain`) |
| `typed::TensorMap::insert_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:129-130` | `TensorMap` arm: payload reused, space rewrapped |
| `typed::TensorMap::insert_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:132-137` | non-`TensorMap` arm: `similar` + blockwise copy |
| `typed::TensorMap::insert_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:158-172` | `insertrightunit` (`seam = Side::Codomain`) |
| `typed::TensorMap::isometry` | TensorKit | 0.17.0 | `tensors/linalg.jl:149-158` | |
| `typed::TensorMap::isomorphism` | TensorKit | 0.17.0 | `tensors/linalg.jl:102-109` | also TK `id(V)` (75-82) as `isomorphism(V, V)` and `unitary` (129-132), whose extra Euclidean check every symtn provider satisfies |
| `typed::TensorMap::left_polar` | TensorKit | 0.17.0 | `factorizations/matrixalgebrakit.jl:204-208` | factor spaces: `w` on the input's homspace, `p` on `domain ← domain` |
| `typed::TensorMap::left_polar` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:8-14` | `DiagonalTensorMap` gets only `copy_input` for the polars — no diagonal polar specialization |
| `typed::TensorMap::leg_dims` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:196-201` | `space(t, i)` |
| `typed::TensorMap::lq_compact` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:29-41,61-66` | divergence: TK's `DiagonalAlgorithm` LQ fast path not adopted (#613 Group 4) |
| `typed::TensorMap::materialize` | TensorKit | 0.17.1 | `tensors/linalg.jl:3,196-201`, `tensors/adjoint.jl:28` | `copy(adjoint(t)) = copy!(similar(t), t)`: one allocation, one conjugate transpose per coupled block |
| `typed::TensorMap::materialize` | TensorKit | 0.17.1 | `tensors/tensor.jl:431` | `copy(::TensorMap)`: fresh storage for an owned dense input |
| `typed::TensorMap::materialize` | TensorKit | 0.17.1 | `tensors/diagonal.jl:111-121` | divergence: `copy(::DiagonalTensorMap)` stays diagonal; `materialize` densifies, as `TensorMap(d)` does |
| `typed::TensorMap::norm` | TensorKit | 0.17.0 | `tensors/linalg.jl:257-275` | `norm(t, p)` and the `_norm` block reduction |
| `typed::TensorMap::qr_compact` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:16-28,61-66` | divergence: TK's `DiagonalAlgorithm` QR fast path not adopted (#613 Group 4) |
| `typed::TensorMap::qr_full` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:16-28,61-66` | same non-adoption as `qr_compact` |
| `typed::TensorMap::rand_with_seed` | TensorKit | 0.17.0 | `tensors/tensor.jl:320-408` | the generated `rand` family; divergence: TK threads a caller-supplied `rng` (overloads at 363-406 inside the generated block), the explicit seed is its counterpart; no seedless form |
| `typed::TensorMap::rank` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:267` | `numind` |
| `typed::TensorMap::re` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:707-717` | `Base.real` |
| `typed::TensorMap::remove_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:186-197` | |
| `typed::TensorMap::right_polar` | TensorKit | 0.17.0 | `factorizations/matrixalgebrakit.jl:210-214` | factor spaces: `p` on `codomain ← codomain`, `wh` on the input's homspace |
| `typed::TensorMap::scalar` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:446-451` | empty payload reads as zero |
| `typed::TensorMap::convert` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:696-705` | `Base.complex` and `convert(TensorMap{T}, t)`; sealed pair list, no complex -> real |
| `typed::TensorMap::to_physical_dense` | TensorKit | 0.17.1 | `tensors/abstracttensor.jl:730-750` | physical expansion; leg order from `spaces/gradedspace.jl` `axes(V, c)`/`sectors(V)` (a dual `V'` in `V`'s order); tests `symtn/tests/physical_dense.rs` (in-test U(1) and Racah CG expansions, sampled dual-leg entries checked against TensorKit 0.17.1) (#1464) |
| `typed::TensorMap::project_physical_dense` | TensorKit | 0.17.1 | `tensors/abstracttensor.jl:752-790` | adjoint projection and coupled-sector normalization; executable fixture provenance is recorded above |
| `typed::TensorMap::twist` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:90-97` | `twist`; in-place `twist!` at 62-78 |
| `typed::TensorMap::twist` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:34-51` | `has_shared_twist` identity-twist detection |
| `typed::TensorMap::twist` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:91-93` | `copy = false` default shares `t` on identity twist |
| `typed::TensorMap::twist` | TensorKit | 0.17.0 | `tensors/diagonal.jl:84-89` | `similar(::DiagonalTensorMap)` preserves diagonal storage, so TK's diagonal twist stays diagonal |
| `typed::TensorMap::zeros_like` | TensorKit | 0.17.0 | `tensors/vectorinterface.jl:7-20` | `zerovector` |
