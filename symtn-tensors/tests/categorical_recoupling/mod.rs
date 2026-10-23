//! Real categorical recoupling fixtures and an oracle that walks a compiled
//! structure through its *public* surface only.
//!
//! These are the structures a provider actually compiles — SU(2) F moves and
//! their fermionic (fZ2 ⊠ SU(2)) counterparts — reached below the typed
//! `TensorMap` layer through
//! [`symtn_tensors::build_tree_pair_transform_group_plan`] and its storage-conjugating compile,
//! so the recoupling matrices are genuine 6j/R data with irrational entries and
//! mixed signs rather than hand-written numbers.
//!
//! The oracle walks `blocks()`, the entries' offsets and the coefficients with
//! a plain triple loop over *raw* block geometry it rebuilds itself: each
//! entry's block is found in the fixture `BlockStructure` by offset, and a
//! source block's axes are permuted by the operation's own permutation. It
//! never reads the compiled layout roles, the recoupling plan, a packed column
//! or a GEMM, so "the host executor agrees with it" and "the device executor
//! agrees with it" are statements about the replayed semantics, not about a
//! shared implementation.

#![allow(dead_code)]

use std::sync::Arc;

use num_complex::Complex64;
use symtn_core::{
    BlockKey, BlockStructure, DegeneracyStructure, FermionParityFusionRule, FusionRule,
    FusionTreeKey, FusionTreePairKey, MultiplicityIndex, ProductFusionRule, RuleIdentity,
    SU2FusionRule, SectorId, SectorStructure,
};
use symtn_operations::{TreeTransformBlock, TreeTransformStructure};
use symtn_tensors::{build_tree_pair_transform_group_plan, TreeTransformOperation};

/// Payload dtypes the categorical fixtures are replayed with, plus the host
/// arithmetic the oracle needs. Deliberately not a symtn trait: the oracle must
/// not share a scalar contract with the code under test.
pub trait TestScalar: Copy + std::fmt::Debug + PartialEq + 'static {
    const NAME: &'static str;
    fn from_parts(re: f64, im: f64) -> Self;
    fn zero() -> Self {
        Self::from_parts(0.0, 0.0)
    }
    fn scale(self, factor: f64) -> Self;
    fn add(self, other: Self) -> Self;
    /// Payload multiplication, for the caller scale `alpha`.
    fn mul(self, other: Self) -> Self;
    fn conjugate(self) -> Self;
    fn distance(self, other: Self) -> f64;
    /// Deterministic, non-degenerate sample for buffer index `index`.
    fn sample(index: usize) -> Self {
        Self::from_parts(1.0 + (index as f64) * 0.5, -0.25 + (index as f64) * 0.125)
    }
    fn is_nan(self) -> bool;
}

impl TestScalar for f64 {
    const NAME: &'static str = "f64";

    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }

    fn scale(self, factor: f64) -> Self {
        self * factor
    }

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn mul(self, other: Self) -> Self {
        self * other
    }

    fn conjugate(self) -> Self {
        self
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).abs()
    }

    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

impl TestScalar for Complex64 {
    const NAME: &'static str = "Complex64";

    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }

    fn scale(self, factor: f64) -> Self {
        self * factor
    }

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn mul(self, other: Self) -> Self {
        self * other
    }

    fn conjugate(self) -> Self {
        Complex64::new(self.re, -self.im)
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }

    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

/// A vacuum-coupled SU(2) tree-pair space: one key per inner-line channel, all
/// legs of `degeneracy` dimensions.
///
/// The domain tree is empty, so the coupled sector must be the vacuum; the
/// channels are the distinct inner-line assignments that reach it. Two or more
/// channels are what makes a leg exchange an F move — each destination tree is a
/// combination of *several* source trees — so the compiled structure carries a
/// `Multi` block with a genuine recoupling matrix.
pub fn su2_space(legs: &[usize], inners: &[Vec<usize>], degeneracy: usize) -> Arc<BlockStructure> {
    let rank = legs.len();
    let keys: Vec<BlockKey> = inners
        .iter()
        .map(|inner| {
            BlockKey::from(FusionTreePairKey::pair(
                FusionTreeKey::try_new_for_rule(
                    &SU2FusionRule,
                    legs.iter().copied().map(SectorId::new).collect::<Vec<_>>(),
                    SU2FusionRule.vacuum(),
                    vec![false; rank],
                    inner.iter().copied().map(SectorId::new).collect::<Vec<_>>(),
                    vec![MultiplicityIndex::ONE; rank - 1],
                )
                .unwrap(),
                FusionTreeKey::try_new_for_rule(
                    &SU2FusionRule,
                    [],
                    SU2FusionRule.vacuum(),
                    [],
                    [],
                    [],
                )
                .unwrap(),
            ))
        })
        .collect();
    let shapes: Vec<Vec<usize>> = (0..keys.len()).map(|_| vec![degeneracy; rank]).collect();
    Arc::new(
        BlockStructure::from_parts(
            SectorStructure::from_keys(rank, keys).unwrap(),
            DegeneracyStructure::packed_column_major(rank, shapes).unwrap(),
        )
        .unwrap(),
    )
}

/// The fermionic rule that can recouple: fZ2 x SU(2).
///
/// Plain fZ2 is `FusionStyleKind::Unique`, so every fusion tree of a key is
/// forced and an F move is 1x1 — a fermionic abelian structure compiles to
/// `Single` blocks only, whatever the operation. Multiplying it by SU(2) keeps
/// `BraidingStyleKind::Fermionic` (`Fermionic ⊞ Bosonic = Fermionic`) while
/// `Unique ⊞ Simple = Simple` supplies the several fusion channels a recoupling
/// needs, so this is the one rule in the workspace whose recoupling matrix
/// carries fermionic signs.
pub type FermionicSu2Rule = ProductFusionRule<FermionParityFusionRule, SU2FusionRule>;

pub fn fermionic_su2_rule() -> FermionicSu2Rule {
    symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
        .unwrap()
}

/// The fZ2 x SU(2) counterpart of [`su2_space`]: every leg is an odd-parity
/// SU(2) irrep, so the parity component of each inner line is forced (odd,
/// even, odd, …) and the channels are the SU(2) ones.
///
/// Each recoupling entry is therefore an SU(2) 6j/R symbol times a fermionic
/// sign.
pub fn fermionic_space(
    rule: &FermionicSu2Rule,
    legs: &[usize],
    inners: &[Vec<usize>],
    degeneracy: usize,
) -> Arc<BlockStructure> {
    let rank = legs.len();
    let odd = SectorId::new(1);
    let even = SectorId::new(0);
    let vacuum = rule.vacuum();
    // Leg i is odd; inner line i therefore alternates even, odd, even, …
    let parity = |index: usize| if index.is_multiple_of(2) { even } else { odd };
    let keys: Vec<BlockKey> = inners
        .iter()
        .map(|inner| {
            let uncoupled: Vec<SectorId> = legs
                .iter()
                .map(|&spin| rule.encode_component_ids(odd, SectorId::new(spin)))
                .collect();
            let inner_sectors: Vec<SectorId> = inner
                .iter()
                .enumerate()
                .map(|(index, &spin)| rule.encode_component_ids(parity(index), SectorId::new(spin)))
                .collect();
            BlockKey::from(FusionTreePairKey::pair(
                FusionTreeKey::try_new_for_rule(
                    rule,
                    uncoupled,
                    vacuum,
                    vec![false; rank],
                    inner_sectors,
                    vec![MultiplicityIndex::ONE; rank - 1],
                )
                .unwrap(),
                FusionTreeKey::try_new_for_rule(rule, [], vacuum, [], [], []).unwrap(),
            ))
        })
        .collect();
    let shapes: Vec<Vec<usize>> = (0..keys.len()).map(|_| vec![degeneracy; rank]).collect();
    Arc::new(
        BlockStructure::from_parts(
            SectorStructure::from_keys(rank, keys).unwrap(),
            DegeneracyStructure::packed_column_major(rank, shapes).unwrap(),
        )
        .unwrap(),
    )
}

/// Four spin-1/2 legs coupling to the vacuum through two inner-line channels —
/// the smallest F move, with a symmetric 2x2 recoupling matrix.
pub fn four_leg_channels() -> Vec<Vec<usize>> {
    vec![vec![0, 1], vec![2, 1]]
}

/// Six spin-1/2 legs coupling to the vacuum through five channels.
///
/// The 5x5 recoupling matrix an exchange produces here is *not* symmetric,
/// which is what lets a test distinguish `U` from `Uᵀ` on provider data — the
/// two-channel matrix above is its own transpose, so it cannot.
pub fn six_leg_channels() -> Vec<Vec<usize>> {
    vec![
        vec![0, 1, 0, 1],
        vec![2, 1, 0, 1],
        vec![2, 1, 2, 1],
        vec![2, 3, 2, 1],
        vec![0, 1, 2, 1],
    ]
}

/// Four odd-parity legs coupling to the even vacuum under plain fZ2.
///
/// Kept as a *negative* fixture: fZ2 is `FusionStyleKind::Unique`, so this
/// compiles to `Single` blocks only, which is why the fermionic recoupling
/// evidence needs [`fermionic_f_move_structure`] instead.
pub fn fermion_parity_structure(degeneracy: usize) -> Arc<BlockStructure> {
    let odd = SectorId::new(1);
    let even = FermionParityFusionRule.vacuum();
    let keys = [BlockKey::from(FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &FermionParityFusionRule,
            [odd; 4],
            even,
            [false; 4],
            [even, odd],
            [MultiplicityIndex::ONE; 3],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&FermionParityFusionRule, [], even, [], [], []).unwrap(),
    ))];
    Arc::new(
        BlockStructure::from_parts(
            SectorStructure::from_keys(4, keys).unwrap(),
            DegeneracyStructure::packed_column_major(4, [vec![degeneracy; 4]]).unwrap(),
        )
        .unwrap(),
    )
}

/// One compiled categorical fixture: the structure the executors replay, the
/// two block structures they validate against, and whether the source is read
/// conjugated.
pub struct Compiled {
    pub name: String,
    pub structure: TreeTransformStructure<f64>,
    pub space: Arc<BlockStructure>,
    pub conjugate: bool,
    /// Destination axis `i` reads source axis `source_axes[i]`: the
    /// operation's codomain then domain permutation.
    pub source_axes: Vec<usize>,
}

impl Compiled {
    pub fn len(&self) -> usize {
        self.space.required_len().unwrap()
    }

    pub fn source<T: TestScalar>(&self) -> Vec<T> {
        (0..self.len()).map(T::sample).collect()
    }

    /// Multi blocks in the compiled structure, as `(src_count, dst_count)`.
    pub fn multi_shapes(&self) -> Vec<(usize, usize)> {
        self.structure
            .blocks()
            .iter()
            .filter_map(|block| match *block {
                TreeTransformBlock::Multi {
                    src_count,
                    dst_count,
                    ..
                } => Some((src_count, dst_count)),
                TreeTransformBlock::Single { .. } => None,
            })
            .collect()
    }

    /// Whether some recoupling matrix is not its own transpose, so an executor
    /// applying `U` where the reference applies `Uᵀ` cannot agree by accident.
    pub fn has_non_symmetric_matrix(&self) -> bool {
        let coefficients = self.structure.gathered_coefficients();
        self.structure.blocks().iter().any(|block| {
            let TreeTransformBlock::Multi {
                src_count,
                dst_count,
                coefficient_start,
                ..
            } = *block
            else {
                return false;
            };
            if src_count != dst_count {
                // Rectangular: a transposed orientation is not even expressible.
                return true;
            }
            (0..dst_count).any(|row| {
                (0..src_count).any(|column| {
                    let upper = coefficients[coefficient_start + row * src_count + column];
                    let lower = coefficients[coefficient_start + column * src_count + row];
                    (upper - lower).abs() > 1e-12
                })
            })
        })
    }
}

/// Compiles `operation` over `space` through the public tree-transform cache,
/// i.e. the same seam the typed layer uses, one level below it.
pub fn compile<R>(
    name: &str,
    rule: &R,
    operation: TreeTransformOperation,
    space: &Arc<BlockStructure>,
    conjugate: bool,
) -> Compiled
where
    R: symtn_core::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn_tensors::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let source_axes: Vec<usize> = operation
        .codomain_permutation()
        .iter()
        .chain(operation.domain_permutation())
        .copied()
        .collect();
    let structure = build_tree_pair_transform_group_plan(rule, operation, space)
        .and_then(|plan| {
            plan.compile_shared_structures_with_storage_conjugation(
                Arc::clone(space),
                Arc::clone(space),
                conjugate,
            )
        })
        .unwrap_or_else(|error| panic!("{name} failed to compile: {error:?}"));
    Compiled {
        name: name.to_string(),
        structure,
        space: Arc::clone(space),
        conjugate,
        source_axes,
    }
}

/// Every categorical fixture the host and device suites replay.
///
/// The permutation `[0, 2, 1, 3, …]` exchanges the two legs that meet at the
/// first inner line, which is what makes the transform an F move rather than a
/// relabelling. `transpose` is the planar kind, so it takes a cyclic rotation;
/// a codomain/domain repartition needs a second tree-pair space and stays with
/// the typed layer.
///
/// Both channel counts are present on purpose. The two-channel matrices are
/// symmetric — real 6j data, but blind to a transposed orientation — while the
/// six-leg five-channel ones are not, and the fermionic six-leg fixture is the
/// one that carries fermionic signs inside a non-symmetric `U`.
pub fn fixtures() -> Vec<Compiled> {
    let rule = fermionic_su2_rule();
    let halves_4 = [1usize; 4];
    let halves_6 = [1usize; 6];
    let four = four_leg_channels();
    let six = six_leg_channels();
    let su2_4 = su2_space(&halves_4, &four, 2);
    let su2_6 = su2_space(&halves_6, &six, 2);
    let fermionic_4 = fermionic_space(&rule, &halves_4, &four, 2);
    let fermionic_6 = fermionic_space(&rule, &halves_6, &six, 2);
    let swap_4 = [0usize, 2, 1, 3];
    let swap_6 = [0usize, 2, 1, 3, 4, 5];
    vec![
        compile(
            "su2_permute",
            &SU2FusionRule,
            TreeTransformOperation::permute(swap_4, []),
            &su2_4,
            false,
        ),
        compile(
            "su2_braid",
            &SU2FusionRule,
            TreeTransformOperation::braid(swap_4, [], 0..4, []),
            &su2_4,
            false,
        ),
        compile(
            "su2_transpose",
            &SU2FusionRule,
            // Planar: `Transpose` rejects a non-planar permutation, so a cyclic
            // rotation is the F move this kind can express here.
            TreeTransformOperation::transpose([1usize, 2, 3, 0], []),
            &su2_4,
            false,
        ),
        compile(
            "su2_braid_conjugated",
            &SU2FusionRule,
            TreeTransformOperation::braid(swap_4, [], 0..4, []),
            &su2_4,
            true,
        ),
        // Degeneracy 1 beside the degeneracy-2 fixtures, so the pack layouts
        // are not all the same shape.
        compile(
            "su2_braid_scalar_degeneracy",
            &SU2FusionRule,
            TreeTransformOperation::braid(swap_4, [], 0..4, []),
            &su2_space(&halves_4, &four, 1),
            false,
        ),
        compile(
            "su2_rank6_permute",
            &SU2FusionRule,
            TreeTransformOperation::permute(swap_6, []),
            &su2_6,
            false,
        ),
        compile(
            "su2_rank6_rotate",
            &SU2FusionRule,
            TreeTransformOperation::permute([1usize, 2, 0, 3, 4, 5], []),
            &su2_6,
            false,
        ),
        compile(
            "fermionic_su2_permute",
            &rule,
            TreeTransformOperation::permute(swap_4, []),
            &fermionic_4,
            false,
        ),
        compile(
            "fermionic_su2_braid",
            &rule,
            TreeTransformOperation::braid(swap_4, [], 0..4, []),
            &fermionic_4,
            false,
        ),
        compile(
            "fermionic_su2_braid_conjugated",
            &rule,
            TreeTransformOperation::braid(swap_4, [], 0..4, []),
            &fermionic_4,
            true,
        ),
        compile(
            "fermionic_su2_rank6_permute",
            &rule,
            TreeTransformOperation::permute(swap_6, []),
            &fermionic_6,
            false,
        ),
        compile(
            "fermionic_su2_rank6_permute_conjugated",
            &rule,
            TreeTransformOperation::permute(swap_6, []),
            &fermionic_6,
            true,
        ),
    ]
}

/// The fixture whose recoupling matrix is not its own transpose, for the
/// orientation control.
pub fn non_symmetric_fixture() -> Compiled {
    fixtures()
        .into_iter()
        .find(|fixture| fixture.name == "su2_rank6_permute")
        .expect("su2_rank6_permute fixture")
}

/// Which block structure a layout entry indexes.
#[derive(Clone, Copy, PartialEq)]
pub enum Side {
    Destination,
    Source,
}

/// Flat positions of one layout entry, in the destination block's column-major
/// order (fastest axis first) — the order pack and scatter agree on.
///
/// The geometry is rebuilt from the fixture alone: the entry's block is the
/// non-empty block at the entry's offset (every fixture block is non-empty, so
/// offsets identify blocks), and a source block is read through
/// `source_axes`, so the walk pairs destination and source elements exactly
/// as the operation's dense permutation does.
pub fn positions(fixture: &Compiled, entry: usize, side: Side) -> Vec<usize> {
    let offset = fixture.structure.layouts().entry(entry).offset;
    let space = &fixture.space;
    let index = (0..space.block_count())
        .find(|&index| {
            let block = space.block(index).unwrap();
            block.shape().iter().product::<usize>() != 0
                && isize::try_from(block.offset()).unwrap() == offset
        })
        .unwrap_or_else(|| panic!("{}: no fixture block at offset {offset}", fixture.name));
    let block = space.block(index).unwrap();
    let axis = |position: usize| match side {
        Side::Destination => position,
        Side::Source => fixture.source_axes[position],
    };
    let shape = (0..block.shape().len())
        .map(|position| block.shape()[axis(position)])
        .collect::<Vec<_>>();
    let strides = (0..block.shape().len())
        .map(|position| block.strides()[axis(position)])
        .collect::<Vec<_>>();
    let count: usize = shape.iter().product();
    (0..count)
        .map(|linear| {
            let mut remaining = linear;
            let mut position = block.offset();
            for (extent, stride) in shape.iter().zip(&strides) {
                position += (remaining % extent) * stride;
                remaining /= extent;
            }
            position
        })
        .collect()
}

/// The expected destination buffer, from the compiled structure's *public*
/// block and layout metadata alone.
///
/// `dst[d] += U[d][s] * [conj] src[s]` per Multi block and
/// `dst = coefficient * [conj] src` per Single block, walked with unbaked
/// strides. `overwrite` starts from a zero buffer, which equals the executors'
/// Overwrite mode for a packed destination space: every element belongs to some
/// destination layout, active ones are assigned and inactive ones are zeroed.
pub fn expected<T: TestScalar>(
    fixture: &Compiled,
    source: &[T],
    destination: &[T],
    overwrite: bool,
) -> Vec<T> {
    expected_scaled(
        fixture,
        source,
        destination,
        overwrite,
        T::from_parts(1.0, 0.0),
    )
}

/// The caller scales the executors must reproduce.
pub fn alphas<T: TestScalar>() -> Vec<T> {
    vec![
        T::from_parts(1.0, 0.0),
        T::from_parts(-2.5, 0.0),
        T::from_parts(0.0, 0.0),
        T::from_parts(-0.0, -0.0),
        T::from_parts(0.5, -1.25),
    ]
}

/// [`expected`] with the caller scale `alpha`, applied where the host applies
/// it: `(alpha * coefficient) * src` per Single block and `alpha * (U x)` per
/// Multi block — never inside the recoupling sum. `alpha = 0` multiplies rather
/// than short-circuiting, so a non-finite source still reaches the destination.
pub fn expected_scaled<T: TestScalar>(
    fixture: &Compiled,
    source: &[T],
    destination: &[T],
    overwrite: bool,
    alpha: T,
) -> Vec<T> {
    let structure = &fixture.structure;
    let coefficients = structure.gathered_coefficients();
    let mut expected = if overwrite {
        vec![T::zero(); destination.len()]
    } else {
        destination.to_vec()
    };
    let read = |position: usize| {
        let value = source[position];
        if fixture.conjugate {
            value.conjugate()
        } else {
            value
        }
    };
    for block in structure.blocks() {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => {
                let coefficient = coefficients[coefficient];
                let dst_positions = positions(fixture, dst_layout, Side::Destination);
                let src_positions = positions(fixture, src_layout, Side::Source);
                assert_eq!(dst_positions.len(), src_positions.len());
                for (dst_position, src_position) in dst_positions.into_iter().zip(src_positions) {
                    let value = alpha.scale(coefficient).mul(read(src_position));
                    expected[dst_position] = expected[dst_position].add(value);
                }
            }
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                ..
            } => {
                for dst_index in 0..dst_count {
                    let dst_positions =
                        positions(fixture, dst_layout_start + dst_index, Side::Destination);
                    // The recoupling sum first, the caller scale once on the
                    // result: `alpha * (U x)`, which is where the host applies
                    // it — at the scatter, not at the pack or the GEMM.
                    let mut column = vec![T::zero(); dst_positions.len()];
                    for src_index in 0..src_count {
                        let coefficient =
                            coefficients[coefficient_start + dst_index * src_count + src_index];
                        let src_positions =
                            positions(fixture, src_layout_start + src_index, Side::Source);
                        assert_eq!(src_positions.len(), dst_positions.len());
                        for (slot, src_position) in column.iter_mut().zip(&src_positions) {
                            *slot = slot.add(read(*src_position).scale(coefficient));
                        }
                    }
                    for (dst_position, value) in dst_positions.iter().zip(column) {
                        expected[*dst_position] = expected[*dst_position].add(alpha.mul(value));
                    }
                }
            }
        }
    }
    expected
}

pub fn assert_close<T: TestScalar>(actual: &[T], expected: &[T], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            left.distance(*right) <= 1e-12,
            "{what}: element {index} is {left:?}, expected {right:?}"
        );
    }
}

/// Host replay of `fixture` in either destination mode, on host slices.
pub fn host_replay<T>(
    fixture: &Compiled,
    source: &[T],
    destination: &[T],
    overwrite: bool,
) -> Vec<T>
where
    T: TestScalar
        + symtn_operations::TreeTransformScalar
        + symtn_operations::RecouplingCoefficientAction<f64>
        + symtn_operations::DenseBlockScalar,
{
    host_replay_scaled(
        fixture,
        source,
        destination,
        overwrite,
        T::from_parts(1.0, 0.0),
    )
}

/// Host replay of `fixture` with the caller scale `alpha`.
pub fn host_replay_scaled<T>(
    fixture: &Compiled,
    source: &[T],
    destination: &[T],
    overwrite: bool,
    alpha: T,
) -> Vec<T>
where
    T: TestScalar
        + symtn_operations::TreeTransformScalar
        + symtn_operations::RecouplingCoefficientAction<f64>
        + symtn_operations::DenseBlockScalar,
{
    let mut kernels = symtn_operations::StridedHostKernelAdapter::default();
    let mut workspace = symtn_tensors::TreeTransformWorkspace::<T>::default();
    let mut data = destination.to_vec();
    let one = T::from_parts(1.0, 0.0);
    if overwrite {
        symtn_operations::tree_transform_structure_overwrite_with_strided_kernel_raw(
            &mut kernels,
            &mut workspace,
            &fixture.structure,
            &fixture.space,
            &fixture.space,
            &mut data,
            source,
            alpha,
        )
        .unwrap();
    } else {
        symtn_operations::tree_transform_structure_with_strided_kernel_raw(
            &mut kernels,
            &mut workspace,
            &fixture.structure,
            &fixture.space,
            &fixture.space,
            &mut data,
            source,
            alpha,
            one,
        )
        .unwrap();
    }
    data
}

/// Test view of the logical coefficient payload (an explicit copy).
trait GatheredCoefficients<T> {
    fn gathered_coefficients(&self) -> Vec<T>;
}

impl<T: Copy> GatheredCoefficients<T> for symtn_tensors::TreeTransformStructure<T> {
    fn gathered_coefficients(&self) -> Vec<T> {
        let mut coefficients = Vec::new();
        self.gather_recoupling_coefficients_into(&mut coefficients);
        coefficients
    }
}
