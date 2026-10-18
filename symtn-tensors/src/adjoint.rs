//! Adjoint (dagger) of a fusion tensor.
//!
//! TensorKit semantics: the adjoint of `t : codomain <- domain` is
//! `t^H : domain <- codomain`, whose coupled-sector blocks are the conjugate
//! transposes of `t`'s blocks (`block(t^H, c) = block(t, c)^H`). Codomain and
//! domain swap as spaces; leg duality flags are unchanged.

use core::ops::{Add, Mul};
use std::sync::Arc;

#[cfg(test)]
use symtn_core::FusionRule;
use symtn_core::{
    BlockKey, BlockStructure, CheckedGenericFusion, CoreError, FusionTreeHomSpace,
    FusionTreePairKey, MultiplicityFreeRigidSymbols,
};

#[cfg(test)]
use crate::contract::{
    checked_layout_primer, checked_metadata_dispatcher, encoded_layout_primer, MetadataOutput,
    MetadataRequest,
};
use crate::contract::{
    dispatch_prepare, BoundDynamicFusionMapSpace, DynamicFusionMapSpace, LayoutKeyBuilder,
};
use crate::{CheckedGenericPlanError, ConjugateValue, FusionOperand, OperationError};
use symtn_operations::overwrite_owned_blocks;

/// Scalar contract of the adjoint materialization: exactly what the shared
/// strided owner `CheckedBlockLayout::tensoradd` requires, in one place.
pub trait AdjointScalar:
    Copy
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + PartialEq
    + num_traits::Zero
    + num_traits::One
    + ConjugateValue
{
}

impl<T> AdjointScalar for T where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + num_traits::Zero
        + num_traits::One
        + ConjugateValue
{
}

/// Dynamic-rank adjoint space (dagger of the homspace): codomain and domain
/// swapped, per-block shapes transposed. Pure metadata — touches no data — so a
/// lazy adjoint can present the correct fresh coupled space in O(blocks)
/// without copying any elements. `adjoint_dyn` is exactly this plus the
/// conjugate-transpose of the block data, and its output data lives in this
/// space's layout.
#[cfg(test)]
pub(crate) fn adjoint_space_dyn<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    adjoint_space_dyn_with_primer(rule, space, encoded_layout_primer::<R>)
}

fn adjoint_space_dyn_with_primer<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    space.validate_rule(rule)?;
    build_adjoint_space_dyn_with_primer(rule, space, primer)
}

fn build_adjoint_space_dyn_with_primer<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let homspace = space.homspace();
    let adjoint_hom =
        FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone());
    let prepared = dispatch_prepare(primer, rule, &adjoint_hom)?;
    let expected_source = homspace
        .coupled_subblock_structure_from_leg_degeneracies(rule)
        .map_err(OperationError::from_core_preserving_context)?;
    validate_adjoint_lazy_source_keys(space.structure(), &expected_source)?;
    validate_adjoint_source_structure_against(space.structure(), &expected_source)?;

    DynamicFusionMapSpace::from_final_homspace_with_prepared(rule, adjoint_hom, prepared)
}

fn build_adjoint_target_space_with_primer<R>(
    rule: &R,
    source_homspace: &FusionTreeHomSpace,
    primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let adjoint_hom = FusionTreeHomSpace::new(
        source_homspace.domain().clone(),
        source_homspace.codomain().clone(),
    );
    let prepared = dispatch_prepare(primer, rule, &adjoint_hom)?;
    DynamicFusionMapSpace::from_final_homspace_with_prepared(rule, adjoint_hom, prepared)
}

#[cfg(test)]
fn build_adjoint_space_dyn_stepwise_oracle<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let nout = space.nout();
    let homspace = space.homspace();
    let adjoint_hom =
        FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone());
    let structure = Arc::clone(space.structure());
    let prepared = dispatch_prepare(primer, rule, &adjoint_hom)?;
    let keys = prepared.keys(rule, &adjoint_hom);
    let shapes = keys
        .iter()
        .map(|key| {
            let source_key = BlockKey::FusionTree(FusionTreePairKey::pair(
                key.domain_tree().clone(),
                key.codomain_tree().clone(),
            ));
            let index = structure
                .find_block_index_by_key(&source_key)
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: Box::new(source_key.clone()),
                })?;
            let source_shape = structure
                .block(index)
                .map_err(OperationError::from_core_preserving_context)?
                .shape();
            let mut shape = source_shape[nout..].to_vec();
            shape.extend_from_slice(&source_shape[..nout]);
            Ok(shape)
        })
        .collect::<Result<Vec<_>, OperationError>>()?;

    DynamicFusionMapSpace::from_degeneracy_shapes_with_key_builder(
        rule,
        adjoint_hom,
        shapes,
        move |_, _| Ok(prepared),
    )
}

fn validate_adjoint_data_extent(
    expected: Result<usize, CoreError>,
    actual: usize,
) -> Result<(), OperationError> {
    let expected = expected.map_err(OperationError::from_core_preserving_context)?;
    if actual != expected {
        return Err(OperationError::ElementCountMismatch { expected, actual });
    }
    Ok(())
}

fn validate_adjoint_source_structure<R>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
    source: &BlockStructure,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let expected = homspace
        .coupled_subblock_structure_from_leg_degeneracies(rule)
        .map_err(OperationError::from_core_preserving_context)?;
    validate_adjoint_source_structure_against(source, &expected)
}

fn validate_adjoint_source_structure_against(
    source: &BlockStructure,
    expected: &BlockStructure,
) -> Result<(), OperationError> {
    if source.content_id() == expected.content_id() {
        return Ok(());
    }
    if source.rank() != expected.rank() {
        return Err(OperationError::Core(CoreError::StructureRankMismatch {
            expected: expected.rank(),
            actual: source.rank(),
        }));
    }
    if source.block_count() != expected.block_count() {
        return Err(OperationError::Core(CoreError::BlockCountMismatch {
            expected: expected.block_count(),
            actual: source.block_count(),
        }));
    }
    for index in 0..expected.block_count() {
        let expected_block = expected
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let source_index = source
            .find_block_index_by_key(expected_block.key())
            .ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(expected_block.key().clone()),
            })?;
        if source_index != index {
            return Err(OperationError::StructureMismatch { tensor: "src" });
        }
        let source_block = source
            .block(source_index)
            .map_err(OperationError::from_core_preserving_context)?;
        if source_block.shape() != expected_block.shape() {
            return Err(OperationError::ShapeMismatch {
                dst: expected_block.shape().to_vec(),
                src: source_block.shape().to_vec(),
            });
        }
        if source_block.strides() != expected_block.strides()
            || source_block.offset() != expected_block.offset()
        {
            return Err(OperationError::StructureMismatch { tensor: "src" });
        }
    }
    Ok(())
}

fn validate_adjoint_lazy_source_keys(
    source: &BlockStructure,
    expected: &BlockStructure,
) -> Result<(), OperationError> {
    // Why not let target construction discover missing source keys: the final
    // HomSpace is now the sole target-layout authority, so it deliberately
    // does not inspect source storage. The metadata path retains its public
    // MissingBlockKey failure before running the stricter canonical preflight.
    for index in 0..expected.block_count() {
        let key = expected
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?
            .key();
        if source.find_block_index_by_key(key).is_none() {
            return Err(OperationError::MissingBlockKey {
                key: Box::new(key.clone()),
            });
        }
    }
    Ok(())
}

/// Dynamic-rank adjoint (dagger): returns the adjoint space (codomain and
/// domain swapped) together with freshly allocated coupled-layout data whose
/// blocks are the conjugate transposes of the source blocks.
#[cfg(test)]
pub(crate) fn adjoint_dyn<R, D>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynamicFusionMapSpace, Vec<D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: AdjointScalar,
{
    adjoint_dyn_with_primer(rule, space, data, encoded_layout_primer::<R>)
}

fn adjoint_dyn_with_primer<R, D>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
    primer: LayoutKeyBuilder<R>,
) -> Result<(DynamicFusionMapSpace, Vec<D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: AdjointScalar,
{
    space.validate_rule(rule)?;
    validate_adjoint_data_extent(space.required_len(), data.len())?;
    validate_adjoint_source_structure(rule, space.homspace(), space.structure())?;
    // Uncached build: the eager adjoint (SVD/eigh consumers) is a separate,
    // out-of-scope path from the cached lazy `adjoint_space_dyn` (#118),
    // and routing it here keeps the replay-cache-key bound off matrix algebra.
    let adjoint_space = build_adjoint_target_space_with_primer(rule, space.homspace(), primer)?;
    let result = materialize_adjoint_data_dyn(space, &adjoint_space, data)?;
    Ok((adjoint_space, result))
}

/// The adjoint block map (#1839 item 4): for every block of an adjoint
/// layout, the parent block it conjugate-transposes, or none for a block that
/// reads as zero. Every storage materializes a lazy adjoint from this one map
/// and owns only its lowering (Host: the `overwrite_owned_blocks` writer;
/// CUDA: strided copies or an element table).
///
/// Borrowed and lookup-only, so building it allocates nothing. It does no
/// provider work: callers derive and admit `adjoint_space` first.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct AdjointBlockMap<'a> {
    parent: FusionOperand<'a>,
    adjoint_space: &'a DynamicFusionMapSpace,
}

/// One adjoint block's parent block, read through
/// [`AdjointBlockSource::stride`] in the adjoint block's axis order.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct AdjointBlockSource<'a> {
    parent: FusionOperand<'a>,
    block: symtn_core::BlockRef<'a>,
}

impl<'a> AdjointBlockMap<'a> {
    /// The map from `space` (the parent) to its admitted `adjoint_space`.
    pub fn new(space: &'a DynamicFusionMapSpace, adjoint_space: &'a DynamicFusionMapSpace) -> Self {
        Self {
            parent: FusionOperand::adjoint(space),
            adjoint_space,
        }
    }

    pub fn parent_space(&self) -> &'a DynamicFusionMapSpace {
        self.parent.storage_space()
    }

    pub fn adjoint_space(&self) -> &'a DynamicFusionMapSpace {
        self.adjoint_space
    }

    /// The parent block of adjoint `block`: [`None`] for a block without a
    /// fusion-tree key (it has no adjoint source and reads as zero), and
    /// [`OperationError::MissingBlockKey`] naming the swapped pair when the
    /// parent lacks it.
    pub fn source(
        &self,
        block: symtn_core::BlockRef<'_>,
    ) -> Result<Option<AdjointBlockSource<'a>>, OperationError> {
        let BlockKey::FusionTree(key) = block.key() else {
            return Ok(None);
        };
        let structure: &'a BlockStructure = self.parent_space().structure();
        let source_index = structure
            .find_block_index_by_adjoint_fusion_tree_pair(key)
            .ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(BlockKey::FusionTree(FusionTreePairKey::pair(
                    key.domain_tree().clone(),
                    key.codomain_tree().clone(),
                ))),
            })?;
        let block = structure
            .block(source_index)
            .map_err(OperationError::from_core_preserving_context)?;
        Ok(Some(AdjointBlockSource {
            parent: self.parent,
            block,
        }))
    }
}

impl AdjointBlockSource<'_> {
    pub fn offset(&self) -> usize {
        self.block.offset()
    }

    /// The parent stride of adjoint axis `axis`.
    pub fn stride(&self, axis: usize) -> Result<usize, OperationError> {
        Ok(self.block.strides()[self.parent.storage_axis(axis)?])
    }
}

/// Materialize adjoint payload bytes between an already-admitted source and
/// destination layout.
///
/// Per logical block this is one strided copy with conjugation from the
/// parent block that [`AdjointBlockMap`] names, through the same owner
/// (`overwrite_owned_blocks`, layout normalized and bounds checked once per
/// block, output left unfilled when its blocks tile it) that the oriented add
/// path uses. The receiver-sized output copy itself is the retained limitation
/// recorded on #1177.
///
/// This contains no provider work: callers must derive and admit
/// `adjoint_space` transactionally before publishing a lazy view.
#[doc(hidden)]
pub fn materialize_adjoint_data_dyn<D>(
    space: &DynamicFusionMapSpace,
    adjoint_space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<Vec<D>, OperationError>
where
    D: AdjointScalar,
{
    validate_adjoint_data_extent(space.required_len(), data.len())?;
    let map = AdjointBlockMap::new(space, adjoint_space);
    overwrite_owned_blocks(adjoint_space.structure(), |block, writer| {
        let Some(source) = map.source(block)? else {
            return writer.zero();
        };
        writer.copy(
            |axis| source.stride(axis),
            data,
            source.offset(),
            true,
            D::one(),
        )
    })
}

/// Dynamic-rank adjoint that retains the exact provider allocation of its
/// checked source space.
pub fn adjoint_bound_dyn<R, D>(
    space: &BoundDynamicFusionMapSpace<R>,
    data: &[D],
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: AdjointScalar,
{
    let (output, data) =
        adjoint_dyn_with_primer(space.provider(), space.space(), data, space.layout_primer())?;
    let output = BoundDynamicFusionMapSpace::from_derived_like(space, output)?;
    Ok((output, data))
}

#[cfg(test)]
/// Generic dynamic-rank adjoint that retains the exact provider allocation of
/// its checked source space.
pub fn adjoint_bound_dyn_generic<R, D>(
    space: &BoundDynamicFusionMapSpace<R>,
    data: &[D],
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), OperationError>
where
    R: FusionRule,
    D: AdjointScalar,
{
    let (output, data) = adjoint_dyn_generic(space.provider(), space.space(), data)?;
    let output =
        BoundDynamicFusionMapSpace::from_derived(Arc::clone(space.provider_arc()), output)?;
    Ok((output, data))
}

/// Lazy-adjoint metadata retaining the exact provider allocation of the
/// checked source space.
pub fn adjoint_bound_space_dyn<R>(
    space: &BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let output =
        adjoint_space_dyn_with_primer(space.provider(), space.space(), space.layout_primer())?;
    BoundDynamicFusionMapSpace::from_derived_like(space, output)
}

#[cfg(test)]
/// Generic lazy-adjoint metadata retaining the exact provider allocation of
/// the checked source space.
pub fn adjoint_bound_space_dyn_generic<R>(
    space: &BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: FusionRule,
{
    let output = adjoint_space_dyn_generic(space.provider(), space.space())?;
    BoundDynamicFusionMapSpace::from_derived(Arc::clone(space.provider_arc()), output)
}

/// Checked Generic lazy-adjoint metadata. Structural admission and all
/// provider queries complete before the derived layout is published.
pub fn adjoint_bound_space_dyn_generic_checked<R>(
    space: &BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynamicFusionMapSpace<R>, CheckedGenericPlanError<R::Error>>
where
    R: CheckedGenericFusion,
{
    BoundDynamicFusionMapSpace::from_final_homspace_generic_checked_with(
        Arc::clone(space.provider_arc()),
        || {
            let homspace = space.space().homspace();
            let adjoint_hom =
                FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone());
            // Why not enumerate for a Complete source: it holds every canonical
            // key, so every swapped adjoint key is present, and a warm call would
            // re-enumerate the trees with provider queries only to confirm it.
            if matches!(
                space.space().admission(),
                symtn_core::FusionSpaceAdmission::Complete(_)
            ) {
                return Ok(adjoint_hom);
            }
            let structure = Arc::clone(space.space().structure());
            let keys = adjoint_hom
                .fusion_tree_keys_generic_checked(space.provider())
                .map_err(CheckedGenericPlanError::from)?;
            for key in keys {
                let source_key = BlockKey::FusionTree(FusionTreePairKey::pair(
                    key.domain_tree().clone(),
                    key.codomain_tree().clone(),
                ));
                if structure.find_block_index_by_key(&source_key).is_none() {
                    return Err(CheckedGenericPlanError::Operation(
                        OperationError::MissingBlockKey {
                            key: Box::new(source_key),
                        },
                    ));
                }
            }
            Ok(adjoint_hom)
        },
    )
}

#[cfg(test)]
/// Generic-fusion (SU(N)) sibling of [`adjoint_space_dyn`]. The adjoint is a
/// pure per-block relabel — codomain and domain trees swap wholesale and each
/// coupled block is transposed — with NO leg bending and NO recoupling, so it
/// is self-duality-independent and identical for a Generic (outer-multiplicity)
/// rule: the only difference from the mult-free path is that the block keys are
/// enumerated multiplicity-aware (vertex-labelled fusion trees). Bound relaxes
/// to [`FusionRule`] accordingly (no F/R symbols are touched).
pub(crate) fn adjoint_space_dyn_generic<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
) -> Result<DynamicFusionMapSpace, OperationError>
where
    R: FusionRule,
{
    space.validate_rule(rule)?;
    let homspace = space.homspace();
    let adjoint_hom =
        FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone());

    let structure = Arc::clone(space.structure());
    let keys = adjoint_hom
        .fusion_tree_keys_generic(rule)
        .map_err(OperationError::from_core_preserving_context)?;
    for key in keys.iter() {
        // Why not rely only on the bound source proof: the cold generic
        // adjoint has historically reported its target-ordered missing key
        // before constructing the target layout.
        let source_key = BlockKey::FusionTree(FusionTreePairKey::pair(
            key.domain_tree().clone(),
            key.codomain_tree().clone(),
        ));
        if structure.find_block_index_by_key(&source_key).is_none() {
            return Err(OperationError::MissingBlockKey {
                key: Box::new(source_key),
            });
        }
    }

    DynamicFusionMapSpace::from_final_homspace_generic(rule, adjoint_hom)
}

#[cfg(test)]
/// Generic-fusion (SU(N)) sibling of [`adjoint_dyn`]: same block relabel +
/// conjugate-transpose, over the multiplicity-aware adjoint space (see
/// [`adjoint_space_dyn_generic`]). The data movement is byte-identical to the
/// mult-free path — no recoupling coefficients enter — so this is the eager
/// materialization TensorKit takes when an SU(N) adjoint is consumed by
/// something other than a contraction.
pub(crate) fn adjoint_dyn_generic<R, D>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    data: &[D],
) -> Result<(DynamicFusionMapSpace, Vec<D>), OperationError>
where
    R: FusionRule,
    D: AdjointScalar,
{
    space.validate_rule(rule)?;
    validate_adjoint_data_extent(space.required_len(), data.len())?;
    let adjoint_space = adjoint_space_dyn_generic(rule, space)?;
    let result = materialize_adjoint_data_dyn(space, &adjoint_space, data)?;
    Ok((adjoint_space, result))
}

#[cfg(test)]
#[path = "../../tests/support/generic_toy_fusion.rs"]
mod generic_toy_fusion;

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::probe::{self, Event};
    use num_complex::Complex64;
    use std::cell::Cell;
    use std::ops::{Add, Mul};
    use symtn_core::{
        BlockSpec, FermionParityFusionRule, FusionProductSpace, FusionTreeKey, Fz2SectorLayout,
        PackedProductCodec, ProductFusionRule, ProductSectorCodec, ProductSectorLayout,
        SU2FusionRule, SU2Irrep, SectorId, SectorLeg, Su2SectorLayout, U1FusionRule, U1Irrep,
        U1SectorLayout, Z2Irrep,
    };

    type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
    type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
    type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

    thread_local! {
        static LOWERED_PRIMER_CALLS: Cell<usize> = const { Cell::new(0) };
        static OUTPUT_ZERO_CALLS: Cell<usize> = const { Cell::new(0) };
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct OutputObservedValue(f64);

    impl Add for OutputObservedValue {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl num_traits::Zero for OutputObservedValue {
        fn zero() -> Self {
            OUTPUT_ZERO_CALLS.with(|calls| calls.set(calls.get() + 1));
            Self(0.0)
        }

        fn is_zero(&self) -> bool {
            self.0 == 0.0
        }
    }

    impl Mul for OutputObservedValue {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self(self.0 * rhs.0)
        }
    }

    impl num_traits::One for OutputObservedValue {
        fn one() -> Self {
            Self(1.0)
        }
    }

    impl strided_kernel::ElementOpApply for OutputObservedValue {}

    impl ConjugateValue for OutputObservedValue {
        fn maybe_conj(self, _conjugate: bool) -> Self {
            self
        }
    }

    use super::generic_toy_fusion::ToyGenericRule;
    use crate::tests::TestTensor;

    fn triple_rule() -> TripleRule {
        TripleRule::new(
            Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
            SU2FusionRule,
        )
    }

    fn triple_sector(parity: u8, charge: i32, twice_spin: usize) -> SectorId {
        TripleCodec::encode(
            Fz2U1Codec::encode(
                Z2Irrep::new(parity).sector_id(),
                U1Irrep::new(charge).sector_id(),
            ),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    }

    fn triple_source(rule: &TripleRule) -> DynamicFusionMapSpace {
        let vacuum = triple_sector(0, 0, 0);
        let charged = triple_sector(1, 1, 1);
        let leg = |dual| SectorLeg::new([(vacuum, 1), (charged, 1)], dual);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false), leg(true)]),
            FusionProductSpace::new([leg(true), leg(false)]),
        );
        checked_layout_primer(rule, &hom).unwrap();
        let count = hom.fusion_tree_keys(rule).len();
        DynamicFusionMapSpace::from_degeneracy_shapes(rule, hom, vec![vec![1; 4]; count]).unwrap()
    }

    fn counting_primer(
        rule: &TripleRule,
        request: MetadataRequest<'_>,
    ) -> Result<MetadataOutput, OperationError> {
        LOWERED_PRIMER_CALLS.with(|calls| calls.set(calls.get() + 1));
        checked_metadata_dispatcher(rule, request)
    }

    // Single-charge U(1) source: one coupled sector, block shape [deg, deg].
    fn u1_source(charge: i32, deg: usize) -> DynamicFusionMapSpace {
        let rule = U1FusionRule;
        let sid = U1Irrep::new(charge).sector_id();
        let leg = || FusionProductSpace::new([SectorLeg::new([(sid, deg)], false)]);
        let hom = FusionTreeHomSpace::new(leg(), leg());
        let count = hom.fusion_tree_keys(&rule).len();
        DynamicFusionMapSpace::from_degeneracy_shapes(&rule, hom, vec![vec![deg, deg]; count])
            .unwrap()
    }

    fn rank_one_source<R>(rule: &R, sector: SectorId, deg: usize) -> DynamicFusionMapSpace
    where
        R: MultiplicityFreeRigidSymbols,
    {
        let leg = || FusionProductSpace::new([SectorLeg::new([(sector, deg)], false)]);
        let hom = FusionTreeHomSpace::new(leg(), leg());
        let count = hom.fusion_tree_keys(rule).len();
        DynamicFusionMapSpace::from_degeneracy_shapes(rule, hom, vec![vec![deg, deg]; count])
            .unwrap()
    }

    fn u1_rank_one_pair(charge: i32) -> FusionTreePairKey {
        let rule = U1FusionRule;
        let sector = U1Irrep::new(charge).sector_id();
        let tree = FusionTreeKey::try_new_for_rule(
            &rule,
            [sector],
            sector,
            [false],
            std::iter::empty(),
            std::iter::empty(),
        )
        .unwrap();
        FusionTreePairKey::pair(tree.clone(), tree)
    }

    fn u1_expert_space(
        homspace: FusionTreeHomSpace,
        structure: BlockStructure,
    ) -> Result<DynamicFusionMapSpace, CoreError> {
        DynamicFusionMapSpace::try_from_layout(homspace, structure)?.try_bind_rule(&U1FusionRule)
    }

    fn u1_expert_tensor(
        homspace: FusionTreeHomSpace,
        structure: BlockStructure,
    ) -> TestTensor<OutputObservedValue> {
        let space = u1_expert_space(homspace, structure).unwrap();
        let len = space.required_len().unwrap();
        TestTensor::from_vec_with_fusion_space(vec![OutputObservedValue(1.0); len], space).unwrap()
    }

    fn toy_generic_bound<const ID: u8>() -> BoundDynamicFusionMapSpace<ToyGenericRule<ID>> {
        let one = SectorId::new(1);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(one, 2)], false),
                SectorLeg::new([(one, 3)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(one, 5)], false)]),
        );
        let provider = Arc::new(ToyGenericRule::<ID>);
        let count = homspace
            .fusion_tree_keys_generic(provider.as_ref())
            .unwrap()
            .len();
        assert_eq!(count, 2);
        BoundDynamicFusionMapSpace::from_degeneracy_shapes_generic(
            provider,
            homspace,
            vec![vec![2, 3, 5]; count],
        )
        .unwrap()
    }

    fn u1_tensor() -> TestTensor<OutputObservedValue> {
        let rule = U1FusionRule;
        let one = U1Irrep::new(1).sector_id();
        let leg = || FusionProductSpace::new([SectorLeg::new([(one, 2)], false)]);
        let homspace = FusionTreeHomSpace::new(leg(), leg());
        let count = homspace.fusion_tree_keys(&rule).len();
        let space =
            DynamicFusionMapSpace::from_degeneracy_shapes(&rule, homspace, vec![vec![2, 2]; count])
                .unwrap();
        let len = space.required_len().unwrap();
        TestTensor::from_vec_with_fusion_space(vec![OutputObservedValue(1.0); len], space).unwrap()
    }

    fn canonical_dynamic_source<R>(rule: &R, homspace: FusionTreeHomSpace) -> DynamicFusionMapSpace
    where
        R: MultiplicityFreeRigidSymbols,
    {
        let structure = homspace
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        let shapes = homspace
            .fusion_tree_keys(rule)
            .iter()
            .map(|key| {
                let index = structure.find_block_index_by_fusion_tree_pair(key).unwrap();
                structure.block(index).unwrap().shape().to_vec()
            })
            .collect::<Vec<_>>();
        DynamicFusionMapSpace::from_degeneracy_shapes(rule, homspace, shapes).unwrap()
    }

    fn assert_complete_structure_eq(actual: &BlockStructure, expected: &BlockStructure) {
        assert_eq!(actual, expected);
        assert_eq!(actual.rank(), expected.rank());
        assert_eq!(actual.block_count(), expected.block_count());
        assert_eq!(
            actual.required_len().unwrap(),
            expected.required_len().unwrap()
        );
        for index in 0..expected.block_count() {
            let actual = actual.block(index).unwrap();
            let expected = expected.block(index).unwrap();
            assert_eq!(actual.key(), expected.key());
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.strides(), expected.strides());
            assert_eq!(actual.offset(), expected.offset());
        }
    }

    fn adjoint_data_oracle(
        source: &DynamicFusionMapSpace,
        target: &DynamicFusionMapSpace,
        data: &[Complex64],
    ) -> Vec<Complex64> {
        let nout = source.nout();
        let nin = source.nin();
        let mut result = vec![Complex64::new(0.0, 0.0); target.required_len().unwrap()];
        for index in 0..target.structure().block_count() {
            let target_block = target.structure().block(index).unwrap();
            let BlockKey::FusionTree(target_key) = target_block.key() else {
                panic!("adjoint oracle requires fusion-tree blocks");
            };
            let source_key = FusionTreePairKey::pair(
                target_key.domain_tree().clone(),
                target_key.codomain_tree().clone(),
            );
            let source_index = source
                .structure()
                .find_block_index_by_fusion_tree_pair(&source_key)
                .unwrap();
            let source_block = source.structure().block(source_index).unwrap();
            let mut indices = vec![0usize; target_block.shape().len()];
            let count = target_block.shape().iter().product::<usize>();
            for _ in 0..count {
                let target_position = target_block.offset()
                    + indices
                        .iter()
                        .zip(target_block.strides())
                        .map(|(&axis, &stride)| axis * stride)
                        .sum::<usize>();
                let source_position = source_block.offset()
                    + indices[nin..]
                        .iter()
                        .zip(&source_block.strides()[..nout])
                        .map(|(&axis, &stride)| axis * stride)
                        .sum::<usize>()
                    + indices[..nin]
                        .iter()
                        .zip(&source_block.strides()[nout..])
                        .map(|(&axis, &stride)| axis * stride)
                        .sum::<usize>();
                result[target_position] = data[source_position].conj();
                for (axis, index) in indices.iter_mut().enumerate() {
                    *index += 1;
                    if *index < target_block.shape()[axis] {
                        break;
                    }
                    *index = 0;
                }
            }
        }
        result
    }

    fn assert_rank_one_final_layout<R>(
        rule: &R,
        sector: SectorId,
        codomain_degeneracy: usize,
        domain_degeneracy: usize,
        codomain_dual: bool,
        domain_dual: bool,
    ) where
        R: MultiplicityFreeRigidSymbols,
    {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                [(sector, codomain_degeneracy)],
                codomain_dual,
            )]),
            FusionProductSpace::new([SectorLeg::new([(sector, domain_degeneracy)], domain_dual)]),
        );
        let source = canonical_dynamic_source(rule, homspace.clone());
        let data = (0..source.required_len().unwrap())
            .map(|index| Complex64::new(index as f64 + 1.0, index as f64 + 0.25))
            .collect::<Vec<_>>();
        let stepwise =
            build_adjoint_space_dyn_stepwise_oracle(rule, &source, encoded_layout_primer::<R>)
                .unwrap();
        let expected_data = adjoint_data_oracle(&source, &stepwise, &data);

        let (dynamic, dynamic_data) = adjoint_dyn(rule, &source, &data).unwrap();
        assert_complete_structure_eq(dynamic.structure(), stepwise.structure());
        assert_eq!(dynamic.nout(), 1);
        assert_eq!(dynamic.nin(), 1);
        assert_eq!(dynamic.homspace(), stepwise.homspace());
        assert_eq!(dynamic_data, expected_data);
    }

    #[test]
    fn final_homspace_is_the_adjoint_layout_authority_for_rank_one_rules() {
        // What: asymmetric U(1), half-integer SU(2), and odd fZ2 typed and
        // dynamic daggers preserve the complete layout and conjugate
        // transpose numbers without a pivotal or fermionic phase.
        assert_rank_one_final_layout(
            &U1FusionRule,
            U1Irrep::new(1).sector_id(),
            2,
            3,
            false,
            false,
        );
        assert_rank_one_final_layout(
            &SU2FusionRule,
            SU2Irrep::from_twice_spin(1).sector_id(),
            3,
            2,
            false,
            true,
        );
        assert_rank_one_final_layout(
            &FermionParityFusionRule,
            Z2Irrep::ODD.sector_id(),
            2,
            3,
            true,
            false,
        );
    }

    #[test]
    fn final_homspace_is_the_adjoint_layout_authority_for_product_rule() {
        // What: a mixed-duality fZ2 x U(1) x SU(2) tensor keeps every ordered
        // block key, shape, stride, offset, and number from the independent
        // source-shape oracle, with no extra categorical phase.
        let rule = triple_rule();
        let vacuum = triple_sector(0, 0, 0);
        let charged = triple_sector(1, 1, 1);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(vacuum, 1), (charged, 2)], false),
                SectorLeg::new([(vacuum, 2), (charged, 1)], true),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(vacuum, 1), (charged, 3)], true),
                SectorLeg::new([(vacuum, 2), (charged, 2)], false),
            ]),
        );
        let source = canonical_dynamic_source(&rule, homspace.clone());
        assert!(source.structure().block_count() > 1);
        let data = (0..source.required_len().unwrap())
            .map(|index| Complex64::new(index as f64 + 0.5, index as f64 + 0.125))
            .collect::<Vec<_>>();
        let stepwise = build_adjoint_space_dyn_stepwise_oracle(
            &rule,
            &source,
            encoded_layout_primer::<TripleRule>,
        )
        .unwrap();
        let expected_data = adjoint_data_oracle(&source, &stepwise, &data);

        let (dynamic, dynamic_data) = adjoint_dyn(&rule, &source, &data).unwrap();
        assert_complete_structure_eq(dynamic.structure(), stepwise.structure());
        assert_eq!(dynamic.nout(), 2);
        assert_eq!(dynamic.nin(), 2);
        assert_eq!(dynamic.homspace(), stepwise.homspace());
        assert_eq!(dynamic_data, expected_data);
    }

    #[test]
    fn eager_adjoint_accepts_exact_multiplicity_free_storage() {
        // What: exact raw storage succeeds for the ordinary U(1), SU(2), fZ2,
        // and fZ2 x U(1) x SU(2) multiplicity-free paths.
        let u1 = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            u1_source(1, 2),
            Arc::new(U1FusionRule),
        )
        .unwrap();
        let u1_len = u1.space().required_len().unwrap();
        let (_, u1_data) = adjoint_bound_dyn(&u1, &vec![1.0; u1_len]).unwrap();
        assert_eq!(u1_data.len(), u1_len);

        let su2_rule = Arc::new(SU2FusionRule);
        let su2 = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            rank_one_source(
                su2_rule.as_ref(),
                SU2Irrep::from_twice_spin(1).sector_id(),
                2,
            ),
            Arc::clone(&su2_rule),
        )
        .unwrap();
        let su2_len = su2.space().required_len().unwrap();
        let (_, su2_data) = adjoint_bound_dyn(&su2, &vec![1.0; su2_len]).unwrap();
        assert_eq!(su2_data.len(), su2_len);

        let fz2_rule = Arc::new(FermionParityFusionRule);
        let fz2 = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            rank_one_source(fz2_rule.as_ref(), Z2Irrep::ODD.sector_id(), 2),
            Arc::clone(&fz2_rule),
        )
        .unwrap();
        let fz2_len = fz2.space().required_len().unwrap();
        let (_, fz2_data) = adjoint_bound_dyn(&fz2, &vec![1.0; fz2_len]).unwrap();
        assert_eq!(fz2_data.len(), fz2_len);

        let rule = triple_rule();
        let product = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            triple_source(&rule),
            Arc::new(rule),
        )
        .unwrap();
        let product_len = product.space().required_len().unwrap();
        let (_, product_data) = adjoint_bound_dyn(&product, &vec![1.0; product_len]).unwrap();
        assert_eq!(product_data.len(), product_len);
    }

    #[test]
    fn rule_binding_rejects_extra_expert_block_before_adjoint_target_work() {
        // What: a locally valid U(1) block outside the source HomSpace cannot
        // acquire the rule identity required to reach adjoint target work.
        let zero = U1Irrep::new(0).sector_id();
        let leg = || FusionProductSpace::new([SectorLeg::new([(zero, 1)], false)]);
        let homspace = FusionTreeHomSpace::new(leg(), leg());
        let structure = BlockStructure::from_blocks_with_rank(
            2,
            vec![
                BlockSpec::column_major_with_key(u1_rank_one_pair(0).into(), vec![1, 1], 0)
                    .unwrap(),
                BlockSpec::column_major_with_key(u1_rank_one_pair(1).into(), vec![1, 1], 1)
                    .unwrap(),
            ],
        )
        .unwrap();
        OUTPUT_ZERO_CALLS.with(|calls| calls.set(0));
        let error = u1_expert_space(homspace, structure).unwrap_err();
        assert_eq!(
            error,
            CoreError::MalformedFusionTree {
                message: "fusion tree uses a sector absent from its HomSpace leg",
            }
        );
        assert_eq!(OUTPUT_ZERO_CALLS.with(Cell::get), 0);
    }

    #[test]
    fn eager_adjoint_keeps_rule_mismatch_precedence_over_canonical_preflight() {
        // What: a wrong rule is rejected before canonical preflight inspects a
        // safe public expert structure whose complete blocks are reordered.
        let zero = U1Irrep::new(0).sector_id();
        let one = U1Irrep::new(1).sector_id();
        let leg = || FusionProductSpace::new([SectorLeg::new([(zero, 1), (one, 1)], false)]);
        let homspace = FusionTreeHomSpace::new(leg(), leg());
        let canonical = homspace
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap();
        let structure = BlockStructure::from_blocks_with_rank(
            2,
            [1, 0]
                .into_iter()
                .map(|index| {
                    let block = canonical.block(index).unwrap();
                    BlockSpec::with_key(
                        block.key().clone(),
                        block.shape().to_vec(),
                        block.strides().to_vec(),
                        block.offset(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        let tensor = u1_expert_tensor(homspace, structure);

        OUTPUT_ZERO_CALLS.with(|calls| calls.set(0));
        let error = tensor.adjoint(&SU2FusionRule).unwrap_err();

        assert!(matches!(
            error,
            OperationError::Core(CoreError::FusionRuleMismatch { .. })
        ));
        assert_eq!(OUTPUT_ZERO_CALLS.with(Cell::get), 0);
    }

    #[test]
    fn eager_adjoint_rejects_missing_expert_block_before_target_work() {
        // What: omitting one HomSpace block fails with the canonical block count
        // before the output vector is allocated or initialized.
        let sectors = [
            (U1Irrep::new(0).sector_id(), 1),
            (U1Irrep::new(1).sector_id(), 1),
        ];
        let leg = || FusionProductSpace::new([SectorLeg::new(sectors, false)]);
        let homspace = FusionTreeHomSpace::new(leg(), leg());
        let structure = BlockStructure::from_blocks_with_rank(
            2,
            vec![
                BlockSpec::column_major_with_key(u1_rank_one_pair(0).into(), vec![1, 1], 0)
                    .unwrap(),
            ],
        )
        .unwrap();
        let tensor = u1_expert_tensor(homspace, structure);

        OUTPUT_ZERO_CALLS.with(|calls| calls.set(0));
        let error = tensor.adjoint(&U1FusionRule).unwrap_err();

        assert_eq!(
            error,
            OperationError::Core(CoreError::BlockCountMismatch {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(OUTPUT_ZERO_CALLS.with(Cell::get), 0);
    }

    #[test]
    fn rule_binding_rejects_wrong_expert_shape_before_adjoint_target_work() {
        // What: a source block whose degeneracy shape disagrees with its
        // HomSpace cannot acquire the rule identity required by adjoint.
        let zero = U1Irrep::new(0).sector_id();
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(zero, 2)], false)]),
            FusionProductSpace::new([SectorLeg::new([(zero, 3)], false)]),
        );
        let structure = BlockStructure::from_blocks_with_rank(
            2,
            vec![
                BlockSpec::column_major_with_key(u1_rank_one_pair(0).into(), vec![2, 2], 0)
                    .unwrap(),
            ],
        )
        .unwrap();

        OUTPUT_ZERO_CALLS.with(|calls| calls.set(0));
        let error = u1_expert_space(homspace, structure).unwrap_err();

        assert_eq!(
            error,
            CoreError::LegDegeneracyMismatch {
                sector: zero,
                expected: 3,
                actual: 2,
            }
        );
        assert_eq!(OUTPUT_ZERO_CALLS.with(Cell::get), 0);
    }

    #[test]
    fn eager_adjoint_rejects_noncanonical_order_and_coverage() {
        // What: complete expert keys still fail when their source block order
        // or storage coverage differs from the canonical coupled layout.
        let sectors = [
            (U1Irrep::new(0).sector_id(), 1),
            (U1Irrep::new(1).sector_id(), 1),
        ];
        let leg = || FusionProductSpace::new([SectorLeg::new(sectors, false)]);
        let homspace = FusionTreeHomSpace::new(leg(), leg());
        let canonical = homspace
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap();

        let reordered = BlockStructure::from_blocks_with_rank(
            2,
            [1, 0]
                .into_iter()
                .map(|index| {
                    let block = canonical.block(index).unwrap();
                    BlockSpec::with_key(
                        block.key().clone(),
                        block.shape().to_vec(),
                        block.strides().to_vec(),
                        block.offset(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        let gapped = BlockStructure::from_blocks_with_rank(
            2,
            (0..canonical.block_count())
                .map(|index| {
                    let block = canonical.block(index).unwrap();
                    BlockSpec::with_key(
                        block.key().clone(),
                        block.shape().to_vec(),
                        block.strides().to_vec(),
                        block.offset() + index,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();

        for structure in [reordered, gapped] {
            let tensor = u1_expert_tensor(homspace.clone(), structure);
            OUTPUT_ZERO_CALLS.with(|calls| calls.set(0));

            let error = tensor.adjoint(&U1FusionRule).unwrap_err();

            assert_eq!(error, OperationError::StructureMismatch { tensor: "src" });
            assert_eq!(OUTPUT_ZERO_CALLS.with(Cell::get), 0);
        }
    }

    #[test]
    fn lazy_adjoint_rejects_noncanonical_source() {
        // What: metadata-only dagger construction rejects noncanonical source
        // ordering after confirming that every canonical key is present.
        let zero = U1Irrep::new(0).sector_id();
        let one = U1Irrep::new(1).sector_id();

        let sectors = [(zero, 1), (one, 1)];
        let double_leg = || FusionProductSpace::new([SectorLeg::new(sectors, false)]);
        let double_homspace = FusionTreeHomSpace::new(double_leg(), double_leg());
        let canonical = double_homspace
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap();
        let reordered = BlockStructure::from_blocks_with_rank(
            2,
            [1, 0]
                .into_iter()
                .map(|index| {
                    let block = canonical.block(index).unwrap();
                    BlockSpec::with_key(
                        block.key().clone(),
                        block.shape().to_vec(),
                        block.strides().to_vec(),
                        block.offset(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        let reordered_tensor = u1_expert_tensor(double_homspace, reordered);
        let reordered = reordered_tensor.dynamic_space().clone();

        let reordered_error = adjoint_space_dyn(&U1FusionRule, &reordered).unwrap_err();
        assert_eq!(
            reordered_error,
            OperationError::StructureMismatch { tensor: "src" }
        );
    }

    #[test]
    fn duplicate_expert_blocks_are_rejected_at_construction() {
        // What: safe expert construction rejects duplicate fusion-tree keys
        // before a malformed source can reach eager adjoint.
        let key = u1_rank_one_pair(0);
        let error = BlockStructure::from_blocks_with_rank(
            2,
            vec![
                BlockSpec::column_major_with_key(key.clone().into(), vec![1, 1], 0).unwrap(),
                BlockSpec::column_major_with_key(key.into(), vec![1, 1], 1).unwrap(),
            ],
        )
        .unwrap_err();

        assert!(matches!(error, CoreError::DuplicateBlockKey { .. }));
    }

    #[test]
    fn eager_adjoint_rejects_inexact_storage_before_target_work() {
        // What: short and oversized ordinary storage fail before target
        // preparation, layout publication, or output zero-fill.
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        symtn_core::clear_structure_caches();
        symtn_core::clear_structure_caches();
        let rule = triple_rule();
        let source = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            triple_source(&rule),
            Arc::new(rule),
        )
        .unwrap()
        .with_test_layout_primer(counting_primer);
        let expected = source.space().required_len().unwrap();

        for actual in [expected - 1, expected + 1] {
            LOWERED_PRIMER_CALLS.with(|calls| calls.set(0));
            OUTPUT_ZERO_CALLS.with(|calls| calls.set(0));
            let scratch_structure_builds_rec = probe::record();
            let data = vec![OutputObservedValue(1.0); actual];

            let error = adjoint_bound_dyn(&source, &data).unwrap_err();

            assert_eq!(
                error,
                OperationError::ElementCountMismatch { expected, actual }
            );
            assert_eq!(LOWERED_PRIMER_CALLS.with(Cell::get), 0);
            assert_eq!(OUTPUT_ZERO_CALLS.with(Cell::get), 0);
            assert_eq!(
                scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
                0
            );
        }
    }

    #[test]
    fn eager_adjoint_keeps_rule_mismatch_precedence_over_extent() {
        // What: source rule validation remains the first eager-adjoint error
        // even when raw storage is also short.
        let source = u1_source(1, 2);
        let error = adjoint_dyn(&SU2FusionRule, &source, &[] as &[f64]).unwrap_err();
        assert!(matches!(
            error,
            OperationError::Core(CoreError::FusionRuleMismatch { .. })
        ));
    }

    #[test]
    fn generic_eager_adjoint_enforces_the_same_extent_boundary() {
        // What: a genuinely outer-multiplicity dynamic space accepts exact
        // storage and rejects short/oversized storage before output zero-fill.
        let source = toy_generic_bound::<1>();
        let expected = source.space().required_len().unwrap();
        let (_, exact) =
            adjoint_bound_dyn_generic(&source, &vec![OutputObservedValue(1.0); expected]).unwrap();
        assert_eq!(exact.len(), expected);

        for actual in [expected - 1, expected + 1] {
            OUTPUT_ZERO_CALLS.with(|calls| calls.set(0));
            let data = vec![OutputObservedValue(1.0); actual];
            let error = adjoint_bound_dyn_generic(&source, &data).unwrap_err();
            assert_eq!(
                error,
                OperationError::ElementCountMismatch { expected, actual }
            );
            assert_eq!(OUTPUT_ZERO_CALLS.with(Cell::get), 0);
        }

        let wrong_rule = ToyGenericRule::<2>;
        let error = adjoint_dyn_generic(&wrong_rule, source.space(), &[] as &[OutputObservedValue])
            .unwrap_err();
        assert!(matches!(
            error,
            OperationError::Core(CoreError::FusionRuleMismatch { .. })
        ));
    }

    #[test]
    fn generic_adjoint_final_homspace_matches_reconstructed_layout() {
        // What: a genuine outer-multiplicity adjoint keeps the reconstructed tree
        // order, shapes, strides, offsets, and storage length when its target
        // layout is derived directly from the swapped final HomSpace.
        let source = toy_generic_bound::<1>();
        let source_space = source.space();
        let source_structure = source_space.structure();
        let source_nout = source_space.nout();
        let homspace = source_space.homspace();
        let adjoint_hom =
            FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone());
        let keys = adjoint_hom
            .fusion_tree_keys_generic(source.provider())
            .unwrap();
        let shapes = keys
            .iter()
            .map(|key| {
                let source_key = BlockKey::FusionTree(FusionTreePairKey::pair(
                    key.domain_tree().clone(),
                    key.codomain_tree().clone(),
                ));
                let source_block = source_structure
                    .find_block_index_by_key(&source_key)
                    .and_then(|index| source_structure.block(index).ok())
                    .unwrap();
                let mut shape = source_block.shape()[source_nout..].to_vec();
                shape.extend_from_slice(&source_block.shape()[..source_nout]);
                shape
            })
            .collect::<Vec<_>>();
        let reconstructed = DynamicFusionMapSpace::from_degeneracy_shapes_generic(
            source.provider(),
            adjoint_hom,
            shapes,
        )
        .unwrap();

        let canonical = adjoint_space_dyn_generic(source.provider(), source_space).unwrap();

        assert_complete_structure_eq(canonical.structure(), reconstructed.structure());
        assert_eq!(canonical.homspace(), reconstructed.homspace());
    }

    #[test]
    fn eager_adjoint_accepts_exact_storage() {
        // What: the eager adjoint keeps its ordinary exact-storage path.
        let exact = u1_tensor();
        let expected = exact.structure().required_len().unwrap();
        OUTPUT_ZERO_CALLS.with(|calls| calls.set(0));
        let output = exact.adjoint(&U1FusionRule).unwrap();
        assert_eq!(output.data().len(), expected);
        // #1290: the canonical output tiles its storage, so no element is
        // zero-filled before its block overwrites it.
        assert_eq!(OUTPUT_ZERO_CALLS.with(Cell::get), 0);
    }

    #[test]
    fn repeated_metadata_calls_ignore_global_operation_cache_reset() {
        // What: repeated eager metadata builds retain the complete structural
        // result before and after an unrelated global operation-cache reset.
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rule = U1FusionRule;
        let src = u1_source(1, 2);
        let first = adjoint_space_dyn(&rule, &src).unwrap();
        let second = adjoint_space_dyn(&rule, &src).unwrap();
        symtn_core::clear_structure_caches();
        let after_reset = adjoint_space_dyn(&rule, &src).unwrap();

        assert_eq!(first.homspace(), second.homspace());
        assert_eq!(first.homspace(), after_reset.homspace());
        assert_complete_structure_eq(first.structure(), second.structure());
        assert_complete_structure_eq(first.structure(), after_reset.structure());
    }

    #[test]
    fn lowered_adjoint_metadata_and_eager_data_match_encoded() {
        // What: one lowered eager metadata build and eager materialization
        // preserve the encoded oracle's layout and data.
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        symtn_core::clear_structure_caches();
        symtn_core::clear_structure_caches();
        let rule = triple_rule();
        let source = triple_source(&rule);

        LOWERED_PRIMER_CALLS.with(|calls| calls.set(0));
        let lowered = adjoint_space_dyn_with_primer(&rule, &source, counting_primer).unwrap();
        assert_eq!(LOWERED_PRIMER_CALLS.with(Cell::get), 1);
        symtn_core::clear_structure_caches();
        symtn_core::clear_structure_caches();
        let encoded_lazy = adjoint_space_dyn(&rule, &source).unwrap();
        assert_eq!(lowered, encoded_lazy);

        let data = (0..source.required_len().unwrap())
            .map(|index| index as f64 + 0.25)
            .collect::<Vec<_>>();
        symtn_core::clear_structure_caches();
        symtn_core::clear_structure_caches();
        let (lowered_space, lowered_data) = adjoint_dyn_with_primer(
            &rule,
            &source,
            &data,
            checked_metadata_dispatcher::<TripleRule>,
        )
        .unwrap();
        symtn_core::clear_structure_caches();
        symtn_core::clear_structure_caches();
        let (encoded_space, encoded_data) = adjoint_dyn(&rule, &source, &data).unwrap();
        assert_eq!(lowered_space, encoded_space);
        assert_eq!(lowered_data, encoded_data);
    }

    #[test]
    fn lowered_adjoint_missing_source_key_abandons_prepared_layout() {
        // What: adjoint source-key mapping fails before its checked target
        // layout receives an ID or publication.
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        symtn_core::clear_structure_caches();
        symtn_core::clear_structure_caches();
        let rule = U1FusionRule;
        let one = U1Irrep::new(1).sector_id();
        let two = U1Irrep::new(2).sector_id();
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(one, 1)], false),
                SectorLeg::new([(one, 1)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(two, 1)], false)]),
        );
        let layout = DynamicFusionMapSpace::try_from_layout(
            homspace,
            BlockStructure::from_blocks_with_rank(3, Vec::new()).unwrap(),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        let source = layout;
        let scratch_structure_builds_rec = probe::record();

        let error = adjoint_space_dyn_with_primer(
            &rule,
            &source,
            checked_metadata_dispatcher::<U1FusionRule>,
        )
        .unwrap_err();

        assert!(matches!(error, OperationError::MissingBlockKey { .. }));
        assert_eq!(
            scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
            0
        );
    }
}
