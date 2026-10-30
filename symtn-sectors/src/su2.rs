use fusion_symbols::{
    category::FusionCategory,
    lie::{GroupId, LieError, LieGroup, LieIrrep},
};
use std::sync::{Arc, OnceLock};

use crate::category_rule::{
    admit_coupled_first, admit_in_order, fusion_algebra_error, AdapterAccess,
    CanonicalUnitAdmission, CategoryAdapterError, CategoryCodec, CategoryOperation,
};
#[cfg(test)]
use crate::{
    CanonicalUnitFusionRule, FusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, SectorCodec,
};
use crate::{
    CheckedFusionAlgebra, FusionAlgebraError, MultiplicityFreeAdmissionMode, PhysicalBasisError,
    PhysicalFusionBasis, RuleIdentity, SectorId,
};

/// Largest doubled spin representable by symtn's compact SU(2) sector encoding.
pub const SU2_MAX_DOUBLED_SPIN: usize = (u8::MAX - 1) as usize;

/// symtn's compact SU(2) sector label (`twice_spin = 2j`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SU2Irrep {
    twice_spin: usize,
}

impl SU2Irrep {
    pub const fn try_from_twice_spin(twice_spin: usize) -> Option<Self> {
        if twice_spin <= SU2_MAX_DOUBLED_SPIN {
            Some(Self { twice_spin })
        } else {
            None
        }
    }

    pub const fn from_twice_spin(twice_spin: usize) -> Self {
        match Self::try_from_twice_spin(twice_spin) {
            Some(irrep) => irrep,
            None => panic!("SU(2) doubled spin exceeds the supported maximum 254"),
        }
    }

    #[inline]
    pub const fn twice_spin(self) -> usize {
        self.twice_spin
    }

    #[inline]
    pub const fn sector_id(self) -> SectorId {
        SectorId::new(self.twice_spin)
    }

    pub const fn try_from_sector_id(sector: SectorId) -> Option<Self> {
        Self::try_from_twice_spin(sector.id())
    }

    pub const fn from_sector_id(sector: SectorId) -> Self {
        match Self::try_from_sector_id(sector) {
            Some(irrep) => irrep,
            None => panic!("SU(2) sector exceeds the supported maximum doubled spin 254"),
        }
    }
}

impl From<SU2Irrep> for SectorId {
    fn from(value: SU2Irrep) -> Self {
        value.sector_id()
    }
}

/// SU(2) fusion data from [`fusion_symbols::lie::LieGroup`] through the
/// shared category adapter, in Racah's Condon--Shortley gauge. This wrapper
/// keeps symtn's compact doubled-spin codec and closure bound; fusion, N,
/// F, R, A, B, dimensions, twists and Frobenius–Schur phase come from the
/// category.
///
/// The physical carrier basis follows the executable convention of
/// TensorKitSectors 0.3.9: zero-based index `k` has doubled projection
/// `dm = dj - 2*k`, hence `dm = dj, dj - 2, ..., -dj`. This is descending in
/// `m`. symtn validates sectors, the fusion channel, its multiplicity and the
/// carrier indices; each entry is then fusion-symbols' scalar
/// `su2::clebsch_gordan`, which is an exact zero for a projection mismatch,
/// without constructing a full CGC tensor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct SU2FusionRule;

const SU2_RACAH_IDENTITY_SCHEMA: u64 = 0x5355_3252_4143_4148;
const SU2_ENCODING_IDENTITY: &[u8] =
    b"symtn:su2:sector-id=dj:max-doubled-spin=254:channels=ascending-step-2";

fn category() -> &'static LieGroup {
    static CATEGORY: OnceLock<LieGroup> = OnceLock::new();
    CATEGORY.get_or_init(|| {
        LieGroup::new(GroupId::su(2).expect("2 is a valid SU(N) rank"))
            .expect("the SU(2) category is supported")
    })
}

fn identity() -> &'static RuleIdentity {
    static IDENTITY: OnceLock<RuleIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        RuleIdentity::from_canonical_bytes::<SU2FusionRule>(
            SU2_RACAH_IDENTITY_SCHEMA,
            Arc::<[u8]>::from(SU2_ENCODING_IDENTITY),
        )
    })
}

fn checked_irrep(sector: SectorId) -> Result<SU2Irrep, FusionAlgebraError> {
    SU2Irrep::try_from_sector_id(sector).ok_or(FusionAlgebraError::InvalidSector { sector })
}

/// The compact label codec and operation-specific closure admission.
#[derive(Debug)]
pub(crate) struct CompactSu2Codec;

impl CompactSu2Codec {
    fn check_closure(left: SectorId, right: SectorId) -> Result<(), FusionAlgebraError> {
        if left.id() + right.id() > SU2_MAX_DOUBLED_SPIN {
            Err(FusionAlgebraError::FusionNotRepresentable { left, right })
        } else {
            Ok(())
        }
    }
}

impl CategoryCodec<LieGroup> for CompactSu2Codec {
    type Sector = SU2Irrep;
    type Error = FusionAlgebraError;
    type Mode = MultiplicityFreeAdmissionMode;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _category: impl Fn() -> &'a LieGroup + Copy,
        _rule: &RuleIdentity,
        label: &SU2Irrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(label.sector_id())
    }

    fn decode_label<'a>(
        &self,
        _category: impl Fn() -> &'a LieGroup + Copy,
        id: SectorId,
    ) -> Result<SU2Irrep, FusionAlgebraError> {
        checked_irrep(id)
    }

    fn object<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        id: SectorId,
    ) -> Result<LieIrrep, FusionAlgebraError> {
        let label = checked_irrep(id)?;
        Ok(category()
            .irrep(&[label.twice_spin() as i64])
            .expect("symtn SU(2) labels fit fusion-symbols' rank-one labels"))
    }

    fn sector_id<'a>(
        &self,
        _category: impl Fn() -> &'a LieGroup + Copy,
        _rule: &RuleIdentity,
        object: &LieIrrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        let sector =
            SectorId::new(object.dynkin().next().expect("SU(2) labels have rank one") as usize);
        checked_irrep(sector).map(SU2Irrep::sector_id)
    }

    fn admit<'a, const N: usize>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        rule: &RuleIdentity,
        operation: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[LieIrrep; N], FusionAlgebraError> {
        let objects = if operation == CategoryOperation::N {
            admit_coupled_first(self, category, ids)?
        } else {
            admit_in_order(self, category, rule, operation, ids)?
        };
        if matches!(operation, CategoryOperation::Fusion | CategoryOperation::N) {
            Self::check_closure(ids[0], ids[1])?;
        }
        Ok(objects)
    }

    fn admit_r<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        _rule: &RuleIdentity,
        [left, right, coupled]: [SectorId; 3],
    ) -> Result<Option<[LieIrrep; 3]>, FusionAlgebraError> {
        let a = self.object(category, left)?;
        let b = self.object(category, right)?;
        Self::check_closure(left, right).unwrap_or_else(|_| {
            panic!("SU(2) fusion closure exceeds the supported maximum doubled spin 254")
        });
        // The compact closure makes every out-of-domain coupled id absent.
        if coupled.id() > SU2_MAX_DOUBLED_SPIN {
            return Ok(None);
        }
        let c = self.object(category, coupled)?;
        let n = category()
            .nsymbol(&a, &b, &c)
            .map_err(|error| fusion_algebra_error(CategoryAdapterError::Provider(error)))?;
        Ok((n != 0).then_some([a, b, c]))
    }

    fn map_error(
        &self,
        _inputs: &[SectorId],
        error: CategoryAdapterError<LieError>,
    ) -> FusionAlgebraError {
        fusion_algebra_error(error)
    }
}

impl crate::seal::Provider for SU2FusionRule {}

crate::category_rule::multiplicity_free_adapter_impls!(SU2FusionRule, SU2Irrep, f64);

impl AdapterAccess for SU2FusionRule {
    type Category = LieGroup;
    type Codec = CompactSu2Codec;

    fn category(&self) -> &LieGroup {
        category()
    }
    fn codec(&self) -> &CompactSu2Codec {
        &CompactSu2Codec
    }
    fn identity(&self) -> &RuleIdentity {
        identity()
    }
    fn vacuum_id(&self) -> SectorId {
        SU2Irrep::from_twice_spin(0).sector_id()
    }
}

impl SU2FusionRule {
    #[doc(hidden)]
    pub fn try_for_each_representable_channel<B>(
        &self,
        left: SU2Irrep,
        right: SU2Irrep,
        mut emit: impl FnMut(SU2Irrep) -> core::ops::ControlFlow<B>,
    ) -> Result<core::ops::ControlFlow<B>, FusionAlgebraError> {
        let [a, b] = self.codec().admit(
            category,
            self.identity(),
            CategoryOperation::Fusion,
            [left.sector_id(), right.sector_id()],
        )?;
        for channel in category()
            .fusion(&a, &b)
            .expect("symtn SU(2) label range cannot overflow fusion-symbols labels")
        {
            let sector = self
                .codec()
                .sector_id(category, self.identity(), &channel)?;
            if let core::ops::ControlFlow::Break(error) = emit(SU2Irrep::from_sector_id(sector)) {
                return Ok(core::ops::ControlFlow::Break(error));
            }
        }
        Ok(core::ops::ControlFlow::Continue(()))
    }
}

impl PhysicalFusionBasis for SU2FusionRule {
    type Scalar = f64;
    // The coefficient authority cannot fail: it returns an exact zero for
    // every forbidden coupling, after symtn's own sector, multiplicity and
    // carrier-index checks.
    type Error = PhysicalBasisError<core::convert::Infallible>;

    fn try_carrier_dimension(&self, sector: SectorId) -> Result<usize, Self::Error> {
        Ok(checked_irrep(sector)?.twice_spin() + 1)
    }

    fn try_fusion_tensor_element(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_basis: usize,
        right_basis: usize,
        coupled_basis: usize,
        multiplicity: usize,
    ) -> Result<Self::Scalar, Self::Error> {
        let fusion_multiplicity = self.try_nsymbol(left, right, coupled)?;
        if multiplicity >= fusion_multiplicity {
            return Err(PhysicalBasisError::FusionMultiplicityOutOfBounds {
                left,
                right,
                coupled,
                multiplicity,
                dimension: fusion_multiplicity,
            });
        }

        let mut doubled_projections = [0_i32; 3];
        for ((sector, index), projection) in [
            (left, left_basis),
            (right, right_basis),
            (coupled, coupled_basis),
        ]
        .into_iter()
        .zip(&mut doubled_projections)
        {
            let doubled_spin = checked_irrep(sector)?.twice_spin();
            let dimension = doubled_spin + 1;
            if index >= dimension {
                return Err(PhysicalBasisError::CarrierIndexOutOfBounds {
                    sector,
                    index,
                    dimension,
                });
            }
            // TensorKitSectors' executable SU(2) convention: index zero is
            // highest weight, so dm = dj, dj - 2, ..., -dj.
            *projection = doubled_spin as i32 - 2 * index as i32;
        }

        // A fusion tensor is dense over carrier indices; a magnetic-number
        // mismatch is the authority's exact structural zero.
        Ok(fusion_symbols::lie::su2::clebsch_gordan(
            left.id() as u32,
            doubled_projections[0],
            right.id() as u32,
            doubled_projections[1],
            coupled.id() as u32,
            doubled_projections[2],
        )
        .to_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_label_boundary_are_unchanged() {
        let bytes = b"symtn:su2:sector-id=dj:max-doubled-spin=254:channels=ascending-step-2";
        assert_eq!(
            SU2FusionRule.rule_identity(),
            RuleIdentity::from_canonical_bytes::<SU2FusionRule>(
                SU2_RACAH_IDENTITY_SCHEMA,
                Arc::from(bytes.as_slice())
            )
        );
        if let Some(id) = (u32::MAX as usize).checked_add(1) {
            let sector = SectorId::new(id);
            assert_eq!(
                SU2FusionRule.decode_sector(sector),
                Err(FusionAlgebraError::InvalidSector { sector })
            );
        }
    }

    #[test]
    fn racah_twist_is_exactly_one() {
        for dj in 0..=40 {
            assert_eq!(
                SU2FusionRule.twist_scalar(SectorId::new(dj)).to_bits(),
                1.0f64.to_bits()
            );
        }
    }

    #[test]
    fn frozen_su2_symbol_and_channel_oracle() {
        let rule = SU2FusionRule;
        let sector = SectorId::new;

        assert_eq!(
            rule.fusion_channels(sector(1), sector(1)).as_slice(),
            &[sector(0), sector(2)]
        );
        assert_eq!(
            rule.try_fusion_channels(sector(127), sector(127))
                .unwrap()
                .last(),
            Some(&sector(254))
        );
        assert_eq!(
            rule.f_symbol_scalar(
                sector(0),
                sector(0),
                sector(0),
                sector(0),
                sector(0),
                sector(0)
            ),
            1.0
        );
        assert_eq!(
            rule.f_symbol_scalar(
                sector(1),
                sector(1),
                sector(1),
                sector(1),
                sector(0),
                sector(0)
            ),
            -0.5
        );
        assert_eq!(rule.r_symbol_scalar(sector(1), sector(1), sector(0)), -1.0);
        assert_eq!(rule.r_symbol_scalar(sector(1), sector(1), sector(255)), 0.0);
        assert_eq!(rule.frobenius_schur_phase_scalar(sector(253)), -1.0);
        assert_eq!(
            rule.f_symbol_scalar(
                sector(1),
                sector(1),
                sector(1),
                sector(1),
                sector(1),
                sector(1)
            ),
            0.0
        );
    }

    #[test]
    fn su2_has_canonical_unit_channels_and_associator() {
        // What: the SU(2) vacuum remains a unique unit at the supported
        // label boundary, with exact unit associator.
        fn accepts_canonical_unit<R: CanonicalUnitFusionRule>(_rule: &R) {}

        let rule = SU2FusionRule;
        let vacuum = rule.vacuum();
        let boundary = SectorId::new(254);
        let sample = SectorId::new(127);
        accepts_canonical_unit(&rule);
        assert_eq!(rule.dual(vacuum), vacuum);
        assert_eq!(
            rule.fusion_channels(vacuum, boundary).as_slice(),
            &[boundary]
        );
        assert_eq!(
            rule.fusion_channels(boundary, vacuum).as_slice(),
            &[boundary]
        );
        assert_eq!(rule.nsymbol(vacuum, boundary, boundary), 1);
        assert_eq!(rule.nsymbol(boundary, vacuum, boundary), 1);
        assert_eq!(rule.nsymbol(vacuum, boundary, vacuum), 0);
        for fused in rule.fusion_channels(sample, sample) {
            assert_eq!(
                rule.f_symbol_scalar(vacuum, sample, sample, fused, sample, fused),
                1.0
            );
            assert_eq!(
                rule.f_symbol_scalar(sample, vacuum, sample, fused, sample, sample),
                1.0
            );
            assert_eq!(
                rule.f_symbol_scalar(sample, sample, vacuum, fused, fused, sample),
                1.0
            );
        }
    }

    #[test]
    fn checked_su2_preserves_validation_order_and_domain() {
        let rule = SU2FusionRule;
        let invalid = SectorId::new(255);
        let later_invalid = SectorId::new(256);

        assert_eq!(
            rule.try_fusion_channels(invalid, SectorId::new(254)),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_fusion_channels(SectorId::new(254), invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_fusion_channels(invalid, later_invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_fusion_channels(SectorId::new(128), SectorId::new(127)),
            Err(FusionAlgebraError::FusionNotRepresentable {
                left: SectorId::new(128),
                right: SectorId::new(127),
            })
        );
        assert_eq!(
            rule.try_nsymbol(SectorId::new(254), invalid, SectorId::new(0)),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_nsymbol(invalid, SectorId::new(0), later_invalid),
            Err(FusionAlgebraError::InvalidSector {
                sector: later_invalid,
            })
        );
    }

    #[test]
    fn su2_physical_basis_matches_tensor_kit_descending_odd_phase_oracle() {
        // What: index zero is highest weight. The asymmetric 1 x 1/2 -> 1/2
        // channel is reversal-odd, so unlike 1/2 x 1/2 -> 1 this fixture fails
        // if the ascending and descending conventions are confused.
        let rule = SU2FusionRule;
        let left = SectorId::new(2);
        let right = SectorId::new(1);
        let coupled = SectorId::new(1);

        assert_eq!(rule.try_carrier_dimension(left), Ok(3));
        assert_eq!(rule.try_carrier_dimension(right), Ok(2));

        let highest = rule
            .try_fusion_tensor_element(left, right, coupled, 0, 1, 0, 0)
            .unwrap();
        let reversed = rule
            .try_fusion_tensor_element(left, right, coupled, 2, 0, 1, 0)
            .unwrap();
        assert!((highest - (2.0_f64 / 3.0).sqrt()).abs() < 1.0e-14);
        assert!((reversed + (2.0_f64 / 3.0).sqrt()).abs() < 1.0e-14);
        assert_eq!(
            highest,
            fusion_symbols::lie::su2::clebsch_gordan(2, 2, 1, -1, 1, 1).to_f64()
        );

        assert_eq!(
            rule.try_fusion_tensor_element(left, right, coupled, 3, 0, 0, 0),
            Err(PhysicalBasisError::CarrierIndexOutOfBounds {
                sector: left,
                index: 3,
                dimension: 3,
            })
        );
    }

    #[test]
    fn su2_physical_basis_is_a_complete_orthonormal_local_transform() {
        // What: all 1 x 1/2 product states and both coupled sectors form one
        // unitary change of basis, not merely a few matching coefficients.
        let rule = SU2FusionRule;
        let left = SectorId::new(2);
        let right = SectorId::new(1);
        let coupled = [(SectorId::new(1), 2), (SectorId::new(3), 4)];
        let coefficient = |left_basis, right_basis, coupled, coupled_basis| {
            rule.try_fusion_tensor_element(
                left,
                right,
                coupled,
                left_basis,
                right_basis,
                coupled_basis,
                0,
            )
            .unwrap()
        };

        for left_basis in 0..3 {
            for right_basis in 0..2 {
                for other_left_basis in 0..3 {
                    for other_right_basis in 0..2 {
                        let mut inner = 0.0;
                        for &(sector, dimension) in &coupled {
                            for basis in 0..dimension {
                                inner += coefficient(left_basis, right_basis, sector, basis)
                                    * coefficient(
                                        other_left_basis,
                                        other_right_basis,
                                        sector,
                                        basis,
                                    );
                            }
                        }
                        let expected = f64::from(
                            left_basis == other_left_basis && right_basis == other_right_basis,
                        );
                        assert!((inner - expected).abs() < 1.0e-14);
                    }
                }
            }
        }

        for &(sector, dimension) in &coupled {
            for basis in 0..dimension {
                for &(other_sector, other_dimension) in &coupled {
                    for other_basis in 0..other_dimension {
                        let mut inner = 0.0;
                        for left_basis in 0..3 {
                            for right_basis in 0..2 {
                                inner += coefficient(left_basis, right_basis, sector, basis)
                                    * coefficient(
                                        left_basis,
                                        right_basis,
                                        other_sector,
                                        other_basis,
                                    );
                            }
                        }
                        let expected = f64::from(sector == other_sector && basis == other_basis);
                        assert!((inner - expected).abs() < 1.0e-14);
                    }
                }
            }
        }
    }

    fn recoupling_overlap(
        rule: &SU2FusionRule,
        [left, middle, right, total]: [SectorId; 4],
        [left_coupled, right_coupled]: [SectorId; 2],
        total_basis: usize,
    ) -> f64 {
        let mut overlap = 0.0;
        for left_basis in 0..rule.try_carrier_dimension(left).unwrap() {
            for middle_basis in 0..rule.try_carrier_dimension(middle).unwrap() {
                for right_basis in 0..rule.try_carrier_dimension(right).unwrap() {
                    let left_tree: f64 = (0..rule.try_carrier_dimension(left_coupled).unwrap())
                        .map(|inner_basis| {
                            rule.try_fusion_tensor_element(
                                left,
                                middle,
                                left_coupled,
                                left_basis,
                                middle_basis,
                                inner_basis,
                                0,
                            )
                            .unwrap()
                                * rule
                                    .try_fusion_tensor_element(
                                        left_coupled,
                                        right,
                                        total,
                                        inner_basis,
                                        right_basis,
                                        total_basis,
                                        0,
                                    )
                                    .unwrap()
                        })
                        .sum();
                    let right_tree: f64 = (0..rule.try_carrier_dimension(right_coupled).unwrap())
                        .map(|inner_basis| {
                            rule.try_fusion_tensor_element(
                                middle,
                                right,
                                right_coupled,
                                middle_basis,
                                right_basis,
                                inner_basis,
                                0,
                            )
                            .unwrap()
                                * rule
                                    .try_fusion_tensor_element(
                                        left,
                                        right_coupled,
                                        total,
                                        left_basis,
                                        inner_basis,
                                        total_basis,
                                        0,
                                    )
                                    .unwrap()
                        })
                        .sum();
                    overlap += left_tree * right_tree;
                }
            }
        }
        overlap
    }

    #[test]
    fn su2_physical_basis_and_f_symbols_share_one_recoupling_gauge() {
        // What: contracting the two local CGC parenthesizations reproduces
        // the provider's F symbol for three spin-1/2 irreps.
        let rule = SU2FusionRule;
        let half = SectorId::new(1);
        for left_coupled in [SectorId::new(0), SectorId::new(2)] {
            for right_coupled in [SectorId::new(0), SectorId::new(2)] {
                for total_basis in 0..2 {
                    let overlap = recoupling_overlap(
                        &rule,
                        [half, half, half, half],
                        [left_coupled, right_coupled],
                        total_basis,
                    );
                    let expected =
                        rule.f_symbol_scalar(half, half, half, half, left_coupled, right_coupled);
                    assert!((overlap - expected).abs() < 1.0e-14);
                }
            }
        }
    }

    #[test]
    fn asymmetric_cgc_recoupling_fixes_f_symbol_orientation() {
        // What: all four external spins differ and the left intermediates
        // {1/2, 3/2} differ from the right {3/2, 5/2}. Swapping the F axes is
        // therefore an inadmissible request, not a hidden matrix transpose.
        let rule = SU2FusionRule;
        let external = [
            SectorId::new(1),
            SectorId::new(2),
            SectorId::new(3),
            SectorId::new(4),
        ];
        let left_intermediates = [SectorId::new(1), SectorId::new(3)];
        let right_intermediates = [SectorId::new(3), SectorId::new(5)];

        for left_coupled in left_intermediates {
            for right_coupled in right_intermediates {
                for total_basis in 0..5 {
                    let overlap = recoupling_overlap(
                        &rule,
                        external,
                        [left_coupled, right_coupled],
                        total_basis,
                    );
                    let expected = rule.f_symbol_scalar(
                        external[0],
                        external[1],
                        external[2],
                        external[3],
                        left_coupled,
                        right_coupled,
                    );
                    assert!((overlap - expected).abs() < 1.0e-14);
                }
            }
        }

        let oriented = rule.f_symbol_scalar(
            external[0],
            external[1],
            external[2],
            external[3],
            left_intermediates[0],
            right_intermediates[1],
        );
        let swapped = rule.f_symbol_scalar(
            external[0],
            external[1],
            external[2],
            external[3],
            right_intermediates[1],
            left_intermediates[0],
        );
        assert!((oriented - swapped).abs() > 0.5);
    }

    #[test]
    #[should_panic(
        expected = "SU(2) fusion closure exceeds the supported maximum doubled spin 254"
    )]
    fn r_symbol_preserves_left_right_closure_validation() {
        let _ =
            SU2FusionRule.r_symbol_scalar(SectorId::new(254), SectorId::new(1), SectorId::new(253));
    }
}
