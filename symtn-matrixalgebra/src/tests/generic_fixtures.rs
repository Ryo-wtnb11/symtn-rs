use super::*;

#[derive(Clone)]
pub(super) struct IdentityQdimRule {
    pub(super) identity: RuleIdentity,
    qdim: f64,
}

impl IdentityQdimRule {
    pub(super) fn new(qdim: f64) -> Self {
        Self {
            identity: RuleIdentity::new_unique::<Self>(),
            qdim,
        }
    }
}

impl symtn_core::testing::Provider for IdentityQdimRule {}

impl FusionRule for IdentityQdimRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        [SectorId::new(0)].into_iter().collect()
    }
}

impl MultiplicityFreeFusionRule for IdentityQdimRule {}

impl MultiplicityFreeFusionSymbols for IdentityQdimRule {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> f64 {
        1.0
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for IdentityQdimRule {
    fn dim_scalar(&self, _: SectorId) -> f64 {
        self.qdim
    }
    fn inv_dim_scalar(&self, _: SectorId) -> f64 {
        self.qdim.recip()
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        self.qdim.sqrt()
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        self.qdim.sqrt().recip()
    }
    fn twist_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
}

#[derive(Clone, Copy)]
pub(super) struct FactorGenericRule;

impl symtn_core::testing::Provider for FactorGenericRule {}

impl FusionRule for FactorGenericRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => [SectorId::new(x)].into_iter().collect(),
            (1, 1) => [SectorId::new(0), SectorId::new(1)].into_iter().collect(),
            _ => SectorVec::new(),
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl GenericFusionSymbols for FactorGenericRule {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        let shape = (
            self.nsymbol(a, b, e),
            self.nsymbol(e, c, d),
            self.nsymbol(b, c, f),
            self.nsymbol(a, f, d),
        );
        let rows = shape.0 * shape.1;
        let cols = shape.2 * shape.3;
        let mut data = vec![0.0; rows * cols];
        for index in 0..rows.min(cols) {
            data[index * cols + index] = 1.0;
        }
        GenericFArray::new(data, shape)
    }

    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        coupled: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        let size = if coupled == SectorId::new(1) { 2 } else { 1 };
        let mut data = vec![0.0; size * size];
        for index in 0..size {
            data[index * size + index] = 1.0;
        }
        GenericRMatrix::new(data, size, size)
    }
}

impl GenericRigidSymbols for FactorGenericRule {
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector == SectorId::new(1) {
            (1.0 + 2.0_f64.sqrt()).sqrt()
        } else {
            1.0
        }
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.sqrt_dim_scalar(sector).recip()
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

pub(super) fn generic_factorization_input(
) -> (BoundDynamicFusionMapSpace<FactorGenericRule>, Vec<f64>) {
    let provider = Arc::new(FactorGenericRule);
    let x = SectorId::new(1);
    let left = SectorLeg::new([(x, 2)], false);
    let unit = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([left, unit.clone()]),
        FusionProductSpace::new([unit.clone(), unit]),
    );
    let space =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(provider, homspace).unwrap();
    let data = (0..space.space().required_len().unwrap())
        .map(|index| 1.0 + index as f64 / 8.0)
        .collect();
    (space, data)
}

/// [`generic_factorization_input`]'s space with codomain and domain
/// exchanged: every coupled block is wide, so the right null space is the
/// shape-based nonempty side.
pub(super) fn wide_generic_factorization_input() -> BoundDynamicFusionMapSpace<FactorGenericRule> {
    let x = SectorId::new(1);
    let left = SectorLeg::new([(x, 2)], false);
    let unit = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([unit.clone(), unit.clone()]),
        FusionProductSpace::new([left, unit]),
    );
    BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(FactorGenericRule), homspace)
        .unwrap()
}

pub(super) fn flattened_block_value<D: FactorScalar>(
    data: &[D],
    block: symtn_core::BlockRef<'_>,
    axes: std::ops::Range<usize>,
    mut flat_index: usize,
) -> (usize, D) {
    let mut offset = block.offset();
    for axis in axes {
        let coordinate = flat_index % block.shape()[axis];
        flat_index /= block.shape()[axis];
        offset += coordinate * block.strides()[axis];
    }
    (offset, data[offset])
}

/// The diagonal `S` of a generic compact SVD, built from its spectrum on
/// `u`'s provider.
pub(super) fn generic_diagonal_factor<R, D>(
    u: &BoundDynFactor<R, D>,
    spectrum: &[SectorSpectrum],
) -> BoundDynFactor<R, D>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let space = spectrum_bond::<CheckedGenericAdmissionMode, _, _>(u.space(), spectrum).unwrap();
    let data = diagonal_bond_data(space.space(), spectrum, &D::from_real).unwrap();
    BoundDynFactor::from_bound(space, data, 1, 1).unwrap()
}

pub(super) struct CheckedOnlyFactorRule {
    pub(super) calls: Cell<usize>,
}

impl CheckedOnlyFactorRule {
    pub(super) fn call<T>(
        &self,
        value: impl FnOnce(&FactorGenericRule) -> T,
    ) -> Result<T, Infallible> {
        self.calls.set(self.calls.get() + 1);
        Ok(value(&FactorGenericRule))
    }
}

impl symtn_core::testing::Provider for CheckedOnlyFactorRule {}

impl CheckedGenericFusion for CheckedOnlyFactorRule {
    type Error = Infallible;

    fn rule_identity(&self) -> RuleIdentity {
        FactorGenericRule.rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FactorGenericRule.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        FactorGenericRule.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        FactorGenericRule.vacuum()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.call(|rule| rule.dual(sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.call(|rule| rule.fusion_channels(left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.call(|rule| rule.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.call(|rule| rule.nsymbol(left, right, coupled))
    }
}

pub(super) fn checked_svd_matrix(
    sector: SectorId,
    complex: bool,
) -> (usize, usize, Vec<Complex64>) {
    match sector.id() {
        0 => (
            2,
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                if complex {
                    Complex64::new(0.0, 4.0)
                } else {
                    Complex64::new(4.0, 0.0)
                },
                Complex64::new(0.0, 0.0),
            ],
        ),
        1 => {
            let a = 3.0 / 2.0_f64.sqrt();
            (
                3,
                2,
                vec![
                    Complex64::new(a, 0.0),
                    if complex {
                        Complex64::new(0.0, a)
                    } else {
                        Complex64::new(a, 0.0)
                    },
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    if complex {
                        Complex64::new(0.0, -2.0)
                    } else {
                        Complex64::new(2.0, 0.0)
                    },
                ],
            )
        }
        id => panic!("unexpected checked-SVD fixture sector {id}"),
    }
}

pub(super) fn generic_svd_truncation_input<D>(
    complex: bool,
) -> (BoundDynamicFusionMapSpace<FactorGenericRule>, Vec<D>)
where
    D: FactorScalar,
{
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 2)], false)]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let mut data = vec![D::zero(); space.space().required_len().unwrap()];
    for index in 0..space.space().structure().block_count() {
        let block = space.space().structure().block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("Generic SVD fixture must use fusion-tree blocks")
        };
        let (rows, cols, matrix) = checked_svd_matrix(key.codomain_tree().coupled(), complex);
        assert_eq!(block.shape(), [rows, cols]);
        for col in 0..cols {
            for row in 0..rows {
                let source_index = row + rows * col;
                let destination =
                    block.offset() + row * block.strides()[0] + col * block.strides()[1];
                data[destination] = D::from_complex64(matrix[source_index]);
            }
        }
    }
    (space, data)
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
pub(super) fn checked_svd_truncation_input<D>(
    complex: bool,
) -> (
    Arc<LateGenericSpy>,
    BoundDynamicFusionMapSpace<LateGenericSpy>,
    Vec<D>,
)
where
    D: FactorScalar,
{
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 2)], false)]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let mut data = vec![D::zero(); checked.space().required_len().unwrap()];
    let structure = checked.space().structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("checked Generic fixture must use fusion-tree blocks")
        };
        let sector = key.codomain_tree().coupled();
        let (rows, cols, matrix) = checked_svd_matrix(sector, complex);
        assert_eq!(block.shape(), [rows, cols]);
        for col in 0..cols {
            for row in 0..rows {
                let source_index = row + rows * col;
                let destination =
                    block.offset() + row * block.strides()[0] + col * block.strides()[1];
                data[destination] = D::from_complex64(matrix[source_index]);
            }
        }
    }
    (provider, checked, data)
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
pub(super) fn checked_svd_wide_input<D>() -> (
    Arc<LateGenericSpy>,
    BoundDynamicFusionMapSpace<LateGenericSpy>,
    Vec<D>,
)
where
    D: FactorScalar,
{
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 3)], false)]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let mut data = vec![D::zero(); checked.space().required_len().unwrap()];
    let structure = checked.space().structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("checked Generic fixture must use fusion-tree blocks")
        };
        let (rows, cols, matrix) = checked_svd_matrix(key.coupled(), true);
        assert_eq!(block.shape(), [cols, rows]);
        for column in 0..rows {
            for row in 0..cols {
                let source = matrix[column + rows * row].conj();
                let destination =
                    block.offset() + row * block.strides()[0] + column * block.strides()[1];
                data[destination] = D::from_complex64(source);
            }
        }
    }
    (provider, checked, data)
}

/// `space`'s exact layout rebound to a checked provider with
/// [`FactorGenericRule`]'s symbols (a [`LateGenericSpy`] that never fails), the
/// provider the checked Generic factorizations take.
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
pub(super) fn bind_checked_layout(
    space: &BoundDynamicFusionMapSpace<FactorGenericRule>,
) -> (
    Arc<LateGenericSpy>,
    BoundDynamicFusionMapSpace<LateGenericSpy>,
) {
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: FactorGenericRule.rule_identity(),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(space.space().clone(), Arc::clone(&provider))
            .unwrap();
    (provider, checked)
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
pub(super) fn bind_checked_only(
    space: &BoundDynamicFusionMapSpace<impl FusionRule>,
) -> (
    Arc<CheckedOnlyFactorRule>,
    BoundDynamicFusionMapSpace<CheckedOnlyFactorRule>,
) {
    let provider = Arc::new(CheckedOnlyFactorRule {
        calls: Cell::new(0),
    });
    let checked = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        space.space().homspace().clone(),
    )
    .unwrap();
    (provider, checked)
}

pub(super) fn generic_values_endomorphism_input() -> (
    BoundDynamicFusionMapSpace<FactorGenericRule>,
    Vec<Complex64>,
    Vec<Complex64>,
) {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let mut hermitian = vec![Complex64::zero(); space.space().required_len().unwrap()];
    let mut general = hermitian.clone();
    for region in regions.iter() {
        let (hermitian_matrix, general_matrix): (&[Complex64], &[Complex64]) =
            match region.coupled().id() {
                0 => (&[Complex64::new(-4.0, 0.0)], &[Complex64::new(2.0, -1.0)]),
                1 => (
                    &[
                        Complex64::new(2.0, 0.0),
                        Complex64::new(0.0, -1.0),
                        Complex64::new(0.0, 1.0),
                        Complex64::new(2.0, 0.0),
                    ],
                    &[
                        Complex64::new(1.0, 1.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(2.0, 0.0),
                        Complex64::new(3.0, -1.0),
                    ],
                ),
                sector => panic!("unexpected Generic values sector {sector}"),
            };
        assert_eq!(region.range().len(), hermitian_matrix.len());
        hermitian[region.range()].copy_from_slice(hermitian_matrix);
        general[region.range()].copy_from_slice(general_matrix);
    }
    (space, hermitian, general)
}

pub(super) fn checked_enumeration_calls<D: FactorScalar>(
    factor: &BoundDynFactor<LateGenericSpy, D>,
) -> usize {
    let homspace = factor.space().space().homspace().clone();
    late_spy_calls(&|probe| {
        homspace
            .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(probe)
            .unwrap();
    })
}
