//! `contract_into` overwrites a destination whose subblocks are stored in a
//! different coupled-sector order than the fresh contraction result.

use std::fmt::Debug;
use std::sync::Arc;
use symtn::typed::ContractSpec;

use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::sector::{
    CheckedFusionAlgebra, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, SectorCodec,
    TypedSectorAdmission,
};
use symtn::typed::FusionAlgebraError;
use symtn::typed::{Complex64, TensorScalar};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

fn assert_same<R, D>(actual: &TensorMap<R, D>, expected: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: OracleScalar,
{
    assert_eq!(actual.codomain(), expected.codomain());
    assert_eq!(actual.domain(), expected.domain());
    assert_eq!(
        actual.materialize().unwrap().dense_data().unwrap().len(),
        expected.materialize().unwrap().dense_data().unwrap().len()
    );
    for (&got, &want) in actual
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.materialize().unwrap().dense_data().unwrap())
    {
        assert!(
            got.error(want) <= 1.0e-11 * (1.0 + want.magnitude()),
            "expected {want:?}, got {got:?}"
        );
    }
}

trait OracleScalar: TensorScalar + Copy + Debug + Default + PartialEq {
    fn sample(value: f64) -> Self;
    fn error(self, expected: Self) -> f64;
    fn magnitude(self) -> f64;
}

impl OracleScalar for f64 {
    fn sample(value: f64) -> Self {
        value
    }

    fn error(self, expected: Self) -> f64 {
        (self - expected).abs()
    }

    fn magnitude(self) -> f64 {
        self.abs()
    }
}

impl OracleScalar for Complex64 {
    fn sample(value: f64) -> Self {
        Complex64::new(value, 0.3 * value - 0.2)
    }

    fn error(self, expected: Self) -> f64 {
        (self - expected).norm()
    }

    fn magnitude(self) -> f64 {
        self.norm()
    }
}

fn matrix<R, D>(
    runtime: &Runtime,
    codomain: &GradedSpace<R>,
    domain: &GradedSpace<R>,
    salt: f64,
    sector_tag: &impl Fn(&<R as TypedSectorAdmission>::Sector) -> f64,
) -> TensorMap<R, D>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: OracleScalar,
{
    TensorMap::from_subblock_fn(runtime, [codomain], [domain], |trees, ij| {
        D::sample(salt + sector_tag(trees.coupled()) + 0.1 * ij[0] as f64 + 0.01 * ij[1] as f64)
    })
    .unwrap()
}

fn assert_reordered_overwrite<R, D>(
    runtime: &Runtime,
    space: &GradedSpace<R>,
    sector_tag: &impl Fn(&<R as TypedSectorAdmission>::Sector) -> f64,
    output_factor: &impl Fn(&<R as TypedSectorAdmission>::Sector) -> f64,
) where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: OracleScalar,
{
    let old_lhs = matrix::<_, D>(runtime, space, space, 11.0, sector_tag);
    let old_rhs = matrix::<_, D>(runtime, space, space, 12.0, sector_tag);
    let mut destination = old_lhs
        .contract(
            &old_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();
    let lhs = matrix::<_, D>(runtime, space, space, 111.0, sector_tag);
    let rhs = matrix::<_, D>(runtime, space, space, 112.0, sector_tag);
    let expected = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();
    let assert_oracle = |actual: &TensorMap<R, D>| {
        assert_eq!(actual.subblock_count(), space.sectors().unwrap().len());
        let lhs_blocks = lhs.subblocks().unwrap().collect::<Vec<_>>();
        let rhs_blocks = rhs.subblocks().unwrap().collect::<Vec<_>>();
        for (trees, output) in actual.subblocks().unwrap() {
            let output_id = actual.provider().try_encode_label(trees.coupled()).unwrap();
            let source_coupled = actual
                .provider()
                .try_dual_id(output_id)
                .and_then(|sector| actual.provider().try_decode_label(sector))
                .unwrap();
            let left = lhs_blocks
                .iter()
                .find(|(candidate, _)| candidate.coupled() == &source_coupled)
                .unwrap()
                .1;
            let right = rhs_blocks
                .iter()
                .find(|(candidate, _)| candidate.coupled() == &source_coupled)
                .unwrap()
                .1;
            for row in 0..output.shape()[0] {
                for column in 0..output.shape()[1] {
                    let mut want = D::default();
                    for contracted in 0..left.shape()[1] {
                        want = want
                            + *left.get(&[column, contracted]).unwrap()
                                * *right.get(&[contracted, row]).unwrap();
                    }
                    want = D::from_real(output_factor(&source_coupled)) * want;
                    let got = *output.get(&[row, column]).unwrap();
                    assert!(
                        got.error(want) <= 1.0e-11 * (1.0 + want.magnitude()),
                        "sector {:?}, ({row}, {column}): expected {want:?}, got {got:?}",
                        trees.coupled()
                    );
                }
            }
        }
    };
    assert_oracle(&expected);
    lhs.contract_into(
        &rhs,
        &ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[1],
            domain: &[0],
        },
        &mut destination,
        D::from_real(1.0),
        D::from_real(0.0),
    )
    .unwrap();
    assert_oracle(&destination);
    assert_same(&destination, &expected);
}

#[test]
fn contract_overwrite_reorders_non_self_dual_unequal_blocks() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let u1_provider = Arc::new(U1FusionRule);
    let u1 = GradedSpace::try_new(
        Arc::clone(&u1_provider),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let u1_tag = |sector: &U1Irrep| f64::from(sector.charge());
    let bosonic_factor = |_: &U1Irrep| 1.0;
    assert_reordered_overwrite::<_, f64>(&runtime, &u1, &u1_tag, &bosonic_factor);
    assert_reordered_overwrite::<_, Complex64>(&runtime, &u1, &u1_tag, &bosonic_factor);

    let product_provider = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let product = GradedSpace::try_new(
        product_provider,
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 1),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
        ],
    )
    .unwrap();
    let product_tag = |sector: &symtn::sector::ProductSector<Z2Irrep, U1Irrep>| {
        f64::from(sector.right().charge())
            + if *sector.left() == Z2Irrep::ODD {
                0.5
            } else {
                0.0
            }
    };
    let fermionic_factor = |sector: &symtn::sector::ProductSector<Z2Irrep, U1Irrep>| {
        if *sector.left() == Z2Irrep::ODD {
            -1.0
        } else {
            1.0
        }
    };
    assert_reordered_overwrite::<_, f64>(&runtime, &product, &product_tag, &fermionic_factor);
    assert_reordered_overwrite::<_, Complex64>(&runtime, &product, &product_tag, &fermionic_factor);
}
