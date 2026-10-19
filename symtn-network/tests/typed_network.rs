use std::fmt::Debug;
use std::sync::Arc;
use symtn::typed::ContractSpec;

use symtn::expert::TensorStorage;
use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::sector::{
    CheckedFusionAlgebra, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, SectorCodec,
    TypedSectorAdmission,
};
use symtn::typed::FusionAlgebraError;
use symtn::typed::{Complex32, Complex64, TensorScalar};
use symtn::typed::{GradedSpace, Runtime, Svd, TensorMap};
use symtn_network::{
    GreedyDenseOptimizer, LabelOrderDenseOptimizer, Network, NetworkExecutionWorkspace,
    PlannedNetwork, TemporaryLabel, TensorId,
};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

fn pair_network() -> Network {
    Network::new(
        vec![vec!["a", "k"], vec!["k", "b"]],
        vec![false, false],
        vec![1, 1],
        vec!["a", "b"],
        Some(1),
    )
    .unwrap()
}

fn plan_accepts_storage<R, D, S>(network: &Network, tensors: &[&TensorMap<R, D, S>])
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    network.plan(tensors, &GreedyDenseOptimizer).unwrap();
}

fn pair_case<R, D>(space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar + numerics::Numeric,
{
    let runtime = Runtime::builder().build().unwrap();
    // The planned network may group the contraction differently from the
    // direct call; the contracted length is bounded by the leg's weighted
    // dimension.
    let terms = space.dim().unwrap().ceil() as usize;
    let lhs = TensorMap::<R, D>::rand_with_seed(&runtime, [space], [space], 1).unwrap();
    let rhs = TensorMap::<R, D>::rand_with_seed(&runtime, [space], [space], 2).unwrap();
    let tensors = [&lhs, &rhs];
    let planned = pair_network()
        .plan(&tensors, &GreedyDenseOptimizer)
        .unwrap();
    let expected = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for _ in 0..2 {
        let actual = planned.execute(&tensors, &mut workspace).unwrap();
        numerics::assert_slices_close(
            "network vs direct contract",
            actual.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            terms,
        );
        assert_eq!(actual.codomain(), expected.codomain());
        assert_eq!(actual.domain(), expected.domain());
    }
}

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

fn assert_chain_oracle<R, D>(
    actual: &TensorMap<R, D>,
    a: &TensorMap<R, D>,
    b: &TensorMap<R, D>,
    c: &TensorMap<R, D>,
) where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: OracleScalar,
{
    assert_eq!(actual.codomain(), a.codomain());
    assert_eq!(actual.domain(), c.domain());
    let codomain = a.codomain();
    let domain = c.domain();
    let expected_sectors = codomain[0]
        .sectors()
        .unwrap()
        .into_iter()
        .filter(|sector| domain[0].degeneracy(sector).unwrap() != 0)
        .collect::<Vec<_>>();
    let actual_sectors = actual
        .subblocks()
        .unwrap()
        .map(|(trees, _)| trees.coupled().clone())
        .collect::<Vec<_>>();
    assert_eq!(actual_sectors, expected_sectors);
    let a_blocks = a.subblocks().unwrap().collect::<Vec<_>>();
    let b_blocks = b.subblocks().unwrap().collect::<Vec<_>>();
    let c_blocks = c.subblocks().unwrap().collect::<Vec<_>>();
    for (trees, output) in actual.subblocks().unwrap() {
        let left = a_blocks
            .iter()
            .find(|(candidate, _)| candidate.coupled() == trees.coupled())
            .map(|(_, block)| *block);
        let middle = b_blocks
            .iter()
            .find(|(candidate, _)| candidate.coupled() == trees.coupled())
            .map(|(_, block)| *block);
        let right = c_blocks
            .iter()
            .find(|(candidate, _)| candidate.coupled() == trees.coupled())
            .map(|(_, block)| *block);
        for row in 0..output.shape()[0] {
            for column in 0..output.shape()[1] {
                let mut expected = D::default();
                if let (Some(left), Some(middle), Some(right)) = (left, middle, right) {
                    for first in 0..left.shape()[1] {
                        for second in 0..middle.shape()[1] {
                            expected = expected
                                + *left.get(&[row, first]).unwrap()
                                    * *middle.get(&[first, second]).unwrap()
                                    * *right.get(&[second, column]).unwrap();
                        }
                    }
                }
                let got = *output.get(&[row, column]).unwrap();
                assert!(
                    got.error(expected) <= 1.0e-11 * (1.0 + expected.magnitude()),
                    "sector {:?}, ({row}, {column}): expected {expected:?}, got {got:?}",
                    trees.coupled()
                );
            }
        }
    }
}

fn assert_chain_case<R, D>(
    plan: &PlannedNetwork,
    workspace: &mut NetworkExecutionWorkspace<R, D>,
    tensors: [&TensorMap<R, D>; 3],
) -> TensorMap<R, D>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: OracleScalar,
{
    let reused = plan.execute(&tensors, workspace).unwrap();
    let fresh = plan.execute(&tensors, &mut Default::default()).unwrap();
    assert_same(&reused, &fresh);
    assert_chain_oracle(&reused, tensors[0], tensors[1], tensors[2]);
    reused
}

fn run_shape_reuse_sequence<R, D>(
    provider: Arc<R>,
    make_space: impl Fn(&Arc<R>, &[(i32, usize)]) -> GradedSpace<R>,
    sector_tag: impl Fn(&<R as TypedSectorAdmission>::Sector) -> f64,
) where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: OracleScalar,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let initial = make_space(&provider, &[(0, 2), (1, 1)]);
    let initial_tensors = [
        matrix(&runtime, &initial, &initial, 1.0, &sector_tag),
        matrix(&runtime, &initial, &initial, 2.0, &sector_tag),
        matrix(&runtime, &initial, &initial, 3.0, &sector_tag),
    ];
    let network = Network::new(
        vec![vec!["a", "b"], vec!["b", "c"], vec!["c", "d"]],
        vec![false; 3],
        vec![1; 3],
        vec!["a", "d"],
        Some(1),
    )
    .unwrap();
    let initial_refs = [
        &initial_tensors[0],
        &initial_tensors[1],
        &initial_tensors[2],
    ];
    let plan = network
        .plan(
            &initial_refs,
            &LabelOrderDenseOptimizer::new(vec!["b".into(), "c".into()]),
        )
        .unwrap();
    assert_eq!(
        plan.plan()
            .steps()
            .iter()
            .map(|step| (step.lhs(), step.rhs()))
            .collect::<Vec<_>>(),
        [
            (TensorId::new(0), TensorId::new(1)),
            (TensorId::new(2), TensorId::new(3)),
        ]
    );

    let mut workspace = NetworkExecutionWorkspace::default();
    assert_chain_case(&plan, &mut workspace, initial_refs);
    for (spec, salt) in [
        (vec![(0, 1), (1, 2)], 11.0),
        (vec![(0, 3), (1, 2)], 21.0),
        (vec![(0, 1)], 31.0),
        (vec![(-1, 2), (2, 1)], 51.0),
        (vec![(0, 0), (1, 2)], 61.0),
        (vec![(0, 2), (1, 1)], 41.0),
    ] {
        let space = make_space(&provider, &spec);
        for value_shift in [0.0, 100.0] {
            let tensors = [
                matrix(&runtime, &space, &space, salt + value_shift, &sector_tag),
                matrix(
                    &runtime,
                    &space,
                    &space,
                    salt + value_shift + 1.0,
                    &sector_tag,
                ),
                matrix(
                    &runtime,
                    &space,
                    &space,
                    salt + value_shift + 2.0,
                    &sector_tag,
                ),
            ];
            assert_chain_case(
                &plan,
                &mut workspace,
                [&tensors[0], &tensors[1], &tensors[2]],
            );
        }
    }

    let x = make_space(&provider, &[(0, 1)]);
    let k = make_space(&provider, &[(0, 1), (1, 1)]);
    let l = make_space(&provider, &[(1, 1), (2, 1)]);
    let y = make_space(&provider, &[(2, 1)]);
    let empty_chain = [
        matrix(&runtime, &x, &k, 71.0, &sector_tag),
        matrix(&runtime, &k, &l, 72.0, &sector_tag),
        matrix(&runtime, &l, &y, 73.0, &sector_tag),
    ];
    let empty = assert_chain_case(
        &plan,
        &mut workspace,
        [&empty_chain[0], &empty_chain[1], &empty_chain[2]],
    );
    assert_eq!(empty.subblock_count(), 0);
    assert!(empty.dense_data().unwrap().is_empty());

    let valid = initial_refs;
    let wrong_bond = make_space(&provider, &[(9, 1)]);
    let invalid_b = matrix(&runtime, &wrong_bond, &initial, 81.0, &sector_tag);
    assert!(plan
        .execute(&[valid[0], &invalid_b, valid[2]], &mut workspace)
        .is_err());
    assert_chain_case(&plan, &mut workspace, valid);

    let wrong_split =
        TensorMap::from_subblock_fn(&runtime, [&initial, &initial], [], |trees, ij| {
            D::sample(91.0 + sector_tag(trees.coupled()) + 0.1 * ij[0] as f64 + 0.01 * ij[1] as f64)
        })
        .unwrap();
    assert!(plan
        .execute(&[valid[0], &wrong_split, valid[2]], &mut workspace)
        .is_err());
    assert_chain_case(&plan, &mut workspace, valid);

    let wrong_rank =
        TensorMap::from_subblock_fn(&runtime, [&initial, &initial], [&initial], |trees, ij| {
            D::sample(
                101.0
                    + sector_tag(trees.coupled())
                    + 0.1 * ij[0] as f64
                    + 0.01 * ij[1] as f64
                    + 0.001 * ij[2] as f64,
            )
        })
        .unwrap();
    assert!(plan
        .execute(&[valid[0], &wrong_rank, valid[2]], &mut workspace)
        .is_err());
    assert_chain_case(&plan, &mut workspace, valid);
}

#[test]
fn u1_workspace_reuse_tracks_chain_values_and_sector_shapes() {
    let provider = Arc::new(U1FusionRule);
    let make_space = |provider: &Arc<_>, spec: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::clone(provider),
            spec.iter()
                .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
        )
        .unwrap()
    };
    run_shape_reuse_sequence::<_, f64>(Arc::clone(&provider), make_space, |sector| {
        f64::from(sector.charge())
    });
    run_shape_reuse_sequence::<_, Complex64>(provider, make_space, |sector| {
        f64::from(sector.charge())
    });
}

#[test]
fn fermion_u1_workspace_reuse_tracks_chain_values_and_sector_shapes() {
    let provider = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let make_space = |provider: &Arc<_>, spec: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::clone(provider),
            spec.iter().map(|&(charge, degeneracy)| {
                (
                    product_sector(
                        if charge.rem_euclid(2) == 0 {
                            Z2Irrep::EVEN
                        } else {
                            Z2Irrep::ODD
                        },
                        U1Irrep::new(charge),
                    ),
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    let sector_tag = |sector: &symtn::sector::ProductSector<Z2Irrep, U1Irrep>| {
        f64::from(sector.right().charge())
            + if *sector.left() == Z2Irrep::ODD {
                0.5
            } else {
                0.0
            }
    };
    run_shape_reuse_sequence::<_, f64>(Arc::clone(&provider), make_space, sector_tag);
    run_shape_reuse_sequence::<_, Complex64>(provider, make_space, sector_tag);
}

#[test]
fn provider_and_dtype_matrix_matches_direct_contract() {
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    pair_case::<_, f64>(&u1);
    pair_case::<_, Complex64>(&u1);

    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    pair_case::<_, f64>(&su2);
    pair_case::<_, Complex64>(&su2);

    let fz2 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    pair_case::<_, f64>(&fz2);
    pair_case::<_, Complex64>(&fz2);

    let product_rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let product = GradedSpace::try_new(
        product_rule,
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 1),
        ],
    )
    .unwrap();
    pair_case::<_, f64>(&product);
    pair_case::<_, Complex64>(&product);
}

#[test]
fn planning_conjugation_uses_checked_effective_duals_without_reading_storage() {
    fn run<R>(
        runtime: &Runtime,
        x0: &GradedSpace<R>,
        x1: &GradedSpace<R>,
        y: &GradedSpace<R>,
        z0: &GradedSpace<R>,
        z1: &GradedSpace<R>,
        seed: u64,
    ) where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
    {
        let a = TensorMap::<R, f64>::rand_with_seed(runtime, [x0, x1], [y], seed).unwrap();
        let b = TensorMap::rand_with_seed(runtime, [x1], [z0, z1], seed + 1).unwrap();
        let adjoint = a.adjoint().unwrap();
        assert_eq!(adjoint.codomain(), vec![y.clone()]);
        assert_eq!(adjoint.domain(), vec![x0.clone(), x1.clone()]);
        let network = Network::new(
            vec![vec!["i0", "i1", "j"], vec!["i1", "k0", "k1"]],
            vec![true, false],
            vec![2, 1],
            vec!["j", "i0", "k0", "k1"],
            Some(2),
        )
        .unwrap();
        let refs = [&a, &b];
        plan_accepts_storage(&network, &refs);
        let actual = network
            .plan(&refs, &GreedyDenseOptimizer)
            .unwrap()
            .execute(&refs, &mut Default::default())
            .unwrap();
        let expected = adjoint
            .contract(
                &b,
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2, 3],
                },
            )
            .unwrap();
        assert_eq!(actual.codomain(), vec![y.clone(), x0.try_dual().unwrap()]);
        assert_eq!(actual.domain(), vec![z0.clone(), z1.clone()]);
        assert_same(&actual, &expected);
    }

    let runtime = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let x0 = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(2), 2)]).unwrap();
    let x1_base = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(-1), 1)]).unwrap();
    let x1 = x1_base.try_dual().unwrap();
    let y = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(1), 3)]).unwrap();
    let z0 = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(-2), 2)]).unwrap();
    let z1 = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 1)]).unwrap();
    run(&runtime, &x0, &x1, &y, &z0, &z1, 748_001);

    let product_rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let product_x0 = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1)],
    )
    .unwrap();
    let product_x1 = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 2)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    let product_y = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2)],
    )
    .unwrap();
    let product_z0 = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::EVEN, U1Irrep::new(-2)), 1)],
    )
    .unwrap();
    let product_z1 = GradedSpace::try_new(
        product_rule,
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 1)],
    )
    .unwrap();
    run(
        &runtime,
        &product_x0,
        &product_x1,
        &product_y,
        &product_z0,
        &product_z1,
        748_003,
    );
}

#[test]
fn single_scalar_split_and_heterogeneous_final_permutation() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = |degeneracy| {
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), degeneracy)]).unwrap()
    };
    let v = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let tensor = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 10).unwrap();

    let single = Network::new(
        vec![vec!["i", "j"]],
        vec![false],
        vec![1],
        vec!["j", "i"],
        Some(1),
    )
    .unwrap();
    let single_plan = single.plan(&[&tensor], &GreedyDenseOptimizer).unwrap();
    assert_same(
        &single_plan
            .execute(&[&tensor], &mut Default::default())
            .unwrap(),
        &tensor.permute(&[1], &[0]).unwrap(),
    );

    let scalar = Network::new(
        vec![vec!["i", "j"], vec!["i", "j"]],
        vec![true, false],
        vec![1, 1],
        vec![],
        Some(0),
    )
    .unwrap();
    let scalar_plan = scalar
        .plan(&[&tensor, &tensor], &GreedyDenseOptimizer)
        .unwrap();
    let value = scalar_plan
        .execute(&[&tensor, &tensor], &mut Default::default())
        .unwrap()
        .scalar()
        .unwrap();
    let norm = tensor.norm(2.0).unwrap();
    assert!((value - norm * norm).abs() <= 1e-12 * (1.0 + norm * norm));
    let other_provider = Arc::new(U1FusionRule);
    let other_v =
        GradedSpace::try_new(other_provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap();
    let other = TensorMap::rand_with_seed(&runtime, [&other_v], [&other_v], 13).unwrap();
    let scalar_from_lhs = scalar_plan
        .execute(&[&tensor, &other], &mut Default::default())
        .unwrap();
    assert!(std::ptr::eq(scalar_from_lhs.provider(), tensor.provider()));
    let scalar_from_other_lhs = scalar_plan
        .execute(&[&other, &tensor], &mut Default::default())
        .unwrap();
    assert!(std::ptr::eq(
        scalar_from_other_lhs.provider(),
        other.provider()
    ));

    let (a, b, bond, c, d) = (space(2), space(3), space(4), space(5), space(6));
    let lhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&a, &b], [&bond], 11).unwrap();
    let rhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&bond], [&c, &d], 12).unwrap();
    let crossed = Network::new(
        vec![vec!["a", "b", "k"], vec!["k", "c", "d"]],
        vec![false, false],
        vec![2, 1],
        vec!["d", "a", "b", "c"],
        Some(2),
    )
    .unwrap();
    let refs = [&lhs, &rhs];
    let plan = crossed.plan(&refs, &GreedyDenseOptimizer).unwrap();
    let expected = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap()
        .permute(&[3, 0], &[1, 2])
        .unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    assert_same(&plan.execute(&refs, &mut workspace).unwrap(), &expected);
    assert_same(&plan.execute(&refs, &mut workspace).unwrap(), &expected);
}

#[test]
fn compact_and_lazy_representation_replay_stays_semantic() {
    let runtime = Runtime::builder().build().unwrap();
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let dense = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 20).unwrap();
    let identity = Network::new(
        vec![vec!["i", "j"]],
        vec![false],
        vec![1],
        vec!["i", "j"],
        Some(1),
    )
    .unwrap();
    let plan = identity.plan(&[&dense], &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    assert_same(&plan.execute(&[&dense], &mut workspace).unwrap(), &dense);
    let lazy = dense.adjoint().unwrap();
    assert_same(&plan.execute(&[&lazy], &mut workspace).unwrap(), &lazy);
    let conjugate = Network::new(
        vec![vec!["i", "j"]],
        vec![true],
        vec![1],
        vec!["j", "i"],
        Some(1),
    )
    .unwrap();
    let lazy_conj_plan = conjugate.plan(&[&lazy], &GreedyDenseOptimizer).unwrap();
    for _ in 0..2 {
        assert_same(
            &lazy_conj_plan.execute(&[&lazy], &mut workspace).unwrap(),
            &dense,
        );
    }

    let Svd { s: compact, .. } = dense.svd_compact(&[0], &[1]).unwrap();
    let compact_plan = identity.plan(&[&compact], &GreedyDenseOptimizer).unwrap();
    let scaled = compact.scale(2.0).unwrap();
    assert_same(
        &compact_plan.execute(&[&compact], &mut workspace).unwrap(),
        &compact,
    );
    assert_same(
        &compact_plan.execute(&[&scaled], &mut workspace).unwrap(),
        &scaled,
    );
    let compact_conj_plan = conjugate.plan(&[&compact], &GreedyDenseOptimizer).unwrap();
    let compact_adjoint = compact.adjoint().unwrap();
    for _ in 0..2 {
        assert_eq!(
            symtn::expert::diagonal_spectrum(
                &compact_conj_plan
                    .execute(&[&compact], &mut workspace)
                    .unwrap()
            )
            .unwrap(),
            symtn::expert::diagonal_spectrum(&compact_adjoint).unwrap()
        );
    }
}

fn u1_space(provider: &Arc<U1FusionRule>, degeneracy: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(Arc::clone(provider), [(U1Irrep::new(0), degeneracy)]).unwrap()
}

#[test]
fn workspace_drift_and_provider_allocation_changes_do_not_leave_stale_results() {
    let runtime1 = Runtime::builder().build().unwrap();
    let runtime2 = Runtime::builder().build().unwrap();
    let provider1 = Arc::new(U1FusionRule);
    let provider2 = Arc::new(U1FusionRule);
    let v1 = u1_space(&provider1, 2);
    let v2 = u1_space(&provider2, 2);
    let larger = u1_space(&provider1, 3);
    let make = |runtime: &Runtime, space: &GradedSpace<U1FusionRule>, seed| {
        TensorMap::<U1FusionRule, f64>::rand_with_seed(runtime, [space], [space], seed).unwrap()
    };
    let (a1, b1) = (make(&runtime1, &v1, 30), make(&runtime1, &v1, 31));
    let (a2, b2) = (make(&runtime1, &v2, 32), make(&runtime1, &v2, 33));
    let (wide_a, wide_b) = (make(&runtime1, &larger, 34), make(&runtime1, &larger, 35));
    let (foreign_a, foreign_b) = (make(&runtime2, &v1, 36), make(&runtime2, &v1, 37));
    let network = pair_network();
    let plan = network.plan(&[&a1, &b1], &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for operands in [
        [&a1, &b1],
        [&a2, &b1],
        [&a1, &b2],
        [&wide_a, &wide_b],
        [&foreign_a, &foreign_b],
    ] {
        let actual = plan.execute(&operands, &mut workspace).unwrap();
        let expected = operands[0]
            .contract(
                operands[1],
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap();
        assert_same(&actual, &expected);
    }

    assert!(plan.execute(&[&a1, &foreign_b], &mut workspace).is_err());
    assert_same(
        &plan.execute(&[&a1, &b1], &mut workspace).unwrap(),
        &a1.contract(
            &b1,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap(),
    );

    let incompatible = u1_space(&provider1, 4);
    let bad = make(&runtime1, &incompatible, 38);
    assert!(plan.execute(&[&a1, &bad], &mut workspace).is_err());
    assert!(plan.execute(&[&a1], &mut workspace).is_err());

    let second_plan = network.plan(&[&a2, &b2], &GreedyDenseOptimizer).unwrap();
    assert_same(
        &second_plan.execute(&[&a2, &b2], &mut workspace).unwrap(),
        &a2.contract(
            &b2,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap(),
    );
}

#[test]
fn one_plan_replays_concurrently_with_distinct_workspaces() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let v = u1_space(&provider, 3);
    let lhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 40).unwrap();
    let rhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 41).unwrap();
    let plan = pair_network()
        .plan(&[&lhs, &rhs], &GreedyDenseOptimizer)
        .unwrap();
    let expected = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    std::thread::scope(|scope| {
        let handles = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let mut workspace = NetworkExecutionWorkspace::default();
                    plan.execute(&[&lhs, &rhs], &mut workspace).unwrap()
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            assert_same(&handle.join().unwrap(), &expected);
        }
    });
}

#[test]
fn fermionic_greedy_chain_keeps_intermediate_on_the_expression_left() {
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)],
    )
    .unwrap();
    let tensors = (0..3)
        .map(|offset| {
            TensorMap::<_, f64>::rand_with_seed(&runtime, [&space], [&space], 750_350 + offset)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let refs = [&tensors[0], &tensors[1], &tensors[2]];
    let network = Network::new(
        vec![vec!["a", "b"], vec!["b", "c"], vec!["c", "d"]],
        vec![false; 3],
        vec![1; 3],
        vec!["a", "d"],
        Some(1),
    )
    .unwrap();
    let planned = network.plan(&refs, &GreedyDenseOptimizer).unwrap();
    let steps = planned.plan().steps();
    assert_eq!(
        (steps[0].lhs(), steps[0].rhs()),
        (TensorId::new(0), TensorId::new(1))
    );
    assert_eq!(
        (steps[1].lhs(), steps[1].rhs()),
        (TensorId::new(3), TensorId::new(2))
    );

    let manual = tensors[0]
        .contract(
            &tensors[1],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .contract(
            &tensors[2],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_same(
        &planned.execute(&refs, &mut Default::default()).unwrap(),
        &manual,
    );
}

#[test]
fn fermionic_interleaved_subtrees_keep_expression_order_and_signs() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let large = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 3)]).unwrap();
    let small = GradedSpace::try_new(provider, [(Z2Irrep::ODD, 2)]).unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&large], [&large], 750_400).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&small], [&small], 750_401).unwrap();
    let c = TensorMap::<_, f64>::rand_with_seed(&runtime, [&large], [&large], 750_402).unwrap();
    let d = TensorMap::<_, f64>::rand_with_seed(&runtime, [&small], [&small], 750_403).unwrap();
    let refs = [&a, &b, &c, &d];
    let network = Network::new(
        vec![
            vec!["a", "x"],
            vec!["b", "y"],
            vec!["x", "c"],
            vec!["y", "d"],
        ],
        vec![false; 4],
        vec![1; 4],
        vec!["a", "b", "c", "d"],
        Some(2),
    )
    .unwrap();
    let planned = network.plan(&refs, &GreedyDenseOptimizer).unwrap();
    let steps = planned.plan().steps();
    assert_eq!(
        (steps[0].lhs(), steps[0].rhs()),
        (TensorId::new(1), TensorId::new(3))
    );
    assert_eq!(
        (steps[1].lhs(), steps[1].rhs()),
        (TensorId::new(0), TensorId::new(2))
    );
    assert_eq!(
        (steps[2].lhs(), steps[2].rhs()),
        (TensorId::new(5), TensorId::new(4))
    );
    assert_eq!(
        steps[2].result_labels(),
        ["a", "b", "c", "d"].map(TemporaryLabel::from)
    );

    let ac = a
        .contract(
            &c,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let bd = b
        .contract(
            &d,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let manual = ac
        .contract(
            &bd,
            &ContractSpec {
                lhs: &[],
                rhs: &[],
                codomain: &[0, 2],
                domain: &[1, 3],
            },
        )
        .unwrap();
    assert_same(
        &planned.execute(&refs, &mut Default::default()).unwrap(),
        &manual,
    );
}

#[test]
fn greedy_order_and_four_site_ring_match_manual_typed_oracles() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = |degeneracy| u1_space(&provider, degeneracy);
    let (va, vb, vc, vd, ve) = (space(4), space(8), space(4), space(2), space(2));
    let chain = [
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&va], [&vb], 51).unwrap(),
        TensorMap::rand_with_seed(&runtime, [&vb], [&vc], 52).unwrap(),
        TensorMap::rand_with_seed(&runtime, [&vc], [&vd], 53).unwrap(),
        TensorMap::rand_with_seed(&runtime, [&vd], [&ve], 54).unwrap(),
    ];
    let chain_refs = [&chain[0], &chain[1], &chain[2], &chain[3]];
    let chain_network = Network::new(
        vec![
            vec!["a", "b"],
            vec!["b", "c"],
            vec!["c", "d"],
            vec!["d", "e"],
        ],
        vec![false; 4],
        vec![1; 4],
        vec!["a", "e"],
        Some(1),
    )
    .unwrap();
    let greedy = chain_network
        .plan(&chain_refs, &GreedyDenseOptimizer)
        .unwrap();
    let naive = chain_network
        .plan(
            &chain_refs,
            &LabelOrderDenseOptimizer::new(vec!["b".into(), "c".into(), "d".into()]),
        )
        .unwrap();
    assert_eq!(
        (
            greedy.plan().steps()[0].lhs(),
            greedy.plan().steps()[0].rhs()
        ),
        (TensorId::new(2), TensorId::new(3))
    );
    assert!(greedy.plan().total_cost() < naive.plan().total_cost());
    let greedy_result = greedy
        .execute(&chain_refs, &mut Default::default())
        .unwrap();
    let naive_result = naive.execute(&chain_refs, &mut Default::default()).unwrap();
    assert_eq!(greedy_result.codomain(), naive_result.codomain());
    assert_eq!(greedy_result.domain(), naive_result.domain());
    assert!(greedy_result
        .dense_data()
        .unwrap()
        .iter()
        .zip(naive_result.dense_data().unwrap())
        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));

    let v = space(3);
    let ring = std::array::from_fn::<_, 4, _>(|index| {
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 60 + index as u64)
            .unwrap()
    });
    let ring_refs = [&ring[0], &ring[1], &ring[2], &ring[3]];
    let ring_network = Network::new(
        vec![
            vec!["a", "b"],
            vec!["b", "c"],
            vec!["c", "d"],
            vec!["d", "a"],
        ],
        vec![false; 4],
        vec![1; 4],
        vec![],
        Some(0),
    )
    .unwrap();
    let planned = ring_network
        .plan(&ring_refs, &GreedyDenseOptimizer)
        .unwrap();
    let manual = ring[0]
        .contract(
            &ring[1],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .contract(
            &ring[2],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .contract(
            &ring[3],
            &ContractSpec {
                lhs: &[1, 0],
                rhs: &[0, 1],
                codomain: &[],
                domain: &[],
            },
        )
        .unwrap();
    let actual = planned
        .execute(&ring_refs, &mut Default::default())
        .unwrap();
    assert_eq!(actual.codomain(), manual.codomain());
    assert_eq!(actual.domain(), manual.domain());
    assert!(actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(manual.dense_data().unwrap())
        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
}
