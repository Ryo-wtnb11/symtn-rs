//! Behaviour of the owned eager contraction output: inactive
//! destination blocks (coupled sectors the contracted bond cannot reach) are
//! exactly `+0.0`, active blocks agree with the destination path
//! `contract_into` on a `+0.0`-prefilled buffer, and an empty
//! support yields an all-zero payload of the full destination length.
//!
//! Owned and destination results are two execution paths of one contraction,
//! so their agreement is the contract; it is checked under the workspace
//! tolerance rule (`docs/testing_numerics.md`) with `terms` the contracted
//! bond length. Inactive blocks are exact zeros by construction.

use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::Arc;
use symtn::typed::ContractSpec;

use num_complex::Complex64;

use crate::numerics;
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap, TensorScalar};

trait Bits: TensorScalar + Debug + numerics::Numeric {
    fn is_positive_zero(self) -> bool;
}

impl Bits for f64 {
    fn is_positive_zero(self) -> bool {
        self.to_bits() == 0
    }
}

impl Bits for Complex64 {
    fn is_positive_zero(self) -> bool {
        self.re.to_bits() == 0 && self.im.to_bits() == 0
    }
}

fn u1(provider: &Arc<U1FusionRule>, sectors: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        sectors.iter().map(|&(q, d)| (U1Irrep::new(q), d)),
    )
    .unwrap()
}

fn su2(provider: &Arc<SU2FusionRule>, sectors: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        sectors
            .iter()
            .map(|&(twice, d)| (SU2Irrep::from_twice_spin(twice), d)),
    )
    .unwrap()
}

/// Every element of one block, walked through the view's own layout.
fn block_values<D: Copy>(view: &symtn::expert::BlockView<'_, D>) -> Vec<D> {
    let shape = view.shape();
    let count: usize = shape.iter().product();
    let mut indices = vec![0usize; shape.len()];
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(*view.get(&indices).unwrap());
        for axis in 0..shape.len() {
            indices[axis] += 1;
            if indices[axis] < shape[axis] {
                break;
            }
            indices[axis] = 0;
        }
    }
    values
}

/// Checks the owned result against the destination path on a zero buffer
/// and that every block whose coupled sector is absent from the operands is
/// exactly `+0.0`. Evaluates to the number of inactive blocks.
macro_rules! check_owned {
    ($output:expr, $expected:expr, $lhs:expr, $rhs:expr, $terms:expr) => {{
        let output = &$output;
        let expected = &$expected;
        numerics::assert_slices_close(
            "owned payload against the destination path",
            output.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            $terms,
        );
        let lhs_coupled: HashSet<_> = (0..$lhs.subblock_count())
            .map(|index| $lhs.subblock_fusion_trees(index).unwrap().coupled().clone())
            .collect();
        let rhs_coupled: HashSet<_> = (0..$rhs.subblock_count())
            .map(|index| $rhs.subblock_fusion_trees(index).unwrap().coupled().clone())
            .collect();
        let mut inactive = 0usize;
        for (trees, view) in output.subblocks().unwrap() {
            if lhs_coupled.contains(trees.coupled()) && rhs_coupled.contains(trees.coupled()) {
                continue;
            }
            inactive += 1;
            assert!(
                block_values(&view).into_iter().all(Bits::is_positive_zero),
                "inactive block {:?} is not +0.0",
                trees.coupled()
            );
        }
        inactive
    }};
}

fn u1_case<D: Bits>(seed: u64) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let open = u1(&provider, &[(-1, 2), (0, 3), (1, 2)]);
    let bond = u1(&provider, &[(0, 2)]);
    let lhs: TensorMap<_, D> =
        TensorMap::rand_with_seed(engine, &runtime, [&open], [&bond], seed).unwrap();
    let rhs: TensorMap<_, D> =
        TensorMap::rand_with_seed(engine, &runtime, [&bond], [&open], seed + 1).unwrap();

    // contract
    let output = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut expected = output.zeros_like(engine).unwrap();
    lhs.contract_into(
        engine,
        &rhs,
        &ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        },
        &mut expected,
        D::from_real(1.0),
        D::from_real(0.0),
    )
    .unwrap();
    assert_eq!(output.dense_data().unwrap().len(), 4 + 9 + 4);
    assert_eq!(check_owned!(output, expected, lhs, rhs, 2), 2);

    // compose
    let composed = lhs.compose(engine, &rhs).unwrap();
    assert_eq!(check_owned!(composed, expected, lhs, rhs, 2), 2);

    // lazy-adjoint operand (oriented owner)
    let parent: TensorMap<_, D> =
        TensorMap::rand_with_seed(engine, &runtime, [&bond], [&open], seed + 2).unwrap();
    let lazy = parent.adjoint(engine).unwrap();
    let output = lazy
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut expected = output.zeros_like(engine).unwrap();
    lazy.contract_into(
        engine,
        &rhs,
        &ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        },
        &mut expected,
        D::from_real(1.0),
        D::from_real(0.0),
    )
    .unwrap();
    assert_eq!(check_owned!(output, expected, lazy, rhs, 2), 2);
    let composed = lazy.compose(engine, &rhs).unwrap();
    assert_eq!(check_owned!(composed, expected, lazy, rhs, 2), 2);
}

#[test]
fn u1_owned_contractions_leave_unreachable_charges_exactly_zero_f64() {
    u1_case::<f64>(1_212_101);
}

#[test]
fn u1_owned_contractions_leave_unreachable_charges_exactly_zero_c64() {
    u1_case::<Complex64>(1_212_201);
}

fn su2_case<D: Bits>(seed: u64) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    // `[a, c] <- [b]` and `[b] <- [a, c]` couple only to spin 1/2 through the
    // bond; the destination `[a, c] <- [a, c]` also couples to spin 3/2.
    let a = su2(&provider, &[(0, 2), (2, 3)]);
    let c = su2(&provider, &[(1, 2)]);
    let b = su2(&provider, &[(1, 3)]);
    let lhs: TensorMap<_, D> =
        TensorMap::rand_with_seed(engine, &runtime, [&a, &c], [&b], seed).unwrap();
    let rhs: TensorMap<_, D> =
        TensorMap::rand_with_seed(engine, &runtime, [&b], [&a, &c], seed + 1).unwrap();

    let output = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap();
    let mut expected = output.zeros_like(engine).unwrap();
    lhs.contract_into(
        engine,
        &rhs,
        &ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 1],
            domain: &[2, 3],
        },
        &mut expected,
        D::from_real(1.0),
        D::from_real(0.0),
    )
    .unwrap();
    assert!(check_owned!(output, expected, lhs, rhs, 3) >= 1);

    let composed = lhs.compose(engine, &rhs).unwrap();
    assert!(check_owned!(composed, expected, lhs, rhs, 3) >= 1);

    let parent: TensorMap<_, D> =
        TensorMap::rand_with_seed(engine, &runtime, [&b], [&a, &c], seed + 2).unwrap();
    let lazy = parent.adjoint(engine).unwrap();
    let output = lazy.compose(engine, &rhs).unwrap();
    let mut expected = output.zeros_like(engine).unwrap();
    lazy.contract_into(
        engine,
        &rhs,
        &ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 1],
            domain: &[2, 3],
        },
        &mut expected,
        D::from_real(1.0),
        D::from_real(0.0),
    )
    .unwrap();
    assert!(check_owned!(output, expected, lazy, rhs, 3) >= 1);
}

#[test]
fn su2_owned_contractions_leave_unreachable_spins_exactly_zero_f64() {
    su2_case::<f64>(1_212_301);
}

#[test]
fn su2_owned_contractions_leave_unreachable_spins_exactly_zero_c64() {
    su2_case::<Complex64>(1_212_401);
}

#[test]
fn empty_support_yields_a_zero_payload_of_the_full_destination_length() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: `[a] <- [b]` with integer `a` and half-integer `b` has no block at
    // all; composing with `[b] <- [a]` still owns the full `[a] <- [a]`
    // payload (spins 0 and 1: 2*2 + 3*3 elements), every element `+0.0`.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let a = su2(&provider, &[(0, 2), (2, 3)]);
    let b = su2(&provider, &[(1, 3)]);
    let lhs: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&a], [&b], 7).unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&b], [&a], 8).unwrap();
    assert_eq!(lhs.subblock_count(), 0);
    let output = lhs.compose(engine, &rhs).unwrap();
    assert_eq!(output.dense_data().unwrap().len(), 4 + 9);
    assert!(output
        .dense_data()
        .unwrap()
        .iter()
        .all(|v| v.to_bits() == 0));
    assert_eq!(
        check_owned!(output, output.zeros_like(engine).unwrap(), lhs, rhs, 1),
        2
    );
}
