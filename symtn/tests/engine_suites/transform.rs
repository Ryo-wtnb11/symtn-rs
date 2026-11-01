//! Structural transforms against oracles independent of any engine: the
//! physical-basis axis permutation (U(1), SU(2)) and the hand-valued fZ2
//! signs of the planar transposes, permutes and repartitions. Device == Host
//! parity of the same call stays in `typed_cuda_transform.rs`.

use super::*;
use crate::common::permute_dense;
use crate::legs::{su2_leg_2_2_1, u1_leg_2_1_2};
use crate::numerics::{self, Numeric};
use num_complex::Complex64;
use std::sync::Arc;
use symtn::sector::{FermionParityFusionRule, Z2Irrep};
use symtn::typed::{GradedSpace, Runtime};

/// Terms per physical entry: a rank-4 SU(2) tensor with spins up to 1 sums
/// over at most four intermediate spins.
const TERMS: usize = 4;

pub(super) fn real_fill<S: std::fmt::Debug>(
    _: &symtn::typed::BlockFusionTrees<S>,
    idx: &[usize],
) -> f64 {
    let mut value = 0.37;
    for (axis, &index) in idx.iter().enumerate() {
        value = value * 1.7 + (index as f64 + 1.0) * (axis as f64 + 2.0);
    }
    value
}

pub(super) fn complex_fill<S: std::fmt::Debug>(
    trees: &symtn::typed::BlockFusionTrees<S>,
    idx: &[usize],
) -> Complex64 {
    let re = real_fill(trees, idx);
    Complex64::new(re, -0.5 * re + 0.25)
}

fn sorted_parts<T: Numeric>(data: &[T]) -> Vec<(f64, f64)> {
    let mut values: Vec<_> = data.iter().map(|v| (v.wide().re, v.wide().im)).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values
}

/// `permute(.., [1, 0], [3, 2])` on `$backend` against the axis permutation
/// of the source's physical dense array. `$recoupling` demands that the
/// reduced payload is not a reordering of the source, so the oracle survived
/// a recoupling.
macro_rules! dense_permute_oracle {
    ($backend:ty, $device:ident, $engine:ident, $what:expr, $tensor:expr, $recoupling:expr) => {{
        let tensor = $tensor;
        let source = <$backend>::upload($device, &tensor);
        let moved = source.permute($device, &[1, 0], &[3, 2]).unwrap();
        let moved = <$backend>::download($device, &moved);
        let physical = tensor.to_physical_dense($engine).unwrap();
        let (shape, data) = permute_dense(&physical.shape, &physical.data, &[1, 0, 3, 2]);
        let actual = moved.to_physical_dense($engine).unwrap();
        assert_eq!(actual.shape, shape, "{} shape", $what);
        let scale = data
            .iter()
            .fold(0.0_f64, |max, v| max.max(v.wide().re.abs()));
        assert!(
            numerics::tolerance::<f64>(TERMS, scale) <= 1e-12,
            "{}: fill scale {scale} loosens the bound past the entry floor",
            $what
        );
        numerics::assert_slices_close($what, &actual.data, &data, TERMS);
        if $recoupling {
            assert_ne!(
                sorted_parts(tensor.dense_data().unwrap()),
                sorted_parts(moved.dense_data().unwrap()),
                "{}: permute must apply coefficients other than 1",
                $what
            );
        }
    }};
}

on_each_engine! {
    /// For a bosonic provider a permute inside the codomain and inside the
    /// domain is an axis permutation of the physical dense array. SU(2) makes
    /// it non-trivial: the reduced replay recouples to produce it.
    fn permute_within_one_side_is_the_physical_axis_permutation<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let u1 = u1_leg_2_1_2();
        let su2 = su2_leg_2_2_1();
        let u1_tensor: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&u1, &u1], [&u1, &u1], real_fill)
                .unwrap();
        dense_permute_oracle!(B, device, engine, "U(1)", u1_tensor, false);
        let su2_tensor: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&su2, &su2], [&su2, &su2], real_fill)
                .unwrap();
        dense_permute_oracle!(B, device, engine, "SU(2) f64", su2_tensor, true);
        let su2_tensor: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
            engine, &runtime, [&su2, &su2], [&su2, &su2], complex_fill,
        )
        .unwrap();
        dense_permute_oracle!(B, device, engine, "SU(2) c64", su2_tensor, true);
    }
}

/// `[leg, leg] <- [leg]` over fZ2, counting fill: four elements across two
/// blocks, all distinct, so the motion and the sign of every element show.
fn fz2_rank_three(
    engine: &mut dyn Engine,
    runtime: &Runtime,
) -> TensorMap<FermionParityFusionRule, f64> {
    let leg = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let mut next = 0.0;
    TensorMap::from_subblock_fn(engine, runtime, [&leg, &leg], [&leg], |_, _| {
        next += 1.0;
        next
    })
    .unwrap()
}

on_each_engine! {
    /// For a non-symmetric braiding a planar transpose and a permute of the
    /// same axes differ: the permute crosses strands and picks up the
    /// fermionic sign, the planar bend does not. Spelling `transpose` as a
    /// `permute` would pass every bosonic test and be wrong here.
    fn fermionic_transposes_bend_where_permutes_braid<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let tensor = fz2_rank_three(engine, &runtime);
        assert_eq!(tensor.dense_data().unwrap(), [1.0, 2.0, 3.0, 4.0]);
        let source = B::upload(device, &tensor);
        macro_rules! values {
            ($moved:expr) => {{
                let moved = $moved;
                B::download(device, &moved).dense_data().unwrap().to_vec()
            }};
        }

        // Full planar rotation (TensorKit's argument-free `transpose`).
        assert_eq!(values!(source.transpose(device, &[2], &[1, 0]).unwrap()), [1.0, 2.0, 4.0, 3.0]);
        assert_eq!(values!(source.permute(device, &[2], &[1, 0]).unwrap()), [1.0, 2.0, -4.0, -3.0]);
        // A different rotation of the planar order.
        assert_eq!(values!(source.transpose(device, &[1, 2], &[0]).unwrap()), [1.0, 4.0, 2.0, 3.0]);
        assert_eq!(values!(source.permute(device, &[1, 2], &[0]).unwrap()), [1.0, 4.0, -2.0, -3.0]);
    }
}

on_each_engine! {
    /// Moving the planar boundary crosses no strands, so `repartition` is
    /// sign free even for a fermionic provider and equals the permute of the
    /// same axes.
    fn fermionic_repartition_is_sign_free<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let tensor = fz2_rank_three(engine, &runtime);
        let source = B::upload(device, &tensor);
        macro_rules! values {
            ($moved:expr) => {{
                let moved = $moved;
                B::download(device, &moved).dense_data().unwrap().to_vec()
            }};
        }

        assert_eq!(values!(source.repartition(device, 0).unwrap()), [1.0, 4.0, 3.0, 2.0]);
        assert_eq!(values!(source.repartition(device, 1).unwrap()), [1.0, 4.0, 3.0, 2.0]);
        assert_eq!(values!(source.repartition(device, 3).unwrap()), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            values!(source.repartition(device, 0).unwrap()),
            values!(source.permute(device, &[], &[2, 1, 0]).unwrap())
        );
    }
}
