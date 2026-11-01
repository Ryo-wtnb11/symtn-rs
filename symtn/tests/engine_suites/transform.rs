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

fn u1_rank_four(
    engine: &mut dyn Engine,
    runtime: &Runtime,
) -> TensorMap<symtn::sector::U1FusionRule, f64> {
    let u1 = u1_leg_2_1_2();
    TensorMap::from_subblock_fn(engine, runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap()
}

/// The spaces and fusion-tree identities of every block, in block order.
fn identity<R, D>(tensor: &TensorMap<R, D>) -> impl PartialEq + std::fmt::Debug
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec,
    D: symtn::typed::TensorScalar,
    <R as symtn::sector::SectorCodec>::Sector: std::fmt::Debug,
{
    (
        tensor
            .codomain()
            .iter()
            .chain(tensor.domain().iter())
            .map(|leg| (leg.is_dual(), leg.degeneracies().to_vec()))
            .collect::<Vec<_>>(),
        (0..tensor.subblock_count())
            .map(|index| tensor.subblock_fusion_trees(index).unwrap())
            .collect::<Vec<_>>(),
    )
}

on_each_engine! {
    /// permute composition, braid inverse, bosonic braid == permute and the
    /// transpose involution, each evaluated entirely on the engine.
    fn transform_laws_hold<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let host = u1_rank_four(engine, &runtime);
        let source = B::upload(device, &host);
        macro_rules! values {
            ($moved:expr) => {{
                let moved = $moved;
                B::download(device, &moved).dense_data().unwrap().to_vec()
            }};
        }
        let close = |what: &str, got: &[f64], want: &[f64]| numerics::assert_slices_close(what, got, want, TERMS);

        let first = [2usize, 0, 3, 1];
        let second = [1usize, 0, 3, 2];
        let stepwise = values!(source
            .permute(device, &first[..2], &first[2..])
            .unwrap()
            .permute(device, &second[..2], &second[2..])
            .unwrap());
        let composed: Vec<usize> = second.iter().map(|&axis| first[axis]).collect();
        let direct = values!(source.permute(device, &composed[..2], &composed[2..]).unwrap());
        close("permute composition", &stepwise, &direct);
        assert_ne!(direct, host.dense_data().unwrap(), "the composition must move the payload");

        // braid o braid^-1 == id, with the levels carried along the strands.
        let axes = [2usize, 0, 3, 1];
        let levels = [4usize, 1, 3, 2];
        let braided = source.braid(device, &axes[..2], &axes[2..], &levels).unwrap();
        let mut inverse = [0usize; 4];
        for (index, &axis) in axes.iter().enumerate() {
            inverse[axis] = index;
        }
        let carried: Vec<usize> = axes.iter().map(|&axis| levels[axis]).collect();
        close(
            "braid inverse round trip",
            &values!(braided.braid(device, &inverse[..2], &inverse[2..], &carried).unwrap()),
            host.dense_data().unwrap(),
        );

        // A bosonic braid is the permute for any levels.
        let permuted = values!(source.permute(device, &axes[..2], &axes[2..]).unwrap());
        close("bosonic braid == permute", &values!(braided), &permuted);

        // Transpose is a bitwise involution that restores spaces and trees.
        let once = source.transpose(device, &[3, 2], &[1, 0]).unwrap();
        let twice = once.transpose(device, &[3, 2], &[1, 0]).unwrap();
        let twice = B::download(device, &twice);
        assert_eq!(twice.dense_data().unwrap(), host.dense_data().unwrap());
        assert_eq!(identity(&twice), identity(&host));
    }
}

on_each_engine! {
    /// Every repartition of a rank-four tensor round-trips bitwise back to
    /// the original split; a bend of an SU(2) tensor carries a scalar that is
    /// neither 1 nor -1, so its payload is not a reordering of the source.
    fn repartition_reaches_every_split_and_round_trips<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let host = u1_rank_four(engine, &runtime);
        let source = B::upload(device, &host);
        for num_codomain in 0..=4 {
            let moved = source.repartition(device, num_codomain).unwrap();
            let back = moved.repartition(device, 2).unwrap();
            let back = B::download(device, &back);
            assert_eq!(back.dense_data().unwrap(), host.dense_data().unwrap(), "split {num_codomain}");
            assert_eq!(identity(&back), identity(&host), "split {num_codomain}");
        }

        let su2 = su2_leg_2_2_1();
        let su2_host: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&su2, &su2], [&su2, &su2], real_fill)
                .unwrap();
        let su2_source = B::upload(device, &su2_host);
        for num_codomain in [0usize, 1, 3, 4] {
            let moved = su2_source.repartition(device, num_codomain).unwrap();
            let moved = B::download(device, &moved);
            assert_ne!(
                sorted_parts(su2_host.dense_data().unwrap()),
                sorted_parts(moved.dense_data().unwrap()),
                "SU(2) repartition to {num_codomain} is a bare reordering"
            );
        }
    }
}

on_each_engine! {
    /// TensorKit 0.17.1: the 2-norm of the complex SU(2) fixture permuted by
    /// `[1, 0], [3, 2]` is a value computed outside symtn.
    fn complex_su2_permute_has_the_tensorkit_norm<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let host = crate::tk_complex_su2::source(engine, &runtime);
        let source = B::upload(device, &host);
        let permuted = source.permute(device, &[1, 0], &[3, 2]).unwrap();
        let permuted = B::download(device, &permuted);
        assert_ne!(
            sorted_parts(host.dense_data().unwrap()),
            sorted_parts(permuted.dense_data().unwrap()),
            "the permute must apply coefficients other than 1"
        );
        let norm = permuted.norm(engine, 2.0).unwrap();
        assert!(
            (norm - 40.741_733_994_626_32).abs() <= 1e-10 * 41.0,
            "TensorKit norm of the permuted tensor: got {norm}"
        );
    }
}

on_each_engine! {
    /// A structure with no admissible block: the output upload, the replay and
    /// the download all survive it, for the lazy adjoint and `*_into` too.
    fn transforms_of_an_empty_tensor_are_empty<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let codomain = u1_leg(&[(1, 2)]);
        let domain = u1_leg(&[(0, 3)]);
        let host: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&codomain], [&domain], |_, _| 1.0)
                .unwrap();
        assert_eq!(host.subblock_count(), 0);
        let source = B::upload(device, &host);

        for (what, moved) in [
            ("permute", source.permute(device, &[1], &[0]).unwrap()),
            ("braid", source.braid(device, &[1], &[0], &[1, 2]).unwrap()),
            ("transpose", source.transpose(device, &[1], &[0]).unwrap()),
            ("repartition", source.repartition(device, 0).unwrap()),
        ] {
            let moved = B::download(device, &moved);
            assert!(moved.materialize(engine).unwrap().dense_data().unwrap().is_empty(), "{what}");
            assert_eq!(moved.subblock_count(), 0, "{what}: expected no blocks");
        }
        let adjoint = source.adjoint(device).unwrap();
        let moved = adjoint.transpose(device, &[1], &[0]).unwrap();
        let moved = B::download(device, &moved);
        assert!(moved.materialize(engine).unwrap().dense_data().unwrap().is_empty());

        let model = host.permute(engine, &[1], &[0]).unwrap();
        let mut destination = B::upload(device, &model);
        drop(model);
        source.permute_into(device, &[1], &[0], &mut destination, 2.0, 0.0).unwrap();
        assert!(B::download(device, &destination).dense_data().unwrap().is_empty());
    }
}

fn u1_leg(charges: &[(i32, usize)]) -> GradedSpace<symtn::sector::U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(symtn::sector::U1FusionRule),
        charges
            .iter()
            .map(|&(c, d)| (symtn::sector::U1Irrep::new(c), d)),
    )
    .unwrap()
}
