//! Real-device gates for typed structural transforms on CUDA tensors
//! (issue #1322): `permute`, `braid`, `transpose`
//! and `repartition` on `TensorMap<R, D, CudaStorage<D>>`.
//!
//! Evidence chain, as the design review fixed it:
//!
//! 1. the device executor below the completed `TreeTransformStructure` is
//!    gated independently by the structure walker in
//!    `symtn-operations/tests/cuda_tree_transform.rs`;
//! 2. the typed *lowering* is gated here, by running the same public call on
//!    the Host and on the device — the Host side carries its own TensorKit and
//!    hand-computed oracles (`typed_facade.rs`, `semantic_suite.rs`,
//!    `tk_complex_su2.rs`, `lazy_adjoint_shared_owner.rs`), which these are
//!    device twins of;
//! 3. a dense oracle independent of both runs where a transform really is a
//!    plain axis permutation of the physical array: a bosonic permute within
//!    one side, for the two providers that have a physical basis (U(1), SU(2)).
//!    Its own agreement with the Host is pinned by
//!    `typed_transform_host_side.rs`, which is not feature gated and so runs
//!    in ordinary CI.
//!
//! Every fixture additionally asserts non-vacuity — either that the payload
//! actually moved, or that the transform is not a bare reordering, which is
//! what proves a coefficient other than `1` or a recoupling block took part.
//!
//! Run with `cargo test -p symtn --features cuda,cpu-faer --test \
//! typed_cuda_transform -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

mod common;
#[path = "../../tests/support"]
mod support {
    use num_complex::{Complex32, Complex64};
    pub mod numerics;
}
use support::numerics;

/// TensorKit's argument-free `transpose(t)`: the full planar rotation, which
/// carries every codomain leg across the boundary and every domain leg back.
macro_rules! full_rotation {
    ($tensor:expr) => {{
        let tensor = &$tensor;
        let codomain_rank = tensor.codomain_rank();
        let codomain_axes: Vec<usize> = (codomain_rank..tensor.rank()).rev().collect();
        let domain_axes: Vec<usize> = (0..codomain_rank).rev().collect();
        (codomain_axes, domain_axes)
    }};
}

macro_rules! full_transpose {
    ($tensor:expr) => {{
        let tensor = &$tensor;
        let (codomain_axes, domain_axes) = full_rotation!(tensor);
        tensor.transpose(&codomain_axes, &domain_axes)
    }};
}

macro_rules! full_transpose_into {
    ($tensor:expr, $destination:expr, $alpha:expr) => {{
        let tensor = &$tensor;
        let (codomain_axes, domain_axes) = full_rotation!(tensor);
        tensor.transpose_into(
            &codomain_axes,
            &domain_axes,
            $destination,
            $alpha,
            Default::default(),
        )
    }};
}

use std::sync::Arc;

use common::permute_dense;

use num_complex::Complex64;
use symtn::sector::CheckedFusionAlgebra;
use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::sector::{MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::typed::{GradedSpace, TensorMap};
use symtn::typed::{Runtime, TensorScalar};

// ---------------------------------------------------------------------------
// Payload comparison
// ---------------------------------------------------------------------------

/// The two device payload dtypes, with the comparisons the gates need.
trait Payload: TensorScalar + Copy + PartialEq + std::fmt::Debug {
    fn parts(self) -> (f64, f64);
    /// The NaN of this dtype, used to poison an overwrite destination.
    fn nan() -> Self;
    fn distance(self, other: Self) -> f64 {
        let (ar, ai) = self.parts();
        let (br, bi) = other.parts();
        ((ar - br).powi(2) + (ai - bi).powi(2)).sqrt()
    }
    fn magnitude(self) -> f64 {
        let (re, im) = self.parts();
        (re * re + im * im).sqrt()
    }
}

impl Payload for f64 {
    fn parts(self) -> (f64, f64) {
        (self, 0.0)
    }
    fn nan() -> Self {
        f64::NAN
    }
}

impl Payload for Complex64 {
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
    fn nan() -> Self {
        Complex64::new(f64::NAN, f64::NAN)
    }
}

fn assert_payload_close<D: Payload>(actual: &[D], expected: &[D], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: payload length");
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            left.distance(right) <= 1e-12 * (1.0 + right.magnitude()),
            "{what}: element {index} is {left:?}, expected {right:?}"
        );
    }
}

/// Sorted payload, used only to compare two payloads as multisets.
fn sorted_values<D: Payload>(data: &[D]) -> Vec<(f64, f64)> {
    let mut values: Vec<(f64, f64)> = data.iter().map(|value| value.parts()).collect();
    values.sort_by(|left, right| left.partial_cmp(right).expect("finite fixture payload"));
    values
}

/// Non-vacuity, weak form: the transform actually moved the payload, so a
/// device replay that wrote nothing at all could not pass.
fn assert_moved<D: Payload>(source: &[D], transformed: &[D], what: &str) {
    assert_eq!(source.len(), transformed.len(), "{what}: length");
    assert!(
        source.iter().zip(transformed).any(|(a, b)| a != b),
        "{what}: the transform left the payload identical, so it proves nothing"
    );
}

/// Non-vacuity, strong form: the transform is not a bare reordering of the
/// source payload, so at least one structural coefficient differs from `1` or
/// a recoupling block mixed several source entries.
fn assert_not_a_reordering<D: Payload>(source: &[D], transformed: &[D], what: &str) {
    assert_ne!(
        sorted_values(source),
        sorted_values(transformed),
        "{what}: the payload is a permutation of the source, so no coefficient \
         other than 1 and no recoupling block took part"
    );
}

// ---------------------------------------------------------------------------
// Layout comparison
// ---------------------------------------------------------------------------

type Layout<S> = (
    Vec<(bool, Vec<usize>)>,
    Vec<symtn::typed::BlockFusionTrees<S>>,
);

fn layout<R, D>(tensor: &TensorMap<R, D>) -> Layout<<R as SectorCodec>::Sector>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    (
        tensor
            .codomain()
            .iter()
            .chain(tensor.domain().iter())
            .map(|leg| (leg.is_dual(), leg.degeneracies().to_vec()))
            .collect(),
        (0..tensor.subblock_count())
            .map(|index| tensor.subblock_fusion_trees(index).unwrap())
            .collect(),
    )
}

/// Runs one transform on the Host and on the device and asserts the device
/// answer is the Host answer — payload to dtype tolerance, spaces and block
/// identities exactly. `$body` is written once and expanded against both
/// receiver types, so the two calls cannot drift.
macro_rules! device_matches_host {
    ($what:expr, $host:expr, |$t:ident| $body:expr) => {{
        let host_source = &$host;
        let device_source = host_source.to_cuda().unwrap();
        let expected = {
            let $t = host_source;
            $body
        }
        .unwrap();
        let actual = {
            let $t = &device_source;
            $body
        }
        .unwrap()
        .to_host()
        .unwrap();
        let what: &str = $what;
        assert_payload_close(
            actual.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            what,
        );
        assert_eq!(layout(&actual), layout(&expected), "{what}: layout");
        actual
    }};
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn runtime() -> Runtime {
    Runtime::builder().cuda(0).build().unwrap()
}

/// Distinct, finite, irrational-ish entries: a permutation of the payload is
/// then visible as a multiset, and no two entries cancel by accident.
fn real_fill(_trees: &symtn::typed::BlockFusionTrees<impl std::fmt::Debug>, idx: &[usize]) -> f64 {
    let mut value = 0.37;
    for (axis, &index) in idx.iter().enumerate() {
        value = value * 1.7 + (index as f64 + 1.0) * (axis as f64 + 2.0);
    }
    value
}

fn complex_fill(
    trees: &symtn::typed::BlockFusionTrees<impl std::fmt::Debug>,
    idx: &[usize],
) -> Complex64 {
    let re = real_fill(trees, idx);
    Complex64::new(re, -0.5 * re + 0.25)
}

fn u1_leg(charges: &[(i32, usize)], dual: bool) -> GradedSpace<U1FusionRule> {
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        charges
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    if dual {
        space.try_dual().unwrap()
    } else {
        space
    }
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

fn fz2_leg() -> GradedSpace<FermionParityFusionRule> {
    GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
    )
    .unwrap()
}

/// `[leg, leg] <- [leg]` over fZ2 with the counting fill of
/// `typed_facade.rs::fermionic_rank_three`, whose transposes and permutes are
/// hand-computed there.
fn fz2_rank_three(runtime: &Runtime) -> TensorMap<FermionParityFusionRule, f64> {
    let leg = fz2_leg();
    let mut next = 0.0;
    TensorMap::from_subblock_fn(runtime, [&leg, &leg], [&leg], |_, _| {
        next += 1.0;
        next
    })
    .unwrap()
}

// ---------------------------------------------------------------------------
// Dense oracle (bosonic permute within one side)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Device gates
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires a real CUDA device"]
fn device_permute_braid_and_planar_match_the_host_for_u1_and_su2() {
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let u1_dual = u1_leg(&[(-1, 1), (0, 2), (1, 1)], true);
    let su2 = su2_leg();

    // U(1) with a dual leg and several coupled sectors, both dtypes.
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1_dual], [&u1, &u1], real_fill).unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1_dual], [&u1, &u1], complex_fill).unwrap();
    assert!(real.subblock_count() >= 2, "multi-block fixture");

    let permuted = device_matches_host!("U1/f64 permute", real, |t| t.permute(&[2, 0], &[1, 3]));
    assert_moved(
        real.dense_data().unwrap(),
        permuted.dense_data().unwrap(),
        "U1/f64 permute",
    );
    // Coefficient-1 f64 block moves are bitwise on device.
    assert_eq!(
        permuted.dense_data().unwrap(),
        real.permute(&[2, 0], &[1, 3])
            .unwrap()
            .dense_data()
            .unwrap(),
        "coefficient-1 f64 permute must move bitwise"
    );
    device_matches_host!("U1/c64 permute", complex, |t| t.permute(&[2, 0], &[1, 3]));
    device_matches_host!("U1/f64 braid", real, |t| t.braid(
        &[2, 0],
        &[1, 3],
        &[3, 1, 4, 2]
    ));
    device_matches_host!("U1/f64 transpose", real, |t| full_transpose!(t));
    device_matches_host!("U1/c64 transpose", complex, |t| full_transpose!(t));
    device_matches_host!("U1/f64 cyclic transpose", real, |t| t
        .transpose(&[1, 3], &[0, 2]));

    // SU(2): recoupling, so Multi blocks and coefficients other than 1.
    let su2_real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2, &su2], real_fill).unwrap();
    let su2_complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2, &su2], complex_fill).unwrap();
    let permuted =
        device_matches_host!("SU2/f64 permute", su2_real, |t| t.permute(&[1, 2], &[3, 0]));
    assert_not_a_reordering(
        su2_real.dense_data().unwrap(),
        permuted.dense_data().unwrap(),
        "SU2/f64 permute",
    );
    let permuted = device_matches_host!("SU2/c64 permute", su2_complex, |t| t
        .permute(&[1, 2], &[3, 0]));
    assert_not_a_reordering(
        su2_complex.dense_data().unwrap(),
        permuted.dense_data().unwrap(),
        "SU2/c64 permute",
    );
    let transposed = device_matches_host!("SU2/f64 transpose", su2_real, |t| full_transpose!(t));
    assert_not_a_reordering(
        su2_real.dense_data().unwrap(),
        transposed.dense_data().unwrap(),
        "SU2/f64 transpose",
    );
}

/// #2192: Unique fusion is `NoCache`, so a degeneracy-only change misses
/// the completed transformer and every block's phase is recomputed. The
/// device permute right after that change, and the warm one after it, must
/// match the Host. The timings are an observation, not a gate.
#[test]
#[ignore = "requires a real CUDA device"]
fn device_u1_permute_after_a_degeneracy_change_matches_the_host() {
    let runtime = runtime();
    let fixture = |outer: usize, inner: usize| {
        let u1 = u1_leg(&[(-1, outer), (0, inner), (1, outer)], false);
        let u1_dual = u1_leg(&[(-1, inner), (0, outer), (1, inner)], true);
        TensorMap::<_, f64>::from_subblock_fn(&runtime, [&u1, &u1_dual], [&u1, &u1], real_fill)
            .unwrap()
    };
    let first = fixture(2, 1);
    device_matches_host!("U1 permute, first degeneracies", first, |t| t
        .permute(&[2, 0], &[1, 3]));
    let changed = fixture(1, 3);
    let device = changed.to_cuda().unwrap();
    for what in ["after a degeneracy change", "warm"] {
        let start = std::time::Instant::now();
        let permuted = device.permute(&[2, 0], &[1, 3]).unwrap();
        let elapsed = start.elapsed();
        let permuted = permuted.to_host().unwrap();
        eprintln!("U1 device permute {what}: {elapsed:?}");
        let expected = changed.permute(&[2, 0], &[1, 3]).unwrap();
        assert_payload_close(
            permuted.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            what,
        );
        assert_eq!(layout(&permuted), layout(&expected), "{what}: layout");
        assert_moved(
            changed.dense_data().unwrap(),
            permuted.dense_data().unwrap(),
            what,
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_permutes_match_the_dense_physical_oracle_within_one_side() {
    // The independent oracle of the acceptance gate: for a bosonic provider, a
    // permute inside the codomain and inside the domain is an axis permutation
    // of the physical dense array. SU(2) makes it non-trivial — the reduced
    // replay recouples to produce it.
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let u1_tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    let device = u1_tensor.to_cuda().unwrap();
    let moved = device.permute(&[1, 0], &[3, 2]).unwrap().to_host().unwrap();
    let source = u1_tensor.to_physical_dense().unwrap();
    let (shape, data) = permute_dense(&source.shape, &source.data, &[1, 0, 3, 2]);
    let actual = moved.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    assert_payload_close(&actual.data, &data, "U(1) device dense oracle");

    let su2 = su2_leg();
    let su2_tensor: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2, &su2], complex_fill).unwrap();
    let device = su2_tensor.to_cuda().unwrap();
    let moved = device.permute(&[1, 0], &[3, 2]).unwrap().to_host().unwrap();
    assert_not_a_reordering(
        su2_tensor.dense_data().unwrap(),
        moved.dense_data().unwrap(),
        "SU(2) device permute",
    );
    let source = su2_tensor.to_physical_dense().unwrap();
    let (shape, data) = permute_dense(&source.shape, &source.data, &[1, 0, 3, 2]);
    let actual = moved.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    assert_payload_close(&actual.data, &data, "SU(2) device dense oracle");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transform_laws_hold_on_device_tensors() {
    // Device twins of `semantic_suite.rs`: permute composition, braid inverse,
    // bosonic braid == permute, transpose involution — all evaluated entirely
    // on device, then compared to the Host's own answer for the same law.
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    let device = host.to_cuda().unwrap();

    // permute ∘ permute == permute(composition)
    let first = [2usize, 0, 3, 1];
    let second = [1usize, 0, 3, 2];
    let stepwise = device
        .permute(&first[..2], &first[2..])
        .unwrap()
        .permute(&second[..2], &second[2..])
        .unwrap()
        .to_host()
        .unwrap();
    let composed: Vec<usize> = second.iter().map(|&axis| first[axis]).collect();
    let direct = device
        .permute(&composed[..2], &composed[2..])
        .unwrap()
        .to_host()
        .unwrap();
    assert_payload_close(
        stepwise.dense_data().unwrap(),
        direct.dense_data().unwrap(),
        "permute composition",
    );
    assert_moved(
        host.dense_data().unwrap(),
        direct.dense_data().unwrap(),
        "permute composition",
    );

    // braid ∘ braid⁻¹ == id, with the levels carried along the strands.
    let axes = [2usize, 0, 3, 1];
    let levels = [4usize, 1, 3, 2];
    let braided = device.braid(&axes[..2], &axes[2..], &levels).unwrap();
    let mut inverse = [0usize; 4];
    for (index, &axis) in axes.iter().enumerate() {
        inverse[axis] = index;
    }
    let carried: Vec<usize> = axes.iter().map(|&axis| levels[axis]).collect();
    let back = braided
        .braid(&inverse[..2], &inverse[2..], &carried)
        .unwrap()
        .to_host()
        .unwrap();
    assert_payload_close(
        back.dense_data().unwrap(),
        host.dense_data().unwrap(),
        "braid inverse round trip",
    );

    // Bosonic provider: braid == permute for any levels.
    let permuted = device.permute(&axes[..2], &axes[2..]).unwrap();
    assert_payload_close(
        braided.to_host().unwrap().dense_data().unwrap(),
        permuted.to_host().unwrap().dense_data().unwrap(),
        "bosonic braid == permute",
    );

    // Transpose is an involution.
    let once = full_transpose!(device).unwrap();
    let twice = full_transpose!(once).unwrap().to_host().unwrap();
    assert_eq!(
        twice.dense_data().unwrap(),
        host.dense_data().unwrap(),
        "transpose involution is bitwise"
    );
    assert_eq!(layout(&twice), layout(&host), "transpose involution layout");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_repartition_reaches_every_split_and_round_trips() {
    // The #1310 residual: codomain/domain repartition, i.e. bends, on device.
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    let device = host.to_cuda().unwrap();
    for num_codomain in 0..=4 {
        let what = format!("U(1) repartition to {num_codomain}");
        let moved = device.repartition(num_codomain).unwrap();
        let expected = host.repartition(num_codomain).unwrap();
        let actual = moved.to_host().unwrap();
        assert_payload_close(
            actual.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            &what,
        );
        assert_eq!(layout(&actual), layout(&expected), "{what}: layout");
        let back = moved.repartition(2).unwrap().to_host().unwrap();
        assert_eq!(
            back.dense_data().unwrap(),
            host.dense_data().unwrap(),
            "{what}: round trip is bitwise"
        );
        assert_eq!(layout(&back), layout(&host), "{what}: round trip layout");
    }

    // SU(2): a bend carries a scalar that is neither 1 nor -1, so the payload
    // is not a reordering of the source.
    let su2 = su2_leg();
    let su2_host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2, &su2], real_fill).unwrap();
    for num_codomain in [0usize, 1, 3, 4] {
        let what = format!("SU(2) repartition to {num_codomain}");
        let moved = device_matches_host!(&what, su2_host, |t| t.repartition(num_codomain));
        assert_not_a_reordering(
            su2_host.dense_data().unwrap(),
            moved.dense_data().unwrap(),
            &what,
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_fermionic_signs_match_the_hand_computed_fixture() {
    // Device twin of `typed_facade.rs::planar_transposes_bend_where_permute_braids`
    // and `repartition_is_sign_free_even_for_a_fermionic_provider`: the values
    // are hand-computed there, not read off any symtn descriptor.
    let runtime = runtime();
    let host = fz2_rank_three(&runtime);
    assert_eq!(host.dense_data().unwrap(), [1.0, 2.0, 3.0, 4.0]);
    let device = host.to_cuda().unwrap();

    assert_eq!(
        full_transpose!(device)
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        [1.0, 2.0, 4.0, 3.0]
    );
    assert_eq!(
        device
            .permute(&[2], &[1, 0])
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        [1.0, 2.0, -4.0, -3.0]
    );
    assert_eq!(
        device
            .transpose(&[1, 2], &[0])
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        [1.0, 4.0, 2.0, 3.0]
    );
    assert_eq!(
        device
            .permute(&[1, 2], &[0])
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        [1.0, 4.0, -2.0, -3.0]
    );
    for num_codomain in [0usize, 1, 3] {
        assert_eq!(
            device
                .repartition(num_codomain)
                .unwrap()
                .to_host()
                .unwrap()
                .dense_data()
                .unwrap(),
            host.repartition(num_codomain)
                .unwrap()
                .dense_data()
                .unwrap(),
            "repartition to {num_codomain} is sign free",
        );
    }
    // Non-vacuity: the permutes above changed signs, so they are not
    // reorderings of the source payload.
    assert_not_a_reordering(
        host.dense_data().unwrap(),
        device
            .permute(&[2], &[1, 0])
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        "fZ2 permute",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_product_providers_match_the_host_for_signs_and_recoupling() {
    let runtime = runtime();

    // fZ2 x U(1): fermionic signs on top of a charge grading.
    let rule = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let leg = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1),
        ],
    )
    .unwrap();
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], real_fill).unwrap();
    let permuted = device_matches_host!("fZ2xU1 permute", host, |t| t.permute(&[2, 1], &[0, 3]));
    assert_not_a_reordering(
        host.dense_data().unwrap(),
        permuted.dense_data().unwrap(),
        "fZ2xU1 permute",
    );
    device_matches_host!("fZ2xU1 transpose", host, |t| full_transpose!(t));
    device_matches_host!("fZ2xU1 braid", host, |t| t.braid(
        &[2, 1],
        &[0, 3],
        &[1, 4, 2, 3]
    ));
    device_matches_host!("fZ2xU1 repartition", host, |t| t.repartition(1));

    // fZ2 (x) SU(2): fermionic signs *and* recoupling in one provider.
    let rule = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
            .unwrap(),
    );
    let leg = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                2,
            ),
            (
                product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(1)),
                1,
            ),
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                1,
            ),
        ],
    )
    .unwrap();
    let host: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], complex_fill).unwrap();
    let permuted = device_matches_host!("fZ2xSU2 permute", host, |t| t.permute(&[1, 2], &[3, 0]));
    assert_not_a_reordering(
        host.dense_data().unwrap(),
        permuted.dense_data().unwrap(),
        "fZ2xSU2 permute",
    );
    let transposed = device_matches_host!("fZ2xSU2 transpose", host, |t| full_transpose!(t));
    assert_not_a_reordering(
        host.dense_data().unwrap(),
        transposed.dense_data().unwrap(),
        "fZ2xSU2 transpose",
    );
    device_matches_host!("fZ2xSU2 repartition", host, |t| t.repartition(3));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_rank_five_transforms_match_the_host() {
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 1)], false);
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1, &u1], [&u1, &u1], real_fill).unwrap();
    assert!(host.subblock_count() >= 3, "multi-block rank-5 fixture");
    let permuted = device_matches_host!("rank-5 permute", host, |t| t.permute(&[4, 1, 0], &[3, 2]));
    assert_moved(
        host.dense_data().unwrap(),
        permuted.dense_data().unwrap(),
        "rank-5 permute",
    );
    device_matches_host!("rank-5 transpose", host, |t| full_transpose!(t));
    device_matches_host!("rank-5 repartition", host, |t| t.repartition(1));

    let su2 = su2_leg();
    let su2_host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2, &su2], [&su2, &su2], real_fill).unwrap();
    let permuted = device_matches_host!("rank-5 SU(2) permute", su2_host, |t| t
        .permute(&[4, 1, 0], &[3, 2]));
    assert_not_a_reordering(
        su2_host.dense_data().unwrap(),
        permuted.dense_data().unwrap(),
        "rank-5 SU(2) permute",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_complex_su2_permute_matches_the_tensorkit_fixture() {
    // Device twin of `tk_complex_su2.rs`: the norm is TensorKit's own value,
    // so this is a value oracle outside symtn entirely.
    let runtime = runtime();
    let space = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let fill = |c0: i64, labels: [i64; 5], idx: &[usize]| {
        let [l1, l2, m1, m2, lc] = labels;
        let value = c0
            + 7 * l1
            + 11 * l2
            + 13 * m1
            + 17 * m2
            + 19 * lc
            + 23 * (idx[0] as i64 + 1)
            + 29 * (idx[1] as i64 + 1)
            + 31 * (idx[2] as i64 + 1)
            + 37 * (idx[3] as i64 + 1);
        (value.rem_euclid(41) - 20) as f64
    };
    let host: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        [&space, &space],
        [&space, &space],
        |trees, idx| {
            let label = |sector: &SU2Irrep| sector.twice_spin() as i64;
            let codomain = trees.codomain_uncoupled();
            let domain = trees.domain_uncoupled();
            let labels = [
                label(&codomain[0]),
                label(&codomain[1]),
                label(&domain[0]),
                label(&domain[1]),
                label(trees.coupled()),
            ];
            Complex64::new(fill(11, labels, idx), fill(17, labels, idx) / 3.0)
        },
    )
    .unwrap();

    let permuted = device_matches_host!("tk SU(2) c64 permute", host, |t| t
        .permute(&[1, 0], &[3, 2]));
    assert_not_a_reordering(
        host.dense_data().unwrap(),
        permuted.dense_data().unwrap(),
        "tk SU(2) c64 permute",
    );
    let norm = permuted.norm(2.0).unwrap();
    assert!(
        (norm - 40.741_733_994_626_32).abs() <= 1e-10 * 41.0,
        "TensorKit norm of the permuted tensor: got {norm}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_lazy_adjoint_transforms_lower_onto_the_parent_like_the_host() {
    // Device twin of `lazy_adjoint_shared_owner.rs`: every transform of a lazy
    // adjoint lowers onto the parent and re-wraps, so it must equal the Host's
    // transform of the same lazy adjoint.
    //
    // The witness is taken against the *lazy adjoint's own* payload, not the
    // parent's: the adjoint alone already rearranges the parent, so comparing
    // with the parent would pass even if the transform did nothing.
    let runtime = runtime();
    let a = u1_leg(&[(-1, 2), (0, 1), (1, 3)], false);
    let b = u1_leg(&[(0, 2), (1, 1)], true);
    let c = u1_leg(&[(-1, 1), (1, 2)], false);
    let d = u1_leg(&[(-2, 2), (-1, 3), (0, 1), (1, 2), (2, 1)], false);
    let u1_parent: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&a, &b, &c], [&d], complex_fill).unwrap();
    assert!(u1_parent.subblock_count() >= 4, "multi-block lazy fixture");
    assert_lazy_adjoint_transforms_match_host(&u1_parent, "U(1) c64 lazy", false);

    // SU(2): the lowered operation recouples, so the transformed adjoint is
    // not a reordering of the adjoint itself.
    let su2 = su2_leg();
    let su2_parent: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2], complex_fill).unwrap();
    assert_lazy_adjoint_transforms_match_host(&su2_parent, "SU(2) c64 lazy", true);

    // fZ2: fermionic signs under a lazy adjoint, real payload.
    let fz2 = fz2_leg();
    let fz2_parent: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&fz2, &fz2], [&fz2], real_fill).unwrap();
    assert_lazy_adjoint_transforms_match_host(&fz2_parent, "fZ2 f64 lazy", true);
}

/// Runs the five transforms on a lazy adjoint, on Host and on device, and
/// asserts they agree.
///
/// The non-vacuity witnesses are counted over the five operations rather than
/// demanded of each: which individual transform of a given fixture moves the
/// flat payload, and which applies a coefficient other than `1`, is a property
/// of that fixture's geometry and provider (fZ2 `repartition`, for instance,
/// is sign free *and* layout preserving on a small fixture). What must hold is
/// that the set as a whole exercises motion, and — for a provider that
/// recouples — a coefficient.
fn assert_lazy_adjoint_transforms_match_host<R, D>(
    parent: &TensorMap<R, D>,
    what: &str,
    recouples: bool,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload + symtn::typed::CudaPayload,
{
    let host_lazy = parent.adjoint().unwrap();
    let device_lazy = parent.to_cuda().unwrap().adjoint().unwrap();

    let rank = host_lazy.rank();
    let split = host_lazy.codomain_rank();
    let mut axes: Vec<usize> = (0..rank).collect();
    axes.rotate_left(1);
    axes.swap(0, 1);
    let (codomain_axes, domain_axes) = axes.split_at(split);
    let levels: Vec<usize> = (0..rank).map(|axis| (axis * 7) % rank + 1).collect();
    let mut cycle: Vec<usize> = (0..split).chain((split..rank).rev()).collect();
    cycle.rotate_left(1);
    let (cyclic_codomain, reversed_domain) = cycle.split_at(split);
    let cyclic_domain: Vec<usize> = reversed_domain.iter().rev().copied().collect();
    let new_split = if split > 1 { split - 1 } else { split + 1 };

    let pairs = [
        (
            "permute",
            host_lazy.permute(codomain_axes, domain_axes).unwrap(),
            device_lazy.permute(codomain_axes, domain_axes).unwrap(),
        ),
        (
            "braid",
            host_lazy
                .braid(codomain_axes, domain_axes, &levels)
                .unwrap(),
            device_lazy
                .braid(codomain_axes, domain_axes, &levels)
                .unwrap(),
        ),
        (
            "repartition",
            host_lazy.repartition(new_split).unwrap(),
            device_lazy.repartition(new_split).unwrap(),
        ),
        (
            "transpose",
            full_transpose!(host_lazy).unwrap(),
            full_transpose!(device_lazy).unwrap(),
        ),
        (
            "cyclic transpose",
            host_lazy
                .transpose(cyclic_codomain, &cyclic_domain)
                .unwrap(),
            device_lazy
                .transpose(cyclic_codomain, &cyclic_domain)
                .unwrap(),
        ),
    ];
    let adjoint_payload = host_lazy
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    let mut moved = 0usize;
    let mut scaled = 0usize;
    for (operation, expected, actual) in pairs {
        let label = format!("{what} {operation}");
        let actual = actual.to_host().unwrap();
        assert_payload_close(
            actual.materialize().unwrap().dense_data().unwrap(),
            expected.materialize().unwrap().dense_data().unwrap(),
            &label,
        );
        assert_eq!(layout(&actual), layout(&expected), "{label}: layout");
        if adjoint_payload
            .iter()
            .zip(actual.materialize().unwrap().dense_data().unwrap())
            .any(|(before, after)| before != after)
        {
            moved += 1;
        }
        if sorted_values(&adjoint_payload)
            != sorted_values(actual.materialize().unwrap().dense_data().unwrap())
        {
            scaled += 1;
        }
    }
    assert!(
        moved >= 1,
        "{what}: no transform moved the adjoint's payload, so none of them          proves anything"
    );
    if recouples {
        assert!(
            scaled >= 1,
            "{what}: no transform applied a coefficient other than 1 or mixed              a recoupling block"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transforms_of_an_empty_tensor_produce_an_empty_tensor() {
    // `required_len == 0`: the output upload, the replay and the download all
    // have to survive a structure with no admissible block at all.
    let runtime = runtime();
    let codomain = u1_leg(&[(1, 2)], false);
    let domain = u1_leg(&[(0, 3)], false);
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&codomain], [&domain], |_, _| 1.0).unwrap();
    assert_eq!(host.subblock_count(), 0);
    assert!(host.materialize().unwrap().dense_data().unwrap().is_empty());
    let device = host.to_cuda().unwrap();

    for (what, actual) in [
        ("permute", device.permute(&[1], &[0]).unwrap()),
        ("braid", device.braid(&[1], &[0], &[1, 2]).unwrap()),
        ("transpose", full_transpose!(device).unwrap()),
        ("repartition", device.repartition(0).unwrap()),
    ] {
        let actual = actual.to_host().unwrap();
        assert!(
            actual
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap()
                .is_empty(),
            "{what}: expected an empty payload"
        );
        assert_eq!(actual.subblock_count(), 0, "{what}: expected no blocks");
    }
    // The lazy-adjoint lowering must survive it too.
    assert!(full_transpose!(device.adjoint().unwrap())
        .unwrap()
        .to_host()
        .unwrap()
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
}

// ---------------------------------------------------------------------------
// `*_into` (issue #1329)
// ---------------------------------------------------------------------------

/// A destination whose every element is NaN, on the model's space. Overwrite
/// mode must clear it — every inactive destination layout included — so a
/// surviving NaN proves a block the replay failed to write.
fn poisoned_like<R, D>(model: &TensorMap<R, D>) -> TensorMap<R, D>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let poisoned = model.scale(D::nan()).unwrap();
    assert!(
        poisoned
            .dense_data()
            .unwrap()
            .iter()
            .all(|value| value.parts().0.is_nan()),
        "the poisoned destination must be all NaN"
    );
    poisoned
}

/// Payload comparison that treats NaN as equal to NaN, which
/// [`assert_payload_close`] cannot: `alpha == 0` over a NaN source, and a
/// destination block the replay left untouched, both have to be compared
/// against the Host's own NaN pattern.
fn assert_payload_matches<D: Payload>(actual: &[D], expected: &[D], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: payload length");
    // Exact equality first: it is the only comparison that works for an
    // infinity, whose difference with itself is NaN. The tolerance arm is
    // then guarded on a finite expectation, because `1e-12 * (1.0 + inf)` is
    // an infinite bound that every non-NaN value would clear.
    let agree = |left: f64, right: f64| {
        left == right
            || (left.is_nan() && right.is_nan())
            || (right.is_finite() && (left - right).abs() <= 1e-12 * (1.0 + right.abs()))
    };
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        let (lr, li) = left.parts();
        let (rr, ri) = right.parts();
        assert!(
            agree(lr, rr) && agree(li, ri),
            "{what}: element {index} is {left:?}, expected {right:?}"
        );
    }
}

/// The positions that hold a NaN, so a NaN *pattern* can be compared rather
/// than merely "some NaN survived".
fn nan_positions<D: Payload>(data: &[D]) -> Vec<usize> {
    data.iter()
        .enumerate()
        .filter(|(_, value)| {
            let (re, im) = value.parts();
            re.is_nan() || im.is_nan()
        })
        .map(|(index, _)| index)
        .collect()
}

/// Runs one `*_into` on the Host and on the device, into two
/// independently built NaN-poisoned destinations on the same space, and
/// asserts the device destination is the Host destination — payload to dtype
/// tolerance (NaN for NaN), spaces and block identities exactly.
///
/// `$body` is written once and expanded against both receiver types, so the
/// Host and device calls cannot drift.
macro_rules! device_overwrite_matches_host {
    ($what:expr, $host:expr, $model:expr, |$t:ident, $d:ident| $body:expr) => {{
        let host_source = &$host;
        let device_source = host_source.to_cuda().unwrap();

        let mut host_destination = poisoned_like(&$model);
        {
            let $t = host_source;
            let $d = &mut host_destination;
            $body
        }
        .unwrap();

        let mut device_destination = poisoned_like(&$model).to_cuda().unwrap();
        {
            let $t = &device_source;
            let $d = &mut device_destination;
            $body
        }
        .unwrap();

        let what: &str = $what;
        let actual = device_destination.to_host().unwrap();
        assert_payload_matches(
            actual.dense_data().unwrap(),
            host_destination.dense_data().unwrap(),
            what,
        );
        assert_eq!(layout(&actual), layout(&host_destination), "{what}: layout");
        actual
    }};
}

/// The real scales the acceptance gate names, `-0.0` included: it compares
/// equal to `0.0` under IEEE, which is exactly the executor's zero test.
const REAL_ALPHAS: [f64; 4] = [1.0, -2.5, 0.0, -0.0];

#[test]
#[ignore = "requires a real CUDA device"]
fn device_into_matches_the_host_for_every_alpha_and_method() {
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let u1_dual = u1_leg(&[(-1, 1), (0, 2), (1, 1)], true);
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1_dual], [&u1, &u1], real_fill).unwrap();
    assert!(real.subblock_count() >= 2, "multi-block fixture");
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1_dual], [&u1, &u1], complex_fill).unwrap();

    let permuted = real.permute(&[2, 0], &[1, 3]).unwrap();
    let transposed = full_transpose!(real).unwrap();
    let cyclic = real.transpose(&[1, 3], &[0, 2]).unwrap();
    let bent = real.repartition(1).unwrap();
    for alpha in REAL_ALPHAS {
        let written = device_overwrite_matches_host!(
            &format!("U1/f64 permute_into alpha={alpha}"),
            real,
            permuted,
            |t, d| t.permute_into(&[2, 0], &[1, 3], d, alpha, Default::default())
        );
        if alpha != 0.0 {
            assert_moved(
                real.dense_data().unwrap(),
                written.dense_data().unwrap(),
                "U1/f64 permute_into",
            );
        }
        device_overwrite_matches_host!(
            &format!("U1/f64 transpose_into alpha={alpha}"),
            real,
            transposed,
            |t, d| full_transpose_into!(t, d, alpha)
        );
        device_overwrite_matches_host!(
            &format!("U1/f64 transpose_into alpha={alpha}"),
            real,
            cyclic,
            |t, d| t.transpose_into(&[1, 3], &[0, 2], d, alpha, Default::default())
        );
        device_overwrite_matches_host!(
            &format!("U1/f64 repartition_into alpha={alpha}"),
            real,
            bent,
            |t, d| t.repartition_into(d, alpha, Default::default())
        );
    }

    // A non-finite caller scale over a finite source. `alpha` reaches the
    // block moves but not the Overwrite zero fills, so the written payload is
    // NaN (or signed infinity) exactly where a block was moved and an exact
    // zero everywhere else — and the device must reproduce the Host's pattern
    // element for element.
    let permuted_model = real.permute(&[2, 0], &[1, 3]).unwrap();
    for alpha in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let written = device_overwrite_matches_host!(
            &format!("U1/f64 non-finite alpha={alpha}"),
            real,
            permuted_model,
            |t, d| t.permute_into(&[2, 0], &[1, 3], d, alpha, Default::default())
        );
        assert!(
            written
                .dense_data()
                .unwrap()
                .iter()
                .any(|value| value.is_nan() || value.is_infinite()),
            "alpha = {alpha} must reach the moved blocks"
        );
    }

    // Complex payload, including a genuinely complex scale.
    let permuted = complex.permute(&[2, 0], &[1, 3]).unwrap();
    let transposed = full_transpose!(complex).unwrap();
    let bent = complex.repartition(3).unwrap();
    let complex_alphas = [
        Complex64::new(1.0, 0.0),
        Complex64::new(-2.5, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(-0.0, -0.0),
        Complex64::new(0.75, -0.25),
    ];
    for alpha in complex_alphas {
        device_overwrite_matches_host!(
            &format!("U1/c64 permute_into alpha={alpha}"),
            complex,
            permuted,
            |t, d| t.permute_into(&[2, 0], &[1, 3], d, alpha, Default::default())
        );
        device_overwrite_matches_host!(
            &format!("U1/c64 transpose_into alpha={alpha}"),
            complex,
            transposed,
            |t, d| full_transpose_into!(t, d, alpha)
        );
        device_overwrite_matches_host!(
            &format!("U1/c64 repartition_into alpha={alpha}"),
            complex,
            bent,
            |t, d| t.repartition_into(d, alpha, Default::default())
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_into_matches_the_host_for_recoupling_and_fermionic_providers() {
    let runtime = runtime();

    // SU(2): Multi blocks, so the caller scale reaches the scatter rather than
    // a Single move, and the coefficients are not 1.
    let su2 = su2_leg();
    let su2_real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2, &su2], real_fill).unwrap();
    let su2_complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2, &su2], complex_fill).unwrap();
    let permuted = su2_real.permute(&[1, 2], &[3, 0]).unwrap();
    let transposed = full_transpose!(su2_real).unwrap();
    let bent = su2_real.repartition(3).unwrap();
    for alpha in REAL_ALPHAS {
        let written = device_overwrite_matches_host!(
            &format!("SU2/f64 permute_into alpha={alpha}"),
            su2_real,
            permuted,
            |t, d| t.permute_into(&[1, 2], &[3, 0], d, alpha, Default::default())
        );
        if alpha != 0.0 {
            assert_not_a_reordering(
                su2_real.dense_data().unwrap(),
                written.dense_data().unwrap(),
                "SU2 *_into",
            );
        }
        device_overwrite_matches_host!(
            &format!("SU2/f64 transpose_into alpha={alpha}"),
            su2_real,
            transposed,
            |t, d| full_transpose_into!(t, d, alpha)
        );
        device_overwrite_matches_host!(
            &format!("SU2/f64 repartition_into alpha={alpha}"),
            su2_real,
            bent,
            |t, d| t.repartition_into(d, alpha, Default::default())
        );
    }
    let su2_permuted_c = su2_complex.permute(&[1, 2], &[3, 0]).unwrap();
    let written = device_overwrite_matches_host!(
        "SU2/c64 permute_into complex alpha",
        su2_complex,
        su2_permuted_c,
        |t, d| t.permute_into(
            &[1, 2],
            &[3, 0],
            d,
            Complex64::new(0.75, -0.25),
            Complex64::new(0.0, 0.0)
        )
    );
    assert_not_a_reordering(
        su2_complex.dense_data().unwrap(),
        written.dense_data().unwrap(),
        "SU2/c64 *_into",
    );

    // fZ2 x U(1): fermionic signs on a charge grading.
    let rule = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let leg = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1),
        ],
    )
    .unwrap();
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], real_fill).unwrap();
    let permuted = host.permute(&[2, 1], &[0, 3]).unwrap();
    let cyclic = host.transpose(&[1, 3], &[0, 2]).unwrap();
    let bent = host.repartition(1).unwrap();
    for alpha in REAL_ALPHAS {
        let written = device_overwrite_matches_host!(
            &format!("fZ2xU1 permute_into alpha={alpha}"),
            host,
            permuted,
            |t, d| t.permute_into(&[2, 1], &[0, 3], d, alpha, Default::default())
        );
        if alpha != 0.0 {
            assert_not_a_reordering(
                host.dense_data().unwrap(),
                written.dense_data().unwrap(),
                "fZ2xU1 *_into",
            );
        }
        device_overwrite_matches_host!(
            &format!("fZ2xU1 transpose_into alpha={alpha}"),
            host,
            cyclic,
            |t, d| t.transpose_into(&[1, 3], &[0, 2], d, alpha, Default::default())
        );
        device_overwrite_matches_host!(
            &format!("fZ2xU1 repartition_into alpha={alpha}"),
            host,
            bent,
            |t, d| t.repartition_into(d, alpha, Default::default())
        );
    }

    // fZ2 (x) SU(2): fermionic signs *and* recoupling, complex payload.
    let rule = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
            .unwrap(),
    );
    let leg = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                2,
            ),
            (
                product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(1)),
                1,
            ),
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                1,
            ),
        ],
    )
    .unwrap();
    let host: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], complex_fill).unwrap();
    let permuted = host.permute(&[1, 2], &[3, 0]).unwrap();
    let transposed = full_transpose!(host).unwrap();
    let bent = host.repartition(3).unwrap();
    for alpha in [
        Complex64::new(1.0, 0.0),
        Complex64::new(-2.5, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(-0.0, -0.0),
        Complex64::new(0.75, -0.25),
    ] {
        let written = device_overwrite_matches_host!(
            &format!("fZ2xSU2 permute_into alpha={alpha}"),
            host,
            permuted,
            |t, d| t.permute_into(&[1, 2], &[3, 0], d, alpha, Default::default())
        );
        if alpha != Complex64::new(0.0, 0.0) {
            assert_not_a_reordering(
                host.dense_data().unwrap(),
                written.dense_data().unwrap(),
                "fZ2xSU2 *_into",
            );
        }
        device_overwrite_matches_host!(
            &format!("fZ2xSU2 transpose_into alpha={alpha}"),
            host,
            transposed,
            |t, d| full_transpose_into!(t, d, alpha)
        );
        device_overwrite_matches_host!(
            &format!("fZ2xSU2 repartition_into alpha={alpha}"),
            host,
            bent,
            |t, d| t.repartition_into(d, alpha, Default::default())
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_into_matches_the_hosts_nan_pattern_for_every_alpha() {
    // At `alpha == 0` device and Host follow VectorInterface's
    // `scale(x, 0) = zero(x) * 0` (#1438): zeros whatever the source holds,
    // as TensorKit's `permute!(tdst, tsrc, p, 0, 0)`. Every other scale
    // multiplies, so a NaN or infinite source poisons the destination
    // exactly where the Host's does.
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let poisoned_source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1], [&u1, &u1], |trees, idx| {
            if idx.iter().sum::<usize>() % 3 == 0 {
                f64::NAN
            } else if idx[0] == 1 {
                f64::INFINITY
            } else {
                real_fill(trees, idx)
            }
        })
        .unwrap();
    assert!(poisoned_source
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.is_nan()));
    assert!(poisoned_source
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.is_infinite()));
    let model = poisoned_source.permute(&[1, 2], &[3, 0]).unwrap();

    // A NaN and an infinite caller scale belong here too: the contract is
    // "alpha equals the Host", not "alpha is finite". `NaN * x` and `inf * 0`
    // are both NaN, so these also widen the NaN set the comparison has to
    // reproduce.
    for alpha in [0.0, -0.0, 1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let written = device_overwrite_matches_host!(
            &format!("NaN source *_into alpha={alpha}"),
            poisoned_source,
            model,
            |t, d| t.permute_into(&[1, 2], &[3, 0], d, alpha, Default::default())
        );
        // The exact NaN *set*, not merely "some NaN survived": `any` would
        // pass if only one of the poisoned blocks propagated. `device ==
        // Host` element by element is already asserted by the macro; this
        // adds that the set is non-empty, so the case cannot pass vacuously.
        let mut host_expected = poisoned_like(&model);
        poisoned_source
            .permute_into(
                &[1, 2],
                &[3, 0],
                &mut host_expected,
                alpha,
                Default::default(),
            )
            .unwrap();
        let expected_nans = nan_positions(host_expected.dense_data().unwrap());
        if alpha == 0.0 {
            assert!(
                written
                    .dense_data()
                    .unwrap()
                    .iter()
                    .all(|value| *value == 0.0),
                "alpha = {alpha}: a zero scale writes zeros"
            );
            continue;
        }
        assert!(
            !expected_nans.is_empty(),
            "alpha = {alpha}: the fixture must propagate at least one NaN"
        );
        assert_eq!(
            nan_positions(written.dense_data().unwrap()),
            expected_nans,
            "alpha = {alpha}: the NaN set must be the Host's exactly"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_into_has_no_identity_short_circuit() {
    // Host `overwrite_tree_transform` has none: an identity axis list still
    // writes `alpha * self` into the caller's destination. A device clone
    // short circuit would silently leave the destination poisoned.
    let runtime = runtime();
    let u1 = u1_leg(&[(-1, 2), (0, 1), (1, 2)], false);
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    let written = device_overwrite_matches_host!("identity permute_into", host, host, |t, d| t
        .permute_into(&[0, 1], &[2, 3], d, -2.5, 0.0));
    assert_payload_close(
        written.dense_data().unwrap(),
        host.scale(-2.5).unwrap().dense_data().unwrap(),
        "identity permute_into writes alpha * self",
    );

    // Same split for `repartition`, and a rank-0 `transpose`: the returning
    // device methods clone there, the overwriting ones must still write.
    let written =
        device_overwrite_matches_host!("same-split repartition_into", host, host, |t, d| t
            .repartition_into(d, 2.0, 0.0));
    assert_payload_close(
        written.dense_data().unwrap(),
        host.scale(2.0).unwrap().dense_data().unwrap(),
        "same-split repartition_into writes alpha * self",
    );

    let square: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1], [&u1], real_fill).unwrap();
    let scalar = square.trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(scalar.rank(), 0);
    let written = device_overwrite_matches_host!(
        "rank-0 transpose_into",
        scalar,
        scalar,
        |t, d| full_transpose_into!(t, d, -0.5)
    );
    assert_payload_close(
        written.dense_data().unwrap(),
        scalar.scale(-0.5).unwrap().dense_data().unwrap(),
        "rank-0 transpose_into writes alpha * self",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_into_of_an_empty_tensor_succeeds() {
    let runtime = runtime();
    let codomain = u1_leg(&[(1, 2)], false);
    let domain = u1_leg(&[(0, 3)], false);
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&codomain], [&domain], |_, _| 1.0).unwrap();
    assert_eq!(host.subblock_count(), 0);
    let model = host.permute(&[1], &[0]).unwrap();
    let source = host.to_cuda().unwrap();
    let mut destination = model.to_cuda().unwrap();
    source
        .permute_into(&[1], &[0], &mut destination, 2.0, 0.0)
        .unwrap();
    assert!(destination
        .to_host()
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_into_reads_a_lazy_source_through_its_parent_like_the_host() {
    // #1870: a lazy-adjoint source of the device `*_into` is read
    // through its parent's device payload (storage-mapped structure, storage
    // conjugation per task), as on Host. Each call is compared against the
    // Host `*_into` of the same lazy source and against the reference steps
    // `alpha * eager(materialize(source)) + beta * dst0`.
    let runtime = runtime();
    let a = u1_leg(&[(-1, 2), (0, 1), (1, 3)], false);
    let b = u1_leg(&[(0, 2), (1, 1)], true);
    let c = u1_leg(&[(-1, 1), (1, 2)], false);
    let d = u1_leg(&[(-2, 2), (-1, 3), (0, 1), (1, 2), (2, 1)], false);
    let u1: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&a, &b, &c], [&d], complex_fill).unwrap();
    let su2 = su2_leg();
    let su2_parent: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2], complex_fill).unwrap();
    let fz2 = fz2_leg();
    let fz2_parent: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&fz2, &fz2], [&fz2], real_fill).unwrap();
    for (alpha, beta) in [
        (Complex64::new(0.75, -0.25), Complex64::new(0.0, 0.0)),
        (Complex64::new(-2.5, 0.0), Complex64::new(0.5, 0.25)),
    ] {
        assert_lazy_into_matches_host(&u1, alpha, beta, "U(1) c64 lazy into");
        assert_lazy_into_matches_host(&su2_parent, alpha, beta, "SU(2) c64 lazy into");
    }
    for (alpha, beta) in [(1.0, 0.0), (-2.5, 0.5), (0.5, 1.0)] {
        assert_lazy_into_matches_host(&fz2_parent, alpha, beta, "fZ2 f64 lazy into");
    }
}

/// Runs permute/braid/transpose/repartition `*_into` of `parent.adjoint()`
/// on Host and on device into the same `dst0` (NaN for `beta = 0`, which
/// must never be read), and asserts device == Host == the reference steps.
fn assert_lazy_into_matches_host<R, D>(parent: &TensorMap<R, D>, alpha: D, beta: D, what: &str)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload + symtn::typed::CudaPayload,
{
    let host_lazy = parent.adjoint().unwrap();
    let device_lazy = parent.to_cuda().unwrap().adjoint().unwrap();
    let owned = host_lazy.materialize().unwrap();
    let rank = host_lazy.rank();
    let split = host_lazy.codomain_rank();
    let mut axes: Vec<usize> = (0..rank).collect();
    axes.rotate_left(1);
    axes.swap(0, 1);
    let (codomain_axes, domain_axes) = axes.split_at(split);
    let levels: Vec<usize> = (0..rank).map(|axis| (axis * 7) % rank + 1).collect();
    let (full_codomain, full_domain) = full_rotation!(host_lazy);
    let new_split = if split > 1 { split - 1 } else { split + 1 };

    macro_rules! case {
        ($operation:expr, |$t:ident| $eager:expr, |$s:ident, $d:ident| $into:expr) => {{
            let label = format!("{what} {} alpha={alpha:?} beta={beta:?}", $operation);
            let oracle_eager = {
                let $t = &owned;
                $eager
            }
            .unwrap();
            let model = oracle_eager.materialize().unwrap();
            // Built afresh per use: a clone would share the payload and trip
            // the unique-ownership check.
            let dst0 = || {
                if beta == D::zero() {
                    poisoned_like(&model)
                } else {
                    model.scale(D::one() + D::one()).unwrap()
                }
            };
            let mut host_destination = dst0();
            {
                let $s = &host_lazy;
                let $d = &mut host_destination;
                $into
            }
            .unwrap();
            let mut device_destination = dst0().to_cuda().unwrap();
            {
                let $s = &device_lazy;
                let $d = &mut device_destination;
                $into
            }
            .unwrap();
            let actual = device_destination.to_host().unwrap();
            assert_payload_matches(
                actual.dense_data().unwrap(),
                host_destination.dense_data().unwrap(),
                &label,
            );
            assert_eq!(layout(&actual), layout(&model), "{label}: layout");
            let expected: Vec<D> = model
                .dense_data()
                .unwrap()
                .iter()
                .zip(dst0().dense_data().unwrap())
                .map(|(&value, &old)| {
                    if beta == D::zero() {
                        alpha * value
                    } else {
                        alpha * value + beta * old
                    }
                })
                .collect();
            assert_payload_close(actual.dense_data().unwrap(), &expected, &label);
        }};
    }

    case!(
        "permute",
        |t| t.permute(codomain_axes, domain_axes),
        |s, d| s.permute_into(codomain_axes, domain_axes, d, alpha, beta)
    );
    case!(
        "braid",
        |t| t.braid(codomain_axes, domain_axes, &levels),
        |s, d| s.braid_into(codomain_axes, domain_axes, &levels, d, alpha, beta)
    );
    case!(
        "transpose",
        |t| t.transpose(&full_codomain, &full_domain),
        |s, d| s.transpose_into(&full_codomain, &full_domain, d, alpha, beta)
    );
    case!("repartition", |t| t.repartition(new_split), |s, d| s
        .repartition_into(d, alpha, beta));
}
