//! Compact LQ and checked-Generic polar write their factors straight into the
//! output storage (#1478): LQ appends each sector's adjoint instead of
//! zero-filling first, and polar runs its GEMMs into the output regions
//! instead of per-sector temporaries that are then copied.

use crate::backend;
use crate::numerics;

use std::hint::black_box;
use std::sync::Arc;
#[cfg(feature = "cgc-gen")]
use symtn::typed::HermitianTol;

use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{Complex64, Runtime};
use symtn::typed::{GradedSpace, Lq, TensorMap, TensorScalar};

use crate::fixtures;

use fixtures::{codomain_axes, domain_axes};

use crate::counting_alloc;

#[derive(Debug)]
#[cfg_attr(
    not(feature = "cgc-gen"),
    expect(
        dead_code,
        reason = "only the checked-Generic contracts read total bytes"
    )
)]
struct Counts {
    bytes: usize,
    zeroed_bytes: usize,
}

fn measured<T>(operation: impl FnOnce() -> T) -> (T, Counts) {
    let (value, allocs) = counting_alloc::measure(operation);
    let counts = Counts {
        bytes: allocs.bytes as usize,
        zeroed_bytes: allocs.zeroed_bytes as usize,
    };
    (value, counts)
}

trait Scalar: TensorScalar + numerics::Numeric {
    const ZERO: Self;
    const ONE: Self;
    const MINUS_ONE: Self;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MINUS_ONE: Self = -1.0;
}

impl Scalar for Complex64 {
    const ZERO: Self = Complex64::new(0.0, 0.0);
    const ONE: Self = Complex64::new(1.0, 0.0);
    const MINUS_ONE: Self = Complex64::new(-1.0, 0.0);
}

/// An owned copy of a (possibly lazy) adjoint: checked-Generic `compose`
/// accepts owned operands only.
macro_rules! owned_adjoint {
    ($engine:ident, $d:ty, $t:expr) => {{
        let t = &$t;
        let adjoint = t.adjoint($engine).unwrap();
        adjoint
            .axpby($engine, <$d as Scalar>::ONE, &adjoint, <$d as Scalar>::ZERO)
            .unwrap()
    }};
}

/// `‖actual − expected‖ ≤ tolerance(terms, ‖expected‖)`; `terms` bounds the
/// floating terms of one entry by the source payload length, and the
/// conditioning factor is one (backward-stable factorizations of random
/// fixtures).
macro_rules! assert_residual {
    ($engine:ident, $what:expr, $d:ty, $actual:expr, $expected:expr, $terms:expr) => {{
        let residual = $actual
            .axpby(
                $engine,
                <$d as Scalar>::ONE,
                $expected,
                <$d as Scalar>::MINUS_ONE,
            )
            .unwrap()
            .norm($engine, 2.0)
            .unwrap();
        let bound = numerics::tolerance::<$d>($terms, $expected.norm($engine, 2.0).unwrap());
        assert!(
            residual <= bound,
            "{}: residual {residual:e} exceeds {bound:e}",
            $what
        );
    }};
}

/// `A = L Q` and `A Qᴴ = L`, which with a full-rank `L` is `Q Qᴴ = 1`
/// (independent reconstruction/orthogonality oracle).
macro_rules! assert_lq {
    ($engine:ident, $d:ty, $a:expr) => {{
        let a = $a;
        let Lq { l, q } = a
            .lq_compact($engine, &codomain_axes(&a), &domain_axes(&a))
            .unwrap();
        let terms = a.dense_data().unwrap().len();
        assert_residual!(
            $engine,
            "A = L Q",
            $d,
            &l.compose($engine, &q).unwrap(),
            &a,
            terms
        );
        let aqh = {
            let adjoint = owned_adjoint!($engine, $d, q);
            a.compose($engine, &adjoint)
        }
        .unwrap();
        assert_residual!($engine, "A Qᴴ = L", $d, &aqh, &l, terms);
    }};
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
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

macro_rules! lq_cases {
    ($engine:ident, $d:ty, $runtime:expr) => {{
        let u1 = u1_leg();
        let a: TensorMap<_, $d> =
            TensorMap::rand_with_seed($engine, $runtime, [&u1], [&u1, &u1], 1478).unwrap();
        assert_lq!($engine, $d, a);
        let su2 = su2_leg();
        let a: TensorMap<_, $d> =
            TensorMap::rand_with_seed($engine, $runtime, [&su2], [&su2, &su2], 1479).unwrap();
        assert_lq!($engine, $d, a);
    }};
}

#[test]
fn multiplicity_free_compact_lq_matches_reconstruction_oracle() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    lq_cases!(engine, f64, &runtime);
    lq_cases!(engine, Complex64, &runtime);
}

// Byte-traffic contract: symtn writes the compact LQ outputs and the adjoint
// scratch once, so it requests no zeroed storage for them. The only zeroed
// request left is the dense backend's R factor (Tenferro
// `faer_linalg::upper_triangle_vec_from_mat`), whose total over sectors is
// the payload of L: 4872 zeroed bytes for this fixture. The counter observes
// `alloc_zeroed` (e.g. `vec![0.0; len]`) for f64 (#1478).
// The zeroed R factor is faer's `upper_triangle_vec_from_mat`; a BLAS provider
// zero-fills its own workspaces (#2107).
#[test]
fn compact_lq_requests_no_zeroed_output_storage() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    if !backend::faer_is_default_backend() {
        return;
    }
    let runtime = Runtime::builder().build().unwrap();
    let leg = u1_leg();
    let a: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg, &leg], [&leg, &leg], 1480).unwrap();
    let warm = a.lq_compact(engine, &[0, 1], &[2, 3]).unwrap();
    let (Lq { l, q }, counts) = measured(|| a.lq_compact(engine, &[0, 1], &[2, 3]).unwrap());
    black_box((&warm, &q));
    let backend_r_bytes = std::mem::size_of_val(l.dense_data().unwrap());
    assert!(
        counts.zeroed_bytes <= backend_r_bytes,
        "{counts:?}, L bytes {backend_r_bytes}"
    );
}

#[cfg(feature = "cgc-gen")]
mod checked_generic {
    use super::*;
    use symtn::sector::LieGroupFusionRule;
    use symtn::typed::{LeftPolar, RightPolar};

    fn su3_leg(
        provider: &Arc<LieGroupFusionRule>,
        irreps: &[(Vec<i64>, usize)],
    ) -> GradedSpace<LieGroupFusionRule> {
        GradedSpace::try_new(Arc::clone(provider), irreps.to_vec()).unwrap()
    }

    /// Four coupled sectors, rows >= columns in each; every degeneracy is
    /// multiplied by `scale`, which leaves the tree structure unchanged.
    fn tall_legs(
        provider: &Arc<LieGroupFusionRule>,
        scale: usize,
    ) -> (
        GradedSpace<LieGroupFusionRule>,
        GradedSpace<LieGroupFusionRule>,
    ) {
        let irreps = [vec![0i64, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
        let rows = irreps
            .iter()
            .cloned()
            .zip([1, 2, 3, 2].map(|d| d * scale))
            .collect::<Vec<_>>();
        let cols = irreps
            .iter()
            .cloned()
            .zip([1, 1, 2, 2].map(|d| d * scale))
            .collect::<Vec<_>>();
        (su3_leg(provider, &rows), su3_leg(provider, &cols))
    }

    fn payload_bytes(tensor: &TensorMap<LieGroupFusionRule, f64>) -> usize {
        std::mem::size_of_val(tensor.dense_data().unwrap())
    }

    /// Left: `A = W P`, right: `A = P W`, with `W` an isometry (`Wᴴ W = 1`
    /// left, `W Wᴴ = 1` right: every eigenvalue of the Gram matrix is one)
    /// and `P` Hermitian positive semidefinite (eigenvalues `>= -tol`). This
    /// excludes trivial splittings such as `W = A`, `P = 1`.
    macro_rules! assert_polar {
        ($engine:ident, $d:ty, $a:expr, $left:expr) => {{
            let a = $a;
            let terms = a.dense_data().unwrap().len();
            let (w, p, product, gram) = if $left {
                let LeftPolar { w, p } = a
                    .left_polar($engine, &codomain_axes(&a), &domain_axes(&a))
                    .unwrap();
                let product = w.compose($engine, &p).unwrap();
                let gram = owned_adjoint!($engine, $d, w).compose($engine, &w).unwrap();
                (w, p, product, gram)
            } else {
                let RightPolar { p, wh: w } = a
                    .right_polar($engine, &codomain_axes(&a), &domain_axes(&a))
                    .unwrap();
                let product = p.compose($engine, &w).unwrap();
                let gram = {
                    let adjoint = owned_adjoint!($engine, $d, w);
                    w.compose($engine, &adjoint)
                }
                .unwrap();
                (w, p, product, gram)
            };
            assert!(!w.dense_data().unwrap().is_empty());
            assert_residual!($engine, "A = polar product", $d, &product, a, terms);
            let p_adjoint = owned_adjoint!($engine, $d, p);
            assert_residual!($engine, "P = Pᴴ", $d, &p_adjoint, &p, terms);
            let one = numerics::tolerance::<$d>(terms, 1.0);
            for spectrum in gram
                .eigh_vals($engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap()
            {
                for value in spectrum.values {
                    assert!(
                        (value - 1.0).abs() <= one,
                        "W is not an isometry: Gram eigenvalue {value:e}"
                    );
                }
            }
            let psd = numerics::tolerance::<$d>(terms, p.norm($engine, 2.0).unwrap());
            let mut p_values = 0;
            for spectrum in p
                .eigh_vals($engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap()
            {
                for value in spectrum.values {
                    p_values += 1;
                    assert!(value >= -psd, "P is not PSD: eigenvalue {value:e}");
                }
            }
            assert!(p_values > 0);
        }};
    }

    macro_rules! cases {
        ($engine:ident, $d:ty, $runtime:expr) => {{
            let provider = Arc::new(
                LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
            );
            let (rows, cols) = tall_legs(&provider, 1);
            let tall: TensorMap<_, $d> =
                TensorMap::rand_with_seed($engine, $runtime, [&rows], [&cols], 1481).unwrap();
            assert_polar!($engine, $d, &tall, true);
            let wide: TensorMap<_, $d> =
                TensorMap::rand_with_seed($engine, $runtime, [&cols], [&rows], 1482).unwrap();
            assert_polar!($engine, $d, &wide, false);
            assert_lq!($engine, $d, wide);
            // Vertex multiplicity: 8 ⊗ 8 contains 8 twice.
            let adjoint = su3_leg(&provider, &[(vec![1, 1], 2)]);
            let single = su3_leg(&provider, &[(vec![1, 1], 1)]);
            let tall: TensorMap<_, $d> =
                TensorMap::rand_with_seed($engine, $runtime, [&adjoint, &adjoint], [&single], 1483)
                    .unwrap();
            assert_eq!(tall.subblock_count(), 2);
            assert_polar!($engine, $d, &tall, true);
            let wide: TensorMap<_, $d> =
                TensorMap::rand_with_seed($engine, $runtime, [&single], [&adjoint, &adjoint], 1484)
                    .unwrap();
            assert_polar!($engine, $d, &wide, false);
            assert_lq!($engine, $d, wide);
        }};
    }

    #[test]
    fn checked_generic_polar_and_lq_match_reconstruction_oracles() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let _serial = crate::counting_alloc::serial();
        let runtime = Runtime::builder().build().unwrap();
        cases!(engine, f64, &runtime);
        cases!(engine, Complex64, &runtime);
    }

    // The polar publication contract (#1478, no per-sector W, P or scaled
    // factor temporary) is observed on the GEMM views in the lib test
    // `checked_generic_polar_writes_its_factors_in_place`, which can inject a
    // recording dense executor.

    // Byte contract (#1494): the leg-degeneracy (facade) layout of a rank-3
    // and a rank-4 multiplicity map is a coupled-sector tiling, so QR, SVD,
    // LQ and `svd_vals` read the payload in place. Before, `FusionTreeKey`
    // Ord admission rejected the facade tree order and each call first packed
    // a copy of the whole payload. The control is the rank-2 map between the
    // fused legs: it has the same coupled blocks, hence the same dense kernel
    // work and output sizes, and was always read in place. The facade call
    // still allocates more tree metadata than the control, but scaling every
    // degeneracy leaves that metadata fixed, so the facade's excess over the
    // control must not grow with the payload; a packed copy grows it by the
    // whole payload growth.
    #[test]
    fn checked_generic_facade_factorizations_read_the_payload_in_place() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let _serial = crate::counting_alloc::serial();
        type Tensor = TensorMap<LieGroupFusionRule, f64>;
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        );
        let mut excess = |scale: usize, rank4: bool| {
            let leg = su3_leg(&provider, &[(vec![1, 1], 2 * scale), (vec![0, 0], scale)]);
            let fused = leg.fuse(&leg).unwrap();
            let (facade, control): (Tensor, Tensor) = if rank4 {
                (
                    TensorMap::rand_with_seed(engine, &runtime, [&leg, &leg], [&leg, &leg], 1495)
                        .unwrap(),
                    TensorMap::rand_with_seed(engine, &runtime, [&fused], [&fused], 1495).unwrap(),
                )
            } else {
                (
                    TensorMap::rand_with_seed(engine, &runtime, [&leg, &leg], [&leg], 1494)
                        .unwrap(),
                    TensorMap::rand_with_seed(engine, &runtime, [&fused], [&leg], 1494).unwrap(),
                )
            };
            assert_eq!(payload_bytes(&facade), payload_bytes(&control));
            let mut counts = |a: &Tensor| {
                black_box((
                    a.qr_compact(engine, &codomain_axes(a), &domain_axes(a))
                        .unwrap(),
                    a.svd_compact(engine, &codomain_axes(a), &domain_axes(a))
                        .unwrap(),
                ));
                black_box((
                    a.lq_compact(engine, &codomain_axes(a), &domain_axes(a))
                        .unwrap(),
                    a.svd_vals(engine, &codomain_axes(a), &domain_axes(a))
                        .unwrap(),
                ));
                [
                    measured(|| {
                        black_box(
                            a.qr_compact(engine, &codomain_axes(a), &domain_axes(a))
                                .unwrap(),
                        )
                    })
                    .1,
                    measured(|| {
                        black_box(
                            a.svd_compact(engine, &codomain_axes(a), &domain_axes(a))
                                .unwrap(),
                        )
                    })
                    .1,
                    measured(|| {
                        black_box(
                            a.lq_compact(engine, &codomain_axes(a), &domain_axes(a))
                                .unwrap(),
                        )
                    })
                    .1,
                    measured(|| {
                        black_box(
                            a.svd_vals(engine, &codomain_axes(a), &domain_axes(a))
                                .unwrap(),
                        )
                    })
                    .1,
                ]
            };
            let (facade_counts, control_counts) = (counts(&facade), counts(&control));
            assert_eq!(facade_counts[3].zeroed_bytes, 0, "{:?}", facade_counts[3]);
            let excess = std::array::from_fn::<isize, 4, _>(|op| {
                facade_counts[op].bytes as isize - control_counts[op].bytes as isize
            });
            (payload_bytes(&facade), excess)
        };
        for rank4 in [false, true] {
            let (small_payload, small) = excess(1, rank4);
            let (large_payload, large) = excess(2, rank4);
            let payload_growth = (large_payload - small_payload) as isize;
            for (op, name) in ["qr_compact", "svd_compact", "lq_compact", "svd_vals"]
                .into_iter()
                .enumerate()
            {
                assert!(
                    large[op] - small[op] < payload_growth,
                    "rank4 {rank4} {name}: facade excess {} -> {}, payload growth {payload_growth}",
                    small[op],
                    large[op]
                );
            }
        }
    }
}
