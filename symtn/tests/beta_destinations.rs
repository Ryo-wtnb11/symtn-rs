//! `dst <- alpha * op(inputs) + beta * dst` for every linear `*_into`.
//!
//! The oracle never touches a destination path: it is the eager operation on
//! the same inputs combined with the destination's prior value through the
//! eager returning `axpby` on fresh tensors (TensorKit `tensorcontract!`,
//! `permute!`, `tensortrace!`, `add!` with `α, β`). Covered: beta in
//! {0, 1, general} and general alpha, a NaN destination under beta = 0,
//! coupled sectors a contraction cannot reach (on the core route, left
//! bitwise untouched under beta = 1 — the old full zero-fill would clear
//! them), U(1), SU(2) and
//! fZ2×U(1) with dual legs, f64/c64/f32/c32, lazy-adjoint inputs, the
//! shared-destination error, and (with `--features cuda -- --ignored`) the
//! device against the Host.

use std::sync::Arc;
use symtn::typed::Engine;

use num_complex::{Complex32, Complex64};
use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{ContractSpec, Error, GradedSpace, Runtime, TensorMap, TensorScalar};

use crate::numerics;

trait Scalar: TensorScalar + numerics::Numeric + PartialEq {
    fn alpha() -> Self;
    fn beta() -> Self;
    fn real(value: f64) -> Self;
    fn bits(self) -> (u64, u64);
}

macro_rules! real_scalar {
    ($t:ty) => {
        impl Scalar for $t {
            fn alpha() -> Self {
                -0.75
            }
            fn beta() -> Self {
                1.5
            }
            fn real(value: f64) -> Self {
                value as $t
            }
            fn bits(self) -> (u64, u64) {
                (self.to_bits().into(), 0)
            }
        }
    };
}
real_scalar!(f64);
real_scalar!(f32);

macro_rules! complex_scalar {
    ($t:ty, $r:ty) => {
        impl Scalar for $t {
            fn alpha() -> Self {
                <$t>::new(0.5, -1.25)
            }
            fn beta() -> Self {
                <$t>::new(-1.5, 0.25)
            }
            fn real(value: f64) -> Self {
                <$t>::new(value as $r, 0.0)
            }
            fn bits(self) -> (u64, u64) {
                (self.re.to_bits().into(), self.im.to_bits().into())
            }
        }
    };
}
complex_scalar!(Complex64, f64);
complex_scalar!(Complex32, f32);

/// Coefficient pairs: beta = 0, -0.0, 1 and general, each with a general
/// alpha, the unit-alpha overwrite every retained-destination caller uses,
/// and a zero alpha with a general beta (`dst <- beta * dst`, the source not
/// read).
fn coefficients<D: Scalar>() -> [(D, D); 6] {
    [
        (D::real(1.0), D::real(0.0)),
        (D::alpha(), D::real(0.0)),
        (D::alpha(), D::real(-0.0)),
        (D::alpha(), D::real(1.0)),
        (D::alpha(), D::beta()),
        (D::real(0.0), D::beta()),
    ]
}

macro_rules! check_into {
    ($engine:ident, $what:expr, $expected:expr, $before:expr, $alpha:expr, $beta:expr, $run:expr, $terms:expr) => {{
        let expected = $expected
            .axpby($engine, $alpha, &$before, $beta)
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec();
        let mut destination = $before.materialize($engine).unwrap();
        #[allow(clippy::redundant_closure_call)]
        ($run)(&mut destination).unwrap();
        numerics::assert_slices_close($what, destination.dense_data().unwrap(), &expected, $terms);
    }};
}

/// One symmetry at one dtype: every `*_into` against its eager oracle.
macro_rules! suite {
    ($engine:ident, $rt:expr, $v:expr, $bond:expr, $d:ty, $label:expr) => {{
        let rt = $rt;
        let v = $v;
        let w = v.try_dual().unwrap();
        let bond = $bond;
        type D = $d;
        let rand = |$engine: &mut dyn Engine,
                    codomain: &[&GradedSpace<_>],
                    domain: &[&GradedSpace<_>],
                    seed: u64| {
            TensorMap::<_, D>::rand_with_seed(
                $engine,
                rt,
                codomain.iter().copied(),
                domain.iter().copied(),
                seed,
            )
            .unwrap()
        };

        // Tree transforms on a rank-4 map with dual legs.
        let t = rand($engine, &[&v, &w], &[&v, &w], 1);
        let t2 = rand($engine, &[&v, &w], &[&v, &w], 2);
        for (alpha, beta) in coefficients::<D>() {
            let what = |op: &str| format!("{} {op} alpha={alpha:?} beta={beta:?}", $label);
            let before = t2.permute($engine, &[2, 0], &[1, 3]).unwrap();
            check_into!(
                $engine,
                &what("permute_into"),
                t.permute($engine, &[2, 0], &[1, 3]).unwrap(),
                before,
                alpha,
                beta,
                |d: &mut TensorMap<_, D>| t.permute_into($engine, &[2, 0], &[1, 3], d, alpha, beta),
                8
            );
            let levels = [0, 1, 2, 3];
            let before = t2.braid($engine, &[1, 0], &[3, 2], &levels).unwrap();
            check_into!(
                $engine,
                &what("braid_into"),
                t.braid($engine, &[1, 0], &[3, 2], &levels).unwrap(),
                before,
                alpha,
                beta,
                |d: &mut TensorMap<_, D>| t.braid_into(
                    $engine,
                    &[1, 0],
                    &[3, 2],
                    &levels,
                    d,
                    alpha,
                    beta
                ),
                8
            );
            let before = t2.transpose($engine, &[1, 3], &[0, 2]).unwrap();
            check_into!(
                $engine,
                &what("transpose_into"),
                t.transpose($engine, &[1, 3], &[0, 2]).unwrap(),
                before,
                alpha,
                beta,
                |d: &mut TensorMap<_, D>| t.transpose_into(
                    $engine,
                    &[1, 3],
                    &[0, 2],
                    d,
                    alpha,
                    beta
                ),
                8
            );
            let before = t2.repartition($engine, 1).unwrap();
            check_into!(
                $engine,
                &what("repartition_into"),
                t.repartition($engine, 1).unwrap(),
                before,
                alpha,
                beta,
                |d: &mut TensorMap<_, D>| t.repartition_into($engine, d, alpha, beta),
                8
            );

            // Trace, owned and lazy adjoint.
            let pairs = [(0, 2)];
            let before = t2.trace_pairs($engine, &pairs).unwrap();
            check_into!(
                $engine,
                &what("trace_pairs_into"),
                t.trace_pairs($engine, &pairs).unwrap(),
                before,
                alpha,
                beta,
                |d: &mut TensorMap<_, D>| t.trace_pairs_into($engine, &pairs, d, alpha, beta),
                16
            );
            let lazy = t.adjoint($engine).unwrap();
            let before = t2
                .adjoint($engine)
                .unwrap()
                .trace_pairs($engine, &pairs)
                .unwrap();
            check_into!(
                $engine,
                &what("lazy trace_pairs_into"),
                lazy.trace_pairs($engine, &pairs).unwrap(),
                before,
                alpha,
                beta,
                |d: &mut TensorMap<_, D>| lazy.trace_pairs_into($engine, &pairs, d, alpha, beta),
                16
            );

            // axpby, owned and lazy adjoint.
            check_into!(
                $engine,
                &what("axpby_into"),
                t.materialize($engine).unwrap(),
                t2,
                alpha,
                beta,
                |d: &mut TensorMap<_, D>| t.axpby_into($engine, d, alpha, beta),
                2
            );
            let before = t2.adjoint($engine).unwrap().materialize($engine).unwrap();
            check_into!(
                $engine,
                &what("lazy axpby_into"),
                lazy.materialize($engine).unwrap(),
                before,
                alpha,
                beta,
                |d: &mut TensorMap<_, D>| lazy.axpby_into($engine, d, alpha, beta),
                2
            );

            // Contraction with an output permutation and split, and with a
            // lazy-adjoint left operand.
            let rhs = rand($engine, &[&v, &w], &[&v], 3);
            let spec = ContractSpec {
                lhs: &[2, 3],
                rhs: &[0, 1],
                codomain: &[2, 0],
                domain: &[1],
            };
            let before = t2
                .contract($engine, &rhs, &spec)
                .unwrap()
                .scale($engine, D::real(0.5))
                .unwrap();
            check_into!(
                $engine,
                &what("contract_into"),
                t.contract($engine, &rhs, &spec).unwrap(),
                before,
                alpha,
                beta,
                |d: &mut TensorMap<_, D>| t.contract_into($engine, &rhs, &spec, d, alpha, beta),
                32
            );
            let before = lazy
                .contract($engine, &rhs, &spec)
                .unwrap()
                .scale($engine, D::real(0.5))
                .unwrap();
            check_into!(
                $engine,
                &what("lazy contract_into"),
                lazy.contract($engine, &rhs, &spec).unwrap(),
                before,
                alpha,
                beta,
                |d: &mut TensorMap<_, D>| lazy.contract_into($engine, &rhs, &spec, d, alpha, beta),
                32
            );
        }

        // Partial block coverage: the bond carries only some sectors, so the
        // coupled sectors it misses are never written by a GEMM.
        let lhs = rand($engine, &[&v], &[&bond], 4);
        let rhs = rand($engine, &[&bond], &[&v], 5);
        let spec = ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        };
        let eager = lhs.contract($engine, &rhs, &spec).unwrap();
        let prior = rand($engine, &[&v], &[&v], 6);
        let unreached: Vec<usize> = eager
            .dense_data()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == D::real(0.0))
            .map(|(index, _)| index)
            .collect();
        assert!(
            !unreached.is_empty(),
            "{}: the fixture must miss a sector",
            $label
        );
        for (alpha, beta) in coefficients::<D>() {
            let what = format!(
                "{} partial contract_into alpha={alpha:?} beta={beta:?}",
                $label
            );
            check_into!(
                $engine,
                &what,
                eager,
                prior,
                alpha,
                beta,
                |d: &mut TensorMap<_, D>| lhs.contract_into($engine, &rhs, &spec, d, alpha, beta),
                8
            );
        }
        // beta = 1 leaves the unreached blocks bit for bit as they were: this
        // spec takes the core route, whose GEMMs write the destination
        // directly and never the sectors they miss.
        let mut destination = prior.materialize($engine).unwrap();
        lhs.contract_into(
            $engine,
            &rhs,
            &spec,
            &mut destination,
            D::alpha(),
            D::real(1.0),
        )
        .unwrap();
        for &index in &unreached {
            assert_eq!(
                destination.dense_data().unwrap()[index].bits(),
                prior.dense_data().unwrap()[index].bits(),
                "{}: beta = 1 wrote an unreached block at {index}",
                $label
            );
        }

        // alpha = 0 never reads the source: a NaN source leaves exactly
        // `beta * dst` (VectorInterface's `scale(x, 0) = zero(x)`).
        let nan_source = t.scale($engine, D::real(f64::NAN)).unwrap();
        let nan_lhs = lhs.scale($engine, D::real(f64::NAN)).unwrap();
        for beta in [D::beta(), D::real(0.0), D::real(1.0)] {
            let zero = D::real(0.0);
            let what = |op: &str| format!("{} NaN-source {op} beta={beta:?}", $label);
            let before = t2.permute($engine, &[2, 0], &[1, 3]).unwrap();
            check_into!(
                $engine,
                &what("permute_into"),
                before.scale($engine, zero).unwrap(),
                before,
                zero,
                beta,
                |d: &mut TensorMap<_, D>| nan_source.permute_into(
                    $engine,
                    &[2, 0],
                    &[1, 3],
                    d,
                    zero,
                    beta
                ),
                2
            );
            let before = t2.trace_pairs($engine, &[(0, 2)]).unwrap();
            check_into!(
                $engine,
                &what("trace_pairs_into"),
                before.scale($engine, zero).unwrap(),
                before,
                zero,
                beta,
                |d: &mut TensorMap<_, D>| nan_source.trace_pairs_into(
                    $engine,
                    &[(0, 2)],
                    d,
                    zero,
                    beta
                ),
                2
            );
            check_into!(
                $engine,
                &what("axpby_into"),
                t2.scale($engine, zero).unwrap(),
                t2,
                zero,
                beta,
                |d: &mut TensorMap<_, D>| nan_source.axpby_into($engine, d, zero, beta),
                2
            );
            check_into!(
                $engine,
                &what("contract_into"),
                prior.scale($engine, zero).unwrap(),
                prior,
                zero,
                beta,
                |d: &mut TensorMap<_, D>| nan_lhs
                    .contract_into($engine, &rhs, &spec, d, zero, beta),
                2
            );
        }

        // beta = 0 never reads the destination: a NaN destination comes back
        // exactly as the eager result, unreached blocks as +0.
        for alpha in [D::real(1.0), D::alpha()] {
            let mut destination = prior.scale($engine, D::real(f64::NAN)).unwrap();
            lhs.contract_into($engine, &rhs, &spec, &mut destination, alpha, D::real(0.0))
                .unwrap();
            let got = destination.dense_data().unwrap();
            assert!(
                !got.iter().any(|value| value.is_nan()),
                "{}: NaN leaked",
                $label
            );
            numerics::assert_slices_close(
                &format!("{} NaN contract_into", $label),
                got,
                eager.scale($engine, alpha).unwrap().dense_data().unwrap(),
                8,
            );
            for &index in &unreached {
                assert_eq!(
                    got[index].bits(),
                    D::real(0.0).bits(),
                    "{}: {index}",
                    $label
                );
            }
            let source = rand($engine, &[&v, &w], &[&v, &w], 7);
            let mut destination = source
                .permute($engine, &[2, 0], &[1, 3])
                .unwrap()
                .scale($engine, D::real(f64::NAN))
                .unwrap();
            source
                .permute_into(
                    $engine,
                    &[2, 0],
                    &[1, 3],
                    &mut destination,
                    alpha,
                    D::real(0.0),
                )
                .unwrap();
            assert!(!destination
                .dense_data()
                .unwrap()
                .iter()
                .any(|value| value.is_nan()));
            let mut destination = source
                .trace_pairs($engine, &[(0, 2)])
                .unwrap()
                .scale($engine, D::real(f64::NAN))
                .unwrap();
            source
                .trace_pairs_into($engine, &[(0, 2)], &mut destination, alpha, D::real(0.0))
                .unwrap();
            assert!(!destination
                .dense_data()
                .unwrap()
                .iter()
                .any(|value| value.is_nan()));
            let mut destination = source.scale($engine, D::real(f64::NAN)).unwrap();
            t.axpby_into($engine, &mut destination, alpha, D::real(0.0))
                .unwrap();
            assert!(!destination
                .dense_data()
                .unwrap()
                .iter()
                .any(|value| value.is_nan()));
        }

        // A clone-shared destination is refused and left as it was.
        let mut destination = prior.materialize($engine).unwrap();
        let clone = destination.clone();
        assert_eq!(
            lhs.contract_into(
                $engine,
                &rhs,
                &spec,
                &mut destination,
                D::real(1.0),
                D::real(0.0)
            ),
            Err(Error::DestinationShared)
        );
        assert_eq!(
            destination.dense_data().unwrap(),
            prior.dense_data().unwrap()
        );
        drop(clone);
    }};
}

fn u1() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let rule = Arc::new(U1FusionRule);
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [(-1, 2), (0, 1), (1, 3)].map(|(q, n)| (U1Irrep::new(q), n)),
    )
    .unwrap();
    let bond = GradedSpace::try_new(rule, [(U1Irrep::new(0), 2)]).unwrap();
    (v, bond)
}

fn su2() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let rule = Arc::new(SU2FusionRule);
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [(0, 2), (1, 2), (2, 1)].map(|(s, n)| (SU2Irrep::from_twice_spin(s), n)),
    )
    .unwrap();
    let bond = GradedSpace::try_new(rule, [(SU2Irrep::from_twice_spin(1), 2)]).unwrap();
    (v, bond)
}

type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

fn fz2u1() -> (GradedSpace<Fz2U1>, GradedSpace<Fz2U1>) {
    let sector = |q: i32| {
        let parity = if q.rem_euclid(2) == 0 {
            Z2Irrep::EVEN
        } else {
            Z2Irrep::ODD
        };
        ProductSector::new(parity, U1Irrep::new(q))
    };
    let rule = Arc::new(Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap());
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [(-1, 2), (0, 1), (1, 2), (2, 1)].map(|(q, n)| (sector(q), n)),
    )
    .unwrap();
    let bond = GradedSpace::try_new(rule, [(sector(1), 2)]).unwrap();
    (v, bond)
}

macro_rules! all_dtypes {
    ($engine:ident, $rt:expr, $legs:expr, $label:expr) => {{
        let (v, bond) = $legs;
        suite!(
            $engine,
            $rt,
            v.clone(),
            bond.clone(),
            f64,
            concat!($label, " f64")
        );
        suite!(
            $engine,
            $rt,
            v.clone(),
            bond.clone(),
            Complex64,
            concat!($label, " c64")
        );
        suite!(
            $engine,
            $rt,
            v.clone(),
            bond.clone(),
            f32,
            concat!($label, " f32")
        );
        suite!($engine, $rt, v, bond, Complex32, concat!($label, " c32"));
    }};
}

#[test]
fn every_linear_into_matches_its_eager_oracle_u1() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    all_dtypes!(engine, &Runtime::builder().build().unwrap(), u1(), "U1");
}

#[test]
fn every_linear_into_matches_its_eager_oracle_su2() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    all_dtypes!(engine, &Runtime::builder().build().unwrap(), su2(), "SU2");
}

#[test]
fn every_linear_into_matches_its_eager_oracle_fz2u1() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    all_dtypes!(
        engine,
        &Runtime::builder().build().unwrap(),
        fz2u1(),
        "fZ2xU1"
    );
}

/// Device `*_into` against the Host `*_into` (already pinned to the eager
/// oracle above) on the same inputs and destination values.
#[cfg(feature = "cuda")]
macro_rules! device_suite {
    ($rt:expr, $v:expr, $bond:expr, $d:ty, $label:expr) => {{
        let rt = $rt;
        let v = $v;
        let w = v.try_dual().unwrap();
        let bond = $bond;
        type D = $d;
        let engine = &mut symtn::typed::CpuEngine::serial();
        let cuda = &mut crate::common::cuda_engine();
        let rand = |engine: &mut dyn symtn::typed::Engine,
                    codomain: &[&GradedSpace<_>],
                    domain: &[&GradedSpace<_>],
                    seed: u64| {
            TensorMap::<_, D>::rand_with_seed(
                engine,
                rt,
                codomain.iter().copied(),
                domain.iter().copied(),
                seed,
            )
            .unwrap()
        };
        macro_rules! same {
            ($what:expr, $prior:expr, $host:expr, $device:expr, $terms:expr) => {{
                let mut host = $prior.materialize(engine).unwrap();
                #[allow(clippy::redundant_closure_call)]
                ($host)(&mut host).unwrap();
                let mut device = $prior.to_cuda(cuda).unwrap();
                #[allow(clippy::redundant_closure_call)]
                ($device)(&mut device).unwrap();
                numerics::assert_slices_close(
                    $what,
                    device.to_host(cuda).unwrap().dense_data().unwrap(),
                    host.dense_data().unwrap(),
                    $terms,
                );
            }};
        }
        let t = rand(engine, &[&v, &w], &[&v, &w], 1);
        let t2 = rand(engine, &[&v, &w], &[&v, &w], 2);
        let rhs = rand(engine, &[&v, &w], &[&v], 3);
        let (dt, drhs) = (t.to_cuda(cuda).unwrap(), rhs.to_cuda(cuda).unwrap());
        let (lazy, dlazy) = (t.adjoint(engine).unwrap(), dt.adjoint(cuda).unwrap());
        let spec = ContractSpec {
            lhs: &[2, 3],
            rhs: &[0, 1],
            codomain: &[2, 0],
            domain: &[1],
        };
        let lhs = rand(engine, &[&v], &[&bond], 4);
        let prhs = rand(engine, &[&bond], &[&v], 5);
        let (dlhs, dprhs) = (lhs.to_cuda(cuda).unwrap(), prhs.to_cuda(cuda).unwrap());
        let partial = ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        };
        let prior_partial = rand(engine, &[&v], &[&v], 6);
        let nan = D::real(f64::NAN);
        for (alpha, beta) in coefficients::<D>() {
            let what = |op: &str| format!("{} device {op} alpha={alpha:?} beta={beta:?}", $label);
            let levels = [0, 1, 2, 3];
            same!(
                &what("permute_into"),
                t2.permute(engine, &[2, 0], &[1, 3]).unwrap(),
                |d: &mut TensorMap<_, D>| t.permute_into(engine, &[2, 0], &[1, 3], d, alpha, beta),
                |d: &mut TensorMap<_, D, _>| dt.permute_into(
                    cuda,
                    &[2, 0],
                    &[1, 3],
                    d,
                    alpha,
                    beta
                ),
                8
            );
            same!(
                &what("braid_into"),
                t2.braid(engine, &[1, 0], &[3, 2], &levels).unwrap(),
                |d: &mut TensorMap<_, D>| t.braid_into(
                    engine,
                    &[1, 0],
                    &[3, 2],
                    &levels,
                    d,
                    alpha,
                    beta
                ),
                |d: &mut TensorMap<_, D, _>| dt.braid_into(
                    cuda,
                    &[1, 0],
                    &[3, 2],
                    &levels,
                    d,
                    alpha,
                    beta
                ),
                8
            );
            same!(
                &what("transpose_into"),
                t2.transpose(engine, &[1, 3], &[0, 2]).unwrap(),
                |d: &mut TensorMap<_, D>| t.transpose_into(
                    engine,
                    &[1, 3],
                    &[0, 2],
                    d,
                    alpha,
                    beta
                ),
                |d: &mut TensorMap<_, D, _>| dt.transpose_into(
                    cuda,
                    &[1, 3],
                    &[0, 2],
                    d,
                    alpha,
                    beta
                ),
                8
            );
            same!(
                &what("repartition_into"),
                t2.repartition(engine, 1).unwrap(),
                |d: &mut TensorMap<_, D>| t.repartition_into(engine, d, alpha, beta),
                |d: &mut TensorMap<_, D, _>| dt.repartition_into(cuda, d, alpha, beta),
                8
            );
            same!(
                &what("trace_pairs_into"),
                t2.trace_pairs(engine, &[(0, 2)]).unwrap(),
                |d: &mut TensorMap<_, D>| t.trace_pairs_into(engine, &[(0, 2)], d, alpha, beta),
                |d: &mut TensorMap<_, D, _>| dt.trace_pairs_into(cuda, &[(0, 2)], d, alpha, beta),
                16
            );
            same!(
                &what("lazy trace_pairs_into"),
                t2.adjoint(engine)
                    .unwrap()
                    .trace_pairs(engine, &[(0, 2)])
                    .unwrap(),
                |d: &mut TensorMap<_, D>| lazy.trace_pairs_into(engine, &[(0, 2)], d, alpha, beta),
                |d: &mut TensorMap<_, D, _>| dlazy.trace_pairs_into(
                    cuda,
                    &[(0, 2)],
                    d,
                    alpha,
                    beta
                ),
                16
            );
            same!(
                &what("axpby_into"),
                t2,
                |d: &mut TensorMap<_, D>| t.axpby_into(engine, d, alpha, beta),
                |d: &mut TensorMap<_, D, _>| dt.axpby_into(cuda, d, alpha, beta),
                2
            );
            same!(
                &what("contract_into"),
                t2.contract(engine, &rhs, &spec)
                    .unwrap()
                    .scale(engine, D::real(0.5))
                    .unwrap(),
                |d: &mut TensorMap<_, D>| t.contract_into(engine, &rhs, &spec, d, alpha, beta),
                |d: &mut TensorMap<_, D, _>| dt.contract_into(cuda, &drhs, &spec, d, alpha, beta),
                32
            );
            same!(
                &what("lazy contract_into"),
                lazy.contract(engine, &rhs, &spec)
                    .unwrap()
                    .scale(engine, D::real(0.5))
                    .unwrap(),
                |d: &mut TensorMap<_, D>| lazy.contract_into(engine, &rhs, &spec, d, alpha, beta),
                |d: &mut TensorMap<_, D, _>| dlazy
                    .contract_into(cuda, &drhs, &spec, d, alpha, beta),
                32
            );
            same!(
                &what("partial contract_into"),
                prior_partial,
                |d: &mut TensorMap<_, D>| lhs
                    .contract_into(engine, &prhs, &partial, d, alpha, beta),
                |d: &mut TensorMap<_, D, _>| dlhs
                    .contract_into(cuda, &dprhs, &partial, d, alpha, beta),
                8
            );
            if beta == D::real(0.0) {
                same!(
                    &what("NaN partial contract_into"),
                    prior_partial.scale(engine, nan).unwrap(),
                    |d: &mut TensorMap<_, D>| lhs
                        .contract_into(engine, &prhs, &partial, d, alpha, beta),
                    |d: &mut TensorMap<_, D, _>| dlhs
                        .contract_into(cuda, &dprhs, &partial, d, alpha, beta),
                    8
                );
                same!(
                    &what("NaN permute_into"),
                    t2.permute(engine, &[2, 0], &[1, 3])
                        .unwrap()
                        .scale(engine, nan)
                        .unwrap(),
                    |d: &mut TensorMap<_, D>| t.permute_into(
                        engine,
                        &[2, 0],
                        &[1, 3],
                        d,
                        alpha,
                        beta
                    ),
                    |d: &mut TensorMap<_, D, _>| dt.permute_into(
                        cuda,
                        &[2, 0],
                        &[1, 3],
                        d,
                        alpha,
                        beta
                    ),
                    8
                );
                same!(
                    &what("NaN lazy contract_into"),
                    lazy.contract(engine, &rhs, &spec)
                        .unwrap()
                        .scale(engine, nan)
                        .unwrap(),
                    |d: &mut TensorMap<_, D>| lazy
                        .contract_into(engine, &rhs, &spec, d, alpha, beta),
                    |d: &mut TensorMap<_, D, _>| dlazy
                        .contract_into(cuda, &drhs, &spec, d, alpha, beta),
                    32
                );
                same!(
                    &what("NaN braid_into"),
                    t2.braid(engine, &[1, 0], &[3, 2], &levels)
                        .unwrap()
                        .scale(engine, nan)
                        .unwrap(),
                    |d: &mut TensorMap<_, D>| t.braid_into(
                        engine,
                        &[1, 0],
                        &[3, 2],
                        &levels,
                        d,
                        alpha,
                        beta
                    ),
                    |d: &mut TensorMap<_, D, _>| dt.braid_into(
                        cuda,
                        &[1, 0],
                        &[3, 2],
                        &levels,
                        d,
                        alpha,
                        beta
                    ),
                    8
                );
                same!(
                    &what("NaN transpose_into"),
                    t2.transpose(engine, &[1, 3], &[0, 2])
                        .unwrap()
                        .scale(engine, nan)
                        .unwrap(),
                    |d: &mut TensorMap<_, D>| t.transpose_into(
                        engine,
                        &[1, 3],
                        &[0, 2],
                        d,
                        alpha,
                        beta
                    ),
                    |d: &mut TensorMap<_, D, _>| dt.transpose_into(
                        cuda,
                        &[1, 3],
                        &[0, 2],
                        d,
                        alpha,
                        beta
                    ),
                    8
                );
                same!(
                    &what("NaN repartition_into"),
                    t2.repartition(engine, 1)
                        .unwrap()
                        .scale(engine, nan)
                        .unwrap(),
                    |d: &mut TensorMap<_, D>| t.repartition_into(engine, d, alpha, beta),
                    |d: &mut TensorMap<_, D, _>| dt.repartition_into(cuda, d, alpha, beta),
                    8
                );
                same!(
                    &what("NaN trace_pairs_into"),
                    t2.trace_pairs(engine, &[(0, 2)])
                        .unwrap()
                        .scale(engine, nan)
                        .unwrap(),
                    |d: &mut TensorMap<_, D>| t.trace_pairs_into(engine, &[(0, 2)], d, alpha, beta),
                    |d: &mut TensorMap<_, D, _>| dt.trace_pairs_into(
                        cuda,
                        &[(0, 2)],
                        d,
                        alpha,
                        beta
                    ),
                    16
                );
                same!(
                    &what("NaN axpby_into"),
                    t2.scale(engine, nan).unwrap(),
                    |d: &mut TensorMap<_, D>| t.axpby_into(engine, d, alpha, beta),
                    |d: &mut TensorMap<_, D, _>| dt.axpby_into(cuda, d, alpha, beta),
                    2
                );
            }
        }
        // alpha = 0 on the device (the executor's zero-scale branch with a
        // general beta): a NaN source is not read, `beta * dst` is left.
        {
            let zero = D::real(0.0);
            let beta = D::beta();
            let (nan_t, dnan_t) = (t.scale(engine, nan).unwrap(), dt.scale(cuda, nan).unwrap());
            let (nan_rhs, dnan_rhs) = (
                rhs.scale(engine, nan).unwrap(),
                drhs.scale(cuda, nan).unwrap(),
            );
            let what = |op: &str| format!("{} device NaN-source {op}", $label);
            same!(
                &what("permute_into"),
                t2.permute(engine, &[2, 0], &[1, 3]).unwrap(),
                |d: &mut TensorMap<_, D>| nan_t.permute_into(
                    engine,
                    &[2, 0],
                    &[1, 3],
                    d,
                    zero,
                    beta
                ),
                |d: &mut TensorMap<_, D, _>| dnan_t.permute_into(
                    cuda,
                    &[2, 0],
                    &[1, 3],
                    d,
                    zero,
                    beta
                ),
                2
            );
            same!(
                &what("contract_into"),
                t2.contract(engine, &rhs, &spec).unwrap(),
                |d: &mut TensorMap<_, D>| t.contract_into(engine, &nan_rhs, &spec, d, zero, beta),
                |d: &mut TensorMap<_, D, _>| dt
                    .contract_into(cuda, &dnan_rhs, &spec, d, zero, beta),
                2
            );
            same!(
                &what("axpby_into"),
                t2,
                |d: &mut TensorMap<_, D>| nan_t.axpby_into(engine, d, zero, beta),
                |d: &mut TensorMap<_, D, _>| dnan_t.axpby_into(cuda, d, zero, beta),
                2
            );
        }
        // A lazy-adjoint axpby source reads its parent's blocks on the device;
        // a zero beta over a NaN destination does not let it through.
        let prior = t2.adjoint(engine).unwrap().materialize(engine).unwrap();
        let poisoned = prior.scale(engine, nan).unwrap();
        for (alpha, beta) in coefficients::<D>() {
            same!(
                &format!(
                    "{} device lazy axpby_into alpha={alpha:?} beta={beta:?}",
                    $label
                ),
                prior,
                |d: &mut TensorMap<_, D>| lazy.axpby_into(engine, d, alpha, beta),
                |d: &mut TensorMap<_, D, _>| dlazy.axpby_into(cuda, d, alpha, beta),
                2
            );
        }
        same!(
            &format!("{} device lazy axpby_into over NaN, beta=0", $label),
            poisoned,
            |d: &mut TensorMap<_, D>| lazy.axpby_into(engine, d, D::beta(), D::real(0.0)),
            |d: &mut TensorMap<_, D, _>| dlazy.axpby_into(cuda, d, D::beta(), D::real(0.0)),
            2
        );
        let mut destination = prior.to_cuda(cuda).unwrap();
        let clone = destination.clone();
        assert_eq!(
            dlazy.materialize(cuda).unwrap().axpby_into(
                cuda,
                &mut destination,
                D::real(1.0),
                D::real(0.0)
            ),
            Err(Error::DestinationShared)
        );
        drop(clone);
    }};
}

#[cfg(feature = "cuda")]
fn device_all(rt: &Runtime) {
    // Why one closure per suite: expanded inline, the twelve suites share
    // one debug-build stack frame, which alone nears the 2 MiB test-thread
    // stack; each closure frame is popped before the next suite runs.
    macro_rules! dtypes {
        ($legs:expr, $label:expr) => {{
            let (v, bond) = $legs;
            let real = || device_suite!(rt, v.clone(), bond.clone(), f64, concat!($label, " f64"));
            real();
            let complex = || {
                device_suite!(
                    rt,
                    v.clone(),
                    bond.clone(),
                    Complex64,
                    concat!($label, " c64")
                )
            };
            complex();
            let single =
                || device_suite!(rt, v.clone(), bond.clone(), f32, concat!($label, " f32"));
            single();
            let single_complex = || {
                device_suite!(
                    rt,
                    v.clone(),
                    bond.clone(),
                    Complex32,
                    concat!($label, " c32")
                )
            };
            single_complex();
        }};
    }
    dtypes!(u1(), "U1");
    dtypes!(su2(), "SU2");
    dtypes!(fz2u1(), "fZ2xU1");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn every_linear_into_matches_the_host_on_the_device() {
    device_all(&Runtime::builder().build().unwrap());
}
