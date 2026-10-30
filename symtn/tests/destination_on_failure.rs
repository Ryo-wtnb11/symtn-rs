//! The destination of a failing `*_into` is left untouched (#1551).
//!
//! Every malformed-input class of every linear destination operation —
//! runtime, rule, space, leg index, capability, destination representation,
//! aliasing and a shared destination — is called on a destination filled
//! with distinct sentinels, and the destination's bytes are compared bit
//! for bit before and after the failing call. The oracle is those bytes
//! alone, independent of every operation under test; each fixture also runs
//! one valid call on a fresh destination, so a rejection is caused by the one
//! malformed input and not by a broken fixture. Host tensors additionally
//! keep their payload pointer. `scale_assign` gets the same destination rule.
//! The device variants run with `--features cuda -- --ignored`.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{U1FusionRule, U1Irrep, ZNFusionRule};
use symtn::typed::{
    ContractSpec, Duality, Error, GradedSpace, Runtime, SectorSpectrum, Side, TensorMap,
};

/// A fresh tensor on `like`'s legs filled with distinct sentinels,
/// alternating quiet NaNs with distinct finite values. Overwriting a NaN
/// changes its bits; a read-modify-write can keep a NaN's payload (IEEE
/// arithmetic propagates it), so the finite neighbours catch a `beta` pass.
macro_rules! sentinel {
    ($rt:expr, $like:expr) => {{
        let engine = &mut symtn_dense::CpuEngine::serial();
        let like = &$like;
        let (codomain, domain) = (like.codomain(), like.domain());
        let mut next = 0u64;
        TensorMap::<_, f64>::from_subblock_fn(engine, $rt, &codomain, &domain, |_, _| {
            next += 1;
            sentinel_value(next)
        })
        .unwrap()
    }};
}

/// Calls `$call` on `$dst`, expects an error matching `$err`, and checks
/// that the destination's legs and payload bits (and, on Host, its payload
/// pointer) are those from before the call.
macro_rules! rejects {
    ($snap:expr, $what:expr, $dst:expr, SpaceMismatch, $call:expr) => {
        rejects!(
            $snap,
            $what,
            $dst,
            Error::Operation(operation)
                if matches!(**operation, symtn::typed::OperationError::SpaceMismatch { .. }),
            $call
        )
    };
    ($snap:expr, $what:expr, $dst:expr, $err:pat $(if $guard:expr)?, $call:expr) => {{
        let mut dst = $dst;
        let (codomain, domain) = (dst.codomain(), dst.domain());
        #[allow(clippy::redundant_closure_call)]
        let before = ($snap)(&dst);
        let result = call(&mut dst, $call);
        // `$err` is `_` where only the rejection, not its kind, is pinned.
        #[allow(clippy::redundant_pattern_matching)]
        let rejected = matches!(&result, Err($err) $(if $guard)?);
        assert!(
            rejected,
            "{}: expected {}, got {result:?}",
            $what,
            stringify!($err)
        );
        #[allow(clippy::redundant_closure_call)]
        let after = ($snap)(&dst);
        assert!(before == after, "{}: destination changed", $what);
        assert!(
            dst.codomain() == codomain && dst.domain() == domain,
            "{}: destination legs changed",
            $what
        );
    }};
}

fn sentinel_value(index: u64) -> f64 {
    if index.is_multiple_of(2) {
        f64::from_bits(0x7ff8_0000_0000_0000 | index)
    } else {
        std::f64::consts::PI * index as f64
    }
}

/// Fixes the destination type before the closure body is checked.
fn call<T, E>(destination: &mut T, f: impl FnOnce(&mut T) -> Result<(), E>) -> Result<(), E> {
    f(destination)
}

fn host_snap<R>(t: &TensorMap<R, f64>) -> (Vec<u64>, usize) {
    let data = t.dense_data().unwrap();
    (
        data.iter().map(|x| x.to_bits()).collect(),
        data.as_ptr() as usize,
    )
}

fn legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
            (U1Irrep::new(-1), 2),
        ],
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    (v, w)
}

/// Every U(1) malformed-input class of every `*_into`. `$lift` puts a Host
/// tensor at the placement under test; `$snap` reads a destination's bytes.
macro_rules! u1_suite {
    ($engine:ident, $ops:ident, $rt:expr, $other_rt:expr, $lift:expr, $snap:expr) => {{
        let (rt, other_rt) = ($rt, $other_rt);
        let lift = $lift;
        let (v, w) = legs();
        let host_t =
            TensorMap::<_, f64>::rand_with_seed($engine, rt, [&v, &w], [&v, &w], 1).unwrap();
        let host_rhs = TensorMap::<_, f64>::rand_with_seed($engine, rt, [&v, &w], [&v], 2).unwrap();
        let (t, rhs) = (lift($ops, &host_t), lift($ops, &host_rhs));
        let lazy = t.adjoint($ops).unwrap();
        let spec = ContractSpec {
            lhs: &[2, 3],
            rhs: &[0, 1],
            codomain: &[2, 0],
            domain: &[1],
        };
        let bad_spec = ContractSpec {
            lhs: &[2, 9],
            rhs: &[0, 1],
            codomain: &[2, 0],
            domain: &[1],
        };
        // Pairs `t`'s domain `v` with `rhs`'s codomain `w`: not mutually dual.
        let undual_spec = ContractSpec {
            lhs: &[2, 3],
            rhs: &[1, 0],
            codomain: &[2, 0],
            domain: &[1],
        };
        let fresh = |e: &mut dyn symtn::typed::Engine, like: &TensorMap<_, f64>| {
            lift(e, &sentinel!(rt, like))
        };
        let foreign = |e: &mut dyn symtn::typed::Engine, like: &TensorMap<_, f64>| {
            lift(e, &sentinel!(other_rt, like))
        };
        // Wrong space: every leg dualized.
        let wrong = |e: &mut dyn symtn::typed::Engine, like: &TensorMap<_, f64>| {
            let dual = |legs: Vec<GradedSpace<_>>| {
                legs.iter()
                    .map(|leg| leg.try_dual().unwrap())
                    .collect::<Vec<_>>()
            };
            let (codomain, domain) = (dual(like.codomain()), dual(like.domain()));
            let mut next = 0u64;
            let fixture_engine = &mut symtn_dense::CpuEngine::serial();
            lift(
                e,
                &TensorMap::<_, f64>::from_subblock_fn(
                    fixture_engine,
                    rt,
                    &codomain,
                    &domain,
                    |_, _| {
                        next += 1;
                        sentinel_value(next)
                    },
                )
                .unwrap(),
            )
        };
        let permuted = host_t.permute($engine, &[2, 0], &[1, 3]).unwrap();
        let braided = host_t
            .braid($engine, &[1, 0], &[3, 2], &[0, 1, 2, 3])
            .unwrap();
        let transposed = host_t.transpose($engine, &[1, 3], &[0, 2]).unwrap();
        let repartitioned = host_t.repartition($engine, 1).unwrap();
        let traced = host_t.trace_pairs($engine, &[(0, 2)]).unwrap();
        let contracted = host_t.contract($engine, &host_rhs, &spec).unwrap();
        let mut fixture_destination = fresh($ops, &permuted);

        // The fixtures are valid: the unmodified call succeeds.
        t.permute_into($ops, &[2, 0], &[1, 3], &mut fixture_destination, 1.0, 0.5)
            .unwrap();
        let mut fixture_destination = fresh($ops, &braided);
        t.braid_into(
            $ops,
            &[1, 0],
            &[3, 2],
            &[0, 1, 2, 3],
            &mut fixture_destination,
            1.0,
            0.5,
        )
        .unwrap();
        let mut fixture_destination = fresh($ops, &transposed);
        t.transpose_into($ops, &[1, 3], &[0, 2], &mut fixture_destination, 1.0, 0.5)
            .unwrap();
        let mut fixture_destination = fresh($ops, &repartitioned);
        t.repartition_into($ops, &mut fixture_destination, 1.0, 0.5)
            .unwrap();
        let mut fixture_destination = fresh($ops, &traced);
        t.trace_pairs_into($ops, &[(0, 2)], &mut fixture_destination, 1.0, 0.5)
            .unwrap();
        let mut fixture_destination = fresh($ops, &contracted);
        t.contract_into($ops, &rhs, &spec, &mut fixture_destination, 1.0, 0.5)
            .unwrap();
        let mut fixture_destination = fresh($ops, &host_t);
        t.axpby_into($ops, &mut fixture_destination, 1.0, 0.5)
            .unwrap();

        // A second handle on the destination's storage for the duration of
        // the call.
        macro_rules! shared {
            ($f:expr) => {
                |d: &mut _| {
                    let clone = d.clone();
                    #[allow(clippy::redundant_closure_call)]
                    let result = ($f)(d);
                    drop(clone);
                    result
                }
            };
        }

        // permute_into
        let op = "permute_into";
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} runtime"),
            foreign($ops, &permuted),
            Error::RuntimeMismatch,
            |d: &mut _| t.permute_into($ops, &[2, 0], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} space"),
            wrong($ops, &permuted),
            SpaceMismatch,
            |d: &mut _| t.permute_into($ops, &[2, 0], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} leg out of range"),
            fresh($ops, &permuted),
            _,
            |d: &mut _| t.permute_into($ops, &[4, 0], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} leg repeated"),
            fresh($ops, &permuted),
            _,
            |d: &mut _| t.permute_into($ops, &[0, 0], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            // Both lanes admit a lazy source (#1870, Host and device) and
            // report the space.
            format!("{op} lazy source, wrong space"),
            wrong($ops, &permuted),
            SpaceMismatch,
            |d: &mut _| lazy.permute_into($ops, &[2, 0], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} alias"),
            fresh($ops, &host_t),
            Error::InvalidArgument(_),
            |d: &mut _| {
                let source = d.clone();
                source.permute_into($ops, &[0, 1], &[2, 3], d, 1.0, 0.5)
            }
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} shared"),
            fresh($ops, &permuted),
            Error::DestinationShared,
            shared!(|d: &mut _| t.permute_into($ops, &[2, 0], &[1, 3], d, 1.0, 0.5))
        );

        // braid_into
        let op = "braid_into";
        let levels = [0, 1, 2, 3];
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} runtime"),
            foreign($ops, &braided),
            Error::RuntimeMismatch,
            |d: &mut _| t.braid_into($ops, &[1, 0], &[3, 2], &levels, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} space"),
            wrong($ops, &braided),
            SpaceMismatch,
            |d: &mut _| t.braid_into($ops, &[1, 0], &[3, 2], &levels, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} levels"),
            fresh($ops, &braided),
            Error::InvalidArgument(_),
            |d: &mut _| t.braid_into($ops, &[1, 0], &[3, 2], &levels[..3], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} leg out of range"),
            fresh($ops, &braided),
            _,
            |d: &mut _| t.braid_into($ops, &[1, 7], &[3, 2], &levels, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} lazy source, wrong space"),
            wrong($ops, &braided),
            SpaceMismatch,
            |d: &mut _| lazy.braid_into($ops, &[1, 0], &[3, 2], &levels, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} shared"),
            fresh($ops, &braided),
            Error::DestinationShared,
            shared!(|d: &mut _| t.braid_into($ops, &[1, 0], &[3, 2], &levels, d, 1.0, 0.5))
        );

        // transpose_into
        let op = "transpose_into";
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} runtime"),
            foreign($ops, &transposed),
            Error::RuntimeMismatch,
            |d: &mut _| t.transpose_into($ops, &[1, 3], &[0, 2], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} space"),
            wrong($ops, &transposed),
            SpaceMismatch,
            |d: &mut _| t.transpose_into($ops, &[1, 3], &[0, 2], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} non-planar"),
            fresh($ops, &transposed),
            _,
            |d: &mut _| t.transpose_into($ops, &[0, 2], &[1, 3], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} leg out of range"),
            fresh($ops, &transposed),
            _,
            |d: &mut _| t.transpose_into($ops, &[1, 8], &[0, 2], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} shared"),
            fresh($ops, &transposed),
            Error::DestinationShared,
            shared!(|d: &mut _| t.transpose_into($ops, &[1, 3], &[0, 2], d, 1.0, 0.5))
        );

        // repartition_into
        let op = "repartition_into";
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} runtime"),
            foreign($ops, &repartitioned),
            Error::RuntimeMismatch,
            |d: &mut _| t.repartition_into($ops, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} space"),
            wrong($ops, &repartitioned),
            SpaceMismatch,
            |d: &mut _| t.repartition_into($ops, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} rank"),
            fresh($ops, &traced),
            Error::InvalidArgument(_),
            |d: &mut _| t.repartition_into($ops, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} lazy source, wrong space"),
            wrong($ops, &repartitioned),
            SpaceMismatch,
            |d: &mut _| lazy.repartition_into($ops, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} shared"),
            fresh($ops, &repartitioned),
            Error::DestinationShared,
            shared!(|d: &mut _| t.repartition_into($ops, d, 1.0, 0.5))
        );

        // trace_pairs_into
        let op = "trace_pairs_into";
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} runtime"),
            foreign($ops, &traced),
            Error::RuntimeMismatch,
            |d: &mut _| t.trace_pairs_into($ops, &[(0, 2)], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} space"),
            wrong($ops, &traced),
            SpaceMismatch,
            |d: &mut _| t.trace_pairs_into($ops, &[(0, 2)], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} leg out of range"),
            fresh($ops, &traced),
            _,
            |d: &mut _| t.trace_pairs_into($ops, &[(0, 9)], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} leg repeated"),
            fresh($ops, &traced),
            _,
            |d: &mut _| t.trace_pairs_into($ops, &[(0, 0)], d, 1.0, 0.5)
        );
        // On the legs the pair would leave, so the duality check itself (in
        // the trace compile, after every destination check) rejects it.
        let undual_traced =
            TensorMap::<_, f64>::rand_with_seed($engine, rt, [&w], [&v], 6).unwrap();
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} legs not dual"),
            fresh($ops, &undual_traced),
            _,
            |d: &mut _| t.trace_pairs_into($ops, &[(0, 3)], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} alias"),
            fresh($ops, &host_t),
            Error::InvalidArgument(_),
            |d: &mut _| {
                let source = d.clone();
                source.trace_pairs_into($ops, &[], d, 1.0, 0.5)
            }
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} shared"),
            fresh($ops, &traced),
            Error::DestinationShared,
            shared!(|d: &mut _| t.trace_pairs_into($ops, &[(0, 2)], d, 1.0, 0.5))
        );
        // One destination check order and wording on every placement: a
        // destination both on the wrong space and shared is a space error.
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} wrong space before shared"),
            wrong($ops, &traced),
            SpaceMismatch,
            shared!(|d: &mut _| t.trace_pairs_into($ops, &[(0, 2)], d, 1.0, 0.5))
        );

        // contract_into
        let op = "contract_into";
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} runtime"),
            foreign($ops, &contracted),
            Error::RuntimeMismatch,
            |d: &mut _| t.contract_into($ops, &rhs, &spec, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} space"),
            wrong($ops, &contracted),
            SpaceMismatch,
            |d: &mut _| t.contract_into($ops, &rhs, &spec, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} leg out of range"),
            fresh($ops, &contracted),
            _,
            |d: &mut _| t.contract_into($ops, &rhs, &bad_spec, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} legs not dual"),
            fresh($ops, &contracted),
            _,
            |d: &mut _| t.contract_into($ops, &rhs, &undual_spec, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} alias"),
            fresh($ops, &contracted),
            Error::InvalidArgument(_),
            |d: &mut _| {
                let lhs = d.clone();
                lhs.contract_into($engine, &rhs, &spec, d, 1.0, 0.5)
            }
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} shared"),
            fresh($ops, &contracted),
            Error::DestinationShared,
            shared!(|d: &mut _| t.contract_into($ops, &rhs, &spec, d, 1.0, 0.5))
        );

        // axpby_into
        let op = "axpby_into";
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} runtime"),
            foreign($ops, &host_t),
            Error::RuntimeMismatch,
            |d: &mut _| t.axpby_into($ops, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} space"),
            wrong($ops, &host_t),
            SpaceMismatch,
            |d: &mut _| t.axpby_into($ops, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} alias"),
            fresh($ops, &host_t),
            Error::InvalidArgument(_),
            |d: &mut _| {
                let source = d.clone();
                source.axpby_into($ops, d, 1.0, 0.5)
            }
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            format!("{op} shared"),
            fresh($ops, &host_t),
            Error::DestinationShared,
            shared!(|d: &mut _| t.axpby_into($ops, d, 1.0, 0.5))
        );
    }};
}

/// Z2 sources against a Z3 destination: the same Rust type, another rule.
macro_rules! rule_suite {
    ($engine:ident, $ops:ident, $rt:expr, $lift:expr, $snap:expr) => {{
        let rt = $rt;
        let lift = $lift;
        let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
        let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
        let a =
            GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 2), (z2.irrep(1), 1)]).unwrap();
        let b =
            GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 2), (z3.irrep(1), 1)]).unwrap();
        let host_t =
            TensorMap::<_, f64>::rand_with_seed($engine, rt, [&a, &a], [&a, &a], 3).unwrap();
        let t = lift($ops, &host_t);
        let rank2 = |e: &mut dyn symtn::typed::Engine, leg: &GradedSpace<ZNFusionRule>| {
            lift(
                e,
                &sentinel!(
                    rt,
                    TensorMap::<_, f64>::rand_with_seed(
                        &mut symtn_dense::CpuEngine::serial(),
                        rt,
                        [leg],
                        [leg],
                        4
                    )
                    .unwrap()
                ),
            )
        };
        let rank4 = |e: &mut dyn symtn::typed::Engine, leg: &GradedSpace<ZNFusionRule>| {
            lift(
                e,
                &sentinel!(
                    rt,
                    TensorMap::<_, f64>::rand_with_seed(
                        &mut symtn_dense::CpuEngine::serial(),
                        rt,
                        [leg, leg],
                        [leg, leg],
                        5
                    )
                    .unwrap()
                ),
            )
        };
        let spec = ContractSpec {
            lhs: &[2, 3],
            rhs: &[0, 1],
            codomain: &[0, 1],
            domain: &[2, 3],
        };
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            "permute_into rule",
            rank4($ops, &b),
            Error::RuleMismatch,
            |d: &mut _| t.permute_into($ops, &[0, 1], &[2, 3], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            "braid_into rule",
            rank4($ops, &b),
            Error::RuleMismatch,
            |d: &mut _| t.braid_into($ops, &[0, 1], &[2, 3], &[0, 1, 2, 3], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            "transpose_into rule",
            rank4($ops, &b),
            Error::RuleMismatch,
            |d: &mut _| t.transpose_into($ops, &[0, 1], &[2, 3], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            "repartition_into rule",
            rank4($ops, &b),
            Error::RuleMismatch,
            |d: &mut _| t.repartition_into($ops, d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            "trace_pairs_into rule",
            rank2($ops, &b),
            Error::RuleMismatch,
            |d: &mut _| t.trace_pairs_into($ops, &[(0, 2)], d, 1.0, 0.5)
        );
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            "contract_into rule",
            rank4($ops, &b),
            Error::RuleMismatch,
            |d: &mut _| t.contract_into($ops, &t, &spec, d, 1.0, 0.5)
        );
        // `axpby_into` compares the operand spaces, which carry the rule.
        rejects!(
            |dst: &_| ($snap)(&mut *$ops, dst),
            "axpby_into rule",
            rank4($ops, &b),
            SpaceMismatch,
            |d: &mut _| t.axpby_into($ops, d, 1.0, 0.5)
        );
        let mut fixture_destination = rank4($ops, &a);
        // The fixture is otherwise valid.
        t.contract_into($ops, &t, &spec, &mut fixture_destination, 1.0, 0.5)
            .unwrap();
        let mut fixture_destination = rank2($ops, &a);
        t.trace_pairs_into($ops, &[(0, 2)], &mut fixture_destination, 1.0, 0.5)
            .unwrap();
    }};
}

#[test]
fn every_host_into_leaves_a_rejected_destination_bit_identical() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = Runtime::builder().build().unwrap();
    let other = Runtime::builder().build().unwrap();
    let lift = |e: &mut dyn symtn::typed::Engine, x: &TensorMap<U1FusionRule, f64>| {
        x.materialize(e).unwrap()
    };
    u1_suite!(engine, engine, &rt, &other, lift, |_, t| host_snap(t));
    let lift = |e: &mut dyn symtn::typed::Engine, x: &TensorMap<ZNFusionRule, f64>| {
        x.materialize(e).unwrap()
    };
    rule_suite!(engine, engine, &rt, lift, |_, t| host_snap(t));
}

/// Host-only representation and capability classes: a compact or lazy
/// destination, a compact source the operation cannot take, and an admitted
/// compact source's destination checks. (A
/// non-symmetric braiding provider cannot name `contract_into` at all: the
/// facade's trait bounds exclude it at compile time.)
#[test]
fn host_capability_and_representation_rejections_leave_the_destination_bit_identical() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = Runtime::builder().build().unwrap();
    let (v, _) = legs();
    let t = TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&v], [&v], 1).unwrap();
    // Inspection and fixture construction run on their own engine, so the
    // operation under test keeps the test's.
    let diag_snap = |d: &TensorMap<U1FusionRule, f64>| {
        d.diagview(&mut symtn_dense::CpuEngine::serial())
            .unwrap()
            .iter()
            .flat_map(|entry| entry.values.iter().map(|x| x.to_bits()))
            .collect::<Vec<_>>()
    };
    let compact = || {
        TensorMap::<_, f64>::diagonal(
            &mut symtn_dense::CpuEngine::serial(),
            &rt,
            &v,
            [
                SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![f64::NAN, 2.0],
                },
                SectorSpectrum {
                    sector: U1Irrep::new(1),
                    values: vec![3.0],
                },
                SectorSpectrum {
                    sector: U1Irrep::new(-1),
                    values: vec![-0.0, 5.0],
                },
            ],
        )
        .unwrap()
    };
    // A compact diagonal destination on the result's space.
    rejects!(
        diag_snap,
        "permute_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.permute_into(engine, &[0], &[1], d, 1.0, 0.5)
    );
    rejects!(
        diag_snap,
        "braid_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.braid_into(engine, &[0], &[1], &[0, 1], d, 1.0, 0.5)
    );
    rejects!(
        diag_snap,
        "transpose_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.transpose_into(engine, &[0], &[1], d, 1.0, 0.5)
    );
    rejects!(
        diag_snap,
        "repartition_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.repartition_into(engine, d, 1.0, 0.5)
    );
    rejects!(
        diag_snap,
        "trace_pairs_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.trace_pairs_into(engine, &[], d, 1.0, 0.5)
    );
    rejects!(
        diag_snap,
        "axpby_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.axpby_into(engine, d, 1.0, 0.5)
    );
    let unit = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    rejects!(
        diag_snap,
        "contract_into compact dst",
        compact(),
        Error::InvalidArgument(_),
        |d: &mut _| t.contract_into(engine, &t, &unit, d, 1.0, 0.5)
    );

    // A lazy-adjoint destination: its parent's bytes are the destination's.
    let parent = sentinel!(&rt, t);
    let lazy_snap = |_: &TensorMap<U1FusionRule, f64>| host_snap(&parent);
    rejects!(
        lazy_snap,
        "permute_into lazy dst",
        parent.adjoint(engine).unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| t.permute_into(engine, &[0], &[1], d, 1.0, 0.5)
    );
    rejects!(
        lazy_snap,
        "repartition_into lazy dst",
        parent.adjoint(engine).unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| t.repartition_into(engine, d, 1.0, 0.5)
    );
    rejects!(
        lazy_snap,
        "trace_pairs_into lazy dst",
        parent.adjoint(engine).unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| t.trace_pairs_into(engine, &[], d, 1.0, 0.5)
    );
    rejects!(
        lazy_snap,
        "axpby_into lazy dst",
        parent.adjoint(engine).unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| t.axpby_into(engine, d, 1.0, 0.5)
    );
    rejects!(
        lazy_snap,
        "contract_into lazy dst",
        parent.adjoint(engine).unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| t.contract_into(engine, &t, &unit, d, 1.0, 0.5)
    );

    // A compact source traced over its only pair is read from the compiled
    // terms, as eager reads it (#1870 C): admitted, and its destination
    // checks run. Its spectrum holds a NaN, so the value is eager's NaN.
    let s = compact();
    let mut scalar = t.trace_pairs(engine, &[(0, 1)]).unwrap();
    s.trace_pairs_into(engine, &[(0, 1)], &mut scalar, 1.0, 0.5)
        .unwrap();
    assert!(s
        .trace_pairs(engine, &[(0, 1)])
        .unwrap()
        .scalar(engine)
        .unwrap()
        .is_nan());
    assert!(scalar.scalar(engine).unwrap().is_nan());
    assert!(
        s.diagview(engine).is_ok(),
        "the compact source stays compact"
    );
    rejects!(
        host_snap,
        "trace_pairs_into compact source, shared",
        t.trace_pairs(engine, &[(0, 1)]).unwrap(),
        Error::DestinationShared,
        |d: &mut TensorMap<_, f64>| {
            let clone = d.clone();
            let result = s.trace_pairs_into(engine, &[(0, 1)], d, 1.0, 0.5);
            drop(clone);
            result
        }
    );
    // A lazy-adjoint source is read through its parent's storage, so a
    // destination that is that parent aliases it (#1870). The adjoint of
    // `[v] ← [v]` has the destination's own space: only the alias rejects.
    rejects!(
        host_snap,
        "permute_into lazy source aliasing its parent",
        sentinel!(&rt, t),
        Error::InvalidArgument(_),
        |d: &mut TensorMap<_, f64>| {
            let source = d.adjoint(engine).unwrap();
            source.permute_into(engine, &[0], &[1], d, 1.0, 0.5)
        }
    );
    // A compact source is admitted (#1870), so its destination checks run:
    // here the space, on every leg dualized.
    let (_, w) = legs();
    let dual = TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&w], [&w], 1).unwrap();
    rejects!(
        host_snap,
        "permute_into compact source, wrong space",
        sentinel!(&rt, dual),
        SpaceMismatch,
        |d: &mut _| s.permute_into(engine, &[0], &[1], d, 1.0, 0.5)
    );
}

/// Checked Generic tree-transform `_into` (#1870): every destination class
/// in the multiplicity-free order — runtime, rule, then (for the destination
/// kind, alias and space) the same `Error` in [`GenericTensorError::Facade`],
/// space before sharing.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_transform_into_leaves_a_rejected_destination_bit_identical() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    use symtn::sector::LieGroupFusionRule;
    use symtn::typed::GenericTensorError;

    let rt = Runtime::builder().build().unwrap();
    let other_rt = Runtime::builder().build().unwrap();
    let su2 =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap());
    let v = GradedSpace::try_new(Arc::clone(&su2), [(vec![0], 2), (vec![1], 1)]).unwrap();
    let w = v.try_dual().unwrap();
    let t = TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&v, &w], [&v], 1).unwrap();
    let permuted = t.permute(engine, &[1], &[2, 0]).unwrap();
    let fresh = |rt: &Runtime| sentinel!(rt, permuted);
    // The same shape under SU(3)'s identity: only the rule differs.
    let su3 =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let su3_leg = |dual: bool| {
        let leg =
            GradedSpace::try_new(Arc::clone(&su3), [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
        if dual {
            leg.try_dual().unwrap()
        } else {
            leg
        }
    };
    let foreign_rule = {
        let mut next = 0u64;
        TensorMap::<_, f64>::from_subblock_fn(
            engine,
            &rt,
            [&su3_leg(true)],
            [&su3_leg(false), &su3_leg(false)],
            |_, _| {
                next += 1;
                sentinel_value(next)
            },
        )
        .unwrap()
    };
    let wrong = {
        let mut next = 0u64;
        TensorMap::<_, f64>::from_subblock_fn(engine, &rt, [&v], [&v, &w], |_, _| {
            next += 1;
            sentinel_value(next)
        })
        .unwrap()
    };
    let into = |engine: &mut dyn symtn::typed::Engine,
                d: &mut TensorMap<LieGroupFusionRule, f64>| {
        t.permute_into(engine, &[1], &[2, 0], d, 1.0, 0.5)
    };

    t.permute_into(engine, &[1], &[2, 0], &mut fresh(&rt), 1.0, 0.5)
        .unwrap();
    rejects!(
        host_snap,
        "checked permute_into runtime",
        fresh(&other_rt),
        GenericTensorError::Facade(Error::RuntimeMismatch),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        host_snap,
        "checked permute_into rule",
        foreign_rule,
        GenericTensorError::Facade(Error::RuleMismatch),
        |d: &mut _| into(engine, d)
    );
    let compact = TensorMap::<_, f64>::diagonal(
        engine,
        &rt,
        &v,
        [
            SectorSpectrum {
                sector: vec![0],
                values: vec![f64::NAN, 2.0],
            },
            SectorSpectrum {
                sector: vec![1],
                values: vec![3.0],
            },
        ],
    )
    .unwrap();
    let compact_snap = |d: &TensorMap<LieGroupFusionRule, f64>| {
        d.diagview(&mut symtn_dense::CpuEngine::serial())
            .unwrap()
            .iter()
            .flat_map(|entry| entry.values.iter().map(|x| x.to_bits()))
            .collect::<Vec<_>>()
    };
    rejects!(
        compact_snap,
        "checked permute_into compact destination",
        compact,
        GenericTensorError::Facade(Error::InvalidArgument(_)),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        |d: &TensorMap<LieGroupFusionRule, f64>| {
            host_snap(
                &d.materialize(&mut symtn_dense::CpuEngine::serial())
                    .unwrap(),
            )
            .0
        },
        "checked permute_into lazy destination",
        permuted.adjoint(engine).unwrap(),
        GenericTensorError::Facade(Error::InvalidArgument(_)),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        host_snap,
        "checked permute_into alias",
        t.clone(),
        GenericTensorError::Facade(Error::InvalidArgument(_)),
        |d: &mut TensorMap<LieGroupFusionRule, f64>| {
            let source = d.clone();
            source.permute_into(engine, &[0, 1], &[2], d, 1.0, 0.5)
        }
    );
    rejects!(
        host_snap,
        "checked permute_into space",
        wrong.clone(),
        GenericTensorError::Facade(Error::Operation(operation))
            if matches!(**operation, symtn::typed::OperationError::SpaceMismatch { .. }),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        host_snap,
        "checked permute_into space before shared",
        wrong,
        GenericTensorError::Facade(Error::Operation(operation))
            if matches!(**operation, symtn::typed::OperationError::SpaceMismatch { .. }),
        |d: &mut TensorMap<LieGroupFusionRule, f64>| {
            let clone = d.clone();
            let result = into(engine, d);
            drop(clone);
            result
        }
    );
    rejects!(
        host_snap,
        "checked permute_into shared",
        fresh(&rt),
        GenericTensorError::Facade(Error::DestinationShared),
        |d: &mut TensorMap<LieGroupFusionRule, f64>| {
            let clone = d.clone();
            let result = into(engine, d);
            drop(clone);
            result
        }
    );
}

/// Checked Generic `contract_into` rejects every malformed destination class
/// in the multiplicity-free order (runtime, rule, destination kind, alias
/// including a lazy operand's parent, the contraction's own errors, space
/// before sharing), the same `Error` in [`GenericTensorError::Facade`].
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_contract_into_leaves_a_rejected_destination_bit_identical() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    use symtn::sector::LieGroupFusionRule;
    use symtn::typed::{ContractSpec, GenericTensorError};

    let rt = Runtime::builder().build().unwrap();
    let other_rt = Runtime::builder().build().unwrap();
    let su2 =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap());
    let v = GradedSpace::try_new(Arc::clone(&su2), [(vec![0], 2), (vec![1], 1)]).unwrap();
    let w = v.try_dual().unwrap();
    let t = TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&v, &w], [&v], 1).unwrap();
    let r = TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&v], [&w], 2).unwrap();
    let undual = TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&w], [&w], 4).unwrap();
    let spec = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[0, 2],
        domain: &[1],
    };
    let contracted = t.contract(engine, &r, &spec).unwrap();
    let fresh = |rt: &Runtime| sentinel!(rt, contracted);
    let su3 =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let su3_leg = GradedSpace::try_new(su3, [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let foreign_rule = sentinel!(
        &rt,
        TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&su3_leg, &su3_leg], [&su3_leg], 3)
            .unwrap()
    );
    let wrong = sentinel!(&rt, t);
    let into = |engine: &mut dyn symtn::typed::Engine,
                d: &mut TensorMap<LieGroupFusionRule, f64>| {
        t.contract_into(engine, &r, &spec, d, 1.0, 0.5)
    };

    t.contract_into(engine, &r, &spec, &mut fresh(&rt), 1.0, 0.5)
        .unwrap();
    rejects!(
        host_snap,
        "checked contract_into runtime",
        fresh(&other_rt),
        GenericTensorError::Facade(Error::RuntimeMismatch),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        host_snap,
        "checked contract_into rule",
        foreign_rule,
        GenericTensorError::Facade(Error::RuleMismatch),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        |d: &TensorMap<LieGroupFusionRule, f64>| {
            host_snap(
                &d.materialize(&mut symtn_dense::CpuEngine::serial())
                    .unwrap(),
            )
            .0
        },
        "checked contract_into lazy destination",
        contracted.adjoint(engine).unwrap(),
        GenericTensorError::Facade(Error::InvalidArgument(_)),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        host_snap,
        "checked contract_into alias",
        fresh(&rt),
        GenericTensorError::Facade(Error::InvalidArgument(_)),
        |d: &mut TensorMap<LieGroupFusionRule, f64>| {
            let lhs = d.clone();
            lhs.contract_into(engine, &r, &spec, d, 1.0, 0.5)
        }
    );
    rejects!(
        host_snap,
        "checked contract_into alias of a lazy operand's parent",
        fresh(&rt),
        GenericTensorError::Facade(Error::InvalidArgument(_)),
        |d: &mut TensorMap<LieGroupFusionRule, f64>| {
            let lazy = d.adjoint(engine).unwrap();
            t.contract_into(engine, &lazy, &spec, d, 1.0, 0.5)
        }
    );
    rejects!(
        host_snap,
        "checked contract_into legs not dual",
        fresh(&rt),
        _,
        |d: &mut TensorMap<LieGroupFusionRule, f64>| t
            .contract_into(engine, &undual, &spec, d, 1.0, 0.5)
    );
    rejects!(
        host_snap,
        "checked contract_into space",
        wrong.clone(),
        GenericTensorError::Facade(Error::Operation(operation))
            if matches!(**operation, symtn::typed::OperationError::SpaceMismatch { .. }),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        host_snap,
        "checked contract_into space before shared",
        wrong,
        GenericTensorError::Facade(Error::Operation(operation))
            if matches!(**operation, symtn::typed::OperationError::SpaceMismatch { .. }),
        |d: &mut TensorMap<LieGroupFusionRule, f64>| {
            let clone = d.clone();
            let result = into(engine, d);
            drop(clone);
            result
        }
    );
    rejects!(
        host_snap,
        "checked contract_into shared",
        fresh(&rt),
        GenericTensorError::Facade(Error::DestinationShared),
        |d: &mut TensorMap<LieGroupFusionRule, f64>| {
            let clone = d.clone();
            let result = into(engine, d);
            drop(clone);
            result
        }
    );
}

/// The checked Generic `trace_pairs_into` rejects every misuse the
/// multiplicity-free one does, in its order (runtime, rule, destination
/// kind, alias including a lazy source's parent, then TensorKit's
/// `space(tdst)` before the pair duality, space before sharing), the same
/// `Error` in [`GenericTensorError::Facade`]; the duality is eager's own
/// staging error.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_trace_pairs_into_leaves_a_rejected_destination_bit_identical() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    use symtn::sector::LieGroupFusionRule;
    use symtn::typed::GenericTensorError;

    let rt = Runtime::builder().build().unwrap();
    let other_rt = Runtime::builder().build().unwrap();
    let su2 =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap());
    let v = GradedSpace::try_new(Arc::clone(&su2), [(vec![0], 2), (vec![1], 1)]).unwrap();
    let w = GradedSpace::try_new(Arc::clone(&su2), [(vec![0], 1), (vec![2], 2)]).unwrap();
    let t = TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&v, &w], [&v], 1).unwrap();
    // The same untraced leg, but the pair joins `v` with `w`.
    let undual = TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&v, &w], [&w], 4).unwrap();
    let traced = t.trace_pairs(engine, &[(0, 2)]).unwrap();
    let fresh = |rt: &Runtime| sentinel!(rt, traced);
    let su3 =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let su3_leg = GradedSpace::try_new(su3, [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let foreign_rule = sentinel!(
        &rt,
        TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&su3_leg], [&su3_leg], 3).unwrap()
    );
    let wrong = sentinel!(
        &rt,
        TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&v], [&v], 5).unwrap()
    );
    let into = |engine: &mut dyn symtn::typed::Engine,
                d: &mut TensorMap<LieGroupFusionRule, f64>| {
        t.trace_pairs_into(engine, &[(0, 2)], d, 1.0, 0.5)
    };
    let shared = |engine: &mut dyn symtn::typed::Engine,
                  d: &mut TensorMap<LieGroupFusionRule, f64>| {
        let clone = d.clone();
        let result = into(engine, d);
        drop(clone);
        result
    };
    let space = |error: &GenericTensorError<_>| {
        matches!(error, GenericTensorError::Facade(Error::Operation(operation))
            if matches!(**operation, symtn::typed::OperationError::SpaceMismatch { .. }))
    };

    t.trace_pairs_into(engine, &[(0, 2)], &mut fresh(&rt), 1.0, 0.5)
        .unwrap();
    rejects!(
        host_snap,
        "checked trace_pairs_into runtime",
        fresh(&other_rt),
        GenericTensorError::Facade(Error::RuntimeMismatch),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        host_snap,
        "checked trace_pairs_into rule",
        foreign_rule,
        GenericTensorError::Facade(Error::RuleMismatch),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        host_snap,
        "checked trace_pairs_into leg out of range",
        fresh(&rt),
        GenericTensorError::Facade(Error::Operation(_)),
        |d: &mut TensorMap<LieGroupFusionRule, f64>| t.trace_pairs_into(
            engine,
            &[(0, 9)],
            d,
            1.0,
            0.5
        )
    );
    rejects!(
        |d: &TensorMap<LieGroupFusionRule, f64>| {
            host_snap(
                &d.materialize(&mut symtn_dense::CpuEngine::serial())
                    .unwrap(),
            )
            .0
        },
        "checked trace_pairs_into lazy destination",
        traced.adjoint(engine).unwrap(),
        GenericTensorError::Facade(Error::InvalidArgument(_)),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        host_snap,
        "checked trace_pairs_into alias of a lazy source's parent",
        sentinel!(&rt, t),
        GenericTensorError::Facade(Error::InvalidArgument(_)),
        |d: &mut TensorMap<LieGroupFusionRule, f64>| {
            let lazy = d.adjoint(engine).unwrap();
            lazy.trace_pairs_into(engine, &[(0, 1)], d, 1.0, 0.5)
        }
    );
    rejects!(
        host_snap,
        "checked trace_pairs_into legs not dual",
        fresh(&rt),
        GenericTensorError::Plan(_),
        |d: &mut TensorMap<LieGroupFusionRule, f64>| {
            let into = undual.trace_pairs_into(engine, &[(0, 2)], d, 1.0, 0.5);
            // Eager's own staging error, in the same wrapper.
            let eager = undual.trace_pairs(engine, &[(0, 2)]).unwrap_err();
            assert_eq!(format!("{into:?}"), format!("{:?}", Err::<(), _>(eager)));
            into
        }
    );
    rejects!(
        host_snap,
        "checked trace_pairs_into space before legs not dual",
        wrong.clone(),
        error if space(error),
        |d: &mut TensorMap<LieGroupFusionRule, f64>| undual.trace_pairs_into(engine, &[(0, 2)], d, 1.0, 0.5)
    );
    rejects!(
        host_snap,
        "checked trace_pairs_into space",
        wrong.clone(),
        error if space(error),
        |d: &mut _| into(engine, d)
    );
    rejects!(
        host_snap,
        "checked trace_pairs_into space before shared",
        wrong,
        error if space(error),
        |d: &mut _| shared(engine, d)
    );
    rejects!(
        host_snap,
        "checked trace_pairs_into shared",
        fresh(&rt),
        GenericTensorError::Facade(Error::DestinationShared),
        |d: &mut _| shared(engine, d)
    );
}

/// `scale_assign` takes the destination rule: in place on the receiver's own
/// storage, `DestinationShared` (receiver untouched) when it is shared.
#[test]
fn scale_assign_is_in_place_or_refuses_a_shared_receiver() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = Runtime::builder().build().unwrap();
    let (v, w) = legs();
    let factor = Complex64::new(0.5, -2.0);

    // Dense: every element times the factor, the payload pointer kept.
    let original =
        TensorMap::<_, Complex64>::rand_with_seed(engine, &rt, [&v], [&v, &w], 1).unwrap();
    let mut dense = original.materialize(engine).unwrap();
    let pointer = dense.dense_data().unwrap().as_ptr();
    dense.scale_assign(engine, factor).unwrap();
    assert_eq!(dense.dense_data().unwrap().as_ptr(), pointer);
    let expected: Vec<_> = original
        .dense_data()
        .unwrap()
        .iter()
        .map(|&x| x * factor)
        .collect();
    assert_eq!(dense.dense_data().unwrap(), expected.as_slice());

    // Shared: refused, both handles unchanged.
    let clone = dense.clone();
    let before = dense.dense_data().unwrap().to_vec();
    assert_eq!(
        dense.scale_assign(engine, factor),
        Err(Error::DestinationShared)
    );
    assert_eq!(dense.dense_data().unwrap(), before.as_slice());
    assert_eq!(clone.dense_data().unwrap(), before.as_slice());
    drop(clone);

    // A unique body whose payload is shared: a unit-leg insertion reuses the
    // payload allocation under a new body.
    let before = dense.dense_data().unwrap().to_vec();
    let mut unit = dense
        .insert_unit(engine, 0, Side::Domain, Duality::Plain)
        .unwrap();
    assert_eq!(
        unit.dense_data().unwrap().as_ptr(),
        dense.dense_data().unwrap().as_ptr()
    );
    assert_eq!(
        unit.scale_assign(engine, factor),
        Err(Error::DestinationShared)
    );
    assert_eq!(unit.dense_data().unwrap(), before.as_slice());
    assert_eq!(dense.dense_data().unwrap(), before.as_slice());
    drop(unit);

    // Compact diagonal: scaled in place, still compact.
    let values = [vec![1.0, -2.0], vec![3.0], vec![0.25, 5.0]];
    let sectors = [U1Irrep::new(0), U1Irrep::new(1), U1Irrep::new(-1)];
    let spectra = || {
        sectors
            .iter()
            .zip(&values)
            .map(|(&sector, values)| SectorSpectrum {
                sector,
                values: values.iter().map(|&x| Complex64::new(x, 0.0)).collect(),
            })
    };
    let mut compact = TensorMap::<_, Complex64>::diagonal(engine, &rt, &v, spectra()).unwrap();
    compact.scale_assign(engine, factor).unwrap();
    let got = compact.diagview(engine).unwrap();
    assert_eq!(got.len(), sectors.len());
    for (&sector, values) in sectors.iter().zip(&values) {
        let entry = got.iter().find(|entry| entry.sector == sector).unwrap();
        let want: Vec<_> = values
            .iter()
            .map(|&x| Complex64::new(x, 0.0) * factor)
            .collect();
        assert_eq!(entry.values, want);
    }
    let clone = compact.clone();
    assert_eq!(
        compact.scale_assign(engine, factor),
        Err(Error::DestinationShared)
    );
    assert_eq!(compact.diagview(engine).unwrap(), got);
    drop(clone);

    // Lazy adjoint: stays lazy; its logical values are `factor * t^H`.
    let parent = TensorMap::<_, Complex64>::rand_with_seed(engine, &rt, [&v], [&v, &w], 2).unwrap();
    let logical = parent.adjoint(engine).unwrap().materialize(engine).unwrap();
    let mut lazy = parent.adjoint(engine).unwrap();
    // The parent handle still shares the storage.
    let before = parent.dense_data().unwrap().to_vec();
    assert_eq!(
        lazy.scale_assign(engine, factor),
        Err(Error::DestinationShared)
    );
    assert_eq!(parent.dense_data().unwrap(), before.as_slice());
    drop(parent);
    // A cloned lazy view shares the adjoint header.
    let view = lazy.clone();
    assert_eq!(
        lazy.scale_assign(engine, factor),
        Err(Error::DestinationShared)
    );
    assert_eq!(
        lazy.materialize(engine).unwrap().dense_data().unwrap(),
        logical.dense_data().unwrap()
    );
    drop(view);
    lazy.scale_assign(engine, factor).unwrap();
    assert!(lazy.dense_data().is_err(), "a lazy adjoint stays lazy");
    let got = lazy.materialize(engine).unwrap();
    let want: Vec<_> = logical
        .dense_data()
        .unwrap()
        .iter()
        .map(|&x| x * factor)
        .collect();
    for (got, want) in got.dense_data().unwrap().iter().zip(&want) {
        assert!(
            (got - want).norm() <= 1e-14 * want.norm().max(1.0),
            "{got} != {want}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn every_device_into_leaves_a_rejected_destination_bit_identical() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    use symtn::typed::CudaStorage;

    let rt = Runtime::builder().build().unwrap();
    let other = Runtime::builder().build().unwrap();
    fn device_snap<R>(
        cuda: &mut dyn symtn::typed::Engine,
        t: &TensorMap<R, f64, CudaStorage<f64>>,
    ) -> Vec<u64> {
        t.to_host(cuda)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect()
    }
    let lift =
        |e: &mut dyn symtn::typed::Engine, x: &TensorMap<U1FusionRule, f64>| x.to_cuda(e).unwrap();
    u1_suite!(engine, cuda, &rt, &other, lift, device_snap);
    let lift =
        |e: &mut dyn symtn::typed::Engine, x: &TensorMap<ZNFusionRule, f64>| x.to_cuda(e).unwrap();
    rule_suite!(engine, cuda, &rt, lift, device_snap);

    // A lazy-adjoint `axpby_into` source is admitted on the device (#1874);
    // the destination checks still reject a shared destination untouched.
    let (v, w) = legs();
    let t = TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&v], [&v, &w], 1).unwrap();
    let lazy = t.to_cuda(cuda).unwrap().adjoint(cuda).unwrap();
    let like = t.adjoint(engine).unwrap().materialize(engine).unwrap();
    rejects!(
        |dst: &_| device_snap(cuda, dst),
        "device axpby_into lazy source, shared dst",
        sentinel!(&rt, like).to_cuda(cuda).unwrap(),
        Error::DestinationShared,
        |d: &mut _| {
            let clone = d.clone();
            let result = lazy.axpby_into(cuda, d, 1.0, 0.5);
            drop(clone);
            result
        }
    );

    // A lazy-adjoint device destination: its parent's bytes are its own.
    let square = TensorMap::<_, f64>::rand_with_seed(engine, &rt, [&v], [&v], 2).unwrap();
    let device_square = square.to_cuda(cuda).unwrap();
    let parent = sentinel!(&rt, square).to_cuda(cuda).unwrap();
    let lazy_snap = |e: &mut dyn symtn::typed::Engine,
                     _: &TensorMap<U1FusionRule, f64, CudaStorage<f64>>| {
        device_snap(e, &parent)
    };
    let unit = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    rejects!(
        |dst: &_| lazy_snap(cuda, dst),
        "device permute_into lazy dst",
        parent.adjoint(cuda).unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| device_square.permute_into(cuda, &[0], &[1], d, 1.0, 0.5)
    );
    rejects!(
        |dst: &_| lazy_snap(cuda, dst),
        "device trace_pairs_into lazy dst",
        parent.adjoint(cuda).unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| device_square.trace_pairs_into(cuda, &[], d, 1.0, 0.5)
    );
    rejects!(
        |dst: &_| lazy_snap(cuda, dst),
        "device axpby_into lazy dst",
        parent.adjoint(cuda).unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| device_square.axpby_into(cuda, d, 1.0, 0.5)
    );
    rejects!(
        |dst: &_| lazy_snap(cuda, dst),
        "device contract_into lazy dst",
        parent.adjoint(cuda).unwrap(),
        Error::InvalidArgument(_),
        |d: &mut _| device_square.contract_into(cuda, &device_square, &unit, d, 1.0, 0.5)
    );
}
