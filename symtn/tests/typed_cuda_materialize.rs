//! Real-device gate for `materialize` on CUDA tensors (#1545).
//!
//! The Host `materialize` is checked against independent oracles in
//! `typed_materialize.rs`. Here the device result must equal it bit for bit
//! (each entry is `1 * [conj] x`), allocate exactly one device payload with no
//! download, and own that payload: the device `*_into`, which
//! rejects a destination aliasing its source, accepts it, and the input is
//! unchanged.
//!
//! Run with `cargo test -p symtn --features cuda,cpu-faer --test \
//! typed_cuda_materialize -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use symtn::expert::cuda_transfer_stats;
use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::__network::NetworkReuseClass;
use symtn::typed::{GradedSpace, TensorMap};
use symtn::typed::{Runtime, TensorScalar};

/// Value classes compared exactly: each real or imaginary part is its bit
/// pattern, except that every NaN is one class. Tenferro's `conj` negates
/// with a float op and cuTENSOR's copy may canonicalize a NaN, so only
/// finite values, infinities and signed zeros are promised bit for bit.
trait Bits: TensorScalar + Copy {
    fn parts(self) -> [Option<u64>; 2];
    fn two() -> Self;
}

fn part_f64(value: f64) -> Option<u64> {
    (!value.is_nan()).then(|| value.to_bits())
}

fn part_f32(value: f32) -> Option<u64> {
    (!value.is_nan()).then(|| u64::from(value.to_bits()))
}

impl Bits for f64 {
    fn parts(self) -> [Option<u64>; 2] {
        [part_f64(self), Some(0)]
    }
    fn two() -> Self {
        2.0
    }
}

impl Bits for f32 {
    fn parts(self) -> [Option<u64>; 2] {
        [part_f32(self), Some(0)]
    }
    fn two() -> Self {
        2.0
    }
}

impl Bits for Complex64 {
    fn parts(self) -> [Option<u64>; 2] {
        [part_f64(self.re), part_f64(self.im)]
    }
    fn two() -> Self {
        Complex64::new(2.0, 0.0)
    }
}

impl Bits for Complex32 {
    fn parts(self) -> [Option<u64>; 2] {
        [part_f32(self.re), part_f32(self.im)]
    }
    fn two() -> Self {
        Complex32::new(2.0, 0.0)
    }
}

fn bits<D: Bits>(data: &[D]) -> Vec<[Option<u64>; 2]> {
    data.iter().map(|value| value.parts()).collect()
}

/// Device `materialize` of `host.to_cuda()` and of its lazy adjoint against
/// the Host results: bit-equal, one fresh payload, no payload upload, no
/// download.
fn check<R, D>(host: &TensorMap<R, D>, what: &str)
where
    R: symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec
        + symtn::sector::TypedSectorAdmission,
    R::Mode: symtn::typed::TypedAdjointSpace<R>
        + symtn::typed::TypedTensorConstructionDispatch<R, D>
        + symtn::typed::TypedTensorTransformDispatch<R, D>,
    D: Bits + symtn::typed::CudaPayload,
{
    assert!(host.subblock_count() >= 2, "{what}: multi-block fixture");
    let host_lazy = host.adjoint().unwrap();
    let device = host.to_cuda().unwrap();
    let lazy = device.adjoint().unwrap();
    assert!(
        symtn::typed::__network::network_reuse_class(&lazy, false)
            == NetworkReuseClass::LazyAdjoint
    );

    let len = host.dense_data().unwrap().len() as u64;
    // A chained input: an owned device copy (a `[len, 1]` gather output, not
    // the flat upload) and its lazy adjoint.
    let rechained = device.materialize().unwrap().adjoint().unwrap();
    // (input, Host expected, device allocations, upload bytes). The counters
    // charge every upload as an allocation too. Owned and real adjoint: the
    // payload plus the gather's 8-byte member index. Complex adjoint: the
    // gathered payload, its coordinate table (`8 * rank * len` bytes for a
    // buffer of rank 1 or 2) and the conjugated payload.
    let (adjoint_allocs, flat_bytes, chained_bytes) = if D::IS_COMPLEX {
        (3, 8 * len, 16 * len)
    } else {
        (2, 8, 8)
    };
    for (input, expected, allocs, bytes, case) in [
        (&device, host.materialize().unwrap(), 2, 8, "owned"),
        (
            &lazy,
            host_lazy.materialize().unwrap(),
            adjoint_allocs,
            flat_bytes,
            "adjoint",
        ),
        (
            &rechained,
            host_lazy.materialize().unwrap(),
            adjoint_allocs,
            chained_bytes,
            "chained adjoint",
        ),
    ] {
        let before = cuda_transfer_stats();
        let owned = input.materialize().unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(
            after.device_allocs - before.device_allocs,
            allocs,
            "{what} {case}"
        );
        // The gather's index table is the only upload.
        assert_eq!(after.h2d_calls - before.h2d_calls, 1, "{what} {case}");
        assert_eq!(after.h2d_bytes - before.h2d_bytes, bytes, "{what} {case}");
        assert_eq!(
            after.d2h_calls, before.d2h_calls,
            "{what} {case}: no download"
        );

        assert!(
            symtn::typed::__network::network_reuse_class(&owned, false)
                == NetworkReuseClass::OwnedDense
        );
        assert_eq!(owned.placement(), input.placement(), "{what} {case}");
        assert!(std::ptr::eq(owned.provider(), input.provider()));
        let back = owned.to_host().unwrap();
        assert_eq!(
            back.codomain(),
            expected.codomain(),
            "{what} {case}: codomain"
        );
        assert_eq!(back.domain(), expected.domain(), "{what} {case}: domain");
        assert_eq!(
            bits(back.dense_data().unwrap()),
            bits(expected.dense_data().unwrap()),
            "{what} {case}: payload"
        );
    }
    // The owned copy is the input's bits.
    assert_eq!(
        bits(
            device
                .materialize()
                .unwrap()
                .to_host()
                .unwrap()
                .dense_data()
                .unwrap()
        ),
        bits(host.dense_data().unwrap()),
        "{what}: owned bits"
    );
    // The adjoint of the lazy adjoint materializes to the parent.
    let twice = lazy
        .adjoint()
        .unwrap()
        .materialize()
        .unwrap()
        .to_host()
        .unwrap();
    assert_eq!(twice.codomain(), host.codomain(), "{what}: adjoint twice");
    assert_eq!(twice.domain(), host.domain(), "{what}: adjoint twice");
    assert_eq!(
        bits(twice.dense_data().unwrap()),
        bits(host.dense_data().unwrap()),
        "{what}: adjoint twice"
    );

    // Independence: the device overwrite rejects a destination that aliases
    // its source, so accepting the copy proves a fresh payload, and the
    // source keeps its values.
    let codomain: Vec<usize> = (0..host.codomain_rank()).collect();
    let domain: Vec<usize> = (host.codomain_rank()..host.rank()).collect();
    let mut destination = device.materialize().unwrap();
    device
        .permute_into(
            &codomain,
            &domain,
            &mut destination,
            D::two(),
            D::from_real(0.0),
        )
        .unwrap();
    assert_eq!(
        bits(device.to_host().unwrap().dense_data().unwrap()),
        bits(host.dense_data().unwrap()),
        "{what}: input unchanged"
    );
    let mut alias = device.clone();
    assert!(device
        .permute_into(&codomain, &domain, &mut alias, D::two(), D::from_real(0.0))
        .is_err());
}

fn fixtures<R, D>(runtime: &Runtime, leg: &GradedSpace<R>, what: &str)
where
    R: symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec
        + symtn::sector::TypedSectorAdmission,
    R::Mode: symtn::typed::TypedAdjointSpace<R>
        + symtn::typed::TypedTensorConstructionDispatch<R, D>
        + symtn::typed::TypedTensorTransformDispatch<R, D>,
    D: Bits + symtn::typed::CudaPayload,
{
    let dual = leg.try_dual().unwrap();
    let square: TensorMap<R, D> =
        TensorMap::rand_with_seed(runtime, [leg, &dual], [&dual, leg], 1545).unwrap();
    check(&square, &format!("{what} 2<-2"));
    // nout != nin: the adjoint swaps unequal codomain and domain ranks.
    let uneven: TensorMap<R, D> =
        TensorMap::rand_with_seed(runtime, [leg], [&dual, leg], 1546).unwrap();
    check(&uneven, &format!("{what} 1<-2"));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_materialize_matches_host_with_one_fresh_payload() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 3),
        ],
    )
    .unwrap();
    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
            (SU2Irrep::from_twice_spin(2), 2),
        ],
    )
    .unwrap();
    let fz2_u1 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 2),
        ],
    )
    .unwrap();
    fixtures::<_, f64>(&runtime, &u1, "U1/f64");
    fixtures::<_, Complex64>(&runtime, &u1, "U1/c64");
    fixtures::<_, f64>(&runtime, &su2, "SU2/f64");
    fixtures::<_, Complex64>(&runtime, &su2, "SU2/c64");
    fixtures::<_, f64>(&runtime, &fz2_u1, "fZ2xU1/f64");
    fixtures::<_, Complex64>(&runtime, &fz2_u1, "fZ2xU1/c64");
    fixtures::<_, f32>(&runtime, &u1, "U1/f32");
    fixtures::<_, Complex32>(&runtime, &u1, "U1/c32");
    fixtures::<_, Complex32>(&runtime, &su2, "SU2/c32");

    // Values a unit-operand contraction would change: complex infinities,
    // and signed zeros in either part. The owned copy and the adjoint
    // must match the Host bit for bit.
    let specials = [
        Complex64::new(f64::INFINITY, 0.0),
        Complex64::new(0.0, f64::NEG_INFINITY),
        Complex64::new(-0.0, 1.0),
        Complex64::new(2.0, -0.0),
        Complex64::new(-0.0, -0.0),
        Complex64::new(f64::INFINITY, f64::INFINITY),
        Complex64::new(-3.5, 0.25),
    ];
    let dual = u1.try_dual().unwrap();
    let mut next = 0;
    let special: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &dual], [&dual, &u1], |_, _| {
            next += 1;
            specials[next % specials.len()]
        })
        .unwrap();
    check(&special, "U1/c64 specials");
    // The Host oracle itself conjugates exactly.
    let host_adjoint = special.adjoint().unwrap().materialize().unwrap();
    assert!(host_adjoint
        .dense_data()
        .unwrap()
        .iter()
        .any(|v| v.re.is_infinite() && v.im == 0.0 && v.im.is_sign_negative()));

    // A NaN imaginary (and real) part stays NaN; only its bits are free.
    let nans = [
        Complex64::new(1.5, f64::NAN),
        Complex64::new(f64::NAN, -2.0),
        Complex64::new(-0.0, 0.5),
        Complex64::new(3.0, -f64::NAN),
    ];
    let mut next = 0;
    let nan: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &dual], [&dual, &u1], |_, _| {
            next += 1;
            nans[next % nans.len()]
        })
        .unwrap();
    check(&nan, "U1/c64 NaN");
    let device_adjoint = nan
        .to_cuda()
        .unwrap()
        .adjoint()
        .unwrap()
        .materialize()
        .unwrap()
        .to_host()
        .unwrap();
    assert!(device_adjoint
        .dense_data()
        .unwrap()
        .iter()
        .any(|v| v.im.is_nan() && v.re == 1.5));
    assert!(device_adjoint
        .dense_data()
        .unwrap()
        .iter()
        .any(|v| v.re.is_nan() && v.im == 2.0));
}
