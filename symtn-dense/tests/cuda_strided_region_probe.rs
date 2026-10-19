//! Device probe for the mode count the strided region accumulation behind
//! symtn's basic device operations (`cuda_region_axpby`) hands to cuTENSOR.
//!
//! `cuda_region_axpby` documents that it imposes no rank limit because Tenferro
//! accepted every mode count up to 72. No adapter-level test can reach a rank
//! that high (a realizable tensor needs extents of at least 2 on every
//! non-unit mode), so this file drives Tenferro directly with unit-padded
//! ranks and records where, if anywhere, the backend starts to reject.
//!
//! The device is the unit under test, so the test is `#[ignore]` like the rest
//! of the device suite.

#![cfg(feature = "cuda")]

use tenferro_gpu::cuda::{upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_tensor::{
    DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorDot, TensorRead, TensorScalar,
    TensorWrite,
};

use symtn_dense::CudaScalar;

fn upload(backend: &CudaBackend, data: Vec<f64>) -> Tensor {
    let len = data.len();
    let host = Tensor::from_vec_col_major(vec![len], data).expect("host tensor");
    upload_tensor(backend.runtime(), &host).expect("upload")
}

#[test]
#[ignore = "requires a real CUDA device"]
fn rank_eight_strided_region_is_accepted_and_the_mode_limit_is_logged() {
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0))
        .expect("CUDA device 0 must be available for the device probe");
    let src = upload(&backend, (0..64).map(|index| index as f64).collect());
    let ones = upload(&backend, vec![1.0]);
    let mut dst = upload(&backend, vec![0.0; 64]);

    // Ranks are padded with unit axes, so the payload stays at 8 elements and
    // only the mode count varies. The first rank that fails is the limit.
    let mut first_failure = None;
    for rank in [8usize, 16, 24, 32, 40, 48, 56, 63, 64, 65, 72] {
        let mut dims = vec![1usize; rank];
        dims[0] = 8;
        // `dst = 1 * src` over the region, as a contraction of the rank-N
        // source plus a trailing unit mode against a 1x1 ones operand.
        let mut view_dims = dims.clone();
        view_dims.push(1);
        let strides = vec![1isize; rank + 1];
        let lhs = f64::typed(&src).expect("source dtype").backend_region_view(
            view_dims.clone(),
            strides.clone(),
            0,
        );
        let rhs =
            f64::typed(&ones)
                .expect("ones dtype")
                .backend_region_view(vec![1, 1], vec![1, 1], 0);
        let out = f64::typed_mut(&mut dst)
            .expect("destination dtype")
            .backend_region_view_mut(view_dims, strides, 0);
        let config = DotGeneralConfig {
            lhs_contracting_dims: vec![rank],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: Vec::new(),
            rhs_batch_dims: Vec::new(),
        };
        let accumulation = DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: 1.0f64.contraction_scalar(),
            beta: 0.0f64.contraction_scalar(),
        };
        let result = match (lhs, rhs, out) {
            (Ok(lhs), Ok(rhs), Ok(out)) => backend.dot_general_read_into_accum(
                TensorRead::from_view(f64::tensor_view(lhs)),
                TensorRead::from_view(f64::tensor_view(rhs)),
                &config,
                accumulation,
                TensorWrite::from_view(f64::tensor_view_mut(out)),
            ),
            (lhs, rhs, out) => Err(lhs
                .err()
                .or(rhs.err())
                .or(out.err())
                .expect("one view failed")),
        };
        match result {
            Ok(()) => println!("rank {rank}: accepted"),
            Err(err) => {
                println!("rank {rank}: rejected with `{err}`");
                if first_failure.is_none() {
                    first_failure = Some((rank, err.to_string()));
                }
            }
        }
    }
    match &first_failure {
        Some((rank, text)) => println!("first rejected rank: {rank} (`{text}`)"),
        None => println!("no mode-count limit reached up to rank 72"),
    }
    // Every rank symtn can reach through a fusion tree is far below any
    // plausible cuTENSOR mode limit; the probe only records where it lies.
    assert!(
        first_failure.as_ref().map_or(usize::MAX, |(rank, _)| *rank) > 8,
        "rank 8 must be usable"
    );
}
