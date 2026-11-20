//! Contract: a facade contraction records every block GEMM it issues, with
//! each job's m/k/n and its BLAS leading dimensions.

use symtn::expert::{gemm_trace, DenseDType, MatrixOp};
use symtn::typed::{Runtime, TensorMap};

use crate::legs::u1_leg;

#[test]
fn compose_records_every_block_gemm_with_its_leading_dimensions() {
    let _serial = crate::counting_alloc::serial();
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let v = u1_leg(&[(-1, 2), (0, 3), (1, 4)]);
    let w = u1_leg(&[(-1, 5), (0, 6), (1, 1)]);
    let x = u1_leg(&[(-1, 3), (0, 2), (1, 7)]);
    let lhs = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&v], [&w], 1).unwrap();
    let rhs = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&w], [&x], 2).unwrap();

    gemm_trace::start(1 << 10);
    lhs.compose(engine, &rhs).unwrap();
    let trace = gemm_trace::take();

    assert_eq!(trace.dropped_calls, 0);
    let mut mkn = Vec::new();
    for call in &trace.calls {
        assert_eq!(call.dtype, DenseDType::F64);
        for job in &call.jobs {
            let identity = |op| op == MatrixOp::Identity;
            assert_eq!(job.ldc, job.m);
            assert_eq!(job.lda, if identity(call.ops[0]) { job.m } else { job.k });
            assert_eq!(job.ldb, if identity(call.ops[1]) { job.k } else { job.n });
            mkn.push((job.m, job.k, job.n));
        }
    }
    mkn.sort_unstable();
    // Per coupled charge: (d_V, d_W) . (d_W, d_X).
    assert_eq!(mkn, [(2, 5, 3), (3, 6, 2), (4, 1, 7)]);
}
