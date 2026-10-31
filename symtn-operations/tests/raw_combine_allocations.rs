//! Allocation gate for the host raw combine actions routed through Strided's
//! raw kernels: `Copy`, `CopyScale` and `Axpy` stay allocation-free at every
//! fused rank up to `RAW_FUSED_RANK_LIMIT`, and above it symtn's own fused
//! walk keeps them allocation-free through raw rank 16.

use std::hint::black_box;

use num_complex::Complex64;
use symtn_operations::tensoradd_raw_strided_kernel;

use crate::counting_alloc;

#[test]
fn raw_combine_actions_allocate_nothing_at_any_rank() {
    let _serial = crate::counting_alloc::serial();
    // What: for ranks 1..=16 (eight above the fused limit) and a transposed
    // source, conjugating copy, scale and axpy allocate nothing on the
    // caller thread. Ranks above 8 stay unfused, so Strided's parallel
    // threshold never applies to them.
    let (alpha, zero, one) = (
        Complex64::new(0.5, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    );
    let mut zero_strides = Vec::new();
    for rank in 1..=16usize {
        let shape = vec![2usize; rank];
        let dst_strides: Vec<isize> = (0..rank).map(|axis| 1 << axis).collect();
        let src_strides: Vec<isize> = dst_strides.iter().rev().copied().collect();
        let len = 1 << rank;
        let src: Vec<Complex64> = (0..len)
            .map(|i| Complex64::new(i as f64, -(i as f64)))
            .collect();
        let mut dst = vec![zero; len];
        for conjugate in [false, true] {
            for (a, b) in [(one, zero), (alpha, zero), (alpha, one)] {
                let mut run = || {
                    tensoradd_raw_strided_kernel(
                        strided_kernel::Exec::serial(),
                        &mut zero_strides,
                        black_box(&mut dst),
                        black_box(&src),
                        &shape,
                        &dst_strides,
                        &src_strides,
                        0,
                        0,
                        conjugate,
                        a,
                        b,
                    )
                    .unwrap()
                };
                run();
                counting_alloc::start();
                run();
                let allocs = counting_alloc::stop();
                assert_eq!(allocs.calls, 0, "rank {rank} alpha {a} beta {b}");
            }
        }
    }
}

#[test]
fn parallel_copy_matches_oracle_without_caller_allocations() {
    let _serial = crate::counting_alloc::serial();
    // What: a transposed 256 x 512 copy (131072 elements, above Strided's
    // 32768-element parallel threshold), plain and conjugating, run inside an
    // explicit 4-thread engine so `CopyPlan` fans out on its token. Every element
    // equals the transposed (conjugated) source bit for bit, and the thread
    // running the call allocates nothing once warm.
    const THREADS: usize = 4;
    let (rows, cols) = (256usize, 512usize);
    let len = rows * cols;
    let shape = [rows, cols];
    let dst_strides = [1isize, rows as isize];
    let src_strides = [cols as isize, 1];
    let src: Vec<Complex64> = (0..len)
        .map(|i| Complex64::new(i as f64 + 0.25, -(i as f64) - 0.5))
        .collect();
    let mut pool = symtn_operations::host_pool::test_pool(THREADS);
    // Why not assert on one counted call: the pool's workers box their
    // OS sleep objects on first use, so a call can pay up to 2 * THREADS pool
    // start-up allocations whenever a worker has not parked or been woken yet;
    // no idle window guarantees that. Those boxes are made once per pool, so a
    // copy that itself allocates can never reach a zero run within that many
    // retries, while an allocation-free one must.
    for conjugate in [false, true] {
        symtn_operations::host_pool::take_host_pool_observations();
        symtn_operations::host_pool::observe_host_pools(true);
        let (dst, allocations) = pool.install(|engine| {
            let mut dst = vec![Complex64::new(0.0, 0.0); len];
            let mut zero_strides = Vec::new();
            let mut run = |dst: &mut [Complex64]| {
                symtn_operations::host_pool::with_exec(engine, |exec| {
                    tensoradd_raw_strided_kernel(
                        exec,
                        &mut zero_strides,
                        black_box(dst),
                        black_box(&src),
                        &shape,
                        &dst_strides,
                        &src_strides,
                        0,
                        0,
                        conjugate,
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                    )
                })
                .unwrap()
            };
            run(&mut dst);
            let mut allocations = Vec::new();
            while allocations.last() != Some(&0) && allocations.len() <= 2 * THREADS {
                dst.fill(Complex64::new(0.0, 0.0));
                counting_alloc::start();
                run(&mut dst);
                let allocs = counting_alloc::stop();
                allocations.push(allocs.calls as usize);
            }
            (dst, allocations)
        });
        symtn_operations::host_pool::observe_host_pools(false);
        // What: the copy ran as a strided region on the engine's full token.
        let observations = symtn_operations::host_pool::take_host_pool_observations();
        assert!(
            !observations.is_empty()
                && observations.iter().all(|o| o.site
                    == symtn_operations::host_pool::HostPoolSite::Strided
                    && o.token_budget == THREADS),
            "{observations:?}"
        );
        assert_eq!(
            allocations.last(),
            Some(&0),
            "conj {conjugate}: {allocations:?}"
        );
        for col in 0..cols {
            for row in 0..rows {
                let source = src[row * cols + col];
                let want = if conjugate { source.conj() } else { source };
                let got = dst[row + col * rows];
                assert_eq!(
                    (got.re.to_bits(), got.im.to_bits()),
                    (want.re.to_bits(), want.im.to_bits()),
                    "conj {conjugate} ({row}, {col})"
                );
            }
        }
    }
}

/// Logical axes `[4, 3, 2, 2, 3, 2, 2, 4]` with a padded destination and a
/// reversed source (nothing fuses): the native rank-8 layout, and the same
/// tensor at raw rank 12 with both extent-4 axes split into contiguous
/// `(2, 2)` pairs and two unit axes inserted.
fn native_and_raw_twelve() -> [(Vec<usize>, Vec<isize>, Vec<isize>); 2] {
    let logical = [4usize, 3, 2, 2, 3, 2, 2, 4];
    let mut dst = Vec::new();
    let mut stride = 1isize;
    for &dim in &logical {
        dst.push(stride);
        stride *= dim as isize + 1;
    }
    let mut src = vec![0isize; logical.len()];
    let mut stride = 1isize;
    for axis in (0..logical.len()).rev() {
        src[axis] = stride;
        stride *= logical[axis] as isize;
    }
    let mut raw = (Vec::new(), Vec::new(), Vec::new());
    for axis in 0..logical.len() {
        if logical[axis] == 4 {
            raw.0.extend([2, 2]);
            raw.1.extend([dst[axis], 2 * dst[axis]]);
            raw.2.extend([src[axis], 2 * src[axis]]);
        } else {
            raw.0.push(logical[axis]);
            raw.1.push(dst[axis]);
            raw.2.push(src[axis]);
        }
        if axis == 2 || axis == 5 {
            raw.0.push(1);
            raw.1.push(11);
            raw.2.push(3);
        }
    }
    [(logical.to_vec(), dst, src), raw]
}

#[test]
fn raw_rank_twelve_fusing_to_eight_allocates_like_native_rank_eight() {
    let _serial = crate::counting_alloc::serial();
    // What: a raw rank-12 layout that fuses to rank 8 makes the
    // same number of warm allocation calls on the caller thread as the
    // native rank-8 layout, for copy, scale, axpy and axpby, plain and
    // conjugating; both are allocation-free.
    let alpha = Complex64::new(0.5, -1.0);
    let (zero, one) = (Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
    let mut zero_strides = Vec::new();
    let src: Vec<Complex64> = (0..4608).map(|i| Complex64::new(i as f64, 0.5)).collect();
    let mut dst = vec![zero; 5 * 4 * 3 * 3 * 4 * 3 * 3 * 5];
    for conjugate in [false, true] {
        for (a, b) in [(one, zero), (alpha, zero), (alpha, one), (alpha, alpha)] {
            let calls = native_and_raw_twelve().map(|(shape, dst_strides, src_strides)| {
                let mut run = || {
                    tensoradd_raw_strided_kernel(
                        strided_kernel::Exec::serial(),
                        &mut zero_strides,
                        black_box(&mut dst),
                        black_box(&src),
                        &shape,
                        &dst_strides,
                        &src_strides,
                        0,
                        0,
                        conjugate,
                        a,
                        b,
                    )
                    .unwrap()
                };
                run();
                counting_alloc::start();
                run();
                counting_alloc::stop().calls
            });
            assert_eq!(calls[1], calls[0], "conj {conjugate} alpha {a} beta {b}");
            assert_eq!(calls[0], 0, "conj {conjugate} alpha {a} beta {b}");
        }
    }
}
