//! Typed allocation gates for the general (Padé) arm of `exp` (issue #780).
//!
//! The operation owns ten `max_c n_c²` matrix buffers and reuses them across
//! coupled sectors. Dense backends may allocate and free their own GEMM/solve
//! scratch on every call, so cumulative allocation slope is not a workspace
//! oracle: it changed with faer on macOS while peak live storage did not.
//! These gates therefore separate cumulative bytes, final-live output, and
//! peak-live scratch.

use std::hint::black_box;
use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Complex64, Runtime};
use symtn::typed::{GradedSpace, TensorMap};

use crate::counting_alloc;

#[derive(Clone, Copy, Debug)]
struct Sample {
    allocation_calls: u64,
    deallocation_calls: u64,
    allocated_bytes: u64,
    deallocated_bytes: u64,
    peak_live_bytes: u64,
    final_live_bytes: u64,
}

impl Sample {
    fn transient_peak_bytes(self) -> u64 {
        self.peak_live_bytes.saturating_sub(self.final_live_bytes)
    }
}

fn measure<T>(operation: impl FnOnce() -> T) -> Sample {
    let (output, allocs) = counting_alloc::measure(|| {
        let output = black_box(operation());
        black_box(&output);
        output
    });
    drop(output);
    assert!(
        allocs.live_bytes >= 0,
        "measurement released an untracked allocation"
    );
    Sample {
        allocation_calls: allocs.calls,
        deallocation_calls: allocs.frees,
        allocated_bytes: allocs.bytes,
        deallocated_bytes: allocs.freed_bytes,
        peak_live_bytes: allocs.peak_live_bytes,
        final_live_bytes: allocs.live_bytes as u64,
    }
}

fn u1_space(sectors: i32, order: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (0..sectors).map(|charge| (U1Irrep::new(charge), order)),
    )
    .unwrap()
}

fn fixture_f64(sectors: i32, order: usize, scale: f64) -> TensorMap<U1FusionRule, f64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = u1_space(sectors, order);
    TensorMap::from_subblock_fn(&runtime, [&space], [&space], |_, indices| {
        scale * (1.0 + indices[0] as f64 - 0.5 * indices[1] as f64)
    })
    .unwrap()
}

fn fixture_c64(sectors: i32, order: usize, scale: f64) -> TensorMap<U1FusionRule, Complex64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = u1_space(sectors, order);
    TensorMap::from_subblock_fn(&runtime, [&space], [&space], |_, indices| {
        scale
            * Complex64::new(
                1.0 + indices[0] as f64 - 0.5 * indices[1] as f64,
                0.25 + 0.125 * indices[0] as f64,
            )
    })
    .unwrap()
}

fn warmed_f64(tensor: &TensorMap<U1FusionRule, f64>) -> Sample {
    black_box(tensor.exp(&[0], &[1]).unwrap());
    measure(|| tensor.exp(&[0], &[1]).unwrap())
}

fn warmed_c64(tensor: &TensorMap<U1FusionRule, Complex64>) -> Sample {
    black_box(tensor.exp(&[0], &[1]).unwrap());
    measure(|| tensor.exp(&[0], &[1]).unwrap())
}

/// Ten matrix buffers are allocated once. Backend scratch may add to the live
/// peak, but adding sectors must add only result storage, not live scratch.
fn assert_one_reused_workspace(
    scalar: &str,
    order: usize,
    scalar_bytes: usize,
    two: Sample,
    four: Sample,
) {
    let block_bytes = (order * order * scalar_bytes) as f64;

    // Cumulative backend churn is linear in sector count and cancels here. The
    // intercept is the once-owned ten-buffer workspace plus small metadata and
    // the O(max_order) balancing vector.
    let intercept = 2.0 * two.allocated_bytes as f64 - four.allocated_bytes as f64;
    let workspace_blocks = intercept / block_bytes;
    assert!(
        (9.0..=11.0).contains(&workspace_blocks),
        "{scalar} order {order}: once-owned allocation is {workspace_blocks:.2} blocks; \
         expected ten workspace buffers; two={two:?}, four={four:?}"
    );

    // Output storage accounts for the sector-count-dependent final-live
    // difference: two extra sectors, hence two extra matrix blocks. Metadata is
    // fixed for this canonical U(1) layout.
    let output_growth =
        four.final_live_bytes.saturating_sub(two.final_live_bytes) as f64 / block_bytes;
    assert!(
        (1.95..=2.05).contains(&output_growth),
        "{scalar} order {order}: two added sectors retain {output_growth:.2} blocks; \
         two={two:?}, four={four:?}"
    );

    // Remove the live result from the peak. The remaining operation plus dense
    // backend scratch must be independent of the number of sequential sectors.
    let two_scratch = two.transient_peak_bytes() as f64 / block_bytes;
    let four_scratch = four.transient_peak_bytes() as f64 / block_bytes;
    assert!(
        (two_scratch - four_scratch).abs() <= 0.10,
        "{scalar} order {order}: transient peak changed with sector count \
         ({two_scratch:.2} vs {four_scratch:.2} blocks); two={two:?}, four={four:?}"
    );

    // Keep every recorded category live in failure diagnostics; cumulative
    // calls/bytes are deliberately not capped because backend algorithms may
    // allocate and free different scratch schedules at different orders.
    assert!(two.allocation_calls >= two.deallocation_calls);
    assert!(four.allocation_calls >= four.deallocation_calls);
    assert!(two.allocated_bytes >= two.deallocated_bytes);
    assert!(four.allocated_bytes >= four.deallocated_bytes);
}

#[test]
fn general_exp_workspace_is_largest_sector_sized_and_reused() {
    let _serial = crate::counting_alloc::serial();

    for (order, scale) in [(48usize, 1.0e-4), (48, 1.0e-2), (96, 1.0e-4)] {
        let two = fixture_f64(2, order, scale);
        let four = fixture_f64(4, order, scale);
        assert_one_reused_workspace(
            "f64",
            order,
            std::mem::size_of::<f64>(),
            warmed_f64(&two),
            warmed_f64(&four),
        );

        let two = fixture_c64(2, order, scale);
        let four = fixture_c64(4, order, scale);
        assert_one_reused_workspace(
            "c64",
            order,
            std::mem::size_of::<Complex64>(),
            warmed_c64(&two),
            warmed_c64(&four),
        );
    }
}

#[test]
fn typed_u1_general_exp_matches_the_upper_triangular_oracle() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        (0..3).map(|charge| (U1Irrep::new(charge), 2)),
    )
    .unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&space], [&space], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 0) => 1.0,
            (0, 1) => 2.0,
            (1, 1) => 3.0,
            _ => 0.0,
        }
    })
    .unwrap();

    let image = source.exp(&[0], &[1]).unwrap();
    let expected = [
        1.0_f64.exp(),
        0.0,
        3.0_f64.exp() - 1.0_f64.exp(),
        3.0_f64.exp(),
    ];
    for block in image.dense_data().unwrap().as_chunks::<4>().0 {
        for (&actual, &expected) in block.iter().zip(&expected) {
            assert!((actual - expected).abs() <= 1.0e-12 * expected.abs().max(1.0));
        }
    }
    assert!(std::ptr::eq(image.provider(), provider.as_ref()));
}
