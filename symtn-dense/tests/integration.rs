//! Behavior tests neither observe nor reset process-global state and run concurrently.

use num_complex::{Complex32, Complex64};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

mod batch_route_values;
mod cuda_copy_region_alignment;
mod cuda_hermitian_admission;
mod cuda_region_axpby;
mod cuda_region_trace;
mod cuda_scalar_dtypes;
mod cuda_spectrum_download;
mod cuda_strided_region_probe;
mod cuda_svd_gauge;
mod dense_exports;
mod eig_equation;
#[path = "../../tests/support/export_scan.rs"]
mod export_scan;
