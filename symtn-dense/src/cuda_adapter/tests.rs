use super::*;

/// Tests one packed CUDA matrix region with the host EIGH rule; the
/// one-region case of [`cuda_hermitian_regions`]. Test support only: no
/// production caller decides a single region.
fn cuda_is_hermitian_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<bool, DenseError> {
    Ok(cuda_hermitian_regions::<D>(ctx, src, &[(offset, n)], default_tolerance::<D>())?[0])
}

/// symtn's default eigh admission tolerance, `eps(real(D))^(3/4)`
/// (`HermitianTol::DEFAULT`), which the device fixtures are expressed in.
fn default_tolerance<D: CudaScalar>() -> f64 {
    <D::Real as CudaRealScalar>::EPSILON.powf(0.75)
}

/// The placement of a buffer of test engine `n`: device ordinal `n`, and a
/// domain of its own per `n`, as two engines have even on one ordinal.
fn on(n: usize) -> CudaPlacement {
    static DOMAINS: std::sync::OnceLock<Vec<AllocationDomainId>> = std::sync::OnceLock::new();
    let domains = DOMAINS.get_or_init(|| (0..4).map(|_| AllocationDomainId::fresh()).collect());
    CudaPlacement {
        ordinal: n,
        domain: domains[n],
    }
}

/// A CUDA engine on device `n`.
fn cuda_engine(n: u32) -> tenferro_gpu::cuda::CudaEngine {
    tenferro_gpu::cuda::CudaEngine::open(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(n)).unwrap()
}

#[test]
fn a_cpu_engine_is_refused_before_any_device_work() {
    // What: a device operation needs the caller's CUDA engine; any other
    // engine is a typed refusal at context construction, so nothing is
    // uploaded or submitted. No GPU is needed to prove it.
    let mut cpu = tenferro_cpu::CpuEngine::serial();
    let before = cuda_transfer_stats();
    let err = CudaDenseContext::new(&mut cpu)
        .err()
        .expect("a CPU engine is refused");
    assert!(
        matches!(
            err,
            DenseError::Unsupported {
                op: "cuda_context",
                ..
            }
        ),
        "{err}"
    );
    assert_eq!(cuda_transfer_stats(), before);
}

#[test]
fn ensure_cuda_device_rejects_another_engine_on_the_same_ordinal() {
    // What: compatibility is the engine's allocation domain, not the device
    // ordinal: a buffer of a second engine on device 0 is refused.
    let other = CudaPlacement {
        ordinal: 0,
        domain: AllocationDomainId::fresh(),
    };
    let err = ensure_cuda_device(on(0), "cuda_matmul", &[("lhs", on(0)), ("rhs", other)])
        .expect_err("a same-ordinal buffer of another engine must be rejected");
    assert!(
        matches!(err, DenseError::Backend { op: "cuda_matmul", ref message, .. } if message.contains("rhs")),
        "{err}"
    );
}

#[test]
fn rectangular_operand_views_use_parent_native_strides() {
    assert_eq!(cuda_operand_view(MatrixOp::Identity, 2, 3), ([1, 2], false));
    assert_eq!(cuda_operand_view(MatrixOp::Adjoint, 2, 3), ([3, 1], true));
    assert_eq!(cuda_operand_view(MatrixOp::Identity, 3, 4), ([1, 3], false));
    assert_eq!(cuda_operand_view(MatrixOp::Adjoint, 3, 4), ([4, 1], true));
}

#[test]
fn a_zero_descriptor_scale_is_rejected_before_any_device_work() {
    // What: `alpha = 0` is the one scale whose backend behaviour is not
    // `alpha * src` — the source read may be skipped, erasing NaN/Inf — so
    // it is a typed rejection rather than a silently different answer. The
    // check needs no device, which is what makes it precede every upload.
    for alpha in [0.0_f64, -0.0] {
        let err = reject_zero_alpha::<f64>("cuda_region_axpby", alpha)
            .expect_err("a zero descriptor scale must be rejected");
        assert!(
            matches!(
                err,
                DenseError::Unsupported {
                    op: "cuda_region_axpby",
                    ..
                }
            ),
            "{err}"
        );
        assert!(
            err.to_string().contains("CudaRegionCoefficient::Zero"),
            "{err}"
        );
    }
    let complex_zero = Complex64::new(-0.0, 0.0);
    assert!(reject_zero_alpha::<Complex64>("cuda_region_axpby", complex_zero).is_err());

    // Every other scale passes, including the non-finite ones: they are
    // ordinary multiplications.
    for alpha in [1.0_f64, -2.5, f64::NAN, f64::INFINITY] {
        assert!(reject_zero_alpha::<f64>("cuda_region_axpby", alpha).is_ok());
    }
    assert!(reject_zero_alpha::<Complex64>("cuda_region_axpby", Complex64::new(0.0, -1.0)).is_ok());
}

#[test]
fn ensure_cuda_device_accepts_matching_operands() {
    assert!(
        ensure_cuda_device(on(0), "op", &[("a", on(0)), ("b", on(0))]).is_ok(),
        "operands on the context device must be accepted"
    );
}

#[test]
fn ensure_cuda_device_rejects_a_foreign_operand() {
    let err = ensure_cuda_device(on(0), "cuda_matmul", &[("lhs", on(0)), ("rhs", on(1))])
        .expect_err("an operand on another device must be rejected");
    match err {
        DenseError::Backend {
            backend,
            op,
            message,
        } => {
            assert_eq!(backend, DenseBackend::Cuda);
            assert_eq!(op, "cuda_matmul");
            assert!(
                message.contains("rhs"),
                "message names the operand: {message}"
            );
            assert!(
                message.contains("device 1"),
                "message names the device: {message}"
            );
        }
        other => panic!("expected a CUDA backend error, got {other:?}"),
    }
}

/// A spectrum produced on `device`, of `len` values (host data stands in for
/// the device buffer: the guard reads only the recorded device).
fn spectrum_on(device: usize, len: usize) -> CudaSpectrum {
    spectrum_at(on(device), len)
}

/// [`spectrum_on`] at an explicit placement, e.g. a real context's.
fn spectrum_at(device: CudaPlacement, len: usize) -> CudaSpectrum {
    CudaSpectrum {
        tensor: Tensor::from_vec_col_major(vec![len], vec![1.0_f64; len]).unwrap(),
        device,
    }
}

fn assert_foreign_spectrum(err: DenseError, op: &str, operand: &str) {
    match err {
        DenseError::Backend {
            backend: DenseBackend::Cuda,
            op: got,
            message,
        } => {
            assert_eq!(got, op);
            assert!(message.contains(operand), "names the operand: {message}");
            assert!(message.contains("device 1"), "names the device: {message}");
        }
        other => panic!("expected a CUDA device mismatch, got {other:?}"),
    }
}

#[test]
fn the_spectrum_device_guard_rejects_any_foreign_spectrum_even_an_empty_one() {
    assert!(ensure_spectra_device(on(0), "op", &[]).is_ok());
    assert!(ensure_spectra_device(on(0), "op", &[spectrum_on(0, 3), spectrum_on(0, 0)]).is_ok());
    let err = ensure_spectra_device(on(0), "op", &[spectrum_on(0, 3), spectrum_on(1, 0)])
        .expect_err("an empty spectrum from device 1 must be rejected");
    assert_foreign_spectrum(err, "op", "spectra[1]");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn every_spectrum_consumer_rejects_a_foreign_spectrum_before_any_device_work() {
    // What: a spectrum recorded on device 1, read through a device-0 context,
    // is rejected by each consumer before it submits or allocates anything —
    // empty ones too, which otherwise skip all device work. One GPU suffices:
    // the guard reads the recorded device, so relabelling stands in for a
    // second producer (the two-device test below uses a real one).
    let mut ctx_engine = cuda_engine(0);
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let here = ctx.device;
    let src = CudaDenseStorage::upload::<f64>(&ctx, &[2.0, 0.5, 0.5, 1.0]).unwrap();
    let foreign = |ctx: &mut CudaDenseContext, n: usize| {
        let (mut values, _) = cuda_eigh_region::<f64>(ctx, &src, 0, n).unwrap();
        values.device = on(1);
        values
    };
    let batched_foreign = |ctx: &mut CudaDenseContext| {
        let (mut values, _) = cuda_eigh_region_batched::<f64>(ctx, &src, 0, 2, 1, 4).unwrap();
        values.device = on(1);
        values
    };
    let mut dst =
        CudaDenseStorage::upload::<Complex64>(&ctx, &[Complex64::new(7.0, 0.0); 4]).unwrap();
    for len in [2, 0] {
        let local = || spectrum_at(here, 0);
        let spectra = [
            local(),
            if len == 0 {
                spectrum_on(1, 0)
            } else {
                foreign(&mut ctx, 2)
            },
        ];
        let before = cuda_transfer_stats();
        let err = cuda_download_spectra::<f64>(&mut ctx, &spectra).unwrap_err();
        assert_foreign_spectrum(err, "cuda_download_spectra", "spectra[1]");
        assert_eq!(cuda_transfer_stats(), before, "download, len {len}");

        let spectra = [
            local(),
            if len == 0 {
                spectrum_on(1, 0)
            } else {
                batched_foreign(&mut ctx)
            },
        ];
        let before = cuda_transfer_stats();
        let err = cuda_download_batched_spectra::<f64>(&mut ctx, &spectra, 1).unwrap_err();
        assert_foreign_spectrum(err, "cuda_download_spectra", "spectra[1]");
        assert_eq!(cuda_transfer_stats(), before, "batched download, len {len}");

        let spectrum = if len == 0 {
            spectrum_on(1, 0)
        } else {
            foreign(&mut ctx, 2)
        };
        let before = cuda_transfer_stats();
        let err =
            cuda_copy_spectrum_into::<Complex64>(&mut ctx, spectrum, &mut dst, 0, 3).unwrap_err();
        assert_foreign_spectrum(err, "cuda_copy_spectrum", "spectrum");
        assert_eq!(cuda_transfer_stats(), before, "copy, len {len}");

        // A local spectrum into a `dst` recorded on device 1: rejected before
        // the complex cast and before the empty-spectrum early return.
        let spectrum = if len == 0 {
            spectrum_at(here, 0)
        } else {
            cuda_eigh_region::<f64>(&mut ctx, &src, 0, 2).unwrap().0
        };
        dst.device = on(1);
        let before = cuda_transfer_stats();
        let err =
            cuda_copy_spectrum_into::<Complex64>(&mut ctx, spectrum, &mut dst, 0, 3).unwrap_err();
        dst.device = here;
        assert_foreign_spectrum(err, "cuda_copy_spectrum", "dst");
        assert_eq!(cuda_transfer_stats(), before, "foreign dst, len {len}");
    }
    assert_eq!(
        dst.download::<Complex64>(&ctx).unwrap(),
        vec![Complex64::new(7.0, 0.0); 4],
        "a rejected copy writes nothing"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_buffer_of_another_engine_on_the_same_device_is_refused() {
    // What: two engines on one ordinal are separate allocation domains. A
    // buffer of the first is refused by the second before any transfer or
    // submission, and stays intact for its own engine.
    let mut first_engine = cuda_engine(0);
    let mut second_engine = cuda_engine(0);
    let first = CudaDenseContext::new(&mut first_engine).unwrap();
    let mut second = CudaDenseContext::new(&mut second_engine).unwrap();
    assert_eq!(first.device(), second.device());
    assert_ne!(first.identity(), second.identity());
    let src = CudaDenseStorage::upload::<f64>(&first, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let mut dst = CudaDenseStorage::upload::<f64>(&first, &[0.0; 4]).unwrap();
    let before = cuda_transfer_stats();
    assert!(src.download::<f64>(&second).is_err());
    assert!(
        cuda_matmul_region_into::<f64>(&mut second, &mut dst, 0, &src, 0, &src, 0, 2, 2, 2)
            .is_err()
    );
    assert!(cuda_all_finite::<f64>(&mut second, &src).is_err());
    assert_eq!(cuda_transfer_stats(), before, "nothing reaches the device");
    assert_eq!(dst.download::<f64>(&first).unwrap(), vec![0.0; 4]);
    assert_eq!(
        src.download::<f64>(&first).unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
#[ignore = "requires two idle CUDA devices"]
fn a_spectrum_from_another_device_is_rejected_by_every_consumer() {
    let mut producer_engine = cuda_engine(1);
    let mut producer = CudaDenseContext::new(&mut producer_engine).unwrap();
    let mut consumer_engine = cuda_engine(0);
    let mut consumer = CudaDenseContext::new(&mut consumer_engine).unwrap();
    let src = CudaDenseStorage::upload::<f64>(&producer, &[2.0, 0.5, 0.5, 1.0]).unwrap();
    let mut eigh = || {
        cuda_eigh_region::<f64>(&mut producer, &src, 0, 2)
            .unwrap()
            .0
    };
    let (values, copied) = (eigh(), eigh());
    let (_, singular, _) = cuda_svd_region::<f64>(&mut producer, &src, 0, 2, 2).unwrap();
    let (batched, _) = cuda_eigh_region_batched::<f64>(&mut producer, &src, 0, 2, 1, 4).unwrap();
    assert_eq!(
        (
            values.device,
            copied.device,
            singular.device,
            batched.device
        ),
        (
            producer.device,
            producer.device,
            producer.device,
            producer.device
        )
    );
    let mut dst = CudaDenseStorage::upload::<f64>(&consumer, &[7.0; 4]).unwrap();

    let before = cuda_transfer_stats();
    let err = cuda_download_spectra::<f64>(&mut consumer, &[values, singular]).unwrap_err();
    assert_foreign_spectrum(err, "cuda_download_spectra", "spectra[0]");
    let err = cuda_download_batched_spectra::<f64>(&mut consumer, &[batched], 1).unwrap_err();
    assert_foreign_spectrum(err, "cuda_download_spectra", "spectra[0]");
    let err = cuda_copy_spectrum_into::<f64>(&mut consumer, copied, &mut dst, 0, 3).unwrap_err();
    assert_foreign_spectrum(err, "cuda_copy_spectrum", "spectrum");
    assert_eq!(
        cuda_transfer_stats(),
        before,
        "no consumer submits anything"
    );
    assert_eq!(dst.download::<f64>(&consumer).unwrap(), vec![7.0; 4]);

    // A local spectrum, empty or not, into a `dst` living on device 1.
    let local_src = CudaDenseStorage::upload::<f64>(&consumer, &[2.0, 0.5, 0.5, 1.0]).unwrap();
    let local = cuda_eigh_region::<f64>(&mut consumer, &local_src, 0, 2)
        .unwrap()
        .0;
    let mut foreign_dst =
        CudaDenseStorage::upload::<Complex64>(&producer, &[Complex64::new(7.0, 0.0); 4]).unwrap();
    for spectrum in [local, spectrum_at(consumer.device, 0)] {
        let before = cuda_transfer_stats();
        let err =
            cuda_copy_spectrum_into::<Complex64>(&mut consumer, spectrum, &mut foreign_dst, 0, 3)
                .unwrap_err();
        assert_foreign_spectrum(err, "cuda_copy_spectrum", "dst");
        assert_eq!(cuda_transfer_stats(), before, "foreign dst submits nothing");
    }
    assert_eq!(
        foreign_dst.download::<Complex64>(&producer).unwrap(),
        vec![Complex64::new(7.0, 0.0); 4]
    );
}

#[test]
fn svd_factor_shape_contract_covers_rectangular_and_bad_backend_results() {
    assert!(validate_svd_factor_shapes(&[4, 3], 3, &[3, 3], 4, 3).is_ok());
    assert!(validate_svd_factor_shapes(&[3, 3], 3, &[3, 4], 3, 4).is_ok());
    assert!(validate_svd_factor_shapes(&[4, 4], 3, &[3, 3], 4, 3).is_err());
    assert!(validate_svd_factor_shapes(&[4, 3], 2, &[3, 3], 4, 3).is_err());
    assert!(validate_svd_factor_shapes(&[4, 3], 3, &[4, 3], 4, 3).is_err());
}

#[test]
fn qr_factor_shapes_must_match_the_requested_compact_problem() {
    assert!(validate_qr_factor_shapes(&[4, 3], &[3, 3], 4, 3).is_ok());
    assert!(validate_qr_factor_shapes(&[4, 4], &[3, 3], 4, 3).is_err());
    assert!(validate_qr_factor_shapes(&[4, 3], &[4, 3], 4, 3).is_err());
}

#[test]
fn eigh_factor_shapes_must_match_the_requested_square_problem() {
    assert!(validate_eigh_factor_shapes(3, &[3, 3], 3).is_ok());
    assert!(validate_eigh_factor_shapes(2, &[3, 3], 3).is_err());
    assert!(validate_eigh_factor_shapes(3, &[3, 2], 3).is_err());
}

/// Each admitted dtype owns its own scalar-operand slot, so a context used
/// from two dtypes never hands one of them the other's payload-typed `1`.
#[test]
fn every_admitted_dtype_owns_a_distinct_operand_slot() {
    let slots = [
        f32::OPERAND_SLOT,
        f64::OPERAND_SLOT,
        Complex32::OPERAND_SLOT,
        Complex64::OPERAND_SLOT,
    ];
    for (index, slot) in slots.iter().enumerate() {
        assert!(*slot < SCALAR_OPERAND_SLOTS);
        assert!(
            !slots[..index].contains(slot),
            "slot {slot} is claimed twice"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_bad_later_gather_row_rejects_the_call_before_any_write() {
    // What: with a valid first entry and an out-of-range later one, the
    // gather is rejected and the destination keeps every sentinel; the
    // valid list alone places the selected columns.
    let mut ctx_engine = cuda_engine(0);
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let (n, members) = (3, 2);
    let mut data = vec![0.0; n * n * members];
    for member in 0..members {
        for i in 0..n {
            data[member * n * n + i + n * i] = (i + 1 + member) as f64;
        }
    }
    let src = CudaDenseStorage::upload_owned(&ctx, data).unwrap();
    let (_, vectors) =
        cuda_eigh_region_batched::<f64>(&mut ctx, &src, 0, n, members, n * n).unwrap();
    let reference = vectors.download::<f64>(&ctx).unwrap();
    let sentinel = vec![7.0; n * n * members];
    let mut dst = CudaDenseStorage::upload(&ctx, &sentinel).unwrap();
    let columns = [2, 0, 1, 1, 2, 0];
    let copies_before = cuda_transfer_stats().copy_calls;
    let bad = cuda_gather_columns_batched_into::<f64>(
        &mut ctx,
        &mut dst,
        0,
        n,
        n * n,
        &vectors,
        n,
        members,
        &columns,
        &[(0, 0, 1), (1, 2, 2)],
    );
    assert!(bad.is_err());
    assert_eq!(
        cuda_transfer_stats().copy_calls,
        copies_before,
        "no submission"
    );
    assert_eq!(dst.download::<f64>(&ctx).unwrap(), sentinel);

    cuda_gather_columns_batched_into::<f64>(
        &mut ctx,
        &mut dst,
        0,
        n,
        n * n,
        &vectors,
        n,
        members,
        &columns,
        &[(0, 0, 1), (1, 1, 2)],
    )
    .unwrap();
    let got = dst.download::<f64>(&ctx).unwrap();
    for member in 0..members {
        for (c, &column) in columns[member * n..][..n].iter().enumerate() {
            for row in 0..n {
                let want = reference[member * n * n + row + n * column];
                assert_eq!(got[member * n * n + row + n * c], want);
            }
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn an_empty_spectrum_copy_touches_nothing_even_for_a_complex_payload() {
    let mut ctx_engine = cuda_engine(0);
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let empty = CudaDenseStorage::upload_owned::<f64>(&ctx, Vec::new()).unwrap();
    let spectrum = CudaSpectrum {
        tensor: empty.tensor,
        device: ctx.device,
    };
    let values = vec![Complex64::new(1.0, 2.0); 4];
    let mut dst = CudaDenseStorage::upload::<Complex64>(&ctx, &values).unwrap();
    let before = cuda_transfer_stats();
    cuda_copy_spectrum_into::<Complex64>(&mut ctx, spectrum, &mut dst, 0, 3).unwrap();
    assert_eq!(cuda_transfer_stats(), before, "no cast, copy or transfer");
    assert_eq!(dst.download::<Complex64>(&ctx).unwrap(), values);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_hermitian_region_is_scaled_and_downloads_only_scalar_metadata() {
    let mut ctx_engine = cuda_engine(0);
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let n = 4;
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i + n * i] = 1.0;
    }
    data[n] = 32.0 * f64::EPSILON;
    data[1] = data[n];
    let storage = CudaDenseStorage::upload::<f64>(&ctx, &data).unwrap();

    let downloads = crate::probe::record();
    assert!(cuda_is_hermitian_region::<f64>(&mut ctx, &storage, 0, n).unwrap());
    assert_eq!(downloads.get(crate::probe::Event::CudaFullDownloadBytes), 0);
    assert!(downloads.get(crate::probe::Event::CudaMetadataDownloadBytes) <= 4 * 8);

    let tol = default_tolerance::<f64>();
    let near_threshold = |ctx: &CudaDenseContext, delta: f64| {
        // For [[1, delta], [0, 1]], `delta / sqrt(2) <= tol * sqrt(2 + delta^2)`
        // changes truth value at delta = 2 tol up to negligible O(delta^2).
        CudaDenseStorage::upload::<f64>(ctx, &[1.0, 0.0, delta, 1.0]).unwrap()
    };
    let below = near_threshold(&ctx, 0.97 * 2.0 * tol);
    let above = near_threshold(&ctx, 1.03 * 2.0 * tol);
    assert!(cuda_is_hermitian_region::<f64>(&mut ctx, &below, 0, 2).unwrap());
    assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &above, 0, 2).unwrap());

    let zero = CudaDenseStorage::upload::<f64>(&ctx, &vec![0.0; n * n]).unwrap();
    assert!(cuda_is_hermitian_region::<f64>(&mut ctx, &zero, 0, n).unwrap());

    // ||A||_F ~ 2 and the half residual is |data[n] - data[1]| / sqrt(2), so
    // admission needs |data[n] - data[1]| <= 2 sqrt(2) tol; this is ~3x over.
    data[n] = data[1] + 8.0 * tol;
    let asymmetric = CudaDenseStorage::upload::<f64>(&ctx, &data).unwrap();
    assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &asymmetric, 0, n).unwrap());

    for scale in [f64::from_bits(0x0010_0000_0000_0000), 2.0_f64.powi(500)] {
        let scaled: Vec<_> = data.iter().map(|value| value * scale).collect();
        let scaled = CudaDenseStorage::upload::<f64>(&ctx, &scaled).unwrap();
        assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &scaled, 0, n).unwrap());
    }

    for bad in [f64::NAN, f64::INFINITY] {
        let mut nonfinite = vec![0.0; n * n];
        nonfinite[0] = bad;
        let nonfinite = CudaDenseStorage::upload::<f64>(&ctx, &nonfinite).unwrap();
        assert!(!cuda_is_hermitian_region::<f64>(&mut ctx, &nonfinite, 0, n).unwrap());
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_transfer_bytes_scale_with_the_payload_dtype() {
    // A complex payload must move the same *number* of buffers as
    // the real one and exactly `size_of::<Complex64>() / size_of::<f64>()`
    // times the bytes for the same element count.
    let mut ctx_engine = cuda_engine(0);
    let ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let elements = 16;
    let real: Vec<f64> = (0..elements).map(|index| index as f64).collect();
    let complex: Vec<Complex64> = (0..elements)
        .map(|index| Complex64::new(index as f64, -(index as f64) - 0.5))
        .collect();

    let real_downloads = crate::probe::record();
    let real_device = CudaDenseStorage::upload::<f64>(&ctx, &real).unwrap();
    assert_eq!(real_device.dtype(), DenseDType::F64);
    assert_eq!(real_device.download::<f64>(&ctx).unwrap(), real);
    let real_bytes = real_downloads.get(crate::probe::Event::CudaFullDownloadBytes);
    let complex_downloads = crate::probe::record();

    let complex_device = CudaDenseStorage::upload::<Complex64>(&ctx, &complex).unwrap();
    assert_eq!(complex_device.dtype(), DenseDType::C64);
    assert_eq!(complex_device.download::<Complex64>(&ctx).unwrap(), complex);
    let complex_bytes = complex_downloads.get(crate::probe::Event::CudaFullDownloadBytes);

    assert_eq!(real_bytes, elements * std::mem::size_of::<f64>());
    assert_eq!(
        complex_bytes,
        real_bytes * std::mem::size_of::<Complex64>() / std::mem::size_of::<f64>()
    );

    // A dtype mismatch is a typed error, never a reinterpretation.
    assert!(matches!(
        complex_device.download::<f64>(&ctx),
        Err(DenseError::DTypeMismatch {
            expected: DenseDType::F64,
            actual: DenseDType::C64,
            ..
        })
    ));
    assert!(matches!(
        real_device.region_view::<Complex64>(4, 4, 4, 0),
        Err(DenseError::DTypeMismatch {
            expected: DenseDType::C64,
            actual: DenseDType::F64,
            ..
        })
    ));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_transfer_counters_attribute_one_upload_download_and_gemm() {
    let mut ctx_engine = cuda_engine(0);
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let n = 2;
    let lhs_host = vec![1.0_f64, 2.0, 3.0, 4.0];
    let rhs_host = vec![5.0_f64, 6.0, 7.0, 8.0];

    reset_cuda_transfer_stats();
    let downloads = crate::probe::record();
    let lhs = CudaDenseStorage::upload::<f64>(&ctx, &lhs_host).unwrap();
    let rhs = CudaDenseStorage::upload::<f64>(&ctx, &rhs_host).unwrap();
    let mut dst = CudaDenseStorage::upload::<f64>(&ctx, &vec![0.0_f64; n * n]).unwrap();
    let after_uploads = cuda_transfer_stats();
    assert_eq!(after_uploads.h2d_calls, 3);
    assert_eq!(
        after_uploads.h2d_bytes,
        (3 * n * n * std::mem::size_of::<f64>()) as u64
    );
    assert_eq!(after_uploads.device_allocs, 3);
    assert_eq!(after_uploads.d2h_calls, 0);
    assert_eq!(after_uploads.gemm_calls, 0);

    cuda_gemm_region_into::<f64>(
        &mut ctx, &mut dst, 0, n, &lhs, 0, n, &rhs, 0, n, n, n, n, 1.0, 0.0,
    )
    .unwrap();
    let after_gemm = cuda_transfer_stats();
    assert_eq!(after_gemm.gemm_calls, 1);
    assert_eq!(after_gemm.h2d_calls, after_uploads.h2d_calls);
    assert_eq!(after_gemm.d2h_calls, 0);

    let values = dst.download::<f64>(&ctx).unwrap();
    let after_download = cuda_transfer_stats();
    assert_eq!(after_download.d2h_calls, 1);
    assert_eq!(
        after_download.d2h_bytes,
        (n * n * std::mem::size_of::<f64>()) as u64
    );
    // Column-major 2x2 product, so the counters above describe a real GEMM.
    assert_eq!(values, vec![23.0, 34.0, 31.0, 46.0]);

    // The finer-grained test counter stays consistent with the always
    // compiled one for the same download.
    assert_eq!(
        downloads.get(crate::probe::Event::CudaFullDownloadBytes) as u64,
        after_download.d2h_bytes
    );

    reset_cuda_transfer_stats();
    assert_eq!(cuda_transfer_stats(), CudaTransferStats::default());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn warm_up_costs_one_gemm_one_solver_call_and_a_bounded_fixed_traffic() {
    // A context built directly here has never submitted work: only
    // `RuntimeBuilder::build` warms one, so this is a cold backend.
    let mut ctx_engine = cuda_engine(0);
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();

    reset_cuda_transfer_stats();
    ctx.warm_up().unwrap();
    let after = cuda_transfer_stats();
    // The documented fixed cost of the warm-up, as the rustdoc states it.
    // `device_allocs` counts the three 1-element uploads plus the
    // eigenvector factor; all four are local to `warm_up` and dropped
    // before it returns, so nothing of this survives construction. There is
    // no live/peak device-buffer observation at this seam to assert on.
    assert_eq!(
        after,
        CudaTransferStats {
            h2d_calls: 3,
            h2d_bytes: 3 * std::mem::size_of::<f64>() as u64,
            d2h_calls: 1,
            d2h_bytes: std::mem::size_of::<f64>() as u64,
            device_allocs: 4,
            gemm_calls: 1,
            solver_calls: 1,
            copy_calls: 0,
            gauge_ops: 0,
        }
    );

    // Warming an already warm context repeats the same bounded work rather
    // than growing with the number of calls.
    reset_cuda_transfer_stats();
    ctx.warm_up().unwrap();
    assert_eq!(cuda_transfer_stats(), after);
    reset_cuda_transfer_stats();
}

#[test]
fn cuda_errors_are_labelled_as_the_cuda_backend() {
    // CUDA failures must not be reported as Tenferro.
    let err = cuda_error("cuda_svd", "boom");
    let text = err.to_string();
    assert!(text.contains("Cuda"), "formatted error names CUDA: {text}");
    assert!(
        !text.contains("Tenferro"),
        "must not mislabel as Tenferro: {text}"
    );
}

/// Builds a payload element from widened components (the imaginary part is
/// dropped for a real payload).
trait FiniteFixture: CudaScalar {
    const MAX: f64;
    const SUBNORMAL: f64;
    fn from_parts(re: f64, im: f64) -> Self;
}
impl FiniteFixture for f32 {
    const MAX: f64 = f32::MAX as f64;
    const SUBNORMAL: f64 = (f32::MIN_POSITIVE / 4.0) as f64;
    fn from_parts(re: f64, _: f64) -> Self {
        re as f32
    }
}
impl FiniteFixture for f64 {
    const MAX: f64 = f64::MAX;
    const SUBNORMAL: f64 = f64::MIN_POSITIVE / 4.0;
    fn from_parts(re: f64, _: f64) -> Self {
        re
    }
}
impl FiniteFixture for Complex32 {
    const MAX: f64 = f32::MAX as f64;
    const SUBNORMAL: f64 = (f32::MIN_POSITIVE / 4.0) as f64;
    fn from_parts(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
}
impl FiniteFixture for Complex64 {
    const MAX: f64 = f64::MAX;
    const SUBNORMAL: f64 = f64::MIN_POSITIVE / 4.0;
    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}

fn assert_finite_scan_decides_every_component<D: FiniteFixture>(ctx: &mut CudaDenseContext) {
    let (max, tiny) = (D::MAX, D::SUBNORMAL);
    let mut cases = vec![
        ((f64::NAN, 0.0), false),
        ((f64::INFINITY, 0.0), false),
        ((f64::NEG_INFINITY, 0.0), false),
        ((max, 0.0), true),
        ((-max, 0.0), true),
        ((tiny, 0.0), true),
        ((-0.0, 0.0), true),
    ];
    if D::IS_COMPLEX {
        cases.extend([
            ((0.0, f64::INFINITY), false),
            ((0.0, f64::NEG_INFINITY), false),
            ((0.0, f64::NAN), false),
            ((f64::NEG_INFINITY, 1.0), false),
            ((max, max), true),
            ((-max, -max), true),
            ((0.0, tiny), true),
        ]);
    }
    for len in [1_usize, 1_000, 100_000] {
        let base: Vec<D> = (0..len)
            .map(|i| D::from_parts((i % 97) as f64 - 48.5, (i % 13) as f64 * 0.25))
            .collect();
        let src = CudaDenseStorage::upload(ctx, &base).unwrap();
        assert!(cuda_all_finite::<D>(ctx, &src).unwrap(), "{len}: finite");
        for &((re, im), finite) in &cases {
            for at in [0, len / 2, len - 1] {
                let mut data = base.clone();
                data[at] = D::from_parts(re, im);
                let src = CudaDenseStorage::upload_owned(ctx, data).unwrap();
                reset_cuda_transfer_stats();
                let decided = cuda_all_finite::<D>(ctx, &src).unwrap();
                let stats = cuda_transfer_stats();
                assert_eq!(decided, finite, "{len} [{at}] = ({re}, {im})");
                assert_eq!(
                    (stats.h2d_calls, stats.d2h_calls, stats.d2h_bytes),
                    (0, 1, std::mem::size_of::<D::Real>() as u64),
                    "one real-lane scalar download, nothing uploaded"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_finite_scan_refuses_exactly_nan_and_infinite_components() {
    // What: `cuda_all_finite` is false exactly when some real or imaginary
    // component is NaN or infinite, wherever it lies, and true for the
    // extremes of the finite range (MAX, MAX+MAX*i, subnormals, -0), at the
    // cost of one real-lane scalar download.
    let mut ctx_engine = cuda_engine(0);
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    assert_finite_scan_decides_every_component::<f32>(&mut ctx);
    assert_finite_scan_decides_every_component::<f64>(&mut ctx);
    assert_finite_scan_decides_every_component::<Complex32>(&mut ctx);
    assert_finite_scan_decides_every_component::<Complex64>(&mut ctx);
    reset_cuda_transfer_stats();
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_finite_scan_reads_only_the_active_payload() {
    // What: a grow-only scratch is judged on its active prefix; a NaN past
    // the active length is stale and ignored, one inside it is refused. An
    // empty payload is finite without a transfer, and a payload of another
    // dtype is a dtype error.
    let mut ctx_engine = cuda_engine(0);
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut src = CudaDenseStorage::upload(&ctx, &[1.0, 2.0, f64::NAN, 4.0]).unwrap();
    src.set_active_len(2).unwrap();
    assert!(cuda_all_finite::<f64>(&mut ctx, &src).unwrap());
    src.set_active_len(3).unwrap();
    assert!(!cuda_all_finite::<f64>(&mut ctx, &src).unwrap());

    src.set_active_len(0).unwrap();
    reset_cuda_transfer_stats();
    assert!(cuda_all_finite::<f64>(&mut ctx, &src).unwrap());
    assert_eq!(cuda_transfer_stats(), CudaTransferStats::default());

    assert!(matches!(
        cuda_all_finite::<f32>(&mut ctx, &src),
        Err(DenseError::DTypeMismatch { .. })
    ));
    reset_cuda_transfer_stats();
}
