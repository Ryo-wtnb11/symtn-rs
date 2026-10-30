use super::*;

#[cfg(feature = "tenferro")]
#[test]
fn dense_tensor_consuming_vectors_preserve_owners_for_every_dtype() {
    let engine = &mut CpuEngine::serial();
    let mut executor = DefaultDenseExecutor::new();
    let shape = [2, 2];
    let strides = [1, 2];

    let f64_data = [1.0, 3.0, 2.0, 4.0];
    let f64 = executor
        .qr(
            engine,
            DenseRead::F64(DenseView::new(&f64_data, &shape, &strides, 0).unwrap()),
        )
        .unwrap()
        .remove(0);
    let f64_ptr = f64.as_f64_slice().unwrap().as_ptr();
    let f64_data = f64.into_f64_vec().unwrap();
    assert_eq!(f64_ptr, f64_data.as_ptr());

    let f32_data = [1.0_f32, 3.0, 2.0, 4.0];
    let f32 = executor
        .qr(
            engine,
            DenseRead::F32(DenseView::new(&f32_data, &shape, &strides, 0).unwrap()),
        )
        .unwrap()
        .remove(0);
    let f32_ptr = f32.as_f32_slice().unwrap().as_ptr();
    let f32_data = f32.into_f32_vec().unwrap();
    assert_eq!(f32_ptr, f32_data.as_ptr());

    let c64_data = [
        Complex64::new(1.0, 1.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(4.0, 0.5),
    ];
    let c64 = executor
        .qr(
            engine,
            DenseRead::C64(DenseView::new(&c64_data, &shape, &strides, 0).unwrap()),
        )
        .unwrap()
        .remove(0);
    let c64_ptr = c64.as_c64_slice().unwrap().as_ptr();
    let c64_data = c64.into_c64_vec().unwrap();
    assert_eq!(c64_ptr, c64_data.as_ptr());

    let c32_data = [
        Complex32::new(1.0, 1.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(2.0, -1.0),
        Complex32::new(4.0, 0.5),
    ];
    let c32 = executor
        .qr(
            engine,
            DenseRead::C32(DenseView::new(&c32_data, &shape, &strides, 0).unwrap()),
        )
        .unwrap()
        .remove(0);
    let c32_ptr = c32.as_c32_slice().unwrap().as_ptr();
    let c32_data = c32.into_c32_vec().unwrap();
    assert_eq!(c32_ptr, c32_data.as_ptr());
}
#[cfg(feature = "tenferro")]
#[test]
fn dense_tensor_consuming_vector_rejects_wrong_dtype() {
    let engine = &mut CpuEngine::serial();
    let mut executor = DefaultDenseExecutor::new();
    let shape = [1, 1];
    let strides = [1, 1];
    let tensor = executor
        .qr(
            engine,
            DenseRead::C64(
                DenseView::new(&[Complex64::new(1.0, 0.0)], &shape, &strides, 0).unwrap(),
            ),
        )
        .unwrap()
        .remove(0);
    assert!(tensor.into_f64_vec().is_err());
}
