//! Value contracts of the arithmetic and contraction entries against hand
//! oracles: the complex inner product is conjugate-linear in its first
//! argument, a fermionic `contract` carries the twist that `compose` does
//! not, and a lazy adjoint contracts as the conjugate transpose.

use super::*;
use num_complex::Complex64;
use std::sync::Arc;
use symtn::sector::{
    FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{ContractSpec, GradedSpace, Runtime};

use crate::numerics;

/// Genuinely complex fill: `im` is never zero for any admitted index tuple.
pub(crate) fn complex_entry(indices: &[usize], seed: f64) -> Complex64 {
    let ramp = indices.iter().map(|&index| index as f64).sum::<f64>();
    Complex64::new(ramp + seed, -(ramp + seed + 0.75))
}

on_each_engine! {
    /// `<a, b>` conjugates `a`: `<b, a> = conj <a, b>`, `<i a, b> = -i <a, b>`,
    /// `<a, i b> = i <a, b>`, and `<a, a> = |a|^2`; the value agrees with the
    /// Host's.
    fn complex_inner_is_conjugate_linear_in_the_first_argument<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let i = Complex64::new(0.0, 1.0);

        for (sectors, seed) in [(0_usize, 1.0_f64), (1, 7.0)] {
            let su2 = GradedSpace::try_new(
                Arc::new(SU2FusionRule),
                [
                    (SU2Irrep::from_twice_spin(0), 1 + sectors),
                    (SU2Irrep::from_twice_spin(1), 2),
                ],
            )
            .unwrap();
            let a = TensorMap::<_, Complex64>::from_subblock_fn(
                engine, &runtime, [&su2], [&su2], |_, indices| complex_entry(indices, seed),
            )
            .unwrap();
            let b = TensorMap::<_, Complex64>::from_subblock_fn(
                engine, &runtime, [&su2], [&su2], |_, indices| complex_entry(indices, seed + 2.0),
            )
            .unwrap();
            let host_inner = a.inner(engine, &b).unwrap();
            let host_norm = a.norm(engine, 2.0).unwrap();
            assert!(host_inner.im != 0.0, "the fixture must exercise a complex inner product");

            let a_device = B::upload(device, &a);
            let b_device = B::upload(device, &b);
            let inner = a_device.inner(device, &b_device).unwrap();
            let tolerance = 1e-12 * (1.0 + host_inner.norm());
            assert!((inner - host_inner).norm() <= tolerance);
            assert!((b_device.inner(device, &a_device).unwrap() - inner.conj()).norm() <= tolerance);
            let scaled = a_device.scale(device, i).unwrap();
            numerics::assert_slices_close(
                "scaled",
                B::download(device, &scaled).dense_data().unwrap(),
                a.scale(engine, i).unwrap().dense_data().unwrap(),
                4,
            );
            assert!((scaled.inner(device, &b_device).unwrap() - (-i) * inner).norm() <= tolerance);
            let b_scaled = b_device.scale(device, i).unwrap();
            assert!((a_device.inner(device, &b_scaled).unwrap() - i * inner).norm() <= tolerance);

            let norm = a_device.norm(device, 2.0).unwrap();
            assert!((norm - host_norm).abs() <= 1e-12 * (1.0 + host_norm));
            let self_inner = a_device.inner(device, &a_device).unwrap();
            assert!(self_inner.im.abs() <= 1e-12 * (1.0 + self_inner.norm()));
            assert!((self_inner.re - norm * norm).abs() <= 1e-12 * (1.0 + self_inner.norm()));
        }
    }
}

on_each_engine! {
    /// Contracting two odd fZ2 legs through a bent pair carries the twist
    /// (-6); `compose` of the same tensors does not (+6).
    fn fermionic_contract_is_minus_six_and_compose_stays_plus_six<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(FermionParityFusionRule);
        let odd = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)]).unwrap();
        let odd_dual = odd.try_dual().unwrap();
        let lhs = TensorMap::from_subblock_fn(engine, &runtime, [&odd], [&odd_dual], |_, _| 2.0)
            .unwrap();
        let rhs = TensorMap::from_subblock_fn(engine, &runtime, [&odd_dual], [&odd], |_, _| 3.0)
            .unwrap();
        let spec = ContractSpec { lhs: &[1], rhs: &[0], codomain: &[0], domain: &[1] };

        let lhs = B::upload(device, &lhs);
        let rhs = B::upload(device, &rhs);
        let contracted = lhs.contract(device, &rhs, &spec).unwrap();
        assert_eq!(B::download(device, &contracted).dense_data().unwrap(), [-6.0]);
        let composed = lhs.compose(device, &rhs).unwrap();
        assert_eq!(B::download(device, &composed).dense_data().unwrap(), [6.0]);
    }
}

on_each_engine! {
    /// `(A^H B)[i, j] = sum_k conj(A[k, i]) B[k, j]` through the lazy adjoint,
    /// by both `compose` and `contract`; U(1) is bosonic so both agree, and the
    /// oracle's imaginary part tells conjugation from transposition.
    fn lazy_adjoint_contraction_is_the_conjugate_transposed_product<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let u1 = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
        let a = TensorMap::<_, Complex64>::from_subblock_fn(
            engine, &runtime, [&u1], [&u1],
            |_, indices| complex_entry(indices, 1.0) * Complex64::new(1.0, indices[1] as f64 + 1.0),
        )
        .unwrap();
        let b = TensorMap::<_, Complex64>::from_subblock_fn(
            engine, &runtime, [&u1], [&u1],
            |_, indices| complex_entry(indices, 3.0) * Complex64::new(indices[0] as f64 + 1.0, -1.0),
        )
        .unwrap();
        assert_eq!(a.subblock_count(), 1);

        let block = |tensor: &TensorMap<U1FusionRule, Complex64>| {
            let block = tensor.subblock(0).unwrap();
            assert_eq!(block.shape(), [2, 2]);
            let (offset, strides) = (block.offset(), block.strides().to_vec());
            let data = tensor.dense_data().unwrap().to_vec();
            move |row: usize, col: usize| data[offset + row * strides[0] + col * strides[1]]
        };
        let (a_block, b_block) = (block(&a), block(&b));
        let mut expected = [[Complex64::new(0.0, 0.0); 2]; 2];
        for (i, row) in expected.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                for k in 0..2 {
                    *entry += a_block(k, i).conj() * b_block(k, j);
                }
            }
        }
        assert!(
            expected.iter().flatten().any(|value| value.im != 0.0),
            "the oracle must distinguish conjugation from transposition"
        );

        let a_device = B::upload(device, &a);
        let b_device = B::upload(device, &b);
        let spec = ContractSpec { lhs: &[1], rhs: &[0], codomain: &[0], domain: &[1] };
        let lazy = a_device.adjoint(device).unwrap();
        let composed = lazy.compose(device, &b_device).unwrap();
        let contracted = lazy.contract(device, &b_device, &spec).unwrap();
        for (label, result) in [
            ("compose", B::download(device, &composed)),
            ("contract", B::download(device, &contracted)),
        ] {
            let result_block = result.subblock(0).unwrap();
            let (offset, strides) = (result_block.offset(), result_block.strides().to_vec());
            for (i, row) in expected.iter().enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    let got = result.dense_data().unwrap()[offset + i * strides[0] + j * strides[1]];
                    assert!(
                        (got - value).norm() <= 1e-12 * (1.0 + value.norm()),
                        "{label} {got:?} != hand {value:?}"
                    );
                }
            }
        }
    }
}
