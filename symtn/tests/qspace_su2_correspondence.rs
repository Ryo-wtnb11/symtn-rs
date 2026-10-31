//! Gauge-invariant correspondence with the pinned QSpace v4 SU(2) spin-half
//! operator. QSpace public master `e87ccd1` produces one rank-3
//! reduced record with labels `(1, 1, 2)`, orientation `('', '*', '*')`, and
//! coefficient `-sqrt(3)/2`. Contracting the operator with its conjugate over
//! the output and vector-operator legs gives `3/4` times the spin-half
//! identity, hence the complete closed norm is `2 * 3/4 = 3/2`.

use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

#[test]
fn spin_half_vector_operator_norm_matches_qspace() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let rule = Arc::new(SU2FusionRule);
    let half =
        GradedSpace::try_new(Arc::clone(&rule), [(SU2Irrep::from_twice_spin(1), 1)]).unwrap();
    let vector =
        GradedSpace::try_new(Arc::clone(&rule), [(SU2Irrep::from_twice_spin(2), 1)]).unwrap();
    let spin =
        TensorMap::from_subblock_fn(engine, &runtime, [&half], [&half, &vector], |trees, _| {
            assert_eq!(trees.codomain_uncoupled(), &[SU2Irrep::from_twice_spin(1)]);
            assert_eq!(
                trees.domain_uncoupled(),
                &[SU2Irrep::from_twice_spin(1), SU2Irrep::from_twice_spin(2),]
            );
            -(3.0_f64).sqrt() / 2.0
        })
        .unwrap();

    let norm = spin.norm(engine, 2.0).unwrap();
    assert!(
        (norm * norm - 1.5).abs() <= 1e-12,
        "norm squared = {}",
        norm * norm
    );
}
