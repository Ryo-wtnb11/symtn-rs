//! Experiment-only smoke: U(1) rank-4 x rank-4 contraction, timed at
//! `ENGINE_SMOKE_THREADS` dense threads. Not for merge.
use std::sync::Arc;
use std::time::Instant;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let threads: usize = std::env::var("ENGINE_SMOKE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let d: usize = std::env::var("ENGINE_SMOKE_D")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let rt = Runtime::builder().dense_threads(threads).build()?;
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (-2..=2).map(|q| (U1Irrep::new(q), d)),
    )?;
    let a: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 1)?;
    let b: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 2)?;
    let spec = ContractSpec {
        lhs: &[2, 3],
        rhs: &[0, 1],
        codomain: &[0, 1],
        domain: &[2, 3],
    };
    let warm = a.contract(&b, &spec)?;
    let reps = 5;
    let start = Instant::now();
    let mut norm = 0.0;
    for _ in 0..reps {
        norm += a.contract(&b, &spec)?.norm(2.0)?;
    }
    let per = start.elapsed() / reps;
    println!(
        "threads={threads} D={d} per_contract={per:?} norm={:.6e} warm_norm={:.6e}",
        norm / reps as f64,
        warm.norm(2.0)?
    );
    Ok(())
}
