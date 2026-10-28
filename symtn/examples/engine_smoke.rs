//! Experiment-only smoke, timed at `ENGINE_SMOKE_THREADS` dense threads. Not
//! for merge. `ENGINE_SMOKE_CASE`:
//! - `dense` (default): U(1) rank-4 x rank-4, charges -2..=2 at degeneracy
//!   `ENGINE_SMOKE_D`, contracted over two legs;
//! - `u1-small`: the same contraction over charges -8..=8 at degeneracy 2;
//! - `su2-small`: SU(2) spins 0..=4 (twice_spin 0..=8) at degeneracy 2.
//!
//! `ENGINE_SMOKE_REPS` overrides the timed repetition count.
use std::sync::Arc;
use std::time::Instant;
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

const SPEC: ContractSpec<'static> = ContractSpec {
    lhs: &[2, 3],
    rhs: &[0, 1],
    codomain: &[0, 1],
    domain: &[2, 3],
};

macro_rules! time {
    ($label:expr, $rt:expr, $v:expr, $reps:expr) => {{
        let reps: u32 = std::env::var("ENGINE_SMOKE_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or($reps);
        let (rt, v): (&Runtime, _) = ($rt, $v);
        let a: TensorMap<_, f64> = TensorMap::rand_with_seed(rt, [&v, &v], [&v, &v], 1)?;
        let b: TensorMap<_, f64> = TensorMap::rand_with_seed(rt, [&v, &v], [&v, &v], 2)?;
        let warm = a.contract(&b, &SPEC)?;
        let start = Instant::now();
        let mut norm = 0.0;
        for _ in 0..reps {
            norm += a.contract(&b, &SPEC)?.norm(2.0)?;
        }
        let per = start.elapsed() / reps;
        println!(
            "{} blocks={} per_contract={per:?} norm={:.6e} warm_norm={:.6e}",
            $label,
            a.subblock_count(),
            norm / f64::from(reps),
            warm.norm(2.0)?
        );
        Ok(())
    }};
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env = |name: &str| std::env::var(name).ok();
    let threads: usize = env("ENGINE_SMOKE_THREADS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let d: usize = env("ENGINE_SMOKE_D")
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let case = env("ENGINE_SMOKE_CASE").unwrap_or_else(|| "dense".into());
    let rt = Runtime::builder().dense_threads(threads).build()?;
    let label = format!("case={case} threads={threads}");
    match case.as_str() {
        "dense" => {
            let v = GradedSpace::try_new(
                Arc::new(U1FusionRule),
                (-2..=2).map(|q| (U1Irrep::new(q), d)),
            )?;
            time!(format!("{label} D={d}"), &rt, v, 5)
        }
        "u1-small" => {
            let v = GradedSpace::try_new(
                Arc::new(U1FusionRule),
                (-8..=8).map(|q| (U1Irrep::new(q), 2)),
            )?;
            time!(label, &rt, v, 50)
        }
        "su2-small" => {
            let v = GradedSpace::try_new(
                Arc::new(SU2FusionRule),
                (0..=8).map(|twice_spin| (SU2Irrep::from_twice_spin(twice_spin), 2)),
            )?;
            time!(label, &rt, v, 20)
        }
        other => Err(format!("unknown ENGINE_SMOKE_CASE {other}").into()),
    }
}
