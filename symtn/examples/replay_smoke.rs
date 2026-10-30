//! Experiment-only replay smoke (F2), not for merge. For each of U(1), SU(2)
//! and U(1)xSU(2), a rank-4 tensor is permuted (recoupling replay) at
//! `REPLAY_SMOKE_THREADS` dense and recoupling threads; prints the warm time per
//! permute and an FNV-1a hash of the result bits, so runs at different thread
//! counts and revisions can be compared bit for bit. `REPLAY_SMOKE_D` sets the
//! degeneracy of every sector (default 4), `REPLAY_SMOKE_REPS` the repetitions.
use std::sync::Arc;
use std::time::Instant;
use symtn::sector::{product_sector, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

fn fnv(data: &[f64]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325, |h, x| {
        x.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
    })
}

macro_rules! run {
    ($label:expr, $rt:expr, $v:expr, $reps:expr) => {{
        let reps: u32 = std::env::var("REPLAY_SMOKE_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or($reps);
        let v = $v;
        let t: TensorMap<_, f64> = TensorMap::rand_with_seed($rt, [&v, &v], [&v, &v], 7)?;
        let warm = t.permute(&[2, 0], &[3, 1])?;
        let start = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(t.permute(&[2, 0], &[3, 1])?);
        }
        let per = start.elapsed() / reps;
        println!(
            "{} len={} per_permute={per:?} hash={:016x}",
            $label,
            warm.dense_data()?.len(),
            fnv(warm.dense_data()?)
        );
    }};
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let threads: usize = std::env::var("REPLAY_SMOKE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let d: usize = std::env::var("REPLAY_SMOKE_D")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let rt = Runtime::builder()
        .dense_threads(threads)
        .recoupling_threads(threads)
        .build()?;
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (-3..=3).map(|q| (U1Irrep::new(q), d)),
    )?;
    run!(format!("u1 threads={threads}"), &rt, u1, 20);
    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        (0..=4).map(|s| (SU2Irrep::from_twice_spin(s), d)),
    )?;
    run!(format!("su2 threads={threads}"), &rt, su2, 20);
    let provider: Arc<symtn::sector::ProductFusionRule<U1FusionRule, SU2FusionRule>> = Arc::new(
        symtn::sector::ProductFusionRule::try_new(U1FusionRule, SU2FusionRule)?,
    );
    let sector = |q, s| product_sector(U1Irrep::new(q), SU2Irrep::from_twice_spin(s));
    let prod = GradedSpace::try_new(
        provider,
        [
            (sector(0, 0), d),
            (sector(1, 1), d),
            (sector(-1, 1), d),
            (sector(0, 2), d),
            (sector(2, 0), d),
        ],
    )?;
    run!(format!("u1xsu2 threads={threads}"), &rt, prod, 20);
    Ok(())
}
