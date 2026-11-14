//! Provider-query timing of the SU(2) rules: `SU2FusionRule` (MF, scalar
//! symbols) against `LieGroupFusionRule(su(2))` in checked Generic mode
//! (`su2cg`, block symbols) and in multiplicity-free mode (`su2mf`, scalar
//! symbols). Run with `cargo bench -p symtn --bench su2_rule_provider`.
//!
//! One child process per (rule, query), so the codec's irrep table and the
//! fusion-symbols caches start cold; the first pass is cold, the next
//! [`WARM_PASSES`] are warm. Labels `2j` lie in `0..=SYMTN_SU2_PROVIDER_MAX`
//! (default 40); every query stays in that domain (channels above it are
//! dropped), so channel, N and F sets are the same for both rules.
//! Allocation counts cover the caller thread. Timings are evidence, never a
//! CI threshold.

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

use symtn::sector::{
    CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols, FusionRule,
    LieGroupFusionRule, MultiplicityFreeAdmissionMode, MultiplicityFreeFusionSymbols,
    SU2FusionRule, SU2Irrep, SectorCodec, SectorId, TypedSectorAdmission,
};

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

const CASE_ENV: &str = "SYMTN_SU2_PROVIDER_CASE";
const MAX_ENV: &str = "SYMTN_SU2_PROVIDER_MAX";
const WARM_PASSES: usize = 5;

/// The queries of one rule, over its own ids.
trait Queries {
    fn encode(&self, twice: usize) -> SectorId;
    fn channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId>;
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize;
    fn f(&self, ids: [SectorId; 6]) -> f64;
}

impl Queries for SU2FusionRule {
    fn encode(&self, twice: usize) -> SectorId {
        self.encode_sector(&SU2Irrep::from_twice_spin(twice))
            .unwrap()
    }
    fn channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
        self.fusion_channels(left, right).into_iter().collect()
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        FusionRule::nsymbol(self, left, right, coupled)
    }
    fn f(&self, [a, b, c, d, e, f]: [SectorId; 6]) -> f64 {
        self.f_symbol_scalar(a, b, c, d, e, f)
    }
}

impl Queries for LieGroupFusionRule<MultiplicityFreeAdmissionMode> {
    fn encode(&self, twice: usize) -> SectorId {
        let label = self.category().irrep(&[twice as i64]).unwrap();
        self.encode_sector(&label).unwrap()
    }
    fn channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
        self.fusion_channels(left, right).into_iter().collect()
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        FusionRule::nsymbol(self, left, right, coupled)
    }
    fn f(&self, [a, b, c, d, e, f]: [SectorId; 6]) -> f64 {
        self.f_symbol_scalar(a, b, c, d, e, f)
    }
}

impl Queries for LieGroupFusionRule {
    fn encode(&self, twice: usize) -> SectorId {
        let label = self.category().irrep(&[twice as i64]).unwrap();
        self.try_encode_label(&label).unwrap()
    }
    fn channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
        self.try_fusion_channels(left, right)
            .unwrap()
            .into_iter()
            .collect()
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.try_nsymbol(left, right, coupled).unwrap()
    }
    fn f(&self, [a, b, c, d, e, f]: [SectorId; 6]) -> f64 {
        let block = self.try_f_symbol_generic(a, b, c, d, e, f).unwrap();
        assert_eq!(block.shape(), (1, 1, 1, 1));
        block.data()[0]
    }
}

/// The admissible F sextuples `(a, b, c, d, e, f)` in `2j` labels, with
/// `e ∈ a⊗b`, `d ∈ e⊗c`, `f ∈ b⊗c`, `d ∈ a⊗f`, all `<= max`, visited in
/// place (about 9·10^7 of them at `max = 40`).
fn for_each_sextuple(max: usize, mut visit: impl FnMut([usize; 6])) {
    let channels = |a: usize, b: usize| (a.abs_diff(b)..=(a + b).min(max)).step_by(2);
    for a in 0..=max {
        for b in 0..=max {
            for e in channels(a, b) {
                for c in 0..=max {
                    for d in channels(e, c) {
                        for f in channels(b, c) {
                            if a.abs_diff(f) <= d && d <= a + f && (a + f + d) % 2 == 0 {
                                visit([a, b, c, d, e, f]);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn pass<T>(f: impl FnOnce() -> T) -> (T, u64, u64, u128) {
    let ((output, ns), allocs) = counting_alloc::measure(|| {
        let started = std::time::Instant::now();
        let output = f();
        (output, started.elapsed().as_nanos())
    });
    (output, allocs.calls, allocs.bytes, ns)
}

fn run<R: Queries>(case: &str, query: &str, max: usize, make: impl FnOnce() -> R) {
    let (rule, calls, bytes, ns) = pass(make);
    println!("case={case} phase=construct ns={ns} allocations={calls} requested_bytes={bytes}");
    let (ids, calls, bytes, ns) = pass(|| {
        (0..=max)
            .map(|twice| rule.encode(twice))
            .collect::<Vec<_>>()
    });
    println!(
        "case={case} phase=encode count={} ns={ns} allocations={calls} requested_bytes={bytes}",
        ids.len()
    );
    let in_domain = |id: &SectorId| ids.contains(id);
    let once = || -> (usize, f64) {
        match query {
            "channels" => {
                let mut count = 0;
                let mut checksum = 0.0;
                for &a in &ids {
                    for &b in &ids {
                        let channels = rule.channels(a, b);
                        count += 1;
                        checksum += channels.iter().filter(|id| in_domain(id)).count() as f64;
                    }
                }
                (count, checksum)
            }
            "nsymbol" => {
                let mut count = 0;
                let mut checksum = 0.0;
                for &a in &ids {
                    for &b in &ids {
                        for &c in &ids {
                            checksum += rule.nsymbol(a, b, c) as f64;
                            count += 1;
                        }
                    }
                }
                (count, checksum)
            }
            "f" => {
                let mut count = 0;
                let mut checksum = 0.0;
                for_each_sextuple(max, |labels| {
                    let value = rule.f(labels.map(|twice| ids[twice]));
                    checksum += value * value;
                    count += 1;
                });
                (count, checksum)
            }
            _ => panic!("unknown query: {query}"),
        }
    };
    let ((count, checksum), calls, bytes, ns) = pass(once);
    println!("case={case} phase=cold count={count} ns={ns} allocations={calls} requested_bytes={bytes} checksum={checksum:.12e}");
    let mut samples = Vec::with_capacity(WARM_PASSES);
    for _ in 0..WARM_PASSES {
        let ((warm_count, warm_checksum), calls, bytes, ns) = pass(once);
        assert_eq!((warm_count, warm_checksum), (count, checksum));
        samples.push((ns, calls, bytes));
    }
    let mut sorted: Vec<u128> = samples.iter().map(|s| s.0).collect();
    sorted.sort_unstable();
    println!(
        "case={case} phase=warm count={count} samples={samples:?} median_ns={}",
        sorted[sorted.len() / 2]
    );
}

fn main() {
    let max: usize = std::env::var(MAX_ENV)
        .ok()
        .map_or(40, |value| value.parse().unwrap());
    let Some(case) = std::env::var_os(CASE_ENV) else {
        for rule in ["su2", "su2cg", "su2mf"] {
            for query in ["channels", "nsymbol", "f"] {
                let case = format!("{rule}/{query}");
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .env(CASE_ENV, &case)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "case={case} status={}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                );
                print!("{}", String::from_utf8_lossy(&output.stdout));
            }
        }
        return;
    };
    let case = case.to_str().unwrap();
    let (rule, query) = case.split_once('/').unwrap();
    println!("case={case} max_twice_spin={max} allocation_scope=caller_thread_requested");
    match rule {
        "su2" => run(case, query, max, || SU2FusionRule),
        "su2cg" => run(case, query, max, || {
            LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
                fusion_symbols::lie::GroupId::su(2).unwrap(),
            )
            .unwrap()
        }),
        "su2mf" => run(case, query, max, || {
            LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(
                fusion_symbols::lie::GroupId::su(2).unwrap(),
            )
            .unwrap()
        }),
        _ => panic!("unknown rule: {rule}"),
    }
}
