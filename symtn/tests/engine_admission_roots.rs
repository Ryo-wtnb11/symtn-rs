//! Admitting the caller's engine takes no execution resource from it: a Host
//! operation that decides no width and runs no parallel section creates no
//! ledger root on a team-backed engine, while an operation that forks on the
//! same engine creates its one root.

#![cfg(feature = "ledger-roots")]

use num_complex::Complex64;
use symtn::sector::U1FusionRule;
use symtn::typed::{CpuEngine, GradedSpace, Runtime, TensorMap};
use symtn_operations::host_pool::eager_test_engine;
use tenferro_cpu::threadteam::ledger;

use crate::legs::u1_leg;

const THREADS: usize = 4;

fn leg() -> GradedSpace<U1FusionRule> {
    u1_leg(&[(-1, 5), (0, 6), (1, 4)])
}

/// A team-backed engine of `THREADS` workers whose cost rule admits every
/// parallel section, so the forking control cannot fall back to serial.
fn team_engine() -> CpuEngine {
    let engine = eager_test_engine(THREADS);
    assert!(engine.has_team(), "the engine holds a reserved team");
    assert_eq!(engine.num_threads(), THREADS);
    engine
}

/// The ledger roots `op` creates on `engine`, setup roots drained first.
fn roots_of<R>(engine: &mut CpuEngine, op: impl FnOnce(&mut CpuEngine) -> R) -> Vec<u64> {
    engine.take_ledger_roots();
    op(engine);
    engine.take_ledger_roots()
}

macro_rules! no_decision_ops_create_no_root {
    ($name:ident, $d:ty) => {
        #[test]
        fn $name() {
            let runtime = Runtime::builder().build().unwrap();
            let v = leg();
            let setup = &mut CpuEngine::serial();
            let x: TensorMap<U1FusionRule, $d> =
                TensorMap::rand_with_seed(setup, &runtime, [&v, &v], [&v], 11).unwrap();
            let mut y = TensorMap::rand_with_seed(setup, &runtime, [&v, &v], [&v], 12).unwrap();
            let physical = x.to_physical_dense(setup).unwrap();
            assert!(x.subblock_count() > 1, "a multi-block fixture");
            let one = <$d>::from(1.0);
            let zero = <$d>::from(0.0);

            let mut engine = team_engine();
            let cases: [(&str, Vec<u64>); 7] = [
                (
                    "axpby_into",
                    roots_of(&mut engine, |e| x.axpby_into(e, &mut y, one, one).unwrap()),
                ),
                (
                    "copy",
                    roots_of(&mut engine, |e| x.axpby_into(e, &mut y, one, zero).unwrap()),
                ),
                ("inner", roots_of(&mut engine, |e| x.inner(e, &y).unwrap())),
                ("norm", roots_of(&mut engine, |e| x.norm(e, 2.0).unwrap())),
                (
                    "scale_assign",
                    roots_of(&mut engine, |e| y.scale_assign(e, one).unwrap()),
                ),
                (
                    "to_physical_dense",
                    roots_of(&mut engine, |e| x.to_physical_dense(e).unwrap()),
                ),
                (
                    "project_physical_dense",
                    roots_of(&mut engine, |e| {
                        x.project_physical_dense(e, &physical).unwrap()
                    }),
                ),
            ];
            let minted: Vec<_> = cases
                .iter()
                .filter(|(_, roots)| !roots.is_empty())
                .map(|(what, roots)| (*what, roots.len()))
                .collect();
            assert!(minted.is_empty(), "roots per operation: {minted:?}");

            // Why the control: zero roots on an engine that never forks would
            // prove nothing. A nonidentity permute replays its blocks across
            // the team on this same engine.
            let roots = roots_of(&mut engine, |e| x.permute(e, &[2, 0], &[1]).unwrap());
            assert_eq!(roots.len(), 1, "one root for the forking permute");
            assert!(ledger::max_live(roots[0]) > 1, "the permute forks");
        }
    };
}

no_decision_ops_create_no_root!(no_decision_ops_create_no_root_f64, f64);
no_decision_ops_create_no_root!(no_decision_ops_create_no_root_c64, Complex64);
