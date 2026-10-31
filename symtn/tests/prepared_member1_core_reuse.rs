//! The acceptance gate of the workspace-owned `B = 1` core bindings: a
//! warmed Host `ComposePlan`/`ContractPlan` workspace prepares no compatible
//! grouped core plan again (affine batches' binary-plan cache misses are
//! outside this guarantee).
//!
//! The observation is the calling thread's allocations with one dense
//! thread: preparing a native grouped plan always allocates its child list on
//! the calling thread, so a call that allocates nothing prepared nothing.
//! New payload values on every call show that payloads and buffers are not
//! part of the binding. Leaving `B = 1` releases the bindings: the return to
//! `B = 1` is a cold call again, and its retained footprint equals the first
//! one. A second workspace of the same plan warms its own bindings, and
//! `retained_bytes` covers what dropping it frees.

// The native grouped route is the faer provider's; a BLAS provider runs the
// grouped jobs through its dot and prepares no plan.
use crate::backend;

use crate::prepared;

use std::cell::RefCell;
use std::fmt::Debug;

use symtn_dense::CpuEngine;

use num_complex::Complex64;
use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::typed::{
    ComposePlan, ContractPlan, ContractSpec, ContractWorkspace, GradedSpace, Runtime,
    StackedTensorMap,
};

use crate::common::Payload;
use crate::counting_alloc;
use prepared::{fz2u1_legs, members, u1_legs};

fn calls(f: impl FnOnce()) -> u64 {
    counting_alloc::measure(f).1.calls
}

fn data<R, D: Payload>(stack: &StackedTensorMap<R, D>) -> Vec<D>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    (0..stack.len())
        .flat_map(|i| {
            stack
                .member(engine, i)
                .unwrap()
                .dense_data()
                .unwrap()
                .to_vec()
        })
        .collect()
}

/// One prepared operation under test: `execute` and `execute_into` over a
/// workspace, for stacks of `count` members with salt `salt`.
trait Prepared<R, D> {
    fn stacks(&self, count: usize, salt: usize)
        -> (StackedTensorMap<R, D>, StackedTensorMap<R, D>);
    fn workspace(&self) -> ContractWorkspace<R, D>;
    fn execute<'w>(
        &self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        workspace: &'w mut ContractWorkspace<R, D>,
    ) -> &'w StackedTensorMap<R, D>;
    fn execute_into(
        &self,
        lhs: &StackedTensorMap<R, D>,
        rhs: &StackedTensorMap<R, D>,
        dst: &mut StackedTensorMap<R, D>,
        workspace: &mut ContractWorkspace<R, D>,
    );
}

fn check<R, D, P>(name: &str, op: &P)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Debug,
    D: Payload,
    P: Prepared<R, D>,
{
    let mut workspace = op.workspace();
    let (lhs, rhs) = op.stacks(1, 1);
    let cold = calls(|| {
        op.execute(&lhs, &rhs, &mut workspace);
    });
    let footprint = workspace.retained_bytes();
    let mut dst = {
        let mut other = op.workspace();
        op.execute(&lhs, &rhs, &mut other);
        other.take_output().unwrap()
    };
    op.execute_into(&lhs, &rhs, &mut dst, &mut workspace);
    for salt in 2..5 {
        let (lhs, rhs) = op.stacks(1, salt);
        let mut reference_ws = op.workspace();
        let want = data(op.execute(&lhs, &rhs, &mut reference_ws));
        let warm = calls(|| {
            op.execute(&lhs, &rhs, &mut workspace);
        });
        assert_eq!(warm, 0, "{name} salt={salt}: warm execute prepared");
        assert_eq!(data(op.execute(&lhs, &rhs, &mut workspace)), want, "{name}");
        let warm_into = calls(|| op.execute_into(&lhs, &rhs, &mut dst, &mut workspace));
        assert_eq!(
            warm_into, 0,
            "{name} salt={salt}: warm execute_into prepared"
        );
        assert_eq!(data(&dst), want, "{name} salt={salt}: execute_into");
        assert_eq!(workspace.retained_bytes(), footprint, "{name}: footprint");
    }
    // A second workspace of the same plan warms independently.
    let mut second = op.workspace();
    assert!(
        calls(|| {
            op.execute(&lhs, &rhs, &mut second);
        }) > 0
    );
    assert_eq!(
        calls(|| {
            op.execute(&lhs, &rhs, &mut second);
        }),
        0,
        "{name}: second workspace warm"
    );
    assert_eq!(
        second.retained_bytes(),
        footprint,
        "{name}: second footprint"
    );
    // The facade query covers everything the warmed workspace owns, its
    // core bindings included: dropping it frees no more than it reports.
    let ((), dropped) = counting_alloc::measure(|| drop(second));
    assert!(
        dropped.freed_bytes > 0 && dropped.freed_bytes as usize <= footprint,
        "{name}: freed {} B against retained_bytes {footprint} B",
        dropped.freed_bytes
    );
    eprintln!(
        "{name}: drop freed {} B of {footprint} B",
        dropped.freed_bytes
    );
    // B = 2 and 17 release the bindings; B = 1 then prepares again.
    for count in [2, 17] {
        let (lhs, rhs) = op.stacks(count, 7);
        op.execute(&lhs, &rhs, &mut workspace);
    }
    let again = calls(|| {
        op.execute(&lhs, &rhs, &mut workspace);
    });
    assert!(
        again >= 2,
        "{name}: the return to B = 1 rebuilds ({again} calls)"
    );
    assert_eq!(workspace.retained_bytes(), footprint, "{name}: B = 1 again");
    assert_eq!(
        calls(|| {
            op.execute(&lhs, &rhs, &mut workspace);
        }),
        0,
        "{name}: warm again"
    );
    eprintln!("{name}: cold={cold} calls, again={again} calls, retained {footprint} B");
}

struct Compose<R, D> {
    engine: RefCell<CpuEngine>,
    runtime: Runtime,
    legs: (GradedSpace<R>, GradedSpace<R>),
    plan: ComposePlan<R, D>,
}

struct Contract<R, D> {
    engine: RefCell<CpuEngine>,
    runtime: Runtime,
    legs: (GradedSpace<R>, GradedSpace<R>),
    plan: ContractPlan<R, D>,
}

macro_rules! prepared_impl {
    ($ty:ident) => {
        impl<R, D> Prepared<R, D> for $ty<R, D>
        where
            R: symtn::sector::TypedSectorAdmission<
                    Sector = <R as SectorCodec>::Sector,
                    Error = symtn::typed::FusionAlgebraError,
                    Mode = symtn::sector::MultiplicityFreeAdmissionMode,
                > + MultiplicityFreeRigidSymbols<Scalar = f64>
                + CheckedFusionAlgebra
                + SectorCodec,
            D: Payload,
        {
            fn stacks(
                &self,
                count: usize,
                salt: usize,
            ) -> (StackedTensorMap<R, D>, StackedTensorMap<R, D>) {
                let (v, w) = (&self.legs.0, &self.legs.1);
                (
                    StackedTensorMap::pack(&members::<R, D>(
                        &self.runtime,
                        &[v, v],
                        &[w],
                        count,
                        salt,
                    ))
                    .unwrap(),
                    StackedTensorMap::pack(&members::<R, D>(
                        &self.runtime,
                        &[w],
                        &[v],
                        count,
                        salt + 1,
                    ))
                    .unwrap(),
                )
            }
            fn workspace(&self) -> ContractWorkspace<R, D> {
                self.plan.workspace().unwrap()
            }
            fn execute<'w>(
                &self,
                lhs: &StackedTensorMap<R, D>,
                rhs: &StackedTensorMap<R, D>,
                workspace: &'w mut ContractWorkspace<R, D>,
            ) -> &'w StackedTensorMap<R, D> {
                self.plan
                    .execute(&mut *self.engine.borrow_mut(), lhs, rhs, workspace)
                    .unwrap()
            }
            fn execute_into(
                &self,
                lhs: &StackedTensorMap<R, D>,
                rhs: &StackedTensorMap<R, D>,
                dst: &mut StackedTensorMap<R, D>,
                workspace: &mut ContractWorkspace<R, D>,
            ) {
                self.plan
                    .execute_into(&mut *self.engine.borrow_mut(), lhs, rhs, dst, workspace)
                    .unwrap()
            }
        }
    };
}
prepared_impl!(Compose);
prepared_impl!(Contract);

fn compose<R, D>(legs: (GradedSpace<R>, GradedSpace<R>)) -> Compose<R, D>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let mut engine = CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = (&legs.0, &legs.1);
    let lhs = StackedTensorMap::pack(&members::<R, D>(&runtime, &[v, v], &[w], 1, 0)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<R, D>(&runtime, &[w], &[v], 1, 1)).unwrap();
    // A first plan fills the process-wide caches and the engine's scratch.
    let warmup = ComposePlan::new(&mut engine, &lhs, &rhs).unwrap();
    warmup
        .execute(&mut engine, &lhs, &rhs, &mut warmup.workspace().unwrap())
        .unwrap();
    let plan = ComposePlan::new(&mut engine, &lhs, &rhs).unwrap();
    Compose {
        engine: RefCell::new(engine),
        runtime,
        legs,
        plan,
    }
}

#[test]
fn warmed_member1_workspace_prepares_no_grouped_core_plan() {
    let _serial = counting_alloc::serial();
    if !backend::faer_is_default_backend() {
        return;
    }
    check("compose U1 f64", &compose::<_, f64>(u1_legs()));
    // Fermionic signs: one grouped submission per coefficient group.
    check("compose fZ2xU1 c64", &compose::<_, Complex64>(fz2u1_legs()));

    // A ContractPlan whose core replays the operands in place.
    let mut engine = CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let legs = u1_legs();
    let (v, w) = (&legs.0, &legs.1);
    let spec = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[0, 1],
        domain: &[2],
    };
    let lhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[v, v], &[w], 1, 0)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[w], &[v], 1, 1)).unwrap();
    let warmup = ContractPlan::new(&mut engine, &lhs, &rhs, &spec).unwrap();
    warmup
        .execute(&mut engine, &lhs, &rhs, &mut warmup.workspace().unwrap())
        .unwrap();
    let plan = ContractPlan::new(&mut engine, &lhs, &rhs, &spec).unwrap();
    check(
        "contract U1 f64 core",
        &Contract {
            engine: RefCell::new(engine),
            runtime,
            legs,
            plan,
        },
    );
}
