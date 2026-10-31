//! Allocation ledger of `planar_contract`, warm, one dense thread.
//!
//! Payload-sized allocations only (at least half the smallest payload), so
//! per-call plan metadata is not counted. The reference is the core alone:
//! `compose` of the operands already transposed by hand, whose payload
//! allocations are its output plus whatever the dense provider's GEMM owns.
//! The hand-written sequence owns every transposed payload. `planar_contract`
//! takes the same work: its operand and output transposes write per-call
//! scratch, as TensorOperations allocates its temporaries per call, so it
//! must match the hand sequence and `planar_contract_into` must match it less
//! the output.

use crate::counting_alloc;
use symtn::typed::Engine;

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{TableRule, TableSector};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

type Table = TensorMap<TableRule, Complex64>;

fn payloads(bytes: usize, f: impl FnOnce()) -> u64 {
    let ((), allocs) = counting_alloc::measure_matching(bytes..=usize::MAX, f);
    allocs.matched_calls
}

type Axes<'a> = (&'a [usize], &'a [usize]);

struct Geometry<'a> {
    rhs: &'a Table,
    spec: ContractSpec<'a>,
    lhs_steps: Axes<'a>,
    rhs_steps: Axes<'a>,
    output_steps: Option<Axes<'a>>,
}

#[test]
fn planar_contract_owns_only_its_output() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(TableRule::fibonacci()),
        [(TableSector::new(1), 6), (TableSector::new(2), 8)],
    )
    .unwrap();
    let a: Table = TensorMap::rand_with_seed(engine, &runtime, [&v, &v], [&v, &v], 1).unwrap();
    let b: Table = TensorMap::rand_with_seed(engine, &runtime, [&v, &v], [&v, &v], 2).unwrap();
    let theta: Table = TensorMap::rand_with_seed(engine, &runtime, [&v, &v, &v], [&v], 3).unwrap();
    let geometries = [
        // `@planar C[b d; h f] := A[a b; c d] * B[c f; a h]`: both operands
        // transposed, the output in core order.
        Geometry {
            rhs: &b,
            spec: ContractSpec {
                lhs: &[0, 2],
                rhs: &[2, 0],
                codomain: &[0, 1],
                domain: &[3, 2],
            },
            lhs_steps: (&[1, 3], &[0, 2]),
            rhs_steps: (&[2, 0], &[3, 1]),
            output_steps: None,
        },
        // The golden-chain step: `self` in place, the output rotated.
        Geometry {
            rhs: &theta,
            spec: ContractSpec {
                lhs: &[2, 3],
                rhs: &[1, 2],
                codomain: &[2, 0, 1],
                domain: &[3],
            },
            lhs_steps: (&[0, 1], &[2, 3]),
            rhs_steps: (&[1, 2], &[0, 3]),
            output_steps: Some((&[2, 0, 1], &[3])),
        },
    ];
    for Geometry {
        rhs,
        spec,
        lhs_steps,
        rhs_steps,
        output_steps,
    } in geometries
    {
        let lhs_core = a.transpose(engine, lhs_steps.0, lhs_steps.1).unwrap();
        let rhs_core = rhs.transpose(engine, rhs_steps.0, rhs_steps.1).unwrap();
        let mut out = a.planar_contract(engine, rhs, &spec).unwrap();
        let payload = std::mem::size_of::<Complex64>()
            * [&a, rhs, &out]
                .iter()
                .map(|t| t.dense_data().unwrap().len())
                .min()
                .unwrap()
            / 2;
        let hand = |engine: &mut dyn Engine, lhs: &Table, rhs: &Table| {
            let core = lhs.compose(engine, rhs).unwrap();
            match output_steps {
                Some((codomain, domain)) => core.transpose(engine, codomain, domain).unwrap(),
                None => core,
            }
        };
        for _ in 0..2 {
            drop(a.planar_contract(engine, rhs, &spec).unwrap());
            a.planar_contract_into(
                engine,
                rhs,
                &spec,
                &mut out,
                Complex64::ONE,
                Complex64::ZERO,
            )
            .unwrap();
            drop(hand(engine, &lhs_core, &rhs_core));
        }
        let core = payloads(payload, || {
            drop(lhs_core.compose(engine, &rhs_core).unwrap())
        });
        let eager = payloads(payload, || {
            drop(a.planar_contract(engine, rhs, &spec).unwrap())
        });
        let into = payloads(payload, || {
            a.planar_contract_into(
                engine,
                rhs,
                &spec,
                &mut out,
                Complex64::ONE,
                Complex64::ZERO,
            )
            .unwrap()
        });
        let hand_written = payloads(payload, || {
            let lhs = a.transpose(engine, lhs_steps.0, lhs_steps.1).unwrap();
            let rhs = rhs.transpose(engine, rhs_steps.0, rhs_steps.1).unwrap();
            drop(hand(engine, &lhs, &rhs));
        });
        assert!(core >= 1, "{spec:?}: the core owns its output");
        assert_eq!(
            eager, hand_written,
            "{spec:?}: eager = the hand sequence's transposes and core"
        );
        assert_eq!(
            into,
            hand_written - 1,
            "{spec:?}: the destination form owns no output payload"
        );
        // An identity `transpose` returns its input, so the golden-chain
        // step's hand sequence copies θ and the output only.
        let transposes = 2;
        assert_eq!(
            hand_written,
            core + transposes,
            "{spec:?}: the hand sequence owns each transposed payload"
        );
    }
}
