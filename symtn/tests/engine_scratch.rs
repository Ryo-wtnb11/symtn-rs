//! Eager Host contraction scratch kept in the caller's CPU engine
//! (`symtn::scratch`): a cold call allocates its route scratch, a warm call
//! allocates none of it, the kept ledger is bounded and trim releases it,
//! and stale scratch never reaches an output.
//!
//! Sizes are the public payload sizes of the operands (`op`) and the output
//! (`out`), all distinct in each fixture, so an allocation of one of them is
//! that buffer or a scratch of its size, never another fixture's.

use std::num::NonZeroUsize;
use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, SU2Irrep, TableRule, TableSector, U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, CpuEngine, GradedSpace, Runtime, TensorMap};
use tenferro_runtime::ExecutionEngine;

use crate::counting_alloc;

/// Allocations of exactly each of `sizes` bytes, summed over the sizes; `f`
/// runs once per size, on `engine`, or on a fresh engine each time for
/// `None`.
fn sized_allocs(
    sizes: &[usize],
    mut engine: Option<&mut CpuEngine>,
    mut f: impl FnMut(&mut CpuEngine),
) -> u64 {
    sizes
        .iter()
        .map(|&bytes| {
            let mut fresh = CpuEngine::serial();
            let engine = match engine.as_deref_mut() {
                Some(engine) => engine,
                None => &mut fresh,
            };
            counting_alloc::measure_matching(bytes..=bytes, || f(engine))
                .1
                .matched_calls
        })
        .sum()
}

fn payload_bytes<R, D>(t: &TensorMap<R, D>) -> usize {
    std::mem::size_of_val(t.dense_data().unwrap())
}

fn u1(base: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(-1, base), (0, base + 1), (1, base)]
            .map(|(charge, n)| (U1Irrep::new(charge).unwrap(), n)),
    )
    .unwrap()
}

fn su2(base: usize) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [(0, base), (1, base + 1), (2, 1)].map(|(twice, n)| (SU2Irrep::from_twice_spin(twice), n)),
    )
    .unwrap()
}

/// The two contraction shapes of every matrix row.
///
/// - `3x3`: `a: [c, q] ← [p]`, `b: [r] ← [c, s]`, contracting `a`'s codomain
///   `c` with `b`'s domain `c`: both operands are transformed into core form.
/// - `4x2`: `a: [p, q] ← [c, r]`, `b: [c] ← [s]`, contracting `c`, with the
///   output `(p, s | q, r)` mixing the operands' legs: the core writes a
///   destination that one output transform moves into the result.
macro_rules! cases {
    ($engine:expr, $runtime:expr, $leg:ident, $d:ty) => {{
        let (c, p, q, r, s) = ($leg(2), $leg(3), $leg(4), $leg(5), $leg(6));
        let three = (
            "3x3",
            TensorMap::<_, $d>::rand_with_seed($engine, $runtime, [&c, &q], [&p], 1).unwrap(),
            TensorMap::<_, $d>::rand_with_seed($engine, $runtime, [&r], [&c, &s], 2).unwrap(),
            ContractSpec {
                lhs: &[0],
                rhs: &[1],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        );
        let four = (
            "4x2",
            TensorMap::<_, $d>::rand_with_seed($engine, $runtime, [&p, &q], [&c, &r], 3).unwrap(),
            TensorMap::<_, $d>::rand_with_seed($engine, $runtime, [&c], [&s], 4).unwrap(),
            ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 3],
                domain: &[1, 2],
            },
        );
        [three, four]
    }};
}

/// Runs `$body` with `(name, a, b, spec, op, out)` for every row of the
/// matrix {U(1), SU(2)} × {f64, c64} × {3x3, 4x2}.
macro_rules! for_each_case {
    (|$name:ident, $a:ident, $b:ident, $spec:ident, $op:ident, $out:ident, $one:ident, $zero:ident| $body:block) => {{
        let setup = &mut CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        for_each_case!(@rule setup, &runtime, u1, f64, $name, $a, $b, $spec, $op, $out, $one, $zero, $body);
        for_each_case!(@rule setup, &runtime, u1, Complex64, $name, $a, $b, $spec, $op, $out, $one, $zero, $body);
        for_each_case!(@rule setup, &runtime, su2, f64, $name, $a, $b, $spec, $op, $out, $one, $zero, $body);
        for_each_case!(@rule setup, &runtime, su2, Complex64, $name, $a, $b, $spec, $op, $out, $one, $zero, $body);
    }};
    (@rule $setup:expr, $runtime:expr, $leg:ident, $d:ty, $name:ident, $a:ident, $b:ident, $spec:ident, $op:ident, $out:ident, $one:ident, $zero:ident, $body:block) => {
        for (shape, $a, $b, $spec) in cases!($setup, $runtime, $leg, $d) {
            let $name = format!("{} {} {shape}", stringify!($leg), stringify!($d));
            let reference = $a.contract($setup, &$b, &$spec).unwrap();
            let $op = [payload_bytes(&$a), payload_bytes(&$b)];
            let $out = payload_bytes(&reference);
            assert!($op[0] != $op[1] && !$op.contains(&$out), "{}: sizes coincide", $name);
            let $one = <$d>::from(1.0);
            let $zero = <$d>::from(0.0);
            $body
        }
    };
}

#[test]
fn cold_plain_contract_into_allocates_route_scratch() {
    let _serial = counting_alloc::serial();
    for_each_case!(|name, a, b, spec, op, out, one, zero| {
        let mut dst = a.contract(&mut CpuEngine::serial(), &b, &spec).unwrap();
        let mut into = |engine: &mut CpuEngine| {
            a.contract_into(engine, &b, &spec, &mut dst, one, zero)
                .unwrap()
        };
        // What: a fresh engine keeps nothing, so the transformed operand (and,
        // for the output transform, the core destination) is allocated; the
        // destination is preallocated, so none of these is the output.
        assert!(sized_allocs(&op, None, &mut into) >= 1, "{name}: op");
        if name.ends_with("4x2") {
            assert!(sized_allocs(&[out], None, &mut into) >= 1, "{name}: out");
        }
    });
}

#[test]
fn warm_plain_contract_allocates_no_operand_or_core_scratch() {
    let _serial = counting_alloc::serial();
    for_each_case!(|name, a, b, spec, op, out, one, zero| {
        let engine = &mut CpuEngine::serial();
        let mut dst = a.contract(engine, &b, &spec).unwrap();
        let sizes = [op[0], op[1], out];
        let mut owned = |engine: &mut CpuEngine| drop(a.contract(engine, &b, &spec).unwrap());
        // What: cold, an owned call allocates more than its one output.
        assert!(sized_allocs(&sizes, None, &mut owned) >= 2, "{name}: cold");
        a.contract_into(engine, &b, &spec, &mut dst, one, zero)
            .unwrap();
        // What: warm, an owned call allocates its output alone, and a
        // destination call nothing of these sizes.
        assert_eq!(
            sized_allocs(&sizes, Some(engine), &mut owned),
            1,
            "{name}: owned"
        );
        let into = |engine: &mut CpuEngine| {
            a.contract_into(engine, &b, &spec, &mut dst, one, zero)
                .unwrap()
        };
        assert_eq!(sized_allocs(&sizes, Some(engine), into), 0, "{name}: into");
    });
}

/// Planar contraction: U(1), which delegates to the ordinary contraction,
/// and Fibonacci, whose planar dispatch transposes both operands.
#[test]
fn warm_planar_contract_allocates_no_route_scratch() {
    let _serial = counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let setup = &mut CpuEngine::serial();
    let spec = ContractSpec {
        lhs: &[0, 2],
        rhs: &[2, 0],
        codomain: &[0, 1],
        domain: &[3, 2],
    };
    macro_rules! check {
        ($name:expr, $v:expr, $w:expr, $d:ty) => {{
            let (v, w) = ($v, $w);
            let a =
                TensorMap::<_, $d>::rand_with_seed(setup, &runtime, [&v, &w], [&v, &w], 1).unwrap();
            let b =
                TensorMap::<_, $d>::rand_with_seed(setup, &runtime, [&v, &w], [&v, &w], 2).unwrap();
            let sizes = [payload_bytes(&a)];
            let out = a.planar_contract(setup, &b, &spec).unwrap();
            assert_ne!(payload_bytes(&out), sizes[0], "{}: sizes coincide", $name);
            let mut call =
                |engine: &mut CpuEngine| drop(a.planar_contract(engine, &b, &spec).unwrap());
            assert!(
                sized_allocs(&sizes, None, &mut call) >= 1,
                "{}: cold",
                $name
            );
            let engine = &mut CpuEngine::serial();
            call(engine);
            assert_eq!(
                sized_allocs(&sizes, Some(engine), &mut call),
                0,
                "{}: warm",
                $name
            );
        }};
    }
    check!("U(1)", u1(3), u1(4), f64);
    let fib = |a, b| {
        GradedSpace::try_new(
            Arc::new(TableRule::fibonacci()),
            [(TableSector::new(1), a), (TableSector::new(2), b)],
        )
        .unwrap()
    };
    check!("Fibonacci", fib(3, 4), fib(5, 2), Complex64);
}

/// The checked Generic route, over SU(3) with fusion multiplicity.
#[cfg(feature = "cgc-gen")]
#[test]
fn warm_checked_generic_contract_allocates_no_route_scratch() {
    let _serial = counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let setup = &mut CpuEngine::serial();
    let leg = |n| {
        GradedSpace::try_new(
            Arc::new(
                symtn::sector::LieGroupFusionRule::new(
                    fusion_symbols::lie::GroupId::su(3).unwrap(),
                )
                .unwrap(),
            ),
            [(vec![1i64, 1], n)],
        )
        .unwrap()
    };
    let (c, p, q, r, s) = (leg(3), leg(5), leg(7), leg(4), leg(6));
    let a = TensorMap::<_, f64>::rand_with_seed(setup, &runtime, [&c, &q], [&p], 1).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(setup, &runtime, [&r], [&c, &s], 2).unwrap();
    let spec = ContractSpec {
        lhs: &[0],
        rhs: &[1],
        codomain: &[0, 1],
        domain: &[2, 3],
    };
    let out = a.contract(setup, &b, &spec).unwrap();
    let sizes = [payload_bytes(&a), payload_bytes(&b)];
    assert!(sizes[0] != sizes[1] && !sizes.contains(&payload_bytes(&out)));
    let mut call = |engine: &mut CpuEngine| drop(a.contract(engine, &b, &spec).unwrap());
    assert!(sized_allocs(&sizes, None, &mut call) >= 1, "cold");
    let engine = &mut CpuEngine::serial();
    call(engine);
    assert_eq!(sized_allocs(&sizes, Some(engine), &mut call), 0, "warm");
}

#[test]
fn engine_scratch_is_bounded_across_route_switching_and_trim_releases_it() {
    let _serial = counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let engine = &mut CpuEngine::serial();
    // An output transform (the core writes a destination) and a copyC route
    // (a default-order temporary, then one permute), on one f64 U(1) lane.
    let shapes = |engine: &mut CpuEngine, scale: usize| {
        let (c, p, q, r, s) = (
            u1(scale),
            u1(scale + 1),
            u1(scale + 2),
            u1(scale + 3),
            u1(scale + 4),
        );
        let a =
            TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&p, &q], [&c, &r], 3).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&c], [&s], 4).unwrap();
        let x = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&p, &q], [&c], 5).unwrap();
        let y = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&c], [&s], 6).unwrap();
        (a, b, x, y)
    };
    let dynamic_tree = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[0, 3],
        domain: &[1, 2],
    };
    let copy_c = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[1, 0],
        domain: &[2],
    };
    let (large, small) = (shapes(engine, 4), shapes(engine, 1));
    type U1 = TensorMap<U1FusionRule, f64>;
    let run = |engine: &mut CpuEngine, (a, b, x, y): &(U1, U1, U1, U1)| {
        let first = a.contract(engine, b, &dynamic_tree).unwrap();
        let second = x.contract(engine, y, &copy_c).unwrap();
        (payload_bytes(&first), payload_bytes(&second))
    };
    let (dynamic_out, copy_c_out) = run(engine, &large);
    let after_large = symtn::scratch::retained_bytes(engine);
    // What: both destinations and the transformed operand are kept.
    assert!(after_large >= dynamic_out + copy_c_out + payload_bytes(&large.0));
    run(engine, &small);
    assert_eq!(
        symtn::scratch::retained_bytes(engine),
        after_large,
        "high-water mark"
    );
    // What: the second large call grows nothing.
    run(engine, &large);
    assert_eq!(symtn::scratch::retained_bytes(engine), after_large);
    // What: trim frees at least what was reported, and leaves nothing.
    let ((), trimmed) = counting_alloc::measure(|| symtn::scratch::trim(engine));
    assert!(trimmed.freed_bytes >= after_large as u64, "{trimmed:?}");
    assert_eq!(symtn::scratch::retained_bytes(engine), 0);
    // What: the next call is cold again.
    let ((), cold) = counting_alloc::measure_matching(dynamic_out..=dynamic_out, || {
        drop(large.0.contract(engine, &large.1, &dynamic_tree).unwrap())
    });
    assert!(
        cold.matched_calls >= 2,
        "output and core destination: {cold:?}"
    );

    // What: a call that fails after taking the scratch leaves only the
    // empty slot charged, which trim releases.
    let warm = symtn::scratch::retained_bytes(engine);
    let mut wrong = small.0.contract(engine, &small.1, &dynamic_tree).unwrap();
    assert!(large
        .0
        .contract_into(engine, &large.1, &dynamic_tree, &mut wrong, 1.0, 0.0)
        .is_err());
    let empty = symtn::scratch::retained_bytes(engine);
    assert!(
        empty > 0 && empty < warm,
        "empty slot {empty} vs warm {warm}"
    );
    symtn::scratch::trim(engine);
    assert_eq!(symtn::scratch::retained_bytes(engine), 0);
}

/// Sets the engine's extension-store byte limit through tenferro's own
/// procedure (the one `symtn::scratch` documents).
fn set_byte_limit(engine: &mut CpuEngine, bytes: usize) {
    let mut store = engine.take_extension_caches();
    store.set_limits(
        store
            .limits()
            .with_max_retained_bytes(NonZeroUsize::new(bytes).unwrap()),
    );
    engine.restore_extension_caches(store);
}

/// Whole-lease dropping: with a budget that fits the f64 lane alone and the
/// c64 lane alone but not both, alternating the two keeps both cold, until
/// the limit is raised.
#[test]
fn whole_lease_dropping_keeps_mixed_lanes_cold_until_the_limit_is_raised() {
    let _serial = counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let engine = &mut CpuEngine::serial();
    let (c, p, q, r, s) = (u1(2), u1(3), u1(4), u1(5), u1(6));
    let a = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&p, &q], [&c, &r], 1).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&c], [&s], 2).unwrap();
    let x =
        TensorMap::<_, Complex64>::rand_with_seed(engine, &runtime, [&p, &q], [&c, &r], 3).unwrap();
    let y = TensorMap::<_, Complex64>::rand_with_seed(engine, &runtime, [&c], [&s], 4).unwrap();
    let spec = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[0, 3],
        domain: &[1, 2],
    };
    let real = |engine: &mut CpuEngine| drop(a.contract(engine, &b, &spec).unwrap());
    let complex = |engine: &mut CpuEngine| drop(x.contract(engine, &y, &spec).unwrap());
    let ledger = |engine: &mut CpuEngine, calls: &[&dyn Fn(&mut CpuEngine)]| {
        symtn::scratch::trim(engine);
        for call in calls {
            call(engine);
        }
        symtn::scratch::retained_bytes(engine)
    };
    let (real_alone, complex_alone) = (ledger(engine, &[&real]), ledger(engine, &[&complex]));
    let both = ledger(engine, &[&real, &complex]);
    let limit = real_alone.max(complex_alone);
    assert!(limit < both, "the fixture must not fit both lanes");
    // The core destinations, one per lane.
    let (real_dst, complex_dst) = (
        payload_bytes(&a.contract(engine, &b, &spec).unwrap()),
        payload_bytes(&x.contract(engine, &y, &spec).unwrap()),
    );
    let core_destination = |engine: &mut CpuEngine, bytes: usize, call: &dyn Fn(&mut CpuEngine)| {
        counting_alloc::measure_matching(bytes..=bytes, || call(engine))
            .1
            .matched_calls
    };

    symtn::scratch::trim(engine);
    set_byte_limit(engine, limit);
    for round in 0..3 {
        // What: each call allocates its output and its core destination.
        assert_eq!(
            core_destination(engine, real_dst, &real),
            2,
            "round {round}: f64"
        );
        assert_eq!(
            core_destination(engine, complex_dst, &complex),
            2,
            "round {round}: c64"
        );
    }

    set_byte_limit(engine, both * 2);
    real(engine);
    complex(engine);
    // What: both lanes fit now, so each call allocates its output alone.
    assert_eq!(core_destination(engine, real_dst, &real), 1, "warm f64");
    assert_eq!(
        core_destination(engine, complex_dst, &complex),
        1,
        "warm c64"
    );
}

type U1Map = TensorMap<U1FusionRule, f64>;

/// `contract` over the physical dense expansions: `lhs_axes` against
/// `rhs_axes`, the open legs (lhs, then rhs) arranged in `order`. Column-major
/// throughout, as `PhysicalDense`.
fn dense_contract(
    a: &symtn::typed::PhysicalDense<f64>,
    b: &symtn::typed::PhysicalDense<f64>,
    (lhs_axes, rhs_axes): (&[usize], &[usize]),
    order: &[usize],
) -> Vec<f64> {
    let open = |shape: &[usize], contracted: &[usize]| {
        (0..shape.len())
            .filter(|axis| !contracted.contains(axis))
            .collect::<Vec<_>>()
    };
    let (open_a, open_b) = (open(&a.shape, lhs_axes), open(&b.shape, rhs_axes));
    let open_dims: Vec<usize> = open_a
        .iter()
        .map(|&axis| a.shape[axis])
        .chain(open_b.iter().map(|&axis| b.shape[axis]))
        .collect();
    let out_dims: Vec<usize> = order.iter().map(|&axis| open_dims[axis]).collect();
    let sum_dims: Vec<usize> = lhs_axes.iter().map(|&axis| a.shape[axis]).collect();
    let decode = |mut linear: usize, dims: &[usize]| {
        dims.iter()
            .map(|&extent| {
                let index = linear % extent;
                linear /= extent;
                index
            })
            .collect::<Vec<_>>()
    };
    let encode = |index: &[usize], dims: &[usize]| {
        index
            .iter()
            .zip(dims)
            .rev()
            .fold(0, |linear, (&i, &extent)| linear * extent + i)
    };
    (0..out_dims.iter().product::<usize>())
        .map(|linear| {
            let out = decode(linear, &out_dims);
            let mut open_index = vec![0; open_dims.len()];
            for (position, &axis) in order.iter().enumerate() {
                open_index[axis] = out[position];
            }
            (0..sum_dims.iter().product::<usize>())
                .map(|k| {
                    let summed = decode(k, &sum_dims);
                    let mut ia = vec![0; a.shape.len()];
                    let mut ib = vec![0; b.shape.len()];
                    for (n, &axis) in open_a.iter().enumerate() {
                        ia[axis] = open_index[n];
                    }
                    for (n, &axis) in open_b.iter().enumerate() {
                        ib[axis] = open_index[open_a.len() + n];
                    }
                    for (n, (&la, &rb)) in lhs_axes.iter().zip(rhs_axes).enumerate() {
                        ia[la] = summed[n];
                        ib[rb] = summed[n];
                    }
                    a.data[encode(&ia, &a.shape)] * b.data[encode(&ib, &b.shape)]
                })
                .sum()
        })
        .collect()
}

/// Stale scratch: right before every checked call, a call of the same route
/// over NaN/Inf operands of the full shapes fills the engine-kept scratch
/// (the transformed operand, the DynamicTree destination, the `copyC`
/// temporary). The finite owned and destination calls that follow, of the
/// same shapes, of a smaller prefix, and with inactive core blocks, still
/// match the dense expansion.
#[test]
fn stale_engine_scratch_never_reaches_the_output() {
    let _serial = counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let engine = &mut CpuEngine::serial();
    let leg = |pairs: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::new(U1FusionRule),
            pairs
                .iter()
                .map(|&(charge, n)| (U1Irrep::new(charge).unwrap(), n)),
        )
        .unwrap()
    };
    let poison = |n: usize| {
        if n.is_multiple_of(2) {
            f64::NAN
        } else {
            f64::INFINITY
        }
    };
    let finite = |n: usize| 0.25 + (n % 7) as f64 * 0.125;
    let filled = |codomain: &[&GradedSpace<U1FusionRule>],
                  domain: &[&GradedSpace<U1FusionRule>],
                  value: &dyn Fn(usize) -> f64| {
        let mut n = 0;
        U1Map::from_subblock_fn(
            &mut CpuEngine::serial(),
            &runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |_, _| {
                n += 1;
                value(n)
            },
        )
        .unwrap()
    };
    // `copyC`: `a[v, v | w] b[w | v] -> [v2, v1 | v3]`, a default-order
    // temporary, then one permute. DynamicTree: `a[v, v | w, v] b[w | v] ->
    // [v1, v4 | v2, v3]`, `a` transformed into core form, the core writing a
    // destination that one output transform moves into the result.
    let routes: [(&str, bool, ContractSpec<'static>); 2] = [
        (
            "copyC",
            false,
            ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[1, 0],
                domain: &[2],
            },
        ),
        (
            "DynamicTree",
            true,
            ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 3],
                domain: &[1, 2],
            },
        ),
    ];
    let large = leg(&[(-1, 3), (0, 2), (1, 3)]);
    let small = leg(&[(-1, 1), (0, 2), (1, 1)]);
    let w = leg(&[(-1, 2), (0, 3), (1, 2)]);
    // With `w` of charge 0 only, the output blocks whose last leg has charge
    // ±1 have no contributing term, though the poisoned run wrote them.
    let neutral = leg(&[(0, 3)]);
    for (route, wide, spec) in routes {
        let operands = |v: &GradedSpace<U1FusionRule>,
                        w: &GradedSpace<U1FusionRule>,
                        value: &dyn Fn(usize) -> f64| {
            let a = if wide {
                filled(&[v, v], &[w, v], value)
            } else {
                filled(&[v, v], &[w], value)
            };
            (a, filled(&[w], &[v], value))
        };
        let poisoned = operands(&large, &w, &poison);
        let poison_scratch = |engine: &mut CpuEngine| {
            let (a, b) = &poisoned;
            let mut dst = a.contract(engine, b, &spec).unwrap();
            a.contract_into(engine, b, &spec, &mut dst, 1.0, 0.0)
                .unwrap();
        };
        let order: Vec<usize> = spec.codomain.iter().chain(spec.domain).copied().collect();
        for (case, v, w) in [
            ("same shapes", &large, &w),
            ("smaller prefix", &small, &w),
            ("inactive core blocks", &large, &neutral),
        ] {
            let what = format!("{route}, {case}");
            let (a, b) = operands(v, w, &finite);
            let want = dense_contract(
                &a.to_physical_dense(&mut CpuEngine::serial()).unwrap(),
                &b.to_physical_dense(&mut CpuEngine::serial()).unwrap(),
                (spec.lhs, spec.rhs),
                &order,
            );
            let dense = |t: &U1Map| t.to_physical_dense(&mut CpuEngine::serial()).unwrap().data;
            poison_scratch(engine);
            let owned = a.contract(engine, &b, &spec).unwrap();
            crate::numerics::assert_slices_close(
                &format!("{what}: owned"),
                &dense(&owned),
                &want,
                16,
            );
            // A destination of the right space that holds non-finite values.
            let (pa, pb) = operands(v, w, &poison);
            let mut dst = pa.contract(&mut CpuEngine::serial(), &pb, &spec).unwrap();
            poison_scratch(engine);
            a.contract_into(engine, &b, &spec, &mut dst, 1.0, 0.0)
                .unwrap();
            crate::numerics::assert_slices_close(&format!("{what}: into"), &dense(&dst), &want, 16);
        }
    }
}
