//! Allocation probes for the typed facade's compact diagonal storage (#570).
//!
//! The `Σ_c k_c` storage claim is asserted by counting bytes through a global
//! allocator while one operation runs, and by `dense_data()` refusing a
//! compact payload. That no operation densifies a compact operand is probed
//! directly in the crate's representation gates.
//!
//! Every measurement is warmed first. The engine's layout and fusion-tree
//! caches allocate on first use, and those allocations belong to the cache, not
//! to the operation under test.

include!("common/predicate_chains.rs");
include!("common/predicate_chain_coefficients.rs");

use std::hint::black_box;
use std::sync::{Arc, OnceLock};
use symtn::typed::ContractSpec;

use symtn::sector::{Z2FusionRule, Z2Irrep};
use symtn::typed::{Complex64, Runtime};
use symtn::typed::{GradedSpace, Svd, TensorMap};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

/// One coupled sector of degeneracy `DEGENERACY`, so the dense payload of a
/// bond factor is exactly `DEGENERACY²` scalars and the compact one
/// `DEGENERACY`. A single sector keeps the arithmetic in the assertions
/// readable; the multi-sector case is covered by the value oracles in
/// `typed_facade.rs`.
const DEGENERACY: usize = 128;

/// `DEGENERACY² * size_of::<f64>()`: one dense f64 bond payload. Every ceiling
/// below is stated against this, because it is the allocation compact storage
/// exists to avoid.
fn dense_payload_bytes() -> u64 {
    (DEGENERACY * DEGENERACY * std::mem::size_of::<f64>()) as u64
}

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::builder().dense_threads(1).build().unwrap())
}

fn leg() -> GradedSpace<Z2FusionRule> {
    GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, DEGENERACY)]).unwrap()
}

fn leg_with(degeneracy: usize) -> GradedSpace<Z2FusionRule> {
    GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, degeneracy)]).unwrap()
}

fn pseudo_random(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

fn source(seed: u64) -> TensorMap<Z2FusionRule, f64> {
    let leg = leg();
    let mut state = seed;
    TensorMap::from_subblock_fn(runtime(), [&leg], [&leg], move |_, _| {
        pseudo_random(&mut state)
    })
    .unwrap()
}

fn complex_source(seed: u64) -> TensorMap<Z2FusionRule, Complex64> {
    let leg = leg();
    let mut state = seed;
    TensorMap::from_subblock_fn(runtime(), [&leg], [&leg], move |_, _| {
        Complex64::new(pseudo_random(&mut state), pseudo_random(&mut state))
    })
    .unwrap()
}

fn measured_bytes<T>(operation: impl FnOnce() -> T) -> u64 {
    let (output, allocs) = counting_alloc::measure(|| black_box(operation()));
    black_box(output);
    allocs.bytes
}

fn constructed_diagonal(degeneracy: usize) -> TensorMap<Z2FusionRule, f64> {
    let bond = leg_with(degeneracy);
    TensorMap::diagonal(
        runtime(),
        &bond,
        [symtn::typed::SectorSpectrum {
            sector: Z2Irrep::EVEN,
            values: vec![1.0; degeneracy],
        }],
    )
    .unwrap()
}

#[test]
fn public_diagonal_constructor_and_readback_stay_compact() {
    let _measurement = counting_alloc::serial();
    black_box(constructed_diagonal(64));
    black_box(constructed_diagonal(DEGENERACY));
    let small = measured_bytes(|| constructed_diagonal(64));
    let large = measured_bytes(|| constructed_diagonal(DEGENERACY));
    assert!(
        large <= small * 3,
        "typed constructor growth is not O(d): {small}, {large}"
    );
    assert!(
        large < dense_payload_bytes(),
        "typed constructor allocated a dense payload: {large}"
    );
    let small_diagonal = constructed_diagonal(64);
    let diagonal = constructed_diagonal(DEGENERACY);
    let small_readback = measured_bytes(|| {
        symtn::expert::diagonal_spectrum(&small_diagonal)
            .unwrap()
            .unwrap()
    });
    let readback = measured_bytes(|| {
        symtn::expert::diagonal_spectrum(&diagonal)
            .unwrap()
            .unwrap()
    });
    assert!(
        readback <= small_readback * 3,
        "typed readback growth is not O(d): {small_readback}, {readback}"
    );
    assert!(
        readback < dense_payload_bytes(),
        "typed readback allocated a dense payload: {readback}"
    );
    assert!(
        diagonal.dense_data().is_err(),
        "typed constructor or readback materialized the dense payload"
    );
}

#[test]
fn labelled_block_inspection_refuses_compact_data_and_borrows_after_materialize() {
    let _measurement = counting_alloc::serial();
    let diagonal = constructed_diagonal(DEGENERACY);

    let refused = measured_bytes(|| diagonal.subblocks().map(|_| ()).is_err());
    assert!(
        refused < dense_payload_bytes(),
        "block inspection materialized the compact payload: {refused}"
    );
    assert!(diagonal.subblocks().is_err());

    let dense = diagonal.materialize().unwrap();
    let borrowed = measured_bytes(|| {
        let blocks = dense.subblocks().unwrap().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 1);
        let (_, values) = &blocks[0];
        assert_eq!(values.shape(), &[DEGENERACY, DEGENERACY]);
        assert_eq!(values.get(&[0, 0]), Some(&1.0));
        assert_eq!(values.get(&[0, 1]), Some(&0.0));
        assert_eq!(values.data().as_ptr(), dense.dense_data().unwrap().as_ptr());
    });
    assert!(
        borrowed < dense_payload_bytes(),
        "block inspection copied the dense payload: {borrowed}"
    );
    assert!(symtn::expert::diagonal_spectrum(&diagonal)
        .unwrap()
        .is_some());
}

#[test]
fn svd_compacts_s_is_built_compact_and_materializes_only_on_demand() {
    // What: `svd_compact` stores `s` as `Σ_c k_c` values: `dense_data()`
    // refuses it, and only the explicit `materialize()` pays for the dense
    // buffer.
    let _measurement = counting_alloc::serial();
    let tensor = source(0x5eed_0001);

    // Warm the factorization path itself; `s` is discarded, only caches persist.
    black_box(tensor.svd_compact(&[0], &[1]).unwrap());

    let s = tensor.svd_compact(&[0], &[1]).unwrap().s;
    assert!(s.dense_data().is_err(), "svd_compact built a dense s");
    let first = measured_bytes(|| s.materialize().unwrap());
    assert!(
        first >= dense_payload_bytes(),
        "materialize() allocated only {first} bytes for a dense s"
    );
}

#[test]
fn a_truncated_s_stays_compact_too() {
    // What: truncation is two-leg `restrict_leg` on `svd_compact`'s `s`, and the
    // restriction keeps the compact storage.
    let _measurement = counting_alloc::serial();
    let tensor = source(0x5eed_0002);
    let truncated = |tensor: &TensorMap<Z2FusionRule, f64>| {
        let s = tensor.svd_compact(&[0], &[1]).unwrap().s;
        let found = s.domain()[0]
            .find_truncated(&s.diagview().unwrap(), &symtn::typed::Truncation::Full)
            .unwrap();
        s.restrict_leg(&[(0, &found.selection), (1, &found.selection)])
            .unwrap()
    };
    black_box(truncated(&tensor));

    let s = truncated(&tensor);
    assert!(
        s.dense_data().is_err(),
        "the truncated s was dense at construction"
    );
}

#[test]
fn a_complex_payloads_s_is_compact_as_well() {
    // What: the compact arm is dtype-generic — `D` is a type parameter, so a
    // c64 spectrum takes exactly the same route with no widening variant.
    let _measurement = counting_alloc::serial();
    let tensor = complex_source(0x5eed_0003);
    black_box(tensor.svd_compact(&[0], &[1]).unwrap());

    let s = tensor.svd_compact(&[0], &[1]).unwrap().s;
    assert!(
        s.dense_data().is_err(),
        "the c64 s was dense at construction"
    );
    let first = measured_bytes(|| s.materialize().unwrap());
    assert!(
        first >= 2 * dense_payload_bytes(),
        "materialize() allocated only {first} bytes for a dense c64 s"
    );
}

/// Runs `operation` once to warm every process-global cache it touches, then
/// measures a second, identical run.
fn warmed_bytes<T>(operation: impl Fn() -> T) -> u64 {
    black_box(operation());
    measured_bytes(&operation)
}

fn spectrum(seed: u64) -> TensorMap<Z2FusionRule, f64> {
    source(seed).svd_compact(&[0], &[1]).unwrap().s
}

#[test]
fn storage_local_compact_operations_never_build_a_dense_payload() {
    // What: scale, adjoint, add(diagonal, diagonal) and compose(D, D) all stay
    // in O(Σ_c k_c). Each allocates its own compact result and nothing else, so
    // the ceiling is far below one dense payload.
    let _measurement = counting_alloc::serial();
    let d = spectrum(0x5eed_0011);
    let ceiling = dense_payload_bytes();

    for (name, bytes) in [
        ("scale", warmed_bytes(|| d.scale(0.5).unwrap())),
        ("adjoint", warmed_bytes(|| d.adjoint().unwrap())),
        ("add", warmed_bytes(|| d.axpby(0.75, &d, -0.5).unwrap())),
        ("compose", warmed_bytes(|| d.compose(&d).unwrap())),
    ] {
        assert!(
            bytes < ceiling,
            "compact {name} allocated at least one dense payload: {bytes} bytes"
        );
    }

    // The reductions allocate nothing at all: they read the stored spectrum
    // rather than its materialization, so there is no destination to own.
    for (name, bytes) in [
        ("norm", warmed_bytes(|| d.norm(2.0).unwrap())),
        ("norm(Inf)", warmed_bytes(|| d.norm(f64::INFINITY).unwrap())),
        // p = 3 is the general arm; p = 2 and p = Inf only delegate to the two
        // above, so they would prove nothing here.
        ("norm(3)", warmed_bytes(|| d.norm(3.0).unwrap())),
        ("tr", warmed_bytes(|| d.tr().unwrap())),
        ("inner", warmed_bytes(|| d.inner(&d).unwrap())),
    ] {
        assert_eq!(bytes, 0, "compact {name} allocated temporary storage");
    }
}

#[test]
fn a_mixed_add_allocates_only_its_own_dense_result() {
    // What: adding a spectrum to a dense tensor on the same bond space scatters
    // straight into the owned result. Materializing the spectrum first would
    // double this, which is what the ceiling rejects.
    let _measurement = counting_alloc::serial();
    let d = spectrum(0x5eed_0012);
    let dense = TensorMap::isomorphism(runtime(), &d.domain(), &d.domain()).unwrap();
    // Reading `dense` must not be what pays for the diagonal: warm nothing on
    // `d` beyond what the operation itself needs.
    let bytes = warmed_bytes(|| d.axpby(0.75, &dense, -0.5).unwrap());

    assert!(
        bytes < dense_payload_bytes() * 3 / 2,
        "diagonal + dense allocated more than one dense payload: {bytes} bytes"
    );
    // Same on the mirrored arm.
    let e = spectrum(0x5eed_0014);
    let bytes = warmed_bytes(|| dense.axpby(0.75, &e, -0.5).unwrap());
    assert!(
        bytes < dense_payload_bytes() * 3 / 2,
        "dense + diagonal allocated more than one dense payload: {bytes} bytes"
    );
}

#[test]
fn absorbing_a_spectrum_through_compose_scales_instead_of_densifying() {
    // What: `u * s` and `s * vh` take the bond-scaling arms. Each allocates its
    // own dense result — `u` and `vh` are dense — but not a second dense buffer
    // for `s`, which is what the ceiling here rejects. The dense GEMM route
    // would need `s` materialized as well.
    let _measurement = counting_alloc::serial();
    let tensor = source(0x5eed_0013);
    let Svd { u, s, vh } = tensor.svd_compact(&[0], &[1]).unwrap();
    let ceiling = dense_payload_bytes() * 3 / 2;

    assert!(
        warmed_bytes(|| u.compose(&s).unwrap()) < ceiling,
        "u * s densified the spectrum"
    );
    assert!(
        warmed_bytes(|| s.compose(&vh).unwrap()) < ceiling,
        "s * vh densified the spectrum"
    );
}

#[test]
fn the_matrix_functions_have_o_rank_diagonal_arms() {
    // What: `exp`, `inv`, `pinv` and `map_diagonal` on a spectrum factor are elementwise
    // on the `Σ_c k_c` stored values, not block work on the `Σ_c k_c²`
    // materialization. The ceiling catches a densified route: no
    // densification is retained, so the warmed run pays for its own.
    //
    // `exp` needs its own probe because its dense fallback materializes a
    // diagonal payload and eigendecomposes it.
    let _measurement = counting_alloc::serial();
    let d = spectrum(0x5eed_0021);
    let ceiling = dense_payload_bytes();

    for (name, bytes) in [
        ("exp", warmed_bytes(|| d.exp(&[0], &[1]).unwrap())),
        ("inv", warmed_bytes(|| d.inv(&[0], &[1]).unwrap())),
        ("pinv", warmed_bytes(|| d.pinv(&[0], &[1], 1e-12).unwrap())),
        (
            "map_diagonal",
            warmed_bytes(|| d.map_diagonal(|x| x.sqrt()).unwrap()),
        ),
    ] {
        assert!(
            bytes < ceiling,
            "compact {name} allocated at least one dense payload: {bytes} bytes"
        );
    }
}

#[test]
fn a_complex_spectrums_matrix_functions_stay_o_rank_too() {
    // What: the arms are dtype-generic, so a c64 spectrum takes the same route
    // — at twice the byte size, which is what the ceiling here accounts for.
    let _measurement = counting_alloc::serial();
    let d = complex_source(0x5eed_0022)
        .svd_compact(&[0], &[1])
        .unwrap()
        .s;
    let ceiling = 2 * dense_payload_bytes();

    for (name, bytes) in [
        ("exp", warmed_bytes(|| d.exp(&[0], &[1]).unwrap())),
        ("inv", warmed_bytes(|| d.inv(&[0], &[1]).unwrap())),
        ("pinv", warmed_bytes(|| d.pinv(&[0], &[1], 1e-12).unwrap())),
        (
            "map_diagonal",
            warmed_bytes(|| d.map_diagonal(|x| x.sqrt()).unwrap()),
        ),
    ] {
        assert!(
            bytes < ceiling,
            "compact c64 {name} allocated at least one dense payload: {bytes} bytes"
        );
    }
}

#[test]
fn contracting_a_spectrum_scales_instead_of_densifying() {
    // What: `contract` against a compact operand takes the same scaling route
    // `compose` does (issue #584) — TensorKit's `lmul!`/`rmul!` on a
    // `DiagonalTensorMap` — rather than materializing the `Σ_c k_c²` block
    // diagonal and running a GEMM.
    //
    // The byte ceiling alone cannot prove it for `t · s` and `s · t`: the
    // scaling route allocates a scaled copy plus the `permute` destination,
    // which is about what the dense route spends on the materialization plus
    // the GEMM result. The crate's representation gates probe the absence of
    // a densification directly; here the two only run as smoke checks.
    let _measurement = counting_alloc::serial();
    let d = spectrum(0x5eed_0031);
    let dense = source(0x5eed_0032);

    // `t · s`: the spectrum scales `t`'s contracted domain leg (`rmul!`).
    black_box(
        dense
            .contract(
                &d,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap(),
    );

    // `s · t`: the mirror image, scaling `t`'s leading codomain leg (`lmul!`).
    let e = spectrum(0x5eed_0033);
    black_box(
        e.contract(
            &dense,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap(),
    );

    // `s · s` stays compact end to end, so here the ceiling *is* decisive: the
    // whole contraction must cost less than a single dense payload.
    let f = spectrum(0x5eed_0034);
    let bytes = warmed_bytes(|| {
        f.contract(
            &f,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
    });
    assert!(
        bytes < dense_payload_bytes(),
        "s * s allocated at least one dense payload: {bytes} bytes"
    );
}

#[test]
fn the_rank_one_swap_keeps_its_source_and_its_result_compact() {
    // What: re-ordering the two legs of a spectrum factor is a per-sector
    // rescaling of the stored values (#585), so it neither reads nor writes a
    // `Σ_c k_c²` payload. Two things are measured, because each catches a
    // different way of getting it wrong: the swap's own byte cost (a densified
    // route would pay for two dense buffers), and that the *result* is still
    // compact (a compact-in, dense-out route could pass the first).
    let _measurement = counting_alloc::serial();
    let d = spectrum(0x5eed_0041);
    let ceiling = dense_payload_bytes();

    for (name, bytes) in [
        ("permute", warmed_bytes(|| d.permute(&[1], &[0]).unwrap())),
        (
            "transpose",
            warmed_bytes(|| d.transpose(&[1], &[0]).unwrap()),
        ),
    ] {
        assert!(
            bytes < ceiling,
            "compact {name} allocated at least one dense payload: {bytes} bytes"
        );
    }

    let swapped = d.transpose(&[1], &[0]).unwrap();
    assert!(
        swapped.dense_data().is_err(),
        "the rank-one swap built a dense result"
    );
}

#[test]
fn exact_identity_keeps_compact_storage_and_braid_publishes_dense_output() {
    // What: exact identities return the source body without
    // allocating or materializing its compact spectrum. An explicit braid
    // reads compact input and publishes an owned dense output.
    let _measurement = counting_alloc::serial();
    let d = spectrum(0x5eed_0042);

    assert_eq!(
        warmed_bytes(|| d.repartition(1).unwrap()),
        0,
        "identity repartition allocated"
    );
    let repartitioned = d.repartition(1).unwrap();
    assert!(
        repartitioned.dense_data().is_err(),
        "identity repartition materialized compact storage"
    );

    let braided = d.braid(&[1], &[0], &[0, 1]).unwrap();
    assert!(
        braided.dense_data().is_ok(),
        "an explicit braid returned compact storage instead of the dense route"
    );
}

#[test]
fn compact_is_posdef_never_builds_a_dense_payload() {
    // What: the positive-definiteness chain on a spectrum factor (#1557:
    // Hermiticity gate, norm, `diagview`) is a comparison over the stored
    // values (#585). It is not allocation-*free* — the Hermiticity gate it
    // opens with is `t - t†`, which owns two compact `Σ_c k_c` results, and
    // that is the route TensorKit takes too — but it must stay far below the
    // `Σ_c k_c²` materialization the eigensolver route needed.
    let _measurement = counting_alloc::serial();
    let ceiling = dense_payload_bytes();

    let d = spectrum(0x5eed_0051);
    let bytes = warmed_bytes(|| is_posdef_compact!(d, 0.0, |v: f64| v));
    assert!(
        bytes < ceiling,
        "compact is_posdef allocated at least one dense payload: {bytes} bytes"
    );

    let e = complex_source(0x5eed_0052)
        .svd_compact(&[0], &[1])
        .unwrap()
        .s;
    let bytes = warmed_bytes(|| is_posdef_compact!(e, 1e-12, |v: Complex64| v.re));
    assert!(
        bytes < 2 * ceiling,
        "compact c64 is_posdef allocated at least one dense payload: {bytes} bytes"
    );
}

#[test]
fn diagonal_conversions_allocate_one_output_and_stay_compact_on_a_spectrum() {
    // What: `zeros_like`, `to_c64` and `re`/`im` are one
    // element-wise pass — on a compact spectrum factor the result stays
    // compact (far below one dense payload).
    let _measurement = counting_alloc::serial();
    let d = spectrum(0x5eed_0021);
    let complex_d = complex_source(0x5eed_0022)
        .svd_compact(&[0], &[1])
        .unwrap()
        .s;
    let ceiling = dense_payload_bytes();

    for (name, bytes) in [
        ("zeros_like", warmed_bytes(|| d.zeros_like().unwrap())),
        ("to_c64", warmed_bytes(|| d.convert::<Complex64>())),
        ("re", warmed_bytes(|| complex_d.re())),
        ("im", warmed_bytes(|| complex_d.im())),
    ] {
        assert!(
            bytes < ceiling,
            "compact {name} allocated at least one dense payload: {bytes} bytes"
        );
    }
}

#[test]
fn diagonal_dense_conversions_allocate_only_their_own_output() {
    // What: on a dense payload, `to_c64` allocates its c64
    // output (twice the f64 bytes) and nothing more; `re`/`im` allocate their
    // f64 output and nothing more. The ceilings reject any hidden second
    // payload.
    let _measurement = counting_alloc::serial();
    let tensor = source(0x5eed_0023);
    let complex_tensor = complex_source(0x5eed_0024);
    let ceiling = dense_payload_bytes();

    let widen = warmed_bytes(|| tensor.convert::<Complex64>());
    assert!(
        (2 * ceiling..3 * ceiling).contains(&widen),
        "dense to_c64 did not allocate exactly its own c64 output: {widen} bytes"
    );
    for (name, bytes) in [
        ("re", warmed_bytes(|| complex_tensor.re())),
        ("im", warmed_bytes(|| complex_tensor.im())),
    ] {
        assert!(
            (ceiling..2 * ceiling).contains(&bytes),
            "dense {name} did not allocate exactly its own f64 output: {bytes} bytes"
        );
    }
}

#[test]
fn diagonal_inspections_allocate_no_payload() {
    // What: the rank family reads the space structure and
    // allocates nothing; `leg_dims` owns only its `Vec<usize>` of rank
    // entries, never a payload.
    let _measurement = counting_alloc::serial();
    let tensor = source(0x5eed_0025);

    for (name, bytes) in [
        ("rank", warmed_bytes(|| tensor.rank())),
        ("codomain_rank", warmed_bytes(|| tensor.codomain_rank())),
        ("domain_rank", warmed_bytes(|| tensor.domain_rank())),
        ("scalar-error", warmed_bytes(|| tensor.scalar().is_err())),
    ] {
        // `scalar` on a tensor with legs formats its error message; everything
        // else is allocation-free. One small ceiling covers both without
        // letting a payload through.
        assert!(bytes < 256, "inspection {name} allocated {bytes} bytes");
    }
    let dims = warmed_bytes(|| tensor.leg_dims().unwrap());
    assert!(dims < 256, "leg_dims allocated {dims} bytes");
}

#[test]
fn the_full_bond_trace_reduces_the_spectrum_without_materializing() {
    // What (issue #604): `trace_pairs` over the only pair of a compact bond
    // factor reduces the stored spectrum in O(Σ_c k_c), preserving the #585
    // regression gate. The warm-up runs on a throwaway twin, so the measured
    // run pays for everything but the process-global caches.
    let _measurement = counting_alloc::serial();
    let ceiling = dense_payload_bytes();

    black_box(spectrum(0x5eed_0081).trace_pairs(&[(0, 1)]).unwrap());
    let d = spectrum(0x5eed_0081);
    let bytes = measured_bytes(|| d.trace_pairs(&[(0, 1)]).unwrap());
    assert!(
        bytes < ceiling,
        "compact trace_pairs allocated at least one dense payload: {bytes} bytes"
    );

    // Same on a c64 spectrum: the arm is dtype-generic.
    let warm = complex_source(0x5eed_0082)
        .svd_compact(&[0], &[1])
        .unwrap()
        .s;
    black_box(warm.trace_pairs(&[(0, 1)]).unwrap());
    let e = complex_source(0x5eed_0082)
        .svd_compact(&[0], &[1])
        .unwrap()
        .s;
    let bytes = measured_bytes(|| e.trace_pairs(&[(0, 1)]).unwrap());
    assert!(
        bytes < 2 * ceiling,
        "compact c64 trace_pairs allocated at least one dense payload: {bytes} bytes"
    );

    // Tracing nothing is the pre-guard clone: no payload either way.
    let f = spectrum(0x5eed_0083);
    assert!(f.trace_pairs(&[]).unwrap().dense_data().is_err());
}

#[test]
fn contract_keeps_the_compact_storage_outcomes() {
    // What: `contract` keeps the compact-storage
    // outcomes documented by its rustdoc.
    let _measurement = counting_alloc::serial();

    // `s · s` with the identity order stays compact end to end: the whole
    // contraction costs less than one dense payload, and the result is
    // compact.
    let d = spectrum(0x5eed_0061);
    let bytes = warmed_bytes(|| {
        d.contract(
            &d,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
    });
    assert!(
        bytes < dense_payload_bytes(),
        "ordered s * s allocated at least one dense payload: {bytes} bytes"
    );
    let product = d
        .contract(
            &d,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert!(
        product.dense_data().is_err(),
        "ordered s * s densified its result"
    );

    // `s · s` with `[1] ← [0]` is the product spectrum and one swap (#1870):
    // the swap keeps it
    // compact, as TensorKit's `DiagonalTensorMap` result (`diagonal.jl:291`),
    // and the whole contraction costs less than one dense payload.
    let e = spectrum(0x5eed_0062);
    let swapped_spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[1],
        domain: &[0],
    };
    let bytes = warmed_bytes(|| e.contract(&e, &swapped_spec).unwrap());
    assert!(
        bytes < dense_payload_bytes(),
        "swapped s * s allocated at least one dense payload: {bytes} bytes"
    );
    let swapped = e.contract(&e, &swapped_spec).unwrap();
    assert!(
        swapped.dense_data().is_err(),
        "swapped s * s densified its result"
    );

    // `t · s` under a non-identity order is still the scaling arm; the
    // representation gates probe that it does not densify the spectrum.
    let f = spectrum(0x5eed_0063);
    let dense = source(0x5eed_0064);
    black_box(
        dense
            .contract(
                &f,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[1],
                    domain: &[0],
                },
            )
            .unwrap(),
    );
}
