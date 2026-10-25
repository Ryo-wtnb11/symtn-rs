//! Checked Generic tree-transform `_into` (#1870): the eager op's
//! staging and commit around a write into the caller's buffer.

use super::*;

type ToyTensor = TensorMap<CheckedOnlyToy, f64>;

fn toy_source(runtime: &Runtime, provider: &Arc<CheckedOnlyToy>) -> ToyTensor {
    let x = GradedSpace::try_new(Arc::clone(provider), [(Label::X, 2)]).unwrap();
    TensorMap::from_subblock_fn(runtime, [&x, &x, &x], [], |trees, indices| {
        trees.codomain_vertices()[0].get() as f64
            + indices
                .iter()
                .enumerate()
                .map(|(axis, &i)| ((axis + 1) * (i + 1)) as f64)
                .sum::<f64>()
    })
    .unwrap()
}

fn bits(tensor: &ToyTensor) -> Vec<u64> {
    tensor
        .dense_data()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

type Eager = fn(&ToyTensor) -> Result<ToyTensor, GenericTensorError<ToyError>>;
type IntoOp = fn(&ToyTensor, &mut ToyTensor, f64, f64) -> Result<(), GenericTensorError<ToyError>>;

/// Why only permute and braid: this toy's planar bends are malformed
/// (`EmptyTransformBlock` in eager too); checked transpose and repartition
/// `_into` run on SU(N) in `into_admission_matrix.rs`, through the same body.
const OPERATIONS: [(&str, Eager, IntoOp); 2] = [
    (
        "permute",
        |t| t.permute(&[1, 0, 2], &[]),
        |t, d, a, b| t.permute_into(&[1, 0, 2], &[], d, a, b),
    ),
    (
        "braid",
        |t| t.braid(&[1, 0, 2], &[], &[0, 1, 2]),
        |t, d, a, b| t.braid_into(&[1, 0, 2], &[], &[0, 1, 2], d, a, b),
    ),
];

/// The cold `_into` asks the provider exactly what cold eager asks (the
/// rule check reads the held identities), and a warm one what warm eager
/// asks; the result is eager's.
#[test]
fn checked_transform_into_makes_the_eager_provider_queries() {
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(70));
    for (name, eager, into) in OPERATIONS {
        // Why a fresh source per cold call: a source admits its trees once
        // (#2233), so a reused one would make the second cold call cheaper.
        let source = toy_source(&runtime, &provider);
        forget_cached_structures();
        reset_provider_queries(&provider);
        let expected = eager(&source).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let cold_eager = provider.queries_since_reset.load(Ordering::Relaxed);
        reset_provider_queries(&provider);
        drop(eager(&source).unwrap());
        let warm_eager = provider.queries_since_reset.load(Ordering::Relaxed);

        let mut destination: ToyTensor = TensorMap::from_subblock_fn(
            &runtime,
            &expected.codomain(),
            &expected.domain(),
            |_, _| f64::NAN,
        )
        .unwrap();
        let source = toy_source(&runtime, &provider);
        forget_cached_structures();
        reset_provider_queries(&provider);
        into(&source, &mut destination, 1.0, 0.0).unwrap();
        let cold_into = provider.queries_since_reset.load(Ordering::Relaxed);
        reset_provider_queries(&provider);
        into(&source, &mut destination, 1.0, 0.0).unwrap();
        let warm_into = provider.queries_since_reset.load(Ordering::Relaxed);

        assert!(cold_eager > 0, "{name}");
        assert_eq!(cold_into, cold_eager, "{name}: cold");
        assert_eq!(warm_into, warm_eager, "{name}: warm");
        for (got, want) in destination
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
        {
            assert!(lazy_close_f64(*got, *want), "{name}: {got} vs {want}");
        }
        assert!(std::ptr::eq(destination.provider(), provider.as_ref()));
    }
}

/// A provider failure in staging is the eager `Plan` error and leaves the
/// destination bit-identical; a destination rejected after staging (shared)
/// drops the stage: nothing reaches the completed-transformer store until a
/// write succeeds.
#[test]
fn checked_transform_into_publishes_nothing_before_the_write() {
    let _serial = crate::counting_alloc::serial();
    if crate::checked_generic_facade::run_isolated_or_return(
        "SYMTN_CHECKED_FACADE_INTO_PUBLICATION",
        "transforms_into::checked_transform_into_publishes_nothing_before_the_write",
    ) {
        return;
    }
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(71));
    let source = toy_source(&runtime, &provider);
    for (name, eager, into) in OPERATIONS {
        let mut destination = eager(&source).unwrap();
        forget_cached_structures();
        assert_eq!(
            crate::checked_generic_facade::completed_transformers().entries(),
            0,
            "{name}"
        );

        provider.fail_algebra.store(true, Ordering::Relaxed);
        let before = bits(&destination);
        let error = into(&source, &mut destination, 1.0, 0.5).unwrap_err();
        provider.fail_algebra.store(false, Ordering::Relaxed);
        assert!(
            matches!(
                error,
                GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
                    ToyError::Algebra
                ))
            ),
            "{name}: {error:?}"
        );
        assert_eq!(bits(&destination), before, "{name}: provider failure");
        assert_eq!(
            crate::checked_generic_facade::completed_transformers().entries(),
            0,
            "{name}: provider failure"
        );

        let shared = destination.clone();
        let error = into(&source, &mut destination, 1.0, 0.5).unwrap_err();
        assert!(
            matches!(
                error,
                GenericTensorError::Facade(symtn::typed::Error::DestinationShared)
            ),
            "{name}: {error:?}"
        );
        drop(shared);
        assert_eq!(bits(&destination), before, "{name}: shared");
        assert_eq!(
            crate::checked_generic_facade::completed_transformers().entries(),
            0,
            "{name}: shared"
        );

        into(&source, &mut destination, 1.0, 0.5).unwrap();
        assert_eq!(
            crate::checked_generic_facade::completed_transformers().entries(),
            1,
            "{name}: committed"
        );
    }
}
