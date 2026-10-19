use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::typed::{GradedSpace, TensorMap};
use symtn_core::{
    BraidingStyleKind, CheckedFusionAlgebra, FusionAlgebraError, FusionProductSpace, FusionRule,
    FusionStyleKind, FusionTreeHomSpace, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, SU2FusionRule, SU2Irrep, SectorCodec, SectorId, SectorLeg,
    SectorVec, TableRule, TableSector,
};

#[test]
fn network_destination_dual_stays_inside_complete_publication_epoch() {
    const FILTER: &str =
        "runtime::tests::network_destination_dual_stays_inside_complete_publication_epoch";
    if std::env::var_os("SYMTN_NETWORK_PRODUCER_RESET").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", FILTER])
            .env("SYMTN_NETWORK_PRODUCER_RESET", "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
            "{stdout} {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(NetworkResetU1 {
        reset_on_dual: std::sync::atomic::AtomicBool::new(false),
    });
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(symtn_core::U1Irrep::new(1), 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
    symtn_core::clear_structure_caches();
    provider.reset_on_dual.store(true, Ordering::SeqCst);
    let stale = source
        .network_zeros_from_effective_legs(std::slice::from_ref(&leg), std::slice::from_ref(&leg))
        .unwrap();
    assert!(!provider.reset_on_dual.load(Ordering::SeqCst));
    assert_eq!(
        symtn_core::structure_cache_info(symtn_core::StructureCacheKind::DegeneracyStructure)
            .entries(),
        0
    );
    let fresh = source
        .network_zeros_from_effective_legs(std::slice::from_ref(&leg), std::slice::from_ref(&leg))
        .unwrap();
    assert_eq!(stale.dense_data().unwrap(), fresh.dense_data().unwrap());
    assert_eq!(
        symtn_core::structure_cache_info(symtn_core::StructureCacheKind::DegeneracyStructure)
            .entries(),
        1
    );
}

struct NetworkResetU1 {
    reset_on_dual: std::sync::atomic::AtomicBool,
}
impl FusionRule for NetworkResetU1 {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        symtn_core::U1FusionRule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        symtn_core::U1FusionRule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        symtn_core::U1FusionRule.vacuum()
    }
    fn dual(&self, s: SectorId) -> SectorId {
        symtn_core::U1FusionRule.dual(s)
    }
    fn fusion_channels(&self, a: SectorId, b: SectorId) -> SectorVec {
        symtn_core::U1FusionRule.fusion_channels(a, b)
    }
}
impl MultiplicityFreeFusionRule for NetworkResetU1 {}
impl CheckedFusionAlgebra for NetworkResetU1 {
    fn try_dual_sector(&self, s: SectorId) -> Result<SectorId, FusionAlgebraError> {
        if self.reset_on_dual.swap(false, Ordering::SeqCst) {
            symtn_core::clear_structure_caches();
        }
        symtn_core::U1FusionRule.try_dual_sector(s)
    }
    fn try_fusion_channels(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        symtn_core::U1FusionRule.try_fusion_channels(a, b)
    }
    fn try_nsymbol(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        symtn_core::U1FusionRule.try_nsymbol(a, b, c)
    }
}
impl SectorCodec for NetworkResetU1 {
    type Sector = symtn_core::U1Irrep;
    fn encode_sector(&self, s: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        symtn_core::U1FusionRule.encode_sector(s)
    }
    fn decode_sector(&self, s: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        symtn_core::U1FusionRule.decode_sector(s)
    }
}
impl MultiplicityFreeFusionSymbols for NetworkResetU1 {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> f64 {
        symtn_core::U1FusionRule.f_symbol_scalar(a, b, c, d, e, f)
    }
    fn r_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> f64 {
        symtn_core::U1FusionRule.r_symbol_scalar(a, b, c)
    }
}
impl MultiplicityFreeRigidSymbols for NetworkResetU1 {
    fn dim_scalar(&self, s: SectorId) -> f64 {
        symtn_core::U1FusionRule.dim_scalar(s)
    }
    fn inv_dim_scalar(&self, s: SectorId) -> f64 {
        symtn_core::U1FusionRule.inv_dim_scalar(s)
    }
    fn sqrt_dim_scalar(&self, s: SectorId) -> f64 {
        symtn_core::U1FusionRule.sqrt_dim_scalar(s)
    }
    fn inv_sqrt_dim_scalar(&self, s: SectorId) -> f64 {
        symtn_core::U1FusionRule.inv_sqrt_dim_scalar(s)
    }
    fn twist_scalar(&self, s: SectorId) -> f64 {
        symtn_core::U1FusionRule.twist_scalar(s)
    }
    fn frobenius_schur_phase_scalar(&self, s: SectorId) -> f64 {
        symtn_core::U1FusionRule.frobenius_schur_phase_scalar(s)
    }
}

#[derive(Clone)]
struct CountingFibonacci {
    structural_calls: Arc<AtomicUsize>,
    layout_calls: Arc<AtomicUsize>,
    malformed_channels: bool,
}

impl CountingFibonacci {
    fn new() -> Self {
        Self {
            structural_calls: Arc::new(AtomicUsize::new(0)),
            layout_calls: Arc::new(AtomicUsize::new(0)),
            malformed_channels: false,
        }
    }

    fn structural_calls(&self) -> usize {
        self.structural_calls.load(Ordering::Relaxed)
    }

    fn layout_calls(&self) -> usize {
        self.layout_calls.load(Ordering::Relaxed)
    }
}

impl CheckedFusionAlgebra for CountingFibonacci {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        self.layout_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().try_dual_sector(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        self.layout_calls.fetch_add(1, Ordering::Relaxed);
        if self.malformed_channels && left == SectorId::new(1) && right == SectorId::new(1) {
            return Ok([SectorId::new(0), SectorId::new(2)].into_iter().collect());
        }
        TableRule::fibonacci().try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        self.layout_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().try_nsymbol(left, right, coupled)
    }
}

impl SectorCodec for CountingFibonacci {
    type Sector = TableSector;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        TableRule::fibonacci().encode_sector(value)
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        TableRule::fibonacci().decode_sector(id)
    }
}

impl FusionRule for CountingFibonacci {
    fn rule_identity(&self) -> RuleIdentity {
        TableRule::fibonacci().rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        TableRule::fibonacci().fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        TableRule::fibonacci().braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        TableRule::fibonacci().vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        self.layout_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().dual(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.layout_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().fusion_channels(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.layout_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().nsymbol(left, right, coupled)
    }
}

impl MultiplicityFreeFusionRule for CountingFibonacci {}

impl MultiplicityFreeFusionSymbols for CountingFibonacci {
    type Scalar = Complex64;

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        self.structural_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().f_symbol_scalar(
            left,
            middle,
            right,
            coupled,
            left_coupled,
            right_coupled,
        )
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        self.structural_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().r_symbol_scalar(left, right, coupled)
    }
}

impl MultiplicityFreeRigidSymbols for CountingFibonacci {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.structural_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().dim_scalar(sector)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.structural_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().inv_dim_scalar(sector)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.structural_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().sqrt_dim_scalar(sector)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.structural_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().inv_sqrt_dim_scalar(sector)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.structural_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().twist_scalar(sector)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.structural_calls.fetch_add(1, Ordering::Relaxed);
        TableRule::fibonacci().frobenius_schur_phase_scalar(sector)
    }
}

// What: the context pool must hold pointer-sized entries. Pooling the
// complete multi-lane execution state by value made Vec pop/push memmoves
// ~70% of a small standalone op's cost (issue #228). Reverting
// `PooledContext` to a by-value alias fails here instead of silently
// reintroducing that tax.
#[test]
fn pooled_context_is_pointer_sized() {
    assert!(std::mem::size_of::<PooledContext>() <= std::mem::size_of::<usize>());
}

// What: every Runtime on one ordinal must share one device lock (#1384),
// and distinct ordinals must not serialize on each other.
#[cfg(feature = "cuda")]
#[test]
fn cuda_device_lock_is_one_per_ordinal() {
    assert!(std::ptr::eq(cuda_device_lock(7), cuda_device_lock(7)));
    assert!(!std::ptr::eq(cuda_device_lock(7), cuda_device_lock(8)));
}

// The default-feature graph includes faer. This control verifies that its
// explicit provider path constructs; the adapter's private route test pins
// the unset compiled-default selection separately.
#[cfg(feature = "cpu-faer")]
#[test]
fn transform_ops_builds_for_every_faer_config() {
    let faer = symtn_dense::CpuBackendKind::Faer;
    let serial = symtn_dense::SharedCpuContext::with_threads(1).expect("serial context");
    let env = symtn_dense::SharedCpuContext::from_env();
    for ctx in [&serial, &env] {
        let ops = make_transform_ops(ctx, None).expect("default transform ops");
        assert!(ops.dense().shares_cpu_context(ctx));
        // Explicit-kind arm: build must succeed; context sharing holds
        // whenever the kind IS the compiled default (all-faer builds), but
        // an explicit non-default kind keeps a private context (see
        // `DefaultDenseExecutor::with_shared_context`), so no sharing
        // assert here — it would flip on blas-featured builds.
        let ops = make_transform_ops(ctx, Some(faer)).expect("faer transform ops");
        drop(ops);
    }
}

#[test]
fn runtime_and_leased_contexts_share_one_cpu_context() {
    let runtime = Runtime::builder().build().expect("runtime");
    let shared = runtime.execution_config().shared_ctx.clone();
    {
        let mut state = runtime.lock();
        assert!(state.shares_cpu_context(&shared));
    }

    let mut lease = runtime.lease_context().expect("lease");
    assert!(lease.context().shares_cpu_context(&shared));
    let mut network_context =
        TensorExecutionContext::for_config(runtime.execution_config()).expect("context");
    assert!(network_context.shares_cpu_context(&shared));
}

// What: a lane built after the runtime carries the configuration the
// eager lanes were given. Deferring construction is the whole reason a
// program that never touches single precision pays nothing for it, and the
// deferral is only sound if the late lane is indistinguishable.
#[test]
fn lazily_built_single_precision_lanes_inherit_the_runtime_configuration() {
    let runtime = Runtime::builder()
        .recoupling_threads(3)
        .build()
        .expect("runtime");
    let shared = runtime.execution_config().shared_ctx.clone();

    let mut state = runtime.lock();
    state.mf.force_single_precision_lanes().expect("mf lanes");
    state
        .generic
        .force_single_precision_lanes()
        .expect("generic lanes");
    assert!(state.recoupling_threads_are(3));
    assert!(state.shares_cpu_context(&shared));
    drop(state);

    let mut context =
        TensorExecutionContext::for_config(runtime.execution_config()).expect("context");
    context.mf.force_single_precision_lanes().expect("mf lanes");
    context
        .generic
        .force_single_precision_lanes()
        .expect("generic lanes");
    assert!(context.recoupling_threads_are(3));
    assert!(context.shares_cpu_context(&shared));
}

#[test]
fn runtime_builder_recoupling_threads_reach_every_runtime_and_context_lane() {
    fn assert_runtime_and_context(runtime: &Runtime, expected: usize) {
        {
            let mut state = runtime.lock();
            assert!(state.recoupling_threads_are(expected));
        }

        let mut context =
            TensorExecutionContext::for_config(runtime.execution_config()).expect("context");
        assert!(context.recoupling_threads_are(expected));
    }

    let configured = Runtime::builder()
        .recoupling_threads(3)
        .build()
        .expect("configured runtime");
    assert_runtime_and_context(&configured, 3);

    let default = Runtime::builder().build().expect("default runtime");
    assert_runtime_and_context(&default, 1);
}

#[test]
fn fibonacci_complex_lane_reuses_nonreal_structural_plan() {
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_FIBONACCI_LANE_ISOLATED",
        "runtime::tests::fibonacci_complex_lane_reuses_nonreal_structural_plan",
    ) {
        return;
    }
    // What: the private Complex64<-Complex64 Runtime lane compiles one
    // Fibonacci braid, then warm replay avoids every structural-symbol
    // query. Layout queries are counted separately and are not claimed to
    // disappear.
    let runtime = Runtime::builder().build().expect("runtime");
    let rule = Arc::new(CountingFibonacci::new());
    let tau = SectorId::new(1);
    let leg = || SectorLeg::new([(tau, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&rule),
        homspace,
    )
    .expect("Fibonacci source");
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let destination = source
        .transformed_multiplicity_free(&operation)
        .expect("Fibonacci braid destination");
    rule.structural_calls.store(0, Ordering::Relaxed);
    rule.layout_calls.store(0, Ordering::Relaxed);
    crate::cache::clear();

    let source_structure = Arc::clone(source.space().structure());
    let destination_structure = Arc::clone(destination.space().structure());
    let source_data = vec![Complex64::new(1.0, 0.0); source.space().required_len().unwrap()];
    let mut destination_data =
        vec![Complex64::new(0.0, 0.0); destination.space().required_len().unwrap()];
    let mut lease = runtime.lease_context().expect("context");
    let context = lease
        .context()
        .complex_multiplicity_free_lane()
        .tree_context_mut();

    context
        .tree_transform_dyn_overwrite_into_ref(
            rule.as_ref(),
            &operation,
            &destination_structure,
            &source_structure,
            &mut destination_data,
            &source_data,
            Complex64::new(1.0, 0.0),
        )
        .expect("cold Fibonacci braid");
    let cold_calls = rule.structural_calls();
    let cold_layout_calls = rule.layout_calls();
    assert!(cold_calls > 0);
    assert!(cold_layout_calls > 0);
    assert!(destination_data.iter().any(|value| value.im != 0.0));
    let cold_destination_data = destination_data.clone();
    let cold = crate::test_cache::completed();
    assert_eq!(cold.entries(), 1);
    assert_eq!(cold.misses(), 1);
    assert_eq!(cold.hits(), 0);

    destination_data.fill(Complex64::new(0.0, 0.0));
    context
        .tree_transform_dyn_overwrite_into_ref(
            rule.as_ref(),
            &operation,
            &destination_structure,
            &source_structure,
            &mut destination_data,
            &source_data,
            Complex64::new(1.0, 0.0),
        )
        .expect("warm Fibonacci braid");
    assert_eq!(destination_data, cold_destination_data);
    assert!(destination_data.iter().any(|value| value.im != 0.0));
    assert_eq!(rule.structural_calls(), cold_calls);
    // Layout/admission queries are observed separately; only structural
    // F/R replay is promised to disappear on a warm cache hit.
    assert!(rule.layout_calls() >= cold_layout_calls);
    let warm = crate::test_cache::completed();
    assert_eq!(warm.entries(), 1);
    assert_eq!(warm.misses(), 1);
    assert_eq!(warm.hits(), 1);
}

#[test]
fn fibonacci_checked_construction_failures_publish_nothing() {
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_FIBONACCI_FAILURES_ISOLATED",
        "runtime::tests::fibonacci_checked_construction_failures_publish_nothing",
    ) {
        return;
    }
    let runtime = Runtime::builder().build().expect("runtime");
    let rule = Arc::new(CountingFibonacci {
        malformed_channels: true,
        ..CountingFibonacci::new()
    });
    let tau = GradedSpace::try_new(Arc::clone(&rule), [(TableSector::new(2), 1)])
        .expect("label admission");
    crate::cache::clear();
    let cache_before = crate::test_cache::completed();
    let callbacks = AtomicUsize::new(0);

    let late = TensorMap::<CountingFibonacci, Complex64>::from_subblock_fn(
        &runtime,
        [&tau, &tau, &tau],
        [&tau],
        |_, _| {
            callbacks.fetch_add(1, Ordering::Relaxed);
            Complex64::new(1.0, 0.0)
        },
    );
    let late = late.unwrap_err();
    assert!(format!("{late:?}").contains("InvalidSector"));
    assert_eq!(callbacks.load(Ordering::Relaxed), 0);
    assert_eq!(crate::test_cache::completed(), cache_before);

    let early = TensorMap::<CountingFibonacci, Complex64>::from_subblock_fn(
        &runtime,
        std::iter::empty(),
        std::iter::empty(),
        |_, _| {
            callbacks.fetch_add(1, Ordering::Relaxed);
            Complex64::new(1.0, 0.0)
        },
    );
    assert!(matches!(
        early,
        Err(Error::InvalidArgument(message))
            if message == "at least one leg is required to infer the fusion provider"
    ));
    assert_eq!(callbacks.load(Ordering::Relaxed), 0);
    assert_eq!(crate::test_cache::completed(), cache_before);

    assert!(TensorMap::<CountingFibonacci, Complex64>::rand_with_seed(
        &runtime,
        [&tau, &tau, &tau],
        [&tau],
        0x9E37_79B9_7F4A_7C15,
    )
    .is_err());
    assert_eq!(crate::test_cache::completed(), cache_before);
}

#[test]
fn runtimes_share_completed_transformers_and_expired_weak_handles_run_eagerly() {
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_RUNTIMES_SHARE_ISOLATED",
        "runtime::tests::runtimes_share_completed_transformers_and_expired_weak_handles_run_eagerly",
    ) {
        return;
    }
    let runtime_a = Runtime::builder().build().unwrap();
    let runtime_b = Runtime::builder().build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let space = GradedSpace::try_new(
        provider,
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap();
    let source_a: TensorMap<SU2FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime_a, [&space, &space], [&space], 475_002).unwrap();
    let source_b: TensorMap<SU2FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime_b, [&space, &space], [&space], 475_002).unwrap();
    crate::cache::clear();
    let expected_a = source_a.permute(&[1], &[2, 0]).unwrap();
    let after_a = crate::test_cache::completed();
    let expected_b = source_b.permute(&[1], &[2, 0]).unwrap();
    let after_b = crate::test_cache::completed();
    // What: one process-global entry serves both Runtimes (#2014).
    assert_eq!((after_a.entries(), after_a.misses()), (1, 1));
    assert_eq!((after_b.entries(), after_b.misses()), (1, 1));
    assert_eq!(after_b.hits(), after_a.hits() + 1);
    assert_eq!(
        expected_a.dense_data().unwrap(),
        expected_b.dense_data().unwrap()
    );

    crate::cache::clear();
    assert_eq!(crate::test_cache::completed().entries(), 0);

    let mut context = TensorExecutionContext::for_config(runtime_b.execution_config()).unwrap();
    let source_space = source_b.test_bound_space();
    let destination_space = expected_b.test_bound_space();
    let rule = Arc::clone(source_space.provider_arc());
    let source_structure = Arc::clone(source_space.space().structure());
    let destination_structure = Arc::clone(destination_space.space().structure());
    let source_data = source_b.dense_data().unwrap().to_vec();
    let expected_data = expected_b.dense_data().unwrap().to_vec();
    let operation = TreeTransformOperation::permute([1], [2, 0]);
    drop(source_a);
    drop(source_b);
    drop(expected_a);
    drop(expected_b);
    drop(runtime_a);
    drop(runtime_b);

    let mut actual = vec![f64::NAN; expected_data.len()];
    context
        .mf
        .f64
        .tree_context_mut()
        .tree_transform_dyn_overwrite_into_ref(
            rule.as_ref(),
            &operation,
            &destination_structure,
            &source_structure,
            &mut actual,
            &source_data,
            1.0,
        )
        .unwrap();
    assert_eq!(actual, expected_data);
}

#[cfg(feature = "cuda")]
#[test]
fn clearing_cuda_caches_without_a_device_leaves_the_semantic_caches_intact() {
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_CLEAR_CUDA_CACHES_ISOLATED",
        "runtime::tests::clearing_cuda_caches_without_a_device_leaves_the_semantic_caches_intact",
    ) {
        return;
    }
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let source: TensorMap<SU2FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 2217).unwrap();
    let _ = source.permute(&[1], &[2, 0]).unwrap();
    let before = crate::test_cache::completed().entries();
    assert!(before > 0);

    runtime.clear_cuda_caches();

    assert_eq!(crate::test_cache::completed().entries(), before);
    assert!(runtime.cuda_tree_transform_stats().is_none());
    assert!(runtime.cuda_contract_scratch_bytes().is_none());
}

/// Compile-level gate for the device-state ownership of #1322: the lease is
/// still a `CudaDenseContext` for every caller that wants only the dense
/// context, and a transform reaches both halves through `split()` without
/// leasing twice. It is never executed — a device would be needed to
/// obtain a lease — but it fails the build if either shape regresses.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn cuda_lease_still_derefs_to_the_dense_context_and_splits(mut lease: CudaLease<'_>) {
    let dense: &mut symtn_dense::CudaDenseContext = &mut lease;
    let _ = dense.device();
    let (dense, executor) = lease.split();
    let _ = executor.retained_device_bytes(dense);
}
