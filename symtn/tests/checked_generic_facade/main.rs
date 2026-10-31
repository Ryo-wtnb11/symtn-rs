use crate::{common, numerics};

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use symtn::typed::ContractSpec;
use symtn::typed::HermitianTol;
#[cfg(feature = "cgc-gen")]
use symtn::typed::Qr;
use symtn::typed::Side;

use symtn::expert::DenseBackend;

fn structure_cache_info(
    kind: symtn::cache::StructureCacheKind,
) -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == kind)
        .unwrap()
}
use symtn::expert::{
    DefaultDenseExecutor, DenseDotConfig, DenseError, DenseExecutor, DenseGemmBatchJob, DenseRead,
    DenseScalar, DenseTensor, DenseWrite, MatrixOp,
};
use symtn::sector::SectorId;
use symtn::sector::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SectorVec, TypedSectorAdmission,
};
use symtn::typed::__network::NetworkReuseClass;
use symtn::typed::CheckedGenericStructureError;
use symtn::typed::{
    CheckedGenericTensorProductError, Eig, Eigh, GradedSpace, LeftPolar, Lq, RightPolar, Svd,
    TensorMap, Truncation, TypedTensorConstructionDispatch,
};
use symtn::typed::{Complex32, Complex64, GenericTensorError, Runtime, SectorSpectrum};

/// Orders process-global cache clears against provider-query and cache-count
/// windows: a test that clears holds the write side for its whole
/// body, a test that counts holds the read side, so a sibling's clear cannot
/// land between a warm-up and a measured call. Cooperative: every new
/// count-dependent test must take `cache_shared` and every new clearing test
/// `cache_exclusive`. Poison-tolerant so one failing
/// test does not cascade.
static CACHE_TEST_LOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());

fn cache_shared() -> std::sync::RwLockReadGuard<'static, ()> {
    CACHE_TEST_LOCK.read().unwrap_or_else(|e| e.into_inner())
}

fn cache_exclusive() -> std::sync::RwLockWriteGuard<'static, ()> {
    CACHE_TEST_LOCK.write().unwrap_or_else(|e| e.into_inner())
}

/// Re-executes exactly one global-cache probe with explicit fresh-process
/// state for assertions on absolute counters (`entries() == 0` after a clear).
fn run_isolated_or_return(isolated_env: &str, test_path: &str) -> bool {
    if std::env::var_os(isolated_env).is_some() {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("checked_generic_facade::{test_path}"),
            "--include-ignored",
        ])
        .env(isolated_env, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
        "isolated test did not execute exactly once: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    true
}

/// The receiver's own split as leg roles: `rows = 0..nout`.
#[cfg(feature = "cgc-gen")]
fn codomain_axes<R, D, S>(t: &symtn::typed::TensorMap<R, D, S>) -> Vec<usize> {
    (0..t.codomain_rank()).collect()
}

/// The receiver's own split as leg roles: `cols = nout..rank`.
#[cfg(feature = "cgc-gen")]
fn domain_axes<R, D, S>(t: &symtn::typed::TensorMap<R, D, S>) -> Vec<usize> {
    (t.codomain_rank()..t.rank()).collect()
}

/// `terms` of the tolerance rule for a product of endomorphisms: an
/// endomorphism payload holds `sum_c n_c^2` entries, so its square root
/// bounds every coupled block's contracted length `n_c`.
fn endomorphism_terms(payload_len: usize) -> usize {
    (payload_len as f64).sqrt().ceil() as usize
}

#[cfg(feature = "cgc-gen")]
trait SunEigInput: symtn::typed::AdvancedLinalgScalar<Eig = Complex64> + fmt::Debug {
    fn to_complex(
        engine: &mut dyn Engine,
        source: &TensorMap<symtn::sector::LieGroupFusionRule, Self>,
    ) -> TensorMap<symtn::sector::LieGroupFusionRule, Complex64>;
}

#[cfg(feature = "cgc-gen")]
impl SunEigInput for f64 {
    fn to_complex(
        engine: &mut dyn Engine,
        source: &TensorMap<symtn::sector::LieGroupFusionRule, Self>,
    ) -> TensorMap<symtn::sector::LieGroupFusionRule, Complex64> {
        source.convert::<Complex64>(engine).unwrap()
    }
}

#[cfg(feature = "cgc-gen")]
impl SunEigInput for Complex64 {
    fn to_complex(
        _engine: &mut dyn Engine,
        source: &TensorMap<symtn::sector::LieGroupFusionRule, Self>,
    ) -> TensorMap<symtn::sector::LieGroupFusionRule, Complex64> {
        source.clone()
    }
}

include!("../common/checked_only_toy.rs");

include!("../common/spy_executor.rs");

// ---------------------------------------------------------------------------
// Lazy-adjoint reads through FusionOperand + the shared strided owner,
// and checked `tr` collapsed onto `weighted_trace`.
// ---------------------------------------------------------------------------

fn lazy_oracle_value(
    trees: &symtn::typed::BlockFusionTrees<Label>,
    indices: &[usize],
) -> Complex64 {
    let tree = (format!("{trees:?}").bytes().fold(0u32, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as u32)
    }) % 97) as f64;
    let mut re = tree;
    let mut im = -0.5 * tree + 0.25;
    for (axis, &index) in indices.iter().enumerate() {
        re += (index as f64 + 1.0) * (axis as f64 + 1.0);
        im += (index as f64 + 1.0) * (axis as f64 + 1.0) * (axis as f64 + 1.0) * 0.5;
    }
    Complex64::new(re, im)
}

fn lazy_close_c64(a: Complex64, b: Complex64) -> bool {
    (a - b).norm() <= 1e-12 * (1.0 + b.norm())
}

fn lazy_close_f64(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + b.abs())
}

mod construction;
mod eig;
mod error_kinds;
mod matrix_fn;
mod nonfinite;
mod null_polar;
mod pinv;
mod qr_lq;
mod reductions;
mod svd;
mod transforms;
mod transforms_into;

/// The process-global completed-transformer cache (`symtn::cache`).
#[cfg_attr(not(feature = "cgc-gen"), allow(dead_code))]
fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}
