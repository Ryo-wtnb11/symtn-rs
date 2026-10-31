use super::*;

pub(crate) fn spilled_smallvec_heap_bytes<A>(values: &SmallVec<A>) -> usize
where
    A: smallvec::Array,
{
    if values.spilled() {
        values
            .capacity()
            .saturating_mul(std::mem::size_of::<A::Item>())
    } else {
        0
    }
}

/// Process-global, strictly monotonic content identity.
///
/// Tensors-layer operation keys use this id directly. It is therefore never
/// reset or reused, even when the complete-layout cache is cleared.
pub(crate) static BLOCK_STRUCTURE_CONTENT_ID: AtomicUsize = AtomicUsize::new(1);

pub(super) fn new_block_structure_content(
    sector: Arc<SectorStructure>,
    degeneracy: DegeneracyStructure,
    required_len: usize,
) -> Arc<BlockStructureContent> {
    Arc::new(BlockStructureContent::new(sector, degeneracy, required_len))
}

/// Clears every process-global structure cache: sector, degeneracy and the
/// registered completed-transformer, composed-coefficient and GEMM-plan
/// caches. The one clear-all behind
/// `symtn::cache::clear`.
///
/// Live HomSpace identities and block-content ids remain valid. A build whose
/// captured epoch straddles the clear cannot publish into any of them; it may
/// still use a current canonical hit. Identity-free prepared sector layouts
/// retain their separate rule: commit may publish under the current epoch.
/// Takes no lock but the reset lock and each cache's publication lock.
#[doc(hidden)]
pub fn clear_structure_caches() {
    let _serial = CORE_RESET_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    CORE_RESET_EPOCH.fetch_add(1, Ordering::SeqCst);
    crate::fusion_space::reset_structure_caches();
    crate::cache::clear_registered_structure_caches();
    #[cfg(test)]
    if let Some(hook) = crate::probe::take_mid_reset_hook() {
        hook();
    }
    CORE_RESET_EPOCH.fetch_add(1, Ordering::SeqCst);
}

/// Reset epoch of the core structure caches; odd while a reset runs.
static CORE_RESET_EPOCH: AtomicUsize = AtomicUsize::new(0);

/// Held across a whole reset, both epoch bumps included.
pub(crate) static CORE_RESET_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The epoch a build records before it performs admission work or forms a
/// key; pass it to `StructureCache::publish`.
#[doc(hidden)]
pub fn core_reset_epoch() -> usize {
    CORE_RESET_EPOCH.load(Ordering::SeqCst)
}

/// Whether a structure built since `epoch` may be published.
pub(crate) fn may_publish_since(epoch: usize) -> bool {
    epoch.is_multiple_of(2) && CORE_RESET_EPOCH.load(Ordering::SeqCst) == epoch
}
