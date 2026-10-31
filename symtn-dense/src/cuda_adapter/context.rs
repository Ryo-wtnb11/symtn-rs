use super::*;

/// The device operands a region call needs but never varies: the ones
/// template, whose first element is the one-element coefficient of an unscaled move
/// and whose packed prefix a trace contracts its diagonal against, and the
/// zero template read as the packed source of a region fill, plus the scaled
/// template a trace with a non-unit scale contracts its diagonal against.
///
/// One pair per payload dtype, created on first use of that dtype rather than
/// at [`CudaDenseContext::warm_up`], so the warm-up's documented fixed cost
/// and its counter deltas are unchanged. Each template grows monotonically to
/// the longest region it has served; it is bounded by the largest single
/// region (fill, or traced extent `prod t_k`) a caller has submitted, not by
/// the number of calls.
///
/// The operands are payload-typed, so the slot is keyed by the dtype
/// ([`CudaScalar::OPERAND_SLOT`]), not by realness: an `f32` call must never
/// be handed the `f64` context `1`.
#[derive(Default)]
pub(super) struct ScalarOperands {
    pub(super) ones: Option<CudaDenseStorage>,
    pub(super) zeros: Option<CudaDenseStorage>,
    /// `α` repeated: the trace's per-element scale (TensorOperations'
    /// `Scaler(α)` before the sum). Filled on the device from the ones
    /// template whenever `α` changes, so only growth uploads.
    pub(super) scaled: Option<CudaDenseStorage>,
    /// The value and length of the filled prefix of `scaled`.
    scaled_prefix: Option<(ContractionScalar, usize)>,
    /// Payload bytes per element of this slot's dtype, recorded when a buffer
    /// is created so [`CudaDenseContext::scalar_operand_bytes`] needs no dtype
    /// parameter. Zero while the slot is empty.
    element_bytes: usize,
}
/// One [`ScalarOperands`] slot per admitted payload dtype.
pub(super) const SCALAR_OPERAND_SLOTS: usize = 4;
/// The scalar operand templates of one engine allocation domain, held by a
/// prepared workspace between calls and lent to each call's
/// [`CudaDenseContext`] through [`CudaDenseContext::swap_scalar_operands`].
///
/// A context is per call, so templates it creates alone are re-uploaded by
/// every call; a workspace that keeps them makes its warm replays
/// upload-free. They are bound to the engine whose context created them: lent
/// to a context on another engine, they are dropped rather than submitted.
#[doc(hidden)]
#[derive(Default)]
pub struct CudaScalarOperands {
    /// [`CudaDenseContext::identity`] of the engine the buffers below belong
    /// to; `None` while they hold none.
    identity: Option<u64>,
    slots: [ScalarOperands; SCALAR_OPERAND_SLOTS],
}
impl CudaScalarOperands {
    /// Keeps the templates only if they belong to the engine `identity`.
    fn bind(&mut self, identity: u64) {
        if self.identity != Some(identity) {
            *self = Self {
                identity: Some(identity),
                ..Self::default()
            };
        }
    }
}
/// One CUDA operation's view of the caller's engine: the borrowed engine every
/// device kernel is submitted through, its runtime handle for explicit
/// transfers, and the invocation-local scalar operands.
///
/// The caller owns the engine; this context only borrows it for the
/// operation, so nothing in symtn stores an engine. Every operand is checked
/// against the engine's allocation domain before any submission: a buffer of
/// another engine, even one on the same device ordinal, is refused.
pub struct CudaDenseContext<'e> {
    pub(super) backend: &'e mut dyn Engine,
    pub(super) runtime: CudaRuntime,
    pub(super) device: CudaPlacement,
    identity: u64,
    operands: CudaScalarOperands,
}
/// The process-unique `u64` of an engine allocation domain, so state keyed by
/// [`CudaDenseContext::identity`] is shared by every operation on one engine
/// and never by two engines (a domain id is never reused).
// ponytail: one entry per engine domain this process has run symtn work on,
// scanned linearly; a map if processes ever open many engines.
fn domain_identity(domain: AllocationDomainId) -> u64 {
    static DOMAINS: std::sync::Mutex<Vec<AllocationDomainId>> = std::sync::Mutex::new(Vec::new());
    let mut domains = DOMAINS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let index = match domains.iter().position(|known| *known == domain) {
        Some(index) => index,
        None => {
            domains.push(domain);
            domains.len() - 1
        }
    };
    index as u64 + 1
}
impl<'e> CudaDenseContext<'e> {
    /// Borrows the caller's engine for one CUDA operation.
    ///
    /// # Errors
    ///
    /// [`DenseError::Unsupported`] when `engine` is not a CUDA engine; nothing
    /// is uploaded or submitted.
    pub fn new(engine: &'e mut dyn Engine) -> Result<Self, DenseError> {
        let runtime = with_cuda_engine(engine, |cuda| cuda.runtime().clone()).ok_or_else(|| {
            DenseError::Unsupported {
                op: "cuda_context",
                message: "device storage needs the caller's CUDA engine".to_string(),
            }
        })?;
        let device = CudaPlacement {
            ordinal: runtime.device_ordinal(),
            domain: runtime.allocation_domain_id(),
        };
        let identity = domain_identity(device.domain);
        Ok(Self {
            backend: engine,
            runtime,
            device,
            identity,
            operands: CudaScalarOperands {
                identity: Some(identity),
                ..CudaScalarOperands::default()
            },
        })
    }

    /// Process-unique identity of the borrowed engine's allocation domain:
    /// equal for every context on one engine, distinct for any two engines
    /// (also on one device ordinal), never reused.
    ///
    /// Device state cached against an engine — an uploaded coefficient
    /// vector, a prepared workspace — is usable only by that engine, and this
    /// is the key that says so.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Refuses `op` unless every buffer in `operands` belongs to this
    /// context's engine (its allocation domain, not only its device ordinal).
    ///
    /// The one admission an operation runs on all of its device inputs and
    /// destinations before it uploads, allocates or writes anything: the
    /// kernel-level checks reject a foreign buffer only at its first
    /// submission, which may come after an output was already uploaded.
    #[doc(hidden)]
    pub fn ensure_owns(
        &self,
        op: &'static str,
        operands: &[&CudaDenseStorage],
    ) -> Result<(), DenseError> {
        operands.iter().try_for_each(|operand| {
            ensure_cuda_device(self.device, op, &[("operand", operand.device)])
        })
    }

    /// Exchanges this context's scalar operand templates with `held`: lent
    /// before a prepared replay, so the replay finds the templates an earlier
    /// call left there, and swapped back after it, so `held` keeps what the
    /// replay grew. Templates `held` keeps for another engine are dropped
    /// first, never submitted here.
    #[doc(hidden)]
    pub fn swap_scalar_operands(&mut self, held: &mut CudaScalarOperands) {
        held.bind(self.identity);
        std::mem::swap(&mut self.operands, held);
    }

    fn operands<D: CudaScalar>(&self) -> &ScalarOperands {
        &self.operands.slots[D::OPERAND_SLOT]
    }

    fn operands_mut<D: CudaScalar>(&mut self) -> &mut ScalarOperands {
        &mut self.operands.slots[D::OPERAND_SLOT]
    }

    /// Uploads this dtype's ones template unless a resident one already holds
    /// at least `len` elements. Its first element is the `1` coefficient of an
    /// unscaled move; a longer prefix is the packed ones operand a trace
    /// contracts its diagonal against ([`cuda_region_trace_accumulate`]).
    pub(super) fn ensure_ones<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
        let usable = self
            .operands::<D>()
            .ones
            .as_ref()
            .is_some_and(|ones| ones.len() >= len);
        if !usable {
            self.operands_mut::<D>().ones = None;
            let ones = CudaDenseStorage::upload_owned(self, vec![D::ONE; len.max(1)])?;
            let slot = self.operands_mut::<D>();
            slot.element_bytes = std::mem::size_of::<D>();
            slot.ones = Some(ones);
        }
        Ok(())
    }

    /// Makes the first `len` elements of this dtype's scaled template equal
    /// `alpha`. Growth uploads `alpha` repeated (one H2D); a resident template
    /// of another value is refilled on the device as `alpha * ones`, one region
    /// submission and no transfer; the same value and length does nothing.
    pub(super) fn ensure_scaled<D: CudaScalar>(
        &mut self,
        alpha: D,
        len: usize,
    ) -> Result<(), DenseError> {
        let tag = alpha.contraction_scalar();
        let slot = self.operands::<D>();
        if slot
            .scaled_prefix
            .is_some_and(|(value, filled)| value == tag && filled >= len)
        {
            return Ok(());
        }
        if slot.scaled.as_ref().is_none_or(|scaled| scaled.len() < len) {
            self.operands_mut::<D>().scaled = None;
            let scaled = CudaDenseStorage::upload_owned(self, vec![alpha; len])?;
            let slot = self.operands_mut::<D>();
            slot.element_bytes = std::mem::size_of::<D>();
            slot.scaled = Some(scaled);
            slot.scaled_prefix = Some((tag, len));
            return Ok(());
        }
        self.ensure_ones::<D>(len)?;
        let region = CudaRegion::packed(&[len], 0)?;
        let Self {
            backend, operands, ..
        } = self;
        let slot = &mut operands.slots[D::OPERAND_SLOT];
        let (Some(ones), Some(scaled)) = (slot.ones.as_ref(), slot.scaled.as_mut()) else {
            return Err(cuda_error(
                "cuda_scaled_template",
                "scalar operands are missing",
            ));
        };
        submit_region_axpby::<D>(
            &mut **backend,
            "cuda_scaled_template",
            ones,
            &region,
            false,
            alpha,
            ones,
            0,
            CudaRegionBeta::Overwrite,
            scaled,
            &region,
        )?;
        slot.scaled_prefix = Some((tag, len));
        Ok(())
    }

    /// Sizes this dtype's scaled template (see [`cuda_region_trace_accumulate`])
    /// for `len` elements up front, together with the ones template of the
    /// same length it is refilled from, so a trace with a non-unit scale
    /// uploads nothing afterwards: a later change of scale is a device
    /// refill. Never shrinks; reported and released with the other scalar
    /// operands.
    pub fn reserve_scaled_template<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
        if len == 0 {
            return Ok(());
        }
        self.ensure_ones::<D>(len)?;
        if self
            .operands::<D>()
            .scaled
            .as_ref()
            .is_some_and(|scaled| scaled.len() >= len)
        {
            return Ok(());
        }
        self.ensure_scaled::<D>(D::ONE, len)
    }

    /// Sizes this dtype's ones template for `len` elements up front, the
    /// counterpart of [`Self::reserve_zero_template`]: a caller that knows the
    /// largest trace extent it will submit reserves once and every later
    /// [`cuda_region_trace_accumulate`] is upload-free. Never shrinks; the
    /// bytes are reported by [`Self::scalar_operand_bytes`] and released by
    /// [`Self::release_scalar_operands`].
    pub fn reserve_ones_template<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
        if len == 0 {
            return Ok(());
        }
        self.ensure_ones::<D>(len)
    }

    /// Uploads this dtype's zero template unless a resident one already holds
    /// at least `len` elements. A prefix of a compact zero buffer is itself a
    /// compact zero buffer, so a longer template serves every shorter fill.
    pub(super) fn ensure_zeros<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
        let usable = self
            .operands::<D>()
            .zeros
            .as_ref()
            .is_some_and(|zeros| zeros.len() >= len);
        if !usable {
            // Drop the short template before allocating the longer one, so the
            // device holds one of them rather than both.
            self.operands_mut::<D>().zeros = None;
            let zeros = CudaDenseStorage::upload_owned(self, vec![D::ZERO; len])?;
            let slot = self.operands_mut::<D>();
            slot.element_bytes = std::mem::size_of::<D>();
            slot.zeros = Some(zeros);
        }
        Ok(())
    }

    /// Sizes this dtype's zero template for `len` elements up front.
    ///
    /// [`cuda_region_zero`] grows the template to the fill it is given, so a
    /// caller that visits `k` regions in ascending size pays `k` uploads for
    /// what is one buffer. A caller that knows its largest region — a device
    /// transform knows the largest inactive destination layout of its
    /// structure on the host, before any replay — reserves once here and pays
    /// none. It never shrinks an already longer template.
    ///
    /// The bytes it pins stay resident until [`Self::release_scalar_operands`]
    /// or the context is dropped, and are reported by
    /// [`Self::scalar_operand_bytes`]. It is the only device zero source;
    /// tenferro-rs#1834's native device fill would remove it.
    pub fn reserve_zero_template<D: CudaScalar>(&mut self, len: usize) -> Result<(), DenseError> {
        if len == 0 {
            return Ok(());
        }
        self.ensure_zeros::<D>(len)
    }

    /// Device bytes the lazily created scalar operands currently pin, over
    /// every dtype. Observability for the caller that owns the device memory
    /// budget; nothing here reads it back.
    pub fn scalar_operand_bytes(&self) -> usize {
        self.operands
            .slots
            .iter()
            .map(|operands| {
                let elements = operands.ones.as_ref().map_or(0, CudaDenseStorage::len)
                    + operands.zeros.as_ref().map_or(0, CudaDenseStorage::len)
                    + operands.scaled.as_ref().map_or(0, CudaDenseStorage::len);
                elements * operands.element_bytes
            })
            .sum()
    }

    /// Frees every lazily created scalar operand. The next region call that
    /// needs one re-creates it, so this is a memory decision, never a
    /// correctness one.
    pub fn release_scalar_operands(&mut self) {
        self.operands = CudaScalarOperands {
            identity: Some(self.identity),
            ..CudaScalarOperands::default()
        };
    }

    /// Hands out the backend and this dtype's scalar operands at once.
    ///
    /// A whole-`self` accessor cannot express this: submitting against a
    /// context-owned operand needs `&mut` on the backend and `&` on the
    /// operand simultaneously, which is only sound because they are disjoint
    /// fields.
    pub(super) fn split_operands<D: CudaScalar>(&mut self) -> (&mut dyn Engine, &ScalarOperands) {
        (&mut *self.backend, &self.operands.slots[D::OPERAND_SLOT])
    }

    /// The CUDA device ordinal of the borrowed engine.
    pub fn device(&self) -> usize {
        self.device.ordinal
    }

    /// Runs the smallest real operation against each backend library this
    /// context can reach, so their one-time initialization is paid here
    /// rather than by the first user operation.
    ///
    /// Why: Tenferro loads cuTENSOR (`dlopen` + `cutensorCreate`) and the
    /// cuSOLVER/cuBLAS handles lazily, per engine, on the first submission
    /// that needs them.
    /// The measured cost is 185-657 ms
    /// (`benchmarks/history/cuda-baseline-2026-09-20.md`), independent of the
    /// operation, provider, dtype and size that happens to pay it. A caller
    /// that wants it outside its first measured operation calls this once on
    /// a fresh engine; nothing is memoized on the symtn side and no result
    /// changes.
    ///
    /// Because both libraries are touched here, a caller that only contracts
    /// still needs cuSOLVER and cuBLAS to be loadable, not just cuTENSOR: a
    /// missing library is the same typed [`DenseError::Backend`] as before,
    /// surfaced by this call instead of by the first factorization. Tenferro resolves the three through
    /// `TENFERRO_CUTENSOR_PATH`, `TENFERRO_CUSOLVER_PATH` and
    /// `TENFERRO_CUBLAS_PATH`.
    ///
    /// What is *not* warmed: tenferro-gpu compiles its kernels with NVRTC per
    /// engine, seconds per kernel module, so the first elementwise, reduction
    /// or copy of a module may still pay that compile. That is a tenferro-gpu
    /// concern, not a symtn one: `TENFERRO_CUDA_ARTIFACT_CACHE=<dir>` persists
    /// the compiled kernels for later engines and processes, and one
    /// long-lived engine per device pays it once. The cuTENSOR plan cache is
    /// also per contraction shape, so a user contraction still builds its own
    /// plan. The region primitive's
    /// scalar operands are not warmed either: they belong to one context and
    /// are created on first use of their dtype.
    ///
    /// Tenferro's runtime-owned
    /// blas1 cuBLAS handle (tenferro-gpu `cubecl/runtime.rs:688`) is also not
    /// warmed, because symtn reaches no blas1 entry point; extend the warm-up
    /// if that changes.
    ///
    /// Observation counters ([`cuda_transfer_stats`]) are per thread and are
    /// deliberately *not* reset here, so a caller's deltas are honest about the
    /// work this seam did. Each call adds exactly: `h2d_calls` 3,
    /// `h2d_bytes` 24, `device_allocs` 4 (three uploads plus the eigenvector
    /// factor), `gemm_calls` 1, `solver_calls` 1, `d2h_calls` 1, `d2h_bytes` 8,
    /// `copy_calls` 0.
    ///
    /// Single precision changes none of this. The warm-up submits `f64` work
    /// only, so a caller that never touches `f32`/[`Complex32`] pays exactly
    /// what it paid before: the libraries this loads (cuTENSOR, cuSOLVER,
    /// cuBLAS) are per engine, not per dtype, and the probe measured
    /// no notable per-dtype NVRTC stall
    /// (`benchmarks/history/cuda-single-precision-probe-2026-09-20.md`,
    /// finding 6). What a single-precision caller still pays on its own first
    /// use is the NVRTC compile of its dtype's kernels and the context's `f32`/
    /// `Complex32` scalar operands, both of which are lazy per dtype.
    ///
    /// Every device buffer created here is dropped before returning.
    pub fn warm_up(&mut self) -> Result<(), DenseError> {
        // cuTENSOR: handle, library load and one plan, through the same seam
        // every contraction uses.
        let lhs = CudaDenseStorage::upload::<f64>(self, &[1.0_f64])?;
        let rhs = CudaDenseStorage::upload::<f64>(self, &[1.0_f64])?;
        let mut dst = CudaDenseStorage::upload::<f64>(self, &[0.0_f64])?;
        cuda_gemm_region_into::<f64>(
            self, &mut dst, 0, 1, &lhs, 0, 1, &rhs, 0, 1, 1, 1, 1, 1.0, 0.0,
        )?;
        // cuSOLVER + cuBLAS: a 1x1 region is trivially Hermitian, so this is a
        // real `eigh` with no host-side precondition to fake.
        let (values, _vectors) = cuda_eigh_region::<f64>(self, &lhs, 0, 1)?;
        cuda_download_spectra::<f64>(self, &[values])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What: templates a workspace kept for one engine are discarded when
    /// lent to a context on another engine, and kept for the same engine.
    /// The slot metadata stands in for the device buffers, which need a GPU.
    #[test]
    fn held_templates_of_another_engine_are_discarded_not_lent() {
        let held = |identity| {
            let mut operands = CudaScalarOperands {
                identity: Some(identity),
                ..CudaScalarOperands::default()
            };
            operands.slots[0].element_bytes = 8;
            operands.slots[0].scaled_prefix = Some((1.0_f64.contraction_scalar(), 4));
            operands
        };

        let mut same = held(1);
        same.bind(1);
        assert_eq!(same.identity, Some(1));
        assert_eq!(same.slots[0].element_bytes, 8, "same engine keeps them");

        let mut other = held(1);
        other.bind(2);
        assert_eq!(other.identity, Some(2));
        assert_eq!(other.slots[0].element_bytes, 0, "another engine drops them");
        assert!(other.slots[0].scaled_prefix.is_none());
        assert!(other
            .slots
            .iter()
            .all(|slot| slot.ones.is_none() && slot.zeros.is_none() && slot.scaled.is_none()));
    }
}
