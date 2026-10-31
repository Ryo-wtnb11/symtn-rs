use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    /// Returns the first leg's provider allocation after proving that every
    /// leg has the same categorical identity.
    fn authority<'a>(legs: &[&'a GradedSpace<R>]) -> Result<&'a Arc<R>, FacadeError<R>> {
        let (first, rest) = legs.split_first().ok_or_else(|| {
            FacadeError::<R>::from(Error::InvalidArgument(
                "at least one leg is required to infer the fusion provider".into(),
            ))
        })?;
        // Equal identities, rather than pointer equality, let separately
        // allocated providers interoperate. This guard precedes every provider
        // query and layout-admission stage.
        let expected_identity = TypedSectorAdmission::typed_rule_identity(first.provider());
        for leg in rest {
            let actual_identity = TypedSectorAdmission::typed_rule_identity(leg.provider());
            if actual_identity != expected_identity {
                return Err(FacadeError::<R>::from(Error::RuleMismatch));
            }
        }
        Ok(first.provider_arc())
    }

    /// Validation half of [`Self::build`]: admits the complete bound layout
    /// without touching payload or runtime RNG state.
    fn build_space(
        provider: Arc<R>,
        codomain: &[&GradedSpace<R>],
        domain: &[&GradedSpace<R>],
    ) -> Result<BoundDynamicFusionMapSpace<R>, FacadeError<R>> {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain.iter().map(|leg| leg.leg().clone())),
            FusionProductSpace::new(domain.iter().map(|leg| leg.leg().clone())),
        );
        <R::Mode as TypedTensorConstructionDispatch<R, D>>::build_construction_root(
            provider, homspace,
        )
    }

    /// Payload half of [`Self::build`]: fills only a fully admitted layout and
    /// validates its final data length before publication.
    fn fill_space(
        runtime: &Runtime,
        space: BoundDynamicFusionMapSpace<R>,
        fill: Fill<'_, D>,
    ) -> Result<Self, FacadeError<R>> {
        let data = apply_fill(space.space(), fill).map_err(FacadeError::<R>::from)?;
        BoundDynamicTensorRef::try_new(&space, &data)
            .map_err(Error::from)
            .map_err(FacadeError::<R>::from)?;
        Ok(Self {
            runtime: runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }

    fn build(
        runtime: &Runtime,
        provider: Arc<R>,
        codomain: &[&GradedSpace<R>],
        domain: &[&GradedSpace<R>],
        fill: Fill<'_, D>,
    ) -> Result<Self, FacadeError<R>> {
        let space = Self::build_space(provider, codomain, domain)?;
        Self::fill_space(runtime, space, fill)
    }

    /// Zero tensor map on `codomain <- domain` (TensorKit `zeros(T, W <- V)`).
    ///
    /// Every leg must have the same rule identity; the first leg's exact
    /// provider allocation becomes the tensor's authority. Identity mismatch
    /// is reported before provider algebra is queried. Layout admission and
    /// payload validation are transactional: failure publishes no tensor.
    ///
    /// # Complexity
    ///
    /// One fusion-tree layout admission plus one `O(stored_len)` zeroed
    /// payload allocation.
    ///
    /// A [`TableRule`](crate::sector::TableRule) payload is `Complex64` even
    /// where F is real (Fibonacci, Ising). This is stricter than TensorKit,
    /// which types real-F data by `fusionscalartype` and only braiding by
    /// `braidingscalartype`: a symtn provider has one `Scalar` for F, R and
    /// the twist, and an operation never changes the payload dtype implicitly.
    ///
    /// ```compile_fail
    /// use std::sync::Arc;
    /// use symtn::sector::{TableRule, TableSector};
    /// use symtn::typed::{GradedSpace, Runtime, TensorMap};
    /// let runtime = Runtime::builder().build().unwrap();
    /// # let engine = &mut symtn::typed::CpuEngine::serial();
    /// let tau = GradedSpace::try_new(
    ///     Arc::new(TableRule::fibonacci()),
    ///     [(TableSector::new(2), 1)],
    /// ).unwrap();
    /// // A complex categorical coefficient cannot act on a real payload.
    /// let _: TensorMap<TableRule, f64> =
    ///     TensorMap::zeros(engine, &runtime, [&tau], [&tau]).unwrap();
    /// ```
    pub fn zeros<'a, Codomain, Domain>(
        engine: &mut dyn Engine,
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
    ) -> Result<Self, FacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        admit_host_engine(engine)?;
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        Self::build(runtime, provider, &codomain, &domain, Fill::Zeros)
    }

    /// Tensor map whose every symmetry-allowed element is produced by
    /// `fill(trees, indices)`, one fusion-tree subblock at a time (the unit
    /// [`Self::subblocks`] reads back; coupled-sector blocks are
    /// [`Self::blocks`]).
    ///
    /// All block keys are decoded exactly once before the first callback.
    /// Therefore a late decode failure invokes `fill` zero times and publishes
    /// neither a partial payload nor a tensor.
    ///
    /// **No TensorKit counterpart.** TensorKit builds an uninitialized,
    /// zeroed, or random tensor and then mutates its blocks; this labelled
    /// callback is symtn's semantic-fixture constructor.
    ///
    /// # Complexity
    ///
    /// One layout admission, one decode per stored block, and one callback per
    /// stored element.
    pub fn from_subblock_fn<'a, Codomain, Domain, F>(
        engine: &mut dyn Engine,
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
        mut fill: F,
    ) -> Result<Self, FacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        F: FnMut(&BlockFusionTrees<R::Sector>, &[usize]) -> D,
        R: 'a,
    {
        admit_host_engine(engine)?;
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        let space = Self::build_space(Arc::clone(&provider), &codomain, &domain)?;
        let structure = space.space().structure();
        let mut labelled = HashMap::with_capacity(structure.block_count());
        for index in 0..structure.block_count() {
            let key = structure
                .block(index)
                .map_err(Error::from)
                .map_err(FacadeError::<R>::from)?
                .key()
                .clone();
            let decoded = decode_block_fusion_trees(provider.as_ref(), &key)?;
            labelled.insert(key, decoded);
        }
        let mut callback = |key: &BlockKey, indices: &[usize]| {
            fill(
                labelled
                    .get(key)
                    .expect("all admitted block keys were decoded before payload fill"),
                indices,
            )
        };
        Self::fill_space(runtime, space, Fill::BlockFn(&mut callback))
    }

    /// Random tensor map using an explicit deterministic splitmix64 seed.
    ///
    /// Every stored real component is uniform on `[-1, 1)`: an `f64` payload
    /// entry uses one draw, while a `Complex64` entry uses independent draws for
    /// its real and imaginary components. This deliberately differs from
    /// TensorKit's default `rand`, whose components are uniform on `[0, 1)`;
    /// pinned source coordinates are recorded in `symtn/references.md`.
    ///
    /// There is no seedless form: TensorKit takes the RNG from the caller, and
    /// the seed is this API's explicit counterpart of that argument.
    ///
    /// Reproducibility is defined for the same symtn version and layout;
    /// it is not a cross-library byte-stream contract. Semantic cross-version
    /// fixtures should use [`Self::from_subblock_fn`].
    ///
    /// # Complexity
    ///
    /// One layout admission and one `O(stored_len)` payload allocation/fill.
    pub fn rand_with_seed<'a, Codomain, Domain>(
        engine: &mut dyn Engine,
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
        seed: u64,
    ) -> Result<Self, FacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        admit_host_engine(engine)?;
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        Self::build(runtime, provider, &codomain, &domain, Fill::Rand(seed))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    /// The fused external sector content of one side of a structural
    /// constructor (TensorKit `fuse`, `spaces/gradedspace.jl:150-158`),
    /// weighted by the provider's `N` symbols. Stored sector content is
    /// already external, so duality is dropped.
    fn fused_content(
        provider: &R,
        legs: &[&GradedSpace<R>],
    ) -> Result<Vec<(symtn_core::SectorId, usize)>, FacadeError<R>> {
        let (first, rest) = legs.split_first().ok_or_else(|| {
            // Keep the empty-side failure local to the typed fusion fold.
            Error::InvalidArgument("fuse_all needs at least one space".into())
        })?;
        let mut fused: Vec<(symtn_core::SectorId, usize)> = first.leg().iter().collect();
        for leg in rest {
            let pairs: Vec<(symtn_core::SectorId, usize)> = leg.leg().iter().collect();
            fused = <R::Mode as TypedTensorModeDispatch<R>>::fuse_sector_content(
                provider, &fused, &pairs,
            )?;
        }
        Ok(fused)
    }

    /// Shared body of the structural constructors: checks the fused fit,
    /// builds zeros and writes the (partial) identity into every
    /// coupled-sector matrix.
    fn structural<'a, C, M>(
        runtime: &Runtime,
        codomain: C,
        domain: M,
        embed: bool,
    ) -> Result<Self, FacadeError<R>>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<&GradedSpace<R>> = codomain.into_iter().collect();
        let domain: Vec<&GradedSpace<R>> = domain.into_iter().collect();
        let legs: Vec<&GradedSpace<R>> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        let fused_codomain = Self::fused_content(&provider, &codomain)?;
        let fused_domain = Self::fused_content(&provider, &domain)?;
        let fits = if embed {
            // TensorKit `domain ≾ codomain`: sectorwise embeddable.
            fused_domain
                .iter()
                .all(|&(sector, deg)| fused_codomain.iter().any(|&(s, d)| s == sector && d >= deg))
        } else {
            // TensorKit `domain ≅ codomain`: identical fused sector content
            // (both sides are SectorId-sorted, so slice equality is content
            // equality).
            fused_codomain == fused_domain
        };
        if !fits {
            return Err(Error::from(symtn_tensors::OperationError::SpaceMismatch {
                message: if embed {
                    "isometry: codomain and domain are not isometrically embeddable (fused sector content differs)"
                } else {
                    "isomorphism: codomain and domain are not isomorphic (fused sector content differs)"
                },
            })
            .into());
        }
        let mut tensor = Self::build(
            runtime,
            Arc::clone(&provider),
            &codomain,
            &domain,
            Fill::Zeros,
        )?;
        // TensorKit's `one!` per coupled-sector block (`tensors/linalg.jl:102-158`),
        // over (fusion tree, vertex, degeneracy) rows and columns in
        // TensorKit's tree order.
        write_identity_blocks_generic(&mut tensor, |sector, dual| {
            let sector = if dual {
                provider
                    .try_dual_id(sector)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?
            } else {
                sector
            };
            Ok::<_, FacadeError<R>>(<R::Mode as TypedTensorModeDispatch<R>>::sector_order_key(
                &provider, sector,
            ))
        })?;
        Ok(tensor)
    }

    /// The canonical structural isomorphism `codomain <- domain` (TensorKit
    /// `isomorphism(W ← V)`; also TensorKit `id(V)` as `isomorphism(V, V)` and
    /// `unitary`, which only adds a Euclidean inner-product check every symtn
    /// provider satisfies): every coupled-sector block is the identity matrix
    /// with rows and columns in TensorKit's fusion-tree order, which requires
    /// the fused codomain and domain to carry identical sector content. Since
    /// [`Self::block`] lists trees in storage order, a block between different
    /// spaces can read as a permuted identity there. Multiplicity-free and
    /// checked-Generic providers alike. The one difference from TensorKit is
    /// SU(N) with three or more legs on one side, where TensorKit's inner-line
    /// order is SUNRepresentations' hash order and symtn uses sector order.
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports, plus `SpaceMismatch`
    /// when the fused codomain and domain differ in sector content
    /// (TensorKit's `SpaceMismatch` on `domain ≅ codomain`).
    ///
    /// # Complexity
    ///
    /// One fused-content fold over the legs, one sort of each coupled block's
    /// trees, plus one `O(stored_len)` payload.
    pub fn isomorphism<'a, C, M>(
        engine: &mut dyn Engine,
        runtime: &Runtime,
        codomain: C,
        domain: M,
    ) -> Result<Self, FacadeError<R>>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        admit_host_engine(engine)?;
        Self::structural(runtime, codomain, domain, false)
    }

    /// The canonical isometry `codomain <- domain` (TensorKit
    /// `isometry(W ← V)`): each
    /// coupled-sector block is the partial identity (the first `cols` columns
    /// of the identity) with rows and columns in TensorKit's fusion-tree
    /// order, so `t† ∘ t = isomorphism(domain, domain)`. Requires
    /// the domain to embed isometrically in the codomain (sectorwise
    /// `deg_domain <= deg_codomain` on the fused content). Multiplicity-free
    /// and checked-Generic providers alike. The one difference from TensorKit
    /// is SU(N) with three or more legs on one side, where TensorKit's
    /// inner-line order is SUNRepresentations' hash order and symtn uses
    /// sector order.
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports, plus `SpaceMismatch`
    /// when the fused domain does not embed sectorwise into the fused
    /// codomain (TensorKit's `SpaceMismatch` on `domain ≾ codomain`).
    ///
    /// # Complexity
    ///
    /// One fused-content fold over the legs, one sort of each coupled block's
    /// trees, plus one `O(stored_len)` payload.
    pub fn isometry<'a, C, M>(
        engine: &mut dyn Engine,
        runtime: &Runtime,
        codomain: C,
        domain: M,
    ) -> Result<Self, FacadeError<R>>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        admit_host_engine(engine)?;
        Self::structural(runtime, codomain, domain, true)
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
    D: TensorScalar,
    S: ElementwiseExec<R, D>,
{
    /// A zero tensor on the same spaces, dtype and storage as `self`
    /// (TensorKit `zerovector`), for every fusion mode and symbol lane.
    /// Dense and compact payloads are freshly initialized to exact positive
    /// zero, independently of non-finite source values. A lazy adjoint zeros
    /// its canonical parent and stays a lazy adjoint.
    pub fn zeros_like(&self, engine: &mut dyn Engine) -> Result<Self, FacadeError<R>> {
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = self.parent_of(view).zeros_like(engine)?;
            return Ok(parent.lazy_adjoint_on(&view.logical_space)?);
        }
        if let Some(spectrum) = self.spectrum() {
            admit_host_engine(engine)?;
            return Ok(self.with_spectrum(
                spectrum
                    .iter()
                    .map(|entry| symtn_matrixalgebra::SectorSpectrum {
                        sector: entry.sector,
                        values: vec![D::from_real(0.0); entry.values.len()],
                    })
                    .collect(),
            ));
        }
        Ok(S::zeros(engine, EXEC, self)?)
    }
}
