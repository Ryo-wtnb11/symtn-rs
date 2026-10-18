use super::*;

// ======================================================================
// Generic-fusion coefficient-vector layer (multi_Fmove /
// multi_associator) plus foldright/foldleft/cycle. Outer-multiplicity
// mirror of the multiplicity-free tree functions below, and of TensorKit
// `basic_manipulations.jl` / `duality_manipulations.jl` (the GenericFusion
// else-branches).
//
// COEFFICIENT-VECTOR INDEX CONVENTION (documented once, referenced below).
// A `multi_Fmove` / `multi_associator` on a splitting tree that splits off
// the leftmost sector `a` leaves a family of standard-form trees, each
// carrying a coefficient *vector* rather than a scalar. The vector index is
// the label `λ` of the TOPMOST fusion vertex `a ⊗ b → c`, where
//   a = tree.uncoupled[0]      (the split-off sector, fixed),
//   b = tail_tree.coupled      (the coupled sector of the (N-1)-leg tail),
//   c = tree.coupled           (the overall coupled sector, fixed),
// so the vector has length `Nsymbol(a, b, c)`. This is exactly TensorKit's
// convention (`basic_manipulations.jl:133-135, 186-187, 346-347`): "vectors
// of length Nsymbol(a,b,c), representing the coefficients associated with the
// different vertex labels λ of the topmost vertex". The λ vertex is NOT part
// of the emitted tail tree — it is the *free* index that a downstream
// operation (fold: the A-move; a further recoupling) contracts against. At
// completion the vector is distributed to scalar coefficients (fold's
// A-matrix contraction collapses it to one scalar per output tree pair).
// ======================================================================

/// A generic `multi_Fmove` / `multi_Fmove_inv` result: each recoupled
/// standard-form tree paired with its coefficient VECTOR (indexed by the
/// topmost `a ⊗ b → c` vertex `λ`; see the convention block above). Aliased to
/// keep the tree-function signatures readable and satisfy
/// `clippy::type_complexity`.
pub(super) type GenericFmoveTerms<S> = Vec<(FusionTreeKey, Vec<S>)>;
pub(super) type GenericTreeTerms<S> = Vec<(FusionTreeKey, S)>;
pub(super) type GenericTreePairTerms<S> = Vec<(FusionTreePairKey, S)>;

/// Enumerate every standard-form fusion tree with the given `uncoupled` legs,
/// `is_dual` flags and `coupled` sector, INCLUDING all outer-multiplicity
/// vertex-label assignments. Generic sibling of
/// [`collect_fusion_trees_for_coupled`] (which uses
/// [`MultiplicityIndex::ONE`] for every vertex and is bounded on
/// `MultiplicityFreeFusionRule`): here each
/// vertex with `Nsymbol > 1` branches over its `1..=N` labels, producing one
/// tree per (innerlines, vertices) combination. This is the enumeration
/// TensorKit's `multi_Fmove` Stage 1 performs inline (`for μ in 1:Nbce′` at
/// `basic_manipulations.jl:265`); factoring it out keeps `generic_multi_fmove_*`
/// structurally identical to the multiplicity-free tree functions.
#[cfg(test)]
pub(crate) fn collect_generic_fusion_trees_for_coupled<R>(
    rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
) -> Vec<FusionTreeKey>
where
    R: FusionRule,
{
    let checked = InfallibleGeneric::new(rule);
    match collect_generic_fusion_trees_for_coupled_frozen_checked(
        &checked, uncoupled, is_dual, effective, coupled,
    ) {
        Ok(trees) => trees,
        Err(CheckedGenericStructureError::Provider(never)) => match never {},
        Err(CheckedGenericStructureError::Core(error)) => {
            panic!("infallible Generic tree enumeration failed unexpectedly: {error}")
        }
    }
}

#[cfg(any(test, feature = "testing"))]
pub(super) fn collect_generic_fusion_trees_for_coupled_frozen<R>(
    rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
) -> Vec<FusionTreeKey>
where
    R: FusionRule,
{
    let checked = InfallibleGeneric::new(rule);
    match collect_generic_fusion_trees_for_coupled_frozen_checked(
        &checked, uncoupled, is_dual, effective, coupled,
    ) {
        Ok(trees) => trees,
        Err(CheckedGenericStructureError::Provider(never)) => match never {},
        Err(CheckedGenericStructureError::Core(error)) => {
            panic!("infallible Generic tree enumeration failed unexpectedly: {error}")
        }
    }
}

pub(super) fn collect_generic_fusion_trees_for_coupled_frozen_checked<R>(
    rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
) -> Result<Vec<FusionTreeKey>, CheckedGenericStructureError<R::Error>>
where
    R: CheckedGenericFusion,
{
    let mut builder = TreeArenaBuilder::sharing();
    push_generic_fusion_trees_for_coupled_checked(
        rule,
        &mut builder,
        uncoupled,
        is_dual,
        effective,
        coupled,
    )?;
    let count = builder.len();
    let arena = builder.freeze();
    Ok((0..count).map(|row| arena.key(row)).collect())
}

/// Push every standard-form tree (all vertex labels) of `uncoupled` into
/// `coupled` as rows of `builder`.
pub(super) fn push_generic_fusion_trees_for_coupled_checked<R>(
    rule: &R,
    builder: &mut TreeArenaBuilder,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
) -> Result<(), CheckedGenericStructureError<R::Error>>
where
    R: CheckedGenericFusion,
{
    // `inner_rev` / `vtx_rev` accumulate outermost-first as the walk descends
    // (the top vertex/innerline is pushed first); the stored key wants
    // innermost-first, so emit reverses both — same discipline as
    // `visit_fusion_trees`, extended to vertex labels.
    let mut inner_rev: Vec<SectorId> = Vec::new();
    let mut vtx_rev: Vec<usize> = Vec::new();
    visit_generic_fusion_trees_checked(
        rule,
        effective,
        coupled,
        &mut inner_rev,
        &mut vtx_rev,
        &mut |inner_rev, vtx_rev| {
            builder.push_parts(
                uncoupled.iter().copied(),
                coupled,
                is_dual.iter().copied(),
                inner_rev.iter().rev().copied(),
                vtx_rev.iter().rev().map(|&label| {
                    MultiplicityIndex::new(label)
                        .expect("enumerated Generic multiplicity labels are one-based")
                }),
            );
        },
    )?;
    Ok(())
}

/// Recursive walker for [`collect_generic_fusion_trees_for_coupled`]. Mirrors
/// [`visit_fusion_trees`] (peels the LAST leg, recursing inward), but at every
/// vertex it iterates `1..=Nsymbol(...)` and records the 1-based label. Vertex
/// labels are stored in [`MultiplicityIndex`] using the same one-based
/// convention that [`mu_index`] decodes.
fn visit_generic_fusion_trees_checked<R, F>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
    inner_rev: &mut Vec<SectorId>,
    vtx_rev: &mut Vec<usize>,
    emit: &mut F,
) -> Result<(), CheckedGenericStructureError<R::Error>>
where
    R: CheckedGenericFusion,
    F: FnMut(&[SectorId], &[usize]),
{
    match effective.len() {
        0 => {
            if coupled == rule.vacuum() {
                emit(inner_rev, vtx_rev);
            }
        }
        1 => {
            if effective[0] == coupled {
                emit(inner_rev, vtx_rev);
            }
        }
        2 => {
            // Base vertex `e0 ⊗ e1 → coupled`, labels 1..=N(e0,e1,coupled).
            let n = rule
                .try_nsymbol(effective[0], effective[1], coupled)
                .map_err(CheckedGenericStructureError::Provider)?;
            for label in 1..=n {
                vtx_rev.push(label);
                emit(inner_rev, vtx_rev);
                vtx_rev.pop();
            }
        }
        _ => {
            let last = effective[effective.len() - 1];
            let front_effective = &effective[..effective.len() - 1];
            // Inner line ranges over `coupled ⊗ dual(last)`; the top vertex
            // `front_coupled ⊗ last → coupled` has `N(front_coupled,last,coupled)`
            // labels. (Same `vertexiterN` structure as the mult-free walker.)
            // `fusion_channels_in_table`: only clean sectors are ever walked
            // (tainted/escaped are an Err upstream), and clean sectors have no
            // tree through a frontier inner line — skipping frontier
            // `front_coupled` candidates drops only provably-dead branches.
            let dual_last = rule
                .try_dual(last)
                .map_err(CheckedGenericStructureError::Provider)?;
            for front_coupled in rule
                .try_fusion_channels_in_table(coupled, dual_last)
                .map_err(CheckedGenericStructureError::Provider)?
            {
                let n_last = rule
                    .try_nsymbol(front_coupled, last, coupled)
                    .map_err(CheckedGenericStructureError::Provider)?;
                if n_last == 0 {
                    continue;
                }
                inner_rev.push(front_coupled);
                for label in 1..=n_last {
                    vtx_rev.push(label);
                    visit_generic_fusion_trees_checked(
                        rule,
                        front_effective,
                        front_coupled,
                        inner_rev,
                        vtx_rev,
                        emit,
                    )?;
                    vtx_rev.pop();
                }
                inner_rev.pop();
            }
        }
    }
    Ok(())
}

#[inline]
pub(super) fn multi_associator_new_cross_channel_is_admissible<R>(
    rule: &R,
    leading: SectorId,
    short_right: SectorId,
    long_right: SectorId,
) -> bool
where
    R: FusionRule,
{
    // Why not recheck all four F vertices: validated/generated trees prove the
    // two stored vertices. At k=2 the left cross is long's stored first vertex;
    // at k>2 it is the previous stage's accepted new cross. Only this right
    // cross is newly introduced at the current stage.
    rule.nsymbol(leading, short_right, long_right) != 0
}

/// Generic-fusion `multi_associator`: the coefficient VECTOR relating a long
/// (`N`-leg) splitting tree to a short (`N-1`-leg) tail tree, indexed by the
/// topmost `a ⊗ short.coupled → long.coupled` vertex `λ` (see the module
/// convention block above). Verbatim mirror of TensorKit `multi_associator`
/// GenericFusion branch (`basic_manipulations.jl:144-166`); the
/// multiplicity-free sibling [`multiplicity_free_multi_associator_scalar`]
/// returns a bare scalar (this is that scalar chain lifted to a length-`Nλ`
/// vector).
///
/// Returns `None` iff the uncoupled/dual tails do not match (the `zero(...)`
/// early return at TK `:141-142`), so callers filter exactly as the mult-free
/// tree functions do.
///
/// Each F slice is read only after its fixed `μ`/`ν`/`κ` indices are inside the
/// returned shape and, for `k > 2`, the incoming vector length equals the `μ`
/// axis; otherwise this is `MalformedFusionTree`. Julia's bounds-checked
/// `view`/`*` in TensorKit throw on the same mismatch. Residual: on the raw
/// infallible path (`InfallibleGenericFR`) a wrong-size final `λ` axis is not
/// detected, because checking it needs the extra `N(a,e′,d)` query that #682
/// keeps out of this path. An intermediate `λ` is compared only with the next
/// F's reported `μ` axis, so a provider that misreports the same `N(a,e′,d)`
/// consistently in both F blocks is not detected either (nor by TensorKit).
pub(crate) fn generic_multi_associator_result<C, L, S>(
    rule: &C,
    long: &L,
    short: &S,
) -> Result<Option<Vec<C::Scalar>>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericFRAccess,
    L: FramedTree + ?Sized,
    S: FramedTree + ?Sized,
{
    let rank = long.uncoupled().len();
    if short.uncoupled().len() + 1 != rank
        || long.uncoupled()[1..] != *short.uncoupled()
        || long.is_dual()[1..] != *short.is_dual()
    {
        return Ok(None);
    }
    if rank == 2 {
        let n = rule
            .try_nsymbol(long.uncoupled()[0], long.uncoupled()[1], long.coupled())
            .map_err(CheckedGenericSymbolError::Provider)?;
        let mut values = vec![C::Scalar::zero(); n];
        let slot = values
            .get_mut(mu_index(long, 0)?)
            .ok_or(CheckedGenericSymbolError::Core(
                CoreError::MalformedFusionTree {
                    message: "multi_associator: vertex label exceeds Nsymbol",
                },
            ))?;
        *slot = C::Scalar::one();
        return Ok(Some(values));
    }
    let first = long.uncoupled()[0];
    // General chain (TK `:150-165`). `values` starts as the length-1 seed and is
    // transformed by one F-slice per interior leg. After each step it is indexed
    // by the current step's `λ` axis (F axis 4, `N(a, e′, d)`), which becomes
    // the next step's `μ` axis (F axis 1, `N(a, b, e)`) — the associator chain.
    let mut coeff = vec![C::Scalar::one()];
    for tensor_kit_k in 2..rank {
        let right_sector = long.uncoupled()[tensor_kit_k]; // c
                                                           // vertex_info(long, k+1) = (e, d); ν = its vertex label.
        let (middle_left, middle_right) =
            fusion_tree_vertex_neighbors_from_parts(long.uncoupled(), long, tensor_kit_k)?;
        let nu0 = mu_index(long, tensor_kit_k - 1)?;
        // vertex_info(short, k) = (b, e′); κ = its vertex label.
        let (short_left, short_right) =
            fusion_tree_vertex_neighbors_from_parts(short.uncoupled(), short, tensor_kit_k - 1)?;
        let kappa0 = mu_index(short, tensor_kit_k - 2)?;
        if rule
            .try_nsymbol(first, short_right, middle_right)
            .map_err(CheckedGenericSymbolError::Provider)?
            == 0
        {
            return Ok(None);
        }
        // F = Fsymbol(a, b, c, d, e, e′); axis order (μ, ν, κ, λ) =
        // (N(a,b,e), N(e,c,d), N(b,c,e′), N(a,e′,d)). Same argument order the
        // mult-free scalar associator passes to `f_symbol_scalar`.
        let f = rule.try_validated_f_symbol_generic(
            first,
            short_left,
            right_sector,
            middle_right,
            middle_left,
            short_right,
        )?;
        let (n_mu, n_nu, n_kappa, n_lambda) = f.shape();
        let mu0 = mu_index(long, 0)?;
        let mu_in_shape = if tensor_kit_k == 2 {
            mu0 < n_mu
        } else {
            coeff.len() == n_mu
        };
        // Why not rely on `GenericFArray::get`: it bounds-checks only the flat
        // index, so an out-of-axis label silently reads a neighbouring entry.
        if !mu_in_shape || nu0 >= n_nu || kappa0 >= n_kappa {
            return Err(CheckedGenericSymbolError::Core(
                CoreError::MalformedFusionTree {
                    message: "multi_associator: Generic F shape disagrees with the tree labels",
                },
            ));
        }
        let mut next = vec![C::Scalar::zero(); n_lambda];
        if tensor_kit_k == 2 {
            // `transpose(view(F, μ:μ, ν, κ, :)) * coeff` (TK `:159-160`): the μ
            // axis is fixed to `long.vertices[0]`, seed has length 1.
            for (lambda, slot) in next.iter_mut().enumerate() {
                *slot = f.get(mu0, nu0, kappa0, lambda).clone() * coeff[0].clone();
            }
        } else {
            // `transpose(view(F, :, ν, κ, :)) * coeff` (TK `:162`): sum over the
            // μ axis (= incoming vector index) into the λ axis.
            for (lambda, slot) in next.iter_mut().enumerate() {
                let mut acc = C::Scalar::zero();
                for (mu, coeff_mu) in coeff.iter().enumerate() {
                    acc = acc + f.get(mu, nu0, kappa0, lambda).clone() * coeff_mu.clone();
                }
                *slot = acc;
            }
        }
        coeff = next;
    }
    Ok(Some(coeff))
}

impl<C: GenericFRAccess> MultiFKernel for GenericK<'_, C> {
    type Moves = GenericFmoveTerms<C::Scalar>;
    type Lift = (SectorId, bool);

    fn unit_tail<T: FramedTree + ?Sized>(&self, _: &T) -> Result<Self::Moves, Self::E> {
        Ok(vec![(
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                self.0.vacuum(),
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<MultiplicityIndex>::new(),
            ),
            vec![C::Scalar::one()],
        )])
    }

    // The one-hot vector over N(a, b, c) at the source vertex μ (TK `:221-233`).
    fn single_tail<T: FramedTree + ?Sized>(&self, tree: &T) -> Result<Self::Moves, Self::E> {
        let n = self
            .0
            .try_nsymbol(tree.uncoupled()[0], tree.uncoupled()[1], tree.coupled())
            .map_err(CheckedGenericSymbolError::Provider)?;
        let mut coefficients = vec![C::Scalar::zero(); n];
        let slot =
            coefficients
                .get_mut(mu_index(tree, 0)?)
                .ok_or(CheckedGenericSymbolError::Core(
                    CoreError::MalformedFusionTree {
                        message: "multi_Fmove: vertex label exceeds Nsymbol",
                    },
                ))?;
        *slot = C::Scalar::one();
        Ok(vec![(
            FusionTreeKey::new(
                [tree.uncoupled()[1]],
                tree.uncoupled()[1],
                [tree.is_dual()[1]],
                [],
                [],
            ),
            coefficients,
        )])
    }

    // Stage 1 over the provider's checked channel table, Stage 2 as the
    // associator vector chain.
    fn tails<T: FramedTree + ?Sized>(&self, tree: &T) -> Result<Self::Moves, Self::E> {
        let rule = self.0;
        let first = tree.uncoupled()[0];
        let tail_uncoupled = &tree.uncoupled()[1..];
        let tail_is_dual = &tree.is_dual()[1..];
        let mut terms = Vec::new();
        let dual_first = rule
            .try_dual(first)
            .map_err(CheckedGenericSymbolError::Provider)?;
        for tail_coupled in rule
            .try_fusion_channels_in_table(dual_first, tree.coupled())
            .map_err(CheckedGenericSymbolError::Provider)?
        {
            for tail in rule.try_fusion_trees(tail_uncoupled, tail_is_dual, tail_coupled)? {
                if let Some(coefficients) = generic_multi_associator_result(rule, tree, &tail)? {
                    terms.push((tail, coefficients));
                }
            }
        }
        Ok(terms)
    }

    fn lift_leading(&(leading, _): &(SectorId, bool)) -> SectorId {
        leading
    }

    fn admit_lift(
        &self,
        leading: SectorId,
        tree_coupled: SectorId,
        coupled: SectorId,
    ) -> Result<(), Self::E> {
        if self
            .0
            .try_nsymbol(leading, tree_coupled, coupled)
            .map_err(CheckedGenericSymbolError::Provider)?
            == 0
        {
            return Err(CheckedGenericSymbolError::Core(CoreError::SectorMismatch {
                expected: coupled,
                actual: tree_coupled,
            }));
        }
        Ok(())
    }

    // Every lifted tree with the conjugated associator vector: TK's inverse
    // Stage 2 applies the adjoint of the same F-slices in the same order
    // (`:437-439, 460-462`), so `w = conj(v)` and no separate inverse chain
    // is needed.
    fn lifts<T: FramedTree + ?Sized>(
        &self,
        &(leading, leading_is_dual): &(SectorId, bool),
        coupled: SectorId,
        tree: &T,
    ) -> Result<Self::Moves, Self::E> {
        let rule = self.0;
        let uncoupled: Vec<SectorId> = std::iter::once(leading)
            .chain(tree.uncoupled().iter().copied())
            .collect();
        let dual: Vec<bool> = std::iter::once(leading_is_dual)
            .chain(tree.is_dual().iter().copied())
            .collect();
        let candidates = rule.try_fusion_trees(&uncoupled, &dual, coupled)?;
        let mut terms = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if let Some(values) = generic_multi_associator_result(rule, &candidate, tree)? {
                terms.push((
                    candidate,
                    values.into_iter().map(|value| value.conj()).collect(),
                ));
            }
        }
        Ok(terms)
    }
}

/// Generic-fusion `multi_Fmove`: recouple a splitting tree to split off its
/// first uncoupled sector, returning `(tail_tree, coeff_vector)` pairs. Mirror
/// of TensorKit `multi_Fmove` GenericFusion branch (`basic_manipulations.jl:
/// 218-232, 234-327`) and structural twin of
/// [`multiplicity_free_multi_fmove_tree`] — same Stage 1 tail enumeration, but
/// coefficients are the `generic_multi_associator` vectors (see the convention
/// block above for the vector index).
#[cfg(test)]
pub(crate) fn generic_multi_fmove_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<GenericFmoveTerms<R::Scalar>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    multi_fmove_surgery(&GenericK(&InfallibleGenericFR(rule)), tree)
        .map_err(map_infallible_generic_symbol_error)
}

/// Generic-fusion `multi_Fmove_inv`: fuse a leading sector `a` onto an existing
/// tree (coupled `b`) to a coupled sector `c`, recoupling into standard-form
/// trees with per-tree coefficient vectors indexed by the topmost INPUT vertex
/// `a ⊗ b → c` (TK `:343-347`). Structural twin of
/// [`multiplicity_free_multi_fmove_inv_tree`].
///
/// Like the mult-free version, the per-candidate coefficient is the
/// `generic_multi_associator(candidate, tree)` vector, CONJUGATED. This is
/// exact because TensorKit's inverse Stage 2 applies the adjoint of the same
/// F-slices in the same order: with `Tₖ = transpose(view(F,:,ν,κ,:))` the
/// forward associator computes `v = Tₙ⋯T₂·seed`, while the inverse computes
/// `w = conj(Tₙ)⋯conj(T₃)·conj(T₂·seed) = conj(v)` (TK `:437-439, 460-462`,
/// the `conj!`/`'` on each factor). No separate inverse F-chain is needed.
#[cfg(test)]
pub(crate) fn generic_multi_fmove_inv_tree<R>(
    rule: &R,
    leading_sector: SectorId,
    coupled: SectorId,
    tree: &FusionTreeKey,
    leading_is_dual: bool,
) -> Result<GenericFmoveTerms<R::Scalar>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    multi_fmove_inv_surgery(
        &GenericK(&InfallibleGenericFR(rule)),
        &(leading_sector, leading_is_dual),
        coupled,
        tree,
    )
    .map_err(map_infallible_generic_symbol_error)
}
