//! The compact-diagonal arms of contraction, composition, trace, transform and
//! twist (TensorKit's `DiagonalTensorMap` methods). Each answers `None` when
//! its operands or destination do not fit, and the dense route runs.
//!
//! Every arm is called from the shared bodies in both modes; the trace arm
//! reads the dense trace's compiled terms.

use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D>,
    D: TensorScalar,
{
    /// The compact arms of [`Self::compose`], or `None` when the operands or
    /// the destination cannot support one and the dense route must run.
    ///
    /// Each arm proves its destination rather than deriving it. Composition
    /// glues `self.codomain <- other.domain`, so:
    ///
    /// - `D * D` — both operands are bond spaces (`codomain == domain`), so
    ///   when the two spaces are equal the destination *is* that space, and
    ///   [`is_diagonal_bond_space`] certifies it can hold a compact result.
    /// - `t * D` — the destination is `t.codomain <- D.domain`; `D` is a bond
    ///   space, so `D.domain == D.codomain`, and requiring that to equal
    ///   `t.domain` makes the destination `t`'s own space. `t`'s payload is
    ///   then `t`'s data with each block's trailing axis scaled.
    /// - `D * t` — the mirror image, scaling `t`'s leading axis.
    ///
    /// Without those equalities the destination is a different space (a dual
    /// leg on the contracted side is the reachable case) and reusing an
    /// operand's would silently produce a tensor on the wrong space, so the
    /// arm declines and the expert layer decides — including by rejecting a
    /// composition that is not one at all.
    pub(super) fn compose_compact(&self, other: &Self) -> Result<Option<Self>, Error> {
        if !self.same_rule(other) {
            return Ok(None);
        }
        let (left, right) = (self.logical_space().space(), other.logical_space().space());
        match (self.spectrum(), other.spectrum()) {
            (Some(lhs), Some(rhs)) => {
                // Both clauses are unreachable today and stay for the reason
                // [`is_diagonal_bond_space`] gives. `left != right` is the
                // weaker one: two compact payloads on unequal bond spaces
                // necessarily carry spectra that differ in their sectors or
                // their lengths, so the elementwise product below would refuse
                // them anyway — just with `spectra_disagree`'s message instead
                // of the expert layer's. Removing it would change which error a
                // caller sees, not whether one is reported.
                if left != right || !is_diagonal_bond_space(left) {
                    return Ok(None);
                }
                if lhs.len() != rhs.len() {
                    return Err(spectra_disagree());
                }
                let product = lhs
                    .iter()
                    .zip(rhs)
                    .map(|(left, right)| {
                        if left.sector != right.sector || left.values.len() != right.values.len() {
                            return Err(spectra_disagree());
                        }
                        Ok(symtn_matrixalgebra::SectorSpectrum {
                            sector: left.sector,
                            values: left
                                .values
                                .iter()
                                .zip(&right.values)
                                .map(|(&a, &b)| a * b)
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(Some(self.with_spectrum(product)))
            }
            // `t * D`: scale each block's trailing axis (TensorKit `rmul!`).
            (None, Some(spectrum)) => {
                if !is_diagonal_bond_space(right)
                    || left.homspace().domain().legs() != right.homspace().codomain().legs()
                {
                    return Ok(None);
                }
                self.scaled_axis(None, spectrum).map(Some)
            }
            // `D * t`: scale each block's leading axis (TensorKit `lmul!`).
            (Some(spectrum), None) => {
                if !is_diagonal_bond_space(left)
                    || right.homspace().codomain().legs() != left.homspace().domain().legs()
                {
                    return Ok(None);
                }
                other.scaled_axis(Some(0), spectrum).map(Some)
            }
            (None, None) => Ok(None),
        }
    }

    /// The compact arm of [`Self::contract`] (issue #584), or `None` when the
    /// operands, the axis pattern or the output order do not fit one and the
    /// dense route must run.
    ///
    /// A one-axis contraction against a compact operand is a bond scaling, so
    /// the spectrum multiplies the *other* operand's contracted leg — O(d·n)
    /// against the dense route's O(d²·n) GEMM on a materialized `Σ_c d_c²`
    /// buffer — and one [`Self::permute`] lays the result out. The permute is
    /// what carries every recoupling and bend, so this adds no mathematics of
    /// its own; it is a scale followed by one permutation.
    ///
    /// The patterns are [`Self::diagonal_contract_pattern`]'s.
    pub(super) fn try_contract_diagonal(
        &self,
        engine: &mut dyn Engine,
        other: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        codomain_rank: usize,
    ) -> Result<Option<Self>, FacadeError<R>> {
        let Some(DiagonalContract { pattern, layout }) =
            self.diagonal_contract_pattern(other, lhs_axes, rhs_axes, output_axes, codomain_rank)
        else {
            return Ok(None);
        };
        match pattern {
            // Contract is permute(compose) here (the pattern excludes the
            // fermionic twist): the product spectrum, then the one permute.
            // Why the permute and not a rebind: `pAB` can move the surviving
            // bond across the codomain/domain split, where rebinding the
            // product spectrum is not equivalent (#453). A swap stays compact
            // through the rank-(1, 1) arm (TensorKit `diagonal.jl:291`);
            // (2, 0)/(0, 2) results are dense.
            DiagonalContractPattern::Product => {
                let Some(product) = self.compose_compact(other)? else {
                    return Ok(None);
                };
                match layout {
                    None => Ok(Some(product)),
                    Some(operation) => product.tree_transform(engine, operation).map(Some),
                }
            }
            DiagonalContractPattern::Scaled {
                scaled,
                axis,
                spectrum,
                rebind,
            } => {
                let completed = scaled.scaled_axis(Some(axis), spectrum)?;
                let completed = match layout {
                    None => completed,
                    Some(operation) => completed.tree_transform(engine, operation)?,
                };
                if !rebind {
                    return Ok(Some(completed));
                }
                // Scaling and permutation deliberately run on `other`; only
                // after both succeed do we rebind their validated owned-dense
                // result to the public contract's exact left authority.
                let TensorMap { repr, .. } = completed;
                let TypedTensorRepr::Owned(body) = repr else {
                    return Ok(None);
                };
                if !matches!(body.data.as_ref(), TypedData::Dense(_)) {
                    return Ok(None);
                }
                let space = self
                    .logical_space()
                    .rebind_validated(&body.space.validated_layout())
                    .map_err(Error::from)?;
                Ok(Some(Self {
                    runtime: self.runtime.clone(),
                    repr: owned_repr(TypedTensorBody::with_shared_payload(
                        space,
                        Arc::clone(&body.data),
                    )),
                }))
            }
        }
    }

    /// The compact arm of [`Self::contract_into`]: the algebra of
    /// [`Self::try_contract_diagonal`] written into `destination`, TensorKit
    /// `blas_contract!` with a `DiagonalTensorMap` operand (`mul!` scaling
    /// its `Diagonal` blocks, plus `copyC`'s `tensoradd!` for a reordered
    /// output). `false` when the dense route must run; `destination` already
    /// passed [`destination_precheck`] against both operands.
    ///
    /// - `t · D` / `D · t` in the scaled operand's order: one pass
    ///   `destination = β·destination + α·scale_axis(t)`, reading a direct `t`
    ///   in place (a lazy adjoint `t` is read into one logical-order buffer
    ///   first, as the eager arm reads it).
    /// - reordered: the scaled `t` in one operation-local buffer, then the
    ///   transform `_into` from it with `(α, β)`.
    /// - `D · D`: the product spectrum, written on the diagonal in its own
    ///   order, or else read by the compact-source transform `_into` onto
    ///   the output order (contract is permute(compose) here: the pattern
    ///   excludes the fermionic twist).
    pub(super) fn try_contract_diagonal_into(
        &self,
        engine: &mut dyn Engine,
        other: &Self,
        spec: &ContractSpec<'_>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<bool, FacadeError<R>> {
        let Some(DiagonalContract { pattern, layout }) = self.diagonal_contract_pattern(
            other,
            spec.lhs,
            spec.rhs,
            &spec.output_axes(),
            spec.codomain.len(),
        ) else {
            return Ok(false);
        };
        let (tensor, operation) = match pattern {
            DiagonalContractPattern::Product => {
                let Some(product) = self.compose_compact(other)? else {
                    return Ok(false);
                };
                let Some(operation) = layout else {
                    // On the bond space itself, as eager's compact `D · D`:
                    // `β·destination`, then `α·s` on each block's diagonal.
                    let space = product.logical_space().space();
                    require_destination_space(destination.logical_space().space(), space)?;
                    let spectrum = product
                        .spectrum()
                        .ok_or_else(|| internal_layout_error("a compact product"))?;
                    let data = destination_slice(destination)?;
                    scale_by_beta(data, beta);
                    if !alpha.is_zero() {
                        add_spectrum_into(space, data, spectrum, alpha)?;
                    }
                    return Ok(true);
                };
                (product, operation)
            }
            DiagonalContractPattern::Scaled {
                scaled,
                axis,
                spectrum,
                ..
            } => match layout {
                None => {
                    require_destination_space(
                        destination.logical_space().space(),
                        scaled.logical_space().space(),
                    )?;
                    let source = scaled.logical_dense_data()?;
                    symtn_matrixalgebra::seam::axpby_axis_by_spectrum_into(
                        scaled.logical_space().space(),
                        destination_slice(destination)?,
                        &source,
                        Some(axis),
                        spectrum,
                        |value| value,
                        alpha,
                        beta,
                    )
                    .map_err(Error::from)?;
                    return Ok(true);
                }
                Some(operation) => (scaled.scaled_axis(Some(axis), spectrum)?, operation),
            },
        };
        let probe_operation = operation.clone();
        tensor.tree_transform_into(
            engine,
            destination,
            alpha,
            beta,
            || Ok(operation),
            |probe| {
                probe(symtn_tensors::TreeTransformOperationView::of(
                    &probe_operation,
                ))
            },
        )?;
        Ok(true)
    }

    /// The one pattern authority of both compact contraction arms, or `None`
    /// when the dense route must run.
    ///
    /// # Which patterns, and why only those
    ///
    /// The engine admits a contracted pair only when the two legs agree on
    /// their raw duality flag on the compose-shaped pairing (one operand's
    /// domain leg against the other's codomain leg), and a compact operand's
    /// leg *is* its bond on both sides. So each arm requires exactly that
    /// pairing and compares the two legs itself: raw equality is the engine's
    /// admissibility condition here, so a mismatch is a contraction the dense
    /// route must reject rather than one this arm may answer, and the arm
    /// declines so the expert layer reports it in its own words. `D · D` is
    /// [`Self::compose_compact`]'s product, which already proves its
    /// destination. These declines are TensorKit's own `copyA`/`copyB` cases
    /// (`tensoroperations.jl:399-409` @cfaa073e): a non-shared pairing, or the
    /// fermionic twist.
    ///
    /// # The twist, and why it is not folded
    ///
    /// [`Self::contract`] applies the fermionic supertrace twist to a **dual**
    /// contracted leg of `other`, where [`Self::compose`] does not. Here the case
    /// cannot arise, so the arm declines instead of carrying arithmetic no test
    /// could reach: a compact payload's bond leg is built non-dual
    /// (`seam::spectrum_bond`), the arms pair it with a *codomain*
    /// leg of `other` whose external duality is exactly its raw flag, and
    /// admissibility forces that flag to equal the bond's. The guard stays
    /// because the first constructor of a compact payload on a dual bond leg —
    /// or of an arm pairing a domain leg of `other` — should decline rather
    /// than silently return a wrong sign.
    fn diagonal_contract_pattern<'t>(
        &'t self,
        other: &'t Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        codomain_rank: usize,
    ) -> Option<DiagonalContract<'t, R, D>> {
        if lhs_axes.len() != 1 || rhs_axes.len() != 1 || !self.same_rule(other) {
            return None;
        }
        let (lhs_axis, rhs_axis) = (lhs_axes[0], rhs_axes[0]);
        if lhs_axis >= self.rank() || rhs_axis >= other.rank() {
            return None;
        }
        // Why the provider rather than a stored flag: `braiding_style` is the
        // rule's own answer.
        let fermionic = <R::Mode as TypedTensorModeDispatch<R>>::braiding_style(self.provider())
            == symtn_core::BraidingStyleKind::Fermionic;
        if fermionic
            && other
                .logical_space()
                .space()
                .homspace()
                .external_axis_is_dual(rhs_axis)
                != Some(false)
        {
            return None;
        }
        let (left_home, right_home) = (
            self.logical_space().space().homspace(),
            other.logical_space().space().homspace(),
        );
        let (pattern, source, scaled_codomain_rank) = match (self.spectrum(), other.spectrum()) {
            (Some(_), Some(_)) => {
                if lhs_axis != 1 || rhs_axis != 0 {
                    return None;
                }
                (DiagonalContractPattern::Product, vec![0, 1], 1)
            }
            // `t · D` (TensorKit `rmul!`): scale `t`'s contracted domain leg,
            // then move it to where the contraction's output order wants it.
            (None, Some(spectrum)) => {
                if rhs_axis != 0 || lhs_axis < self.codomain_rank() {
                    return None;
                }
                if left_home.domain().legs()[lhs_axis - self.codomain_rank()]
                    != right_home.codomain().legs()[0]
                {
                    return None;
                }
                let mut source: Vec<usize> = (0..self.rank()).filter(|&a| a != lhs_axis).collect();
                source.push(lhs_axis);
                let pattern = DiagonalContractPattern::Scaled {
                    scaled: self,
                    axis: lhs_axis,
                    spectrum,
                    rebind: false,
                };
                (pattern, source, self.codomain_rank())
            }
            // `D · t` (TensorKit `lmul!`): the mirror image, scaling the
            // contracted codomain leg of `t` at whatever position it sits.
            (Some(spectrum), None) => {
                if lhs_axis != 1 || rhs_axis >= other.codomain_rank() {
                    return None;
                }
                if left_home.domain().legs()[0] != right_home.codomain().legs()[rhs_axis] {
                    return None;
                }
                let mut source = vec![rhs_axis];
                source.extend((0..other.rank()).filter(|&a| a != rhs_axis));
                let pattern = DiagonalContractPattern::Scaled {
                    scaled: other,
                    axis: rhs_axis,
                    spectrum,
                    rebind: true,
                };
                (pattern, source, other.codomain_rank())
            }
            (None, None) => return None,
        };
        let layout = output_layout(&source, output_axes, codomain_rank, scaled_codomain_rank)?;
        Some(DiagonalContract { pattern, layout })
    }

    /// This tensor's dense payload in its logical orientation: borrowed when
    /// stored so, a lazy adjoint read through its parent into one buffer.
    fn logical_dense_data(&self) -> Result<std::borrow::Cow<'_, [D]>, Error> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let (operand, source) = self.fusion_operand_and_data();
            return Ok(std::borrow::Cow::Owned(
                symtn_tensors::oriented_fusion_add_owned(
                    self.logical_space().space().structure(),
                    operand,
                    &source,
                    operand,
                    &source,
                    D::from_real(1.0),
                    D::from_real(0.0),
                )?,
            ));
        }
        Ok(self
            .owned_body()
            .expect("owned scaled-axis input")
            .materialized_dense_data())
    }

    /// This tensor with one bond axis of every block scaled by `spectrum`,
    /// on its own space. `axis = None` scales the trailing axis, `Some(0)` the
    /// leading one, exactly as the seam names them.
    fn scaled_axis(
        &self,
        axis: Option<usize>,
        spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
    ) -> Result<Self, Error> {
        let mut data = self.logical_dense_data()?.into_owned();
        symtn_matrixalgebra::seam::scale_axis_by_spectrum_mapped(
            self.logical_space().space(),
            &mut data,
            axis,
            spectrum,
            |value| value,
        )?;
        Ok(self.with_data(data))
    }

    /// Whether two operands' providers are the same rule. The compact paths
    /// below skip the expert layer, which is where a mismatch would otherwise
    /// be caught, so they have to ask themselves. Why the admitted identity
    /// rather than the provider's: it *is* the provider's (#2046), and reading
    /// it makes no provider call.
    fn same_rule(&self, other: &Self) -> bool {
        self.logical_space().space().admission().rule_identity()
            == other.logical_space().space().admission().rule_identity()
    }
}

/// A compact contraction pattern and how its default-order result is laid
/// out on the requested output: `None` in that order already, else the one
/// permute that moves it.
struct DiagonalContract<'t, R, D> {
    pattern: DiagonalContractPattern<'t, R, D>,
    layout: Option<TreeTransformOperation>,
}

enum DiagonalContractPattern<'t, R, D> {
    /// `D · D` on the shared pairing: the product spectrum.
    Product,
    /// `t · D` or `D · t`: `scaled`'s `axis` multiplied by `spectrum`;
    /// `rebind` when `scaled` is the right operand, whose eager result is
    /// rebound to the left authority.
    Scaled {
        scaled: &'t TensorMap<R, D>,
        axis: usize,
        spectrum: &'t [symtn_matrixalgebra::SectorSpectrum<D>],
        rebind: bool,
    },
}

/// The permute taking a result whose axes are `source` (the contraction's
/// default output order, split at `source_codomain_rank`) to `output_axes`
/// split at `codomain_rank`: `Some(None)` when it is the identity, `None`
/// when `output_axes` is not a permutation of `0..source.len()`. A malformed
/// order declines rather than errors: the dense route validates it and
/// reports it, and one error message beats two.
fn output_layout(
    source: &[usize],
    output_axes: &[usize],
    codomain_rank: usize,
    source_codomain_rank: usize,
) -> Option<Option<TreeTransformOperation>> {
    let mut sorted = output_axes.to_vec();
    sorted.sort_unstable();
    if sorted.iter().copied().ne(0..source.len()) {
        return None;
    }
    let ordered: Vec<usize> = output_axes.iter().map(|&axis| source[axis]).collect();
    if codomain_rank == source_codomain_rank && ordered.iter().copied().eq(0..source.len()) {
        return Some(None);
    }
    Some(Some(TreeTransformOperation::permute(
        ordered[..codomain_rank].iter().copied(),
        ordered[codomain_rank..].iter().copied(),
    )))
}

/// `twist` of a compact diagonal, or `None` for any other payload. A bond
/// space's two legs both carry the block's coupled sector, so the per-block
/// factor collapses to `θ(sector)^|legs|`; the space is unchanged, so the
/// payload stays compact. `theta` is the mode's staged `twist_values`; the
/// caller has already returned a shared clone when every value is one.
pub(super) fn twist_spectrum<R, D, C>(
    tensor: &TensorMap<R, D>,
    legs: &[usize],
    inverse: bool,
    theta: impl Fn(SectorId) -> C,
) -> Option<TensorMap<R, D>>
where
    D: TensorScalar,
    C: TwistCoefficient<D>,
{
    let spectrum = tensor.spectrum()?;
    let sector_factor = |sector: SectorId| -> C {
        let factor = legs
            .iter()
            .map(|_| theta(sector))
            .fold(C::one(), |product, factor| product * factor);
        twist_factor_with_inverse(factor, inverse)
    };
    let scaled = spectrum
        .iter()
        .map(|entry| {
            let factor: D = sector_factor(entry.sector).as_data();
            symtn_matrixalgebra::SectorSpectrum {
                sector: entry.sector,
                values: entry.values.iter().map(|&value| value * factor).collect(),
            }
        })
        .collect();
    Some(tensor.with_spectrum_on(tensor.logical_space().clone(), scaled))
}

/// The compact rank-(1,1) transforms, one body for both modes; `None` for
/// every other payload or operation.
///
/// - **swap** (`permute`/`transpose` with `[1] | [0]`): TensorKit `cfaa073e`
///   `permute(d::DiagonalTensorMap, ...)` / `transpose` (`diagonal.jl:215-273`)
///   keep the result diagonal on `dual(d.domain)`; it stays compact here on
///   the mode's staged destination.
/// - **braid**: TensorKit has no diagonal `braid`; `similar(t, T, V)`
///   (`abstracttensor.jl:614-618`) makes it a dense `TensorMap`. The dense
///   result is read from the spectrum, without a `Σ_c k_c²` source.
///
/// Both read the coefficient from the same compiled structure the dense
/// route replays, staged and committed once by
/// [`TypedTensorTransformDispatch::transform_bond_spectrum`]. A braid whose
/// coefficient is not a finite non-zero `D` (an R symbol beyond `f32`) is
/// that structure's dense replay of the densified spectrum, as on the dense
/// route.
pub(super) fn transform_rank_one_diagonal<R, D>(
    engine: &mut dyn Engine,
    tensor: &TensorMap<R, D>,
    operation: &TreeTransformOperation,
) -> Result<Option<TensorMap<R, D>>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D>,
    D: TensorScalar,
{
    let Some(spectrum) = tensor.spectrum() else {
        return Ok(None);
    };
    let (codomain_rank, domain_rank) = (tensor.codomain_rank(), tensor.domain_rank());
    let braid =
        crate::tensor_product::is_rank_one_diagonal_braid(codomain_rank, domain_rank, operation);
    if !braid
        && !crate::tensor_product::is_rank_one_diagonal_swap(codomain_rank, domain_rank, operation)
    {
        return Ok(None);
    }
    let BondTransform {
        destination,
        output,
    } = <R::Mode as TypedTensorTransformDispatch<R, D>>::transform_bond_spectrum(
        engine, tensor, spectrum, operation, braid,
    )?;
    let data = match (output, braid) {
        (BondOutput::Spectrum(spectrum), false) => {
            return Ok(Some(tensor.with_spectrum_on(destination, spectrum)));
        }
        // Why unreachable: rank-one trees compile to one `Single` term per
        // block, and a swap does not ask for representability.
        (BondOutput::Dense(_), false) => {
            return Err(internal_layout_error(
                "a rank-one diagonal swap resolved to a non-bijective structure",
            )
            .into());
        }
        (BondOutput::Spectrum(spectrum), true) => {
            symtn_matrixalgebra::seam::diagonal_bond_data(destination.space(), &spectrum, &|v| v)?
        }
        (BondOutput::Dense(data), true) => data,
    };
    Ok(Some(tensor.published(destination, data)))
}

/// The full trace of a rank-(1,1) compact diagonal over its only pair, read
/// from the dense trace's compiled terms, or `None` when the terms are not
/// one diagonal read of a spectrum block each (#604, #1866).
///
/// The one owned trace offers only a scalar destination, so with a rank-two
/// source every term reads the trace `Σ_i s_i` of one source block and adds
/// it with its coefficient — exactly the dense execution's work, minus the
/// `k²` buffer. Why not the per-sector `dim(c) · θ(c)` formula: each mode
/// rounds its trace channel factor differently, and only the compiled
/// coefficients keep compact and dense on one authority per mode. The sum is
/// widened into `Complex64` and narrowed once, so it equals the dense
/// route's `D` accumulation within dtype tolerance, not bitwise. A term
/// whose block is not this spectrum's (`get`, coupled sector, square shape
/// of the spectrum's length) declines, and the dense route executes in the
/// same transaction.
pub(super) fn full_trace_terms<D: TensorScalar>(
    structure: &symtn_tensors::TensorTraceFusionStructure<f64>,
    source: &symtn_core::BlockStructure,
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
) -> Option<D> {
    if structure.src_rank() != 2 {
        return None;
    }
    let mut total = num_complex::Complex64::new(0.0, 0.0);
    for term in structure.terms() {
        let entry = spectrum.get(term.src_block())?;
        let block = source.block(term.src_block()).ok()?;
        let &[rows, columns] = block.shape() else {
            return None;
        };
        if term.src_key().codomain_tree().coupled() != entry.sector
            || rows != columns
            || entry.values.len() != rows
        {
            return None;
        }
        let mut partial = D::Wide::from_real(0.0);
        for &value in &entry.values {
            partial = partial + value.widen();
        }
        total += partial.widen_complex() * *term.coefficient();
    }
    Some(D::from_complex64(total))
}
