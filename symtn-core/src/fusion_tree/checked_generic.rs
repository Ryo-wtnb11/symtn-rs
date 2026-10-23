use super::*;

/// Expert error returned by checked Generic-fusion structural construction.
///
/// The provider source remains typed; core structural errors retain their
/// established variants instead of being flattened into a string envelope.
#[derive(Debug)]
pub enum CheckedGenericStructureError<E> {
    Provider(E),
    Core(CoreError),
}

impl<E> From<CoreError> for CheckedGenericStructureError<E> {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl<E: fmt::Display> fmt::Display for CheckedGenericStructureError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(_) => formatter.write_str("fusion provider failed"),
            Self::Core(_) => formatter.write_str("invalid tensor structure"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CheckedGenericStructureError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::Core(error) => Some(error),
        }
    }
}

/// Fallible Generic-symbol access error used internally by checked row
/// lowering. Native testing providers bypass the production adapters, so
/// their categorical shape failures remain distinct from provider failures.
#[doc(hidden)]
#[derive(Debug)]
pub enum CheckedGenericSymbolError<E> {
    Provider(E),
    Shape {
        symbol: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    Core(CoreError),
}

impl<E> From<CoreError> for CheckedGenericSymbolError<E> {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl<E: fmt::Display> fmt::Display for CheckedGenericSymbolError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(_) => formatter.write_str("fusion provider failed"),
            Self::Shape {
                symbol,
                expected,
                actual,
            } => write!(
                formatter,
                "{symbol} shape mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::Core(_) => formatter.write_str("invalid tensor structure"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CheckedGenericSymbolError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::Core(error) => Some(error),
            Self::Shape { .. } => None,
        }
    }
}

#[cfg(any(test, feature = "testing"))]
pub(super) fn map_infallible_generic_symbol_error(
    error: CheckedGenericSymbolError<std::convert::Infallible>,
) -> CoreError {
    match error {
        CheckedGenericSymbolError::Provider(never) => match never {},
        CheckedGenericSymbolError::Core(error) => error,
        CheckedGenericSymbolError::Shape { symbol, .. } => CoreError::MalformedFusionTree {
            message: if symbol == "F" {
                "Generic F-symbol shape mismatch"
            } else {
                "Generic symbol shape mismatch"
            },
        },
    }
}

pub(crate) trait GenericFRAccess {
    type Scalar: CategoricalScalar;
    type Error;
    fn fusion_style(&self) -> FusionStyleKind;
    fn braiding_style(&self) -> BraidingStyleKind;
    fn vacuum(&self) -> SectorId;
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error>;
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error>;
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error>;
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error>;
    fn try_validated_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, CheckedGenericSymbolError<Self::Error>>;
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error>;
    /// Every standard-form tree over the externals with the given coupled
    /// sector, in canonical order.
    fn try_fusion_trees(
        &self,
        uncoupled: &[SectorId],
        is_dual: &[bool],
        coupled: SectorId,
    ) -> Result<Vec<FusionTreeKey>, CheckedGenericSymbolError<Self::Error>>;
}

pub(crate) trait GenericRigidAccess: GenericFRAccess {
    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error>;
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error>;
    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error>;
    fn try_b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>>;
    #[allow(dead_code)]
    fn try_a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>>;
}

#[cfg(any(test, feature = "testing"))]
pub(crate) struct InfallibleGenericFR<'a, R>(pub(crate) &'a R);

#[cfg(any(test, feature = "testing"))]
impl<R> GenericFRAccess for InfallibleGenericFR<'_, R>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    type Scalar = R::Scalar;
    type Error = std::convert::Infallible;
    fn fusion_style(&self) -> FusionStyleKind {
        self.0.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.0.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.0.vacuum()
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(self.0.dual(sector))
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        Ok(self.0.nsymbol(a, b, c))
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.0.fusion_channels(a, b))
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error> {
        Ok(self.0.f_symbol_generic(a, b, c, d, e, f))
    }
    fn try_validated_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        Ok(self.0.f_symbol_generic(a, b, c, d, e, f))
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        Ok(self.0.r_symbol_generic(a, b, c))
    }
    fn try_fusion_trees(
        &self,
        uncoupled: &[SectorId],
        is_dual: &[bool],
        coupled: SectorId,
    ) -> Result<Vec<FusionTreeKey>, CheckedGenericSymbolError<Self::Error>> {
        Ok(collect_generic_fusion_trees_for_coupled_frozen(
            self.0, uncoupled, is_dual, uncoupled, coupled,
        ))
    }
}

#[cfg(any(test, feature = "testing"))]
impl<R> GenericRigidAccess for InfallibleGenericFR<'_, R>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.0.sqrt_dim_scalar(sector))
    }
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.0.inv_sqrt_dim_scalar(sector))
    }
    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        Ok(self.0.frobenius_schur_phase_scalar(sector))
    }
    fn try_b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        checked_generic_b_symbol(self, a, b, c)
    }
    fn try_a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        checked_generic_a_symbol(self, a, b, c)
    }
}

impl<P: CheckedGenericRigidSymbols> GenericFRAccess for P {
    type Scalar = P::Scalar;
    type Error = P::Error;
    fn fusion_style(&self) -> FusionStyleKind {
        CheckedGenericFusion::fusion_style(self)
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        CheckedGenericFusion::braiding_style(self)
    }
    fn vacuum(&self) -> SectorId {
        CheckedGenericFusion::vacuum(self)
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        CheckedGenericFusion::try_dual(self, sector)
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        CheckedGenericFusion::try_nsymbol(self, a, b, c)
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        CheckedGenericFusion::try_fusion_channels_in_table(self, a, b)
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error> {
        CheckedGenericRigidSymbols::try_f_symbol_generic(self, a, b, c, d, e, f)
    }
    fn try_validated_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        checked_generic_f_symbol(self, a, b, c, d, e, f)
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        CheckedGenericRigidSymbols::try_r_symbol_generic(self, a, b, c)
    }
    fn try_fusion_trees(
        &self,
        uncoupled: &[SectorId],
        is_dual: &[bool],
        coupled: SectorId,
    ) -> Result<Vec<FusionTreeKey>, CheckedGenericSymbolError<Self::Error>> {
        collect_generic_fusion_trees_for_coupled_frozen_checked(
            self, uncoupled, is_dual, uncoupled, coupled,
        )
        .map_err(map_checked_generic_structure_error)
    }
}

impl<P: CheckedGenericRigidSymbols> GenericRigidAccess for P {
    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        CheckedGenericRigidSymbols::try_sqrt_dim_scalar(self, sector)
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        CheckedGenericRigidSymbols::try_inv_sqrt_dim_scalar(self, sector)
    }

    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        CheckedGenericRigidSymbols::try_frobenius_schur_phase_scalar(self, sector)
    }

    fn try_b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        let provider = CheckedGenericSymbolError::Provider;
        let _rows = CheckedGenericFusion::try_nsymbol(self, a, b, c).map_err(provider)?;
        let dual_b = CheckedGenericFusion::try_dual(self, b).map_err(provider)?;
        let _cols = CheckedGenericFusion::try_nsymbol(self, c, dual_b, a).map_err(provider)?;
        let symbol =
            CheckedGenericRigidSymbols::try_b_symbol_generic(self, a, b, c).map_err(provider)?;
        #[cfg(any(test, feature = "testing"))]
        if symbol.shape() != (_rows, _cols) {
            let (rows, cols) = symbol.shape();
            return Err(CheckedGenericSymbolError::Shape {
                symbol: "B",
                expected: vec![_rows, _cols],
                actual: vec![rows, cols],
            });
        }
        Ok(symbol)
    }

    fn try_a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        let provider = CheckedGenericSymbolError::Provider;
        let _rows = CheckedGenericFusion::try_nsymbol(self, a, b, c).map_err(provider)?;
        let dual_a = CheckedGenericFusion::try_dual(self, a).map_err(provider)?;
        let _cols = CheckedGenericFusion::try_nsymbol(self, dual_a, c, b).map_err(provider)?;
        let symbol =
            CheckedGenericRigidSymbols::try_a_symbol_generic(self, a, b, c).map_err(provider)?;
        #[cfg(any(test, feature = "testing"))]
        if symbol.shape() != (_rows, _cols) {
            let (rows, cols) = symbol.shape();
            return Err(CheckedGenericSymbolError::Shape {
                symbol: "A",
                expected: vec![_rows, _cols],
                actual: vec![rows, cols],
            });
        }
        Ok(symbol)
    }
}

/// Validate a Generic fusion-tree pair without requiring the infallible
/// [`FusionRule`] contract.
pub fn validate_generic_fusion_tree_pair_checked<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
) -> Result<(), CheckedGenericStructureError<C::Error>>
where
    C: CheckedGenericFusion,
{
    validate_generic_fusion_tree_checked(rule, tree_pair.codomain_tree().as_view())?;
    validate_generic_fusion_tree_trusted_style(rule, tree_pair.domain_tree().as_view())?;
    validate_fusion_tree_pair_coupled(tree_pair.codomain_tree(), tree_pair.domain_tree())?;
    Ok(())
}

/// Admit one Generic fusion tree through the provider: the single-tree half
/// of [`validate_generic_fusion_tree_pair_checked`].
pub(super) fn validate_generic_fusion_tree_checked<C>(
    rule: &C,
    tree: FusionTreeRef<'_>,
) -> Result<(), CheckedGenericStructureError<C::Error>>
where
    C: CheckedGenericFusion,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        }
        .into());
    }
    validate_generic_fusion_tree_trusted_style(rule, tree)
}

fn validate_generic_fusion_tree_trusted_style<C>(
    rule: &C,
    tree: FusionTreeRef<'_>,
) -> Result<(), CheckedGenericStructureError<C::Error>>
where
    C: CheckedGenericFusion,
{
    validate_fusion_tree_key_shape(tree)?;
    validate_fusion_tree_checked_after_shape(
        tree,
        rule.vacuum(),
        |left, right| {
            rule.try_fusion_channels(left, right)
                .map(drop)
                .map_err(CheckedGenericStructureError::Provider)
        },
        |left, right, coupled| {
            rule.try_nsymbol(left, right, coupled)
                .map_err(CheckedGenericStructureError::Provider)
        },
    )
}

pub(super) fn map_checked_generic_structure_error<E>(
    error: CheckedGenericStructureError<E>,
) -> CheckedGenericSymbolError<E> {
    match error {
        CheckedGenericStructureError::Provider(error) => CheckedGenericSymbolError::Provider(error),
        CheckedGenericStructureError::Core(error) => CheckedGenericSymbolError::Core(error),
    }
}

pub(super) fn checked_generic_f_symbol<C>(
    rule: &C,
    a: SectorId,
    b: SectorId,
    c: SectorId,
    d: SectorId,
    e: SectorId,
    f: SectorId,
) -> Result<GenericFArray<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericFRAccess,
{
    let provider = CheckedGenericSymbolError::Provider;
    #[cfg(any(test, feature = "testing"))]
    let expected = [
        rule.try_nsymbol(a, b, e).map_err(provider)?,
        rule.try_nsymbol(e, c, d).map_err(provider)?,
        rule.try_nsymbol(b, c, f).map_err(provider)?,
        rule.try_nsymbol(a, f, d).map_err(provider)?,
    ];
    let symbol = rule
        .try_f_symbol_generic(a, b, c, d, e, f)
        .map_err(provider)?;
    #[cfg(any(test, feature = "testing"))]
    {
        let (mu, nu, kappa, lambda) = symbol.shape();
        let actual = [mu, nu, kappa, lambda];
        if actual != expected {
            return Err(CheckedGenericSymbolError::Shape {
                symbol: "F",
                expected: expected.to_vec(),
                actual: actual.to_vec(),
            });
        }
    }
    Ok(symbol)
}

pub(super) fn checked_generic_r_symbol<C>(
    rule: &C,
    a: SectorId,
    b: SectorId,
    c: SectorId,
) -> Result<GenericRMatrix<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericFRAccess,
{
    #[cfg(any(test, feature = "testing"))]
    let expected = [rule.try_nsymbol(a, b, c), rule.try_nsymbol(b, a, c)]
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(CheckedGenericSymbolError::Provider)?;
    let symbol = rule
        .try_r_symbol_generic(a, b, c)
        .map_err(CheckedGenericSymbolError::Provider)?;
    #[cfg(any(test, feature = "testing"))]
    {
        let (rows, cols) = symbol.shape();
        let actual = vec![rows, cols];
        if actual != expected {
            return Err(CheckedGenericSymbolError::Shape {
                symbol: "R",
                expected,
                actual,
            });
        }
    }
    Ok(symbol)
}

/// `B^{ab}_c[μ,ν] = √d_a·√d_b/√d_c · F(a,b,b̄,a,c,1)[μ,ν,0,0]`,
/// testing oracle: TensorKitSectors `Bsymbol_from_Fsymbol`
/// (`sectors.jl:543-552`, v0.3.6). The reshape drops the trailing `F` axes
/// because `N(b,b̄,1) = N(a,1,a) = 1`; F is shape-validated first.
#[cfg(any(test, feature = "testing"))]
fn checked_generic_b_symbol<C>(
    rule: &C,
    a: SectorId,
    b: SectorId,
    c: SectorId,
) -> Result<GenericRMatrix<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    let rows = rule
        .try_nsymbol(a, b, c)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let dual_b = rule
        .try_dual(b)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let cols = rule
        .try_nsymbol(c, dual_b, a)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let vacuum = rule.vacuum();
    let f = checked_generic_f_symbol(rule, a, b, dual_b, a, c, vacuum)?;
    let factor = rule
        .try_sqrt_dim_scalar(a)
        .map_err(CheckedGenericSymbolError::Provider)?
        * rule
            .try_sqrt_dim_scalar(b)
            .map_err(CheckedGenericSymbolError::Provider)?
        * rule
            .try_inv_sqrt_dim_scalar(c)
            .map_err(CheckedGenericSymbolError::Provider)?;
    let mut data = Vec::with_capacity(rows * cols);
    for mu in 0..rows {
        for nu in 0..cols {
            data.push(factor.clone() * f.get(mu, nu, 0, 0).clone());
        }
    }
    Ok(GenericRMatrix::new(data, rows, cols))
}

/// `A^{ab}_c[κ,λ] = √d_a·√d_b/√d_c · conj(κ_a · F(ā,a,b,b,1,c)[0,0,κ,λ])`,
/// testing oracle: TensorKitSectors `Asymbol_from_Fsymbol`
/// (`sectors.jl:501-511`, v0.3.6). Here the leading `F` axes are singletons.
#[cfg(any(test, feature = "testing"))]
fn checked_generic_a_symbol<C>(
    rule: &C,
    a: SectorId,
    b: SectorId,
    c: SectorId,
) -> Result<GenericRMatrix<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    let rows = rule
        .try_nsymbol(a, b, c)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let dual_a = rule
        .try_dual(a)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let cols = rule
        .try_nsymbol(dual_a, c, b)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let vacuum = rule.vacuum();
    let f = checked_generic_f_symbol(rule, dual_a, a, b, b, vacuum, c)?;
    let factor = rule
        .try_sqrt_dim_scalar(a)
        .map_err(CheckedGenericSymbolError::Provider)?
        * rule
            .try_sqrt_dim_scalar(b)
            .map_err(CheckedGenericSymbolError::Provider)?
        * rule
            .try_inv_sqrt_dim_scalar(c)
            .map_err(CheckedGenericSymbolError::Provider)?;
    let fs = rule
        .try_frobenius_schur_phase_scalar(a)
        .map_err(CheckedGenericSymbolError::Provider)?;
    let mut data = Vec::with_capacity(rows * cols);
    for kappa in 0..rows {
        for lambda in 0..cols {
            data.push(factor.clone() * (fs.clone() * f.get(0, 0, kappa, lambda).clone()).conj());
        }
    }
    Ok(GenericRMatrix::new(data, rows, cols))
}
