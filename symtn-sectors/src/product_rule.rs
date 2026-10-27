use core::marker::PhantomData;
use core::ops::Mul;

use fusion_symbols::category::FusionCategory;
use fusion_symbols::product::Product;

use crate::category_rule::{forward, require_braiding_or_panic, violated, AdapterAccess};
use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CategoricalScalar, CategoryCodec, CategoryFailure,
    CategoryProvider, CheckedCanonicalUnitFusionRule, CheckedFusionAlgebra,
    CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FusionAlgebraError, FusionRule, FusionStyleKind, GenericFArray,
    GenericRMatrix, MultiplicityFreeAdmissionMode, MultiplicityFreeFusionRule,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, ProductSector,
    ProductSectorCodecError, RuleIdentity, SectorCodec, SectorId, SectorOrderKey, SectorVec,
    TensorKitProductCodec, TypedSectorAdmission,
};

mod sealed {
    pub struct Token;
}

/// Selects a product's admission mode and error from its factors' modes.
///
/// Implemented for the four pairs of [`MultiplicityFreeAdmissionMode`] and
/// [`CheckedGenericAdmissionMode`]; sealed.
pub trait ProductAdmission<LeftError, RightError> {
    /// The product's admission mode.
    type Mode;
    /// The product's error.
    type Error: std::error::Error + Send + Sync + 'static;
    #[doc(hidden)]
    fn __sealed(_: sealed::Token);
    #[doc(hidden)]
    fn left(error: LeftError) -> Self::Error;
    #[doc(hidden)]
    fn right(error: RightError) -> Self::Error;
    #[doc(hidden)]
    fn codec(error: ProductSectorCodecError) -> Self::Error;
    #[doc(hidden)]
    fn adapter<E: std::error::Error + Send + Sync + 'static>(
        error: crate::CategoryAdapterError<E>,
    ) -> Self::Error;
}

type MF = MultiplicityFreeAdmissionMode;
type CG = CheckedGenericAdmissionMode;

impl ProductAdmission<FusionAlgebraError, FusionAlgebraError> for (MF, MF) {
    type Mode = MF;
    type Error = FusionAlgebraError;
    fn __sealed(_: sealed::Token) {}
    fn left(error: FusionAlgebraError) -> FusionAlgebraError {
        error
    }
    fn right(error: FusionAlgebraError) -> FusionAlgebraError {
        error
    }
    fn codec(error: ProductSectorCodecError) -> FusionAlgebraError {
        FusionAlgebraError::ProductCodec(error)
    }
    fn adapter<E: std::error::Error + Send + Sync + 'static>(
        error: crate::CategoryAdapterError<E>,
    ) -> FusionAlgebraError {
        crate::category_rule::fusion_algebra_error(error)
    }
}

macro_rules! generic_pair {
    ($($pair:ty),+) => {$(
        impl<L, R> ProductAdmission<L, R> for $pair
        where
            L: std::error::Error + Send + Sync + 'static,
            R: std::error::Error + Send + Sync + 'static,
        {
            type Mode = CG;
            type Error = ProductFusionRuleError<L, R>;
            fn __sealed(_: sealed::Token) {}
            fn left(error: L) -> Self::Error { ProductFusionRuleError::Left(error) }
            fn right(error: R) -> Self::Error { ProductFusionRuleError::Right(error) }
            fn codec(error: ProductSectorCodecError) -> Self::Error { ProductFusionRuleError::Codec(error) }
            fn adapter<E: std::error::Error + Send + Sync + 'static>(
                error: crate::CategoryAdapterError<E>,
            ) -> Self::Error {
                use crate::CategoryAdapterError as A;
                match error {
                    A::DimensionNotRepresentable { sector } => {
                        ProductFusionRuleError::DimensionNotRepresentable { sector }
                    }
                    A::Provider(error) => ProductFusionRuleError::Category(CategoryFailure::new(error)),
                    other => ProductFusionRuleError::Category(CategoryFailure::new(other)),
                }
            }
        }
    )+};
}
generic_pair!((MF, CG), (CG, MF), (CG, CG));

/// The error of a product with a checked-Generic factor.
///
/// Holds the factors' own errors, product id packing, and representability of
/// `f64` dimensions. Product-category failures retain their original source.
#[derive(Debug)]
#[non_exhaustive]
pub enum ProductFusionRuleError<LeftError, RightError> {
    /// The left factor rejected its component.
    Left(LeftError),
    /// The right factor rejected its component.
    Right(RightError),
    /// The product id cannot be packed or unpacked.
    Codec(ProductSectorCodecError),
    /// The product dimension of `sector` is not a finite positive `f64`.
    DimensionNotRepresentable { sector: SectorId },
    /// The category or adapter failed; the original failure is retained as the source.
    Category(CategoryFailure),
}

impl<L: std::error::Error, R: std::error::Error> core::fmt::Display
    for ProductFusionRuleError<L, R>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Left(_) => f.write_str("the left factor rejected its component"),
            Self::Right(_) => f.write_str("the right factor rejected its component"),
            Self::Codec(_) => f.write_str("the product sector id cannot be packed"),
            Self::DimensionNotRepresentable { sector } => {
                write!(
                    f,
                    "the dimension of {sector:?} is not a finite positive f64"
                )
            }
            Self::Category(_) => f.write_str("the product category failed"),
        }
    }
}

impl<L, R> std::error::Error for ProductFusionRuleError<L, R>
where
    L: std::error::Error + 'static,
    R: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Left(error) => Some(error),
            Self::Right(error) => Some(error),
            Self::Codec(error) => Some(error),
            Self::DimensionNotRepresentable { .. } => None,
            Self::Category(error) => Some(error),
        }
    }
}

type ModeOf<L, R> = <(
    <L as TypedSectorAdmission>::Mode,
    <R as TypedSectorAdmission>::Mode,
) as ProductAdmission<
    <L as TypedSectorAdmission>::Error,
    <R as TypedSectorAdmission>::Error,
>>::Mode;
type ErrorOf<L, R> = <(
    <L as TypedSectorAdmission>::Mode,
    <R as TypedSectorAdmission>::Mode,
) as ProductAdmission<
    <L as TypedSectorAdmission>::Error,
    <R as TypedSectorAdmission>::Error,
>>::Error;
type CatObj<T> = <<T as CategoryProvider>::Category as FusionCategory>::Object;
type CatErr<T> = <<T as CategoryProvider>::Category as FusionCategory>::Error;
type Pair<L, R> = (
    <L as TypedSectorAdmission>::Mode,
    <R as TypedSectorAdmission>::Mode,
);
type ProductCategory<L, R> = Product<(
    <L as CategoryProvider>::Category,
    <R as CategoryProvider>::Category,
)>;

/// Private: the product codec over the factors' providers.
#[derive(Clone, Debug)]
pub(crate) struct ProductCodec<L, R, P> {
    left: L,
    right: R,
    packing: PhantomData<P>,
}

impl<L, R, P> ProductCodec<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error>,
    P: crate::ProductSectorCodec,
{
    fn decode_ids<const N: usize>(
        &self,
        ids: [SectorId; N],
    ) -> Result<([SectorId; N], [SectorId; N]), ErrorOf<L, R>> {
        let mut left = [SectorId::new(0); N];
        let mut right = [SectorId::new(0); N];
        for (k, id) in ids.into_iter().enumerate() {
            (left[k], right[k]) = P::decode_checked(id)
                .map_err(<Pair<L, R> as ProductAdmission<L::Error, R::Error>>::codec)?;
        }
        Ok((left, right))
    }
}

impl<L, R, P> CategoryCodec<ProductCategory<L, R>> for ProductCodec<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
    type Sector = ProductSector<L::Sector, R::Sector>;
    type Error = ErrorOf<L, R>;
    type Mode = ModeOf<L, R>;
    type Unit = crate::CanonicalUnitAdmission;
    fn encode_label<'a>(
        &self,
        _category: impl Fn() -> &'a ProductCategory<L, R> + Copy,
        _rule: &crate::RuleIdentity,
        label: &Self::Sector,
    ) -> Result<SectorId, Self::Error> {
        type A<L, R> = Pair<L, R>;
        let l = self
            .left
            .try_encode_label(label.left())
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::left)?;
        let r = self
            .right
            .try_encode_label(label.right())
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::right)?;
        P::encode_checked(l, r).map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::codec)
    }
    fn decode_label<'a>(
        &self,
        _category: impl Fn() -> &'a ProductCategory<L, R> + Copy,
        id: SectorId,
    ) -> Result<Self::Sector, Self::Error> {
        type A<L, R> = Pair<L, R>;
        let (l, r) = P::decode_checked(id)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::codec)?;
        let left = self
            .left
            .try_decode_label(l)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::left)?;
        let right = self
            .right
            .try_decode_label(r)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::right)?;
        Ok(ProductSector::new(left, right))
    }
    fn object<'a>(
        &self,
        _category: impl Fn() -> &'a ProductCategory<L, R> + Copy,
        id: SectorId,
    ) -> Result<<ProductCategory<L, R> as FusionCategory>::Object, Self::Error> {
        type A<L, R> = Pair<L, R>;
        let (l, r) = P::decode_checked(id)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::codec)?;
        let left = self
            .left
            .object(l)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::left)?;
        let right = self
            .right
            .object(r)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::right)?;
        Ok((left, right))
    }
    fn sector_id<'a>(
        &self,
        _category: impl Fn() -> &'a ProductCategory<L, R> + Copy,
        _rule: &crate::RuleIdentity,
        object: &<ProductCategory<L, R> as FusionCategory>::Object,
    ) -> Result<SectorId, Self::Error> {
        type A<L, R> = Pair<L, R>;
        let l = self
            .left
            .sector_id(&object.0)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::left)?;
        let r = self
            .right
            .sector_id(&object.1)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::right)?;
        P::encode_checked(l, r).map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::codec)
    }
    fn admit<'a, const N: usize>(
        &self,
        _category: impl Fn() -> &'a ProductCategory<L, R> + Copy,
        _rule: &crate::RuleIdentity,
        operation: crate::CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[<ProductCategory<L, R> as FusionCategory>::Object; N], Self::Error> {
        type A<L, R> = Pair<L, R>;
        let (left_ids, right_ids) = self.decode_ids(ids)?;
        let left = self
            .left
            .admit(operation, left_ids)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::left)?;
        let right = self
            .right
            .admit(operation, right_ids)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::right)?;
        let mut right = right.into_iter();
        Ok(left.map(|l| (l, right.next().expect("same length"))))
    }
    fn admit_r<'a>(
        &self,
        _category: impl Fn() -> &'a ProductCategory<L, R> + Copy,
        _rule: &crate::RuleIdentity,
        ids: [SectorId; 3],
    ) -> Result<Option<[<ProductCategory<L, R> as FusionCategory>::Object; 3]>, Self::Error> {
        type A<L, R> = Pair<L, R>;
        let mut left_ids = [SectorId::new(0); 3];
        let mut right_ids = [SectorId::new(0); 3];
        for (k, id) in ids.into_iter().enumerate() {
            let (l, r) = P::decode_checked(id)
                .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::codec)?;
            left_ids[k] = l;
            right_ids[k] = r;
        }
        let left = self
            .left
            .admit_r(left_ids)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::left)?;
        let right = self
            .right
            .admit_r(right_ids)
            .map_err(<A<L, R> as ProductAdmission<L::Error, R::Error>>::right)?;
        let (Some(left), Some(right)) = (left, right) else {
            return Ok(None);
        };
        let mut right = right.into_iter();
        Ok(Some(left.map(|l| (l, right.next().expect("same length")))))
    }
    fn sector_order_key(&self, id: SectorId) -> crate::SectorOrderKey {
        let (l, r) =
            P::decode_checked(id).expect("product fusion rule received an invalid product sector");
        crate::SectorOrderKey::product(
            self.left.sector_order_key(l),
            self.right.sector_order_key(r),
        )
    }
    fn map_error(
        &self,
        inputs: &[SectorId],
        error: crate::CategoryAdapterError<<ProductCategory<L, R> as FusionCategory>::Error>,
    ) -> Self::Error {
        use crate::CategoryAdapterError as E;
        use fusion_symbols::product::ProductError as PE;
        type A<L, R> = Pair<L, R>;
        match error {
            E::Provider(PE::Factor0(e)) => <A<L, R> as ProductAdmission<L::Error, R::Error>>::left(
                self.left
                    .map_error(&components::<P>(inputs, 0), E::Provider(e)),
            ),
            E::Provider(PE::Factor1(e)) => {
                <A<L, R> as ProductAdmission<L::Error, R::Error>>::right(
                    self.right
                        .map_error(&components::<P>(inputs, 1), E::Provider(e)),
                )
            }
            other => <A<L, R> as ProductAdmission<L::Error, R::Error>>::adapter(other),
        }
    }
}

/// The `factor` components of product ids that unpack; the error context only.
fn components<P: crate::ProductSectorCodec>(inputs: &[SectorId], factor: usize) -> Vec<SectorId> {
    inputs
        .iter()
        .filter_map(|&id| P::decode_checked(id).ok())
        .map(|(l, r)| if factor == 0 { l } else { r })
        .collect()
}

/// Deligne product of two category-backed providers, composed by
/// [`fusion_symbols::product::Product`].
///
/// Two multiplicity-free factors admit a multiplicity-free product; any
/// checked-Generic factor selects checked-Generic admission. Products can nest.
/// Fusion channels vary the first factor fastest. F/R/A/B multiplicity axes
/// vary the last factor fastest in row-major storage. Scalar promotion,
/// fermionic signs, and category errors follow `fusion_symbols::product`.
/// Tensor operations order sectors by the factors' TensorKit order keys,
/// independently of the product id codec and channel enumeration.
///
/// The default [`TensorKitProductCodec`] packs component ids. A different
/// [`crate::ProductSectorCodec`] changes the rule identity, not its mathematics.
/// Factor order and nesting remain explicit in the Rust type and rule identity.
#[derive(Clone)]
pub struct ProductFusionRule<L, R, P = TensorKitProductCodec>
where
    L: CategoryProvider,
    R: CategoryProvider,
{
    category: std::sync::Arc<ProductCategory<L, R>>,
    codec: ProductCodec<L, R, P>,
    identity: crate::RuleIdentity,
    vacuum: SectorId,
}

impl<L, R, P> ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error>,
    P: crate::ProductSectorCodec + 'static,
{
    /// The left factor provider.
    pub const fn left_rule(&self) -> &L {
        &self.codec.left
    }

    /// The right factor provider.
    pub const fn right_rule(&self) -> &R {
        &self.codec.right
    }

    /// Packs component ids, panicking when the codec cannot represent them.
    pub fn encode_component_ids(&self, left: SectorId, right: SectorId) -> SectorId {
        P::encode(left, right)
    }

    /// Packs component ids with the product admission error.
    pub fn try_encode_component_ids(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorId, ErrorOf<L, R>> {
        P::encode_checked(left, right)
            .map_err(<Pair<L, R> as ProductAdmission<L::Error, R::Error>>::codec)
    }

    /// Unpacks component ids, or returns `None` for an invalid product id.
    pub fn decode_component_ids(&self, sector: SectorId) -> Option<(SectorId, SectorId)> {
        P::decode(sector)
    }
}

impl<L, R, P> ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
    /// The Deligne product of `left` and `right`.
    ///
    /// Fails with `MissingCanonicalUnit` when a factor's category does not
    /// certify a canonical unit, and with `IncompatibleFusionStyle` or
    /// `UnsupportedBraidingStyle` when the product reports a style symtn does
    /// not know. Products of multiplicity-free factors are multiplicity-free,
    /// so admitted factors already fix the mode's multiplicity check.
    pub fn try_new(
        left: L,
        right: R,
    ) -> Result<Self, crate::CategoryAdmissionError<ErrorOf<L, R>>> {
        let category = std::sync::Arc::new(Product::new((
            left.category().clone(),
            right.category().clone(),
        )));
        crate::category_rule::admit_styles(
            category.fusion_style(),
            category.braiding_style(),
            false,
        )?;
        if !category.has_canonical_unit() {
            return Err(crate::CategoryAdmissionError::MissingCanonicalUnit);
        }
        let identity = crate::RuleIdentity::compose_with_codec::<P>(
            left.typed_rule_identity(),
            right.typed_rule_identity(),
        );
        let codec = ProductCodec {
            left,
            right,
            packing: PhantomData,
        };
        let vacuum = codec
            .sector_id(|| &*category, &identity, &category.unit())
            .map_err(crate::CategoryAdmissionError::Codec)?;
        Ok(Self {
            category,
            codec,
            identity,
            vacuum,
        })
    }
}

impl<L, R, P> crate::seal::Provider for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
}

// Private adapter access; braiding and associator gauge come from fs Product.
impl<L, R, P> AdapterAccess for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
    type Category = ProductCategory<L, R>;
    type Codec = ProductCodec<L, R, P>;
    fn category(&self) -> &Self::Category {
        &self.category
    }
    fn codec(&self) -> &Self::Codec {
        &self.codec
    }
    fn identity(&self) -> &crate::RuleIdentity {
        &self.identity
    }
    fn vacuum_id(&self) -> SectorId {
        self.vacuum
    }

    fn admit_vertex_move(
        &self,
        operation: crate::CategoryOperation,
        ids: [SectorId; 3],
    ) -> Result<
        crate::category_rule::AdmittedObjects<<Self::Category as FusionCategory>::Object, 3>,
        ErrorOf<L, R>,
    > {
        let (left_ids, right_ids) = self.codec.decode_ids(ids)?;
        let (left, left_dual) = crate::category_rule::VertexMoveAdmission::admit_vertex_move(
            &self.codec.left,
            operation,
            left_ids,
        )
        .map_err(<Pair<L, R> as ProductAdmission<L::Error, R::Error>>::left)?;
        let (right, right_dual) = crate::category_rule::VertexMoveAdmission::admit_vertex_move(
            &self.codec.right,
            operation,
            right_ids,
        )
        .map_err(<Pair<L, R> as ProductAdmission<L::Error, R::Error>>::right)?;
        let mut right = right.into_iter();
        Ok((
            left.map(|l| (l, right.next().expect("same length"))),
            left_dual.zip(right_dual),
        ))
    }
}

impl<L, R, P> core::fmt::Debug for ProductFusionRule<L, R, P>
where
    L: CategoryProvider + core::fmt::Debug,
    R: CategoryProvider + core::fmt::Debug,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ProductFusionRule")
            .field("left", &self.codec.left)
            .field("right", &self.codec.right)
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

impl<L, R, P> TypedSectorAdmission for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
    type Sector = ProductSector<L::Sector, R::Sector>;
    type Error = ErrorOf<L, R>;
    type Mode = ModeOf<L, R>;

    fn typed_rule_identity(&self) -> RuleIdentity {
        self.identity().clone()
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        self.codec()
            .encode_label(|| &*self.category, self.identity(), sector)
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        self.codec().decode_label(|| &*self.category, sector)
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        forward::dual(self, sector)
    }
}

impl<L, R, P> FusionRule for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error, Mode = MF, Error = FusionAlgebraError>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
    fn rule_identity(&self) -> RuleIdentity {
        self.identity().clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        forward::multiplicity_free_style(self)
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        forward::braiding_style_kind(AdapterAccess::category(self).braiding_style())
    }

    fn vacuum(&self) -> SectorId {
        self.vacuum_id()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        self.dual_or_panic(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.fusion_channels_or_panic(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.nsymbol_or_panic(left, right, coupled)
    }

    fn sector_order_key(&self, sector: SectorId) -> SectorOrderKey {
        self.codec().sector_order_key(sector)
    }
}

impl<L, R, P> CheckedFusionAlgebra for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error, Mode = MF, Error = FusionAlgebraError>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        forward::dual(self, sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        forward::fusion_channels(self, left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        forward::nsymbol(self, left, right, coupled)
    }
}

impl<L, R, P> SectorCodec for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error, Mode = MF, Error = FusionAlgebraError>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
    type Sector = ProductSector<L::Sector, R::Sector>;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        self.try_encode_label(value)
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        self.try_decode_label(id)
    }
}

impl<L, R, P> MultiplicityFreeFusionRule for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error, Mode = MF, Error = FusionAlgebraError>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
}

impl<L, R, P> MultiplicityFreeFusionSymbols for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error, Mode = MF, Error = FusionAlgebraError>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
    <ProductCategory<L, R> as FusionCategory>::Scalar: CategoricalScalar + Send + Sync,
{
    type Scalar = <ProductCategory<L, R> as FusionCategory>::Scalar;

    fn has_trivial_associator_gauge(&self) -> bool {
        AdapterAccess::category(self).has_trivial_associator_gauge()
    }

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        violated(
            "MultiplicityFreeFusionSymbols::f_symbol_scalar",
            forward::f_scalar(
                self,
                [left, middle, right, coupled, left_coupled, right_coupled],
            ),
        )
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        require_braiding_or_panic(self, "r_symbol");
        violated(
            "MultiplicityFreeFusionSymbols::r_symbol_scalar",
            forward::r_scalar(self, [left, right, coupled]),
        )
    }
}

impl<L, R, P> MultiplicityFreeRigidSymbols for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error, Mode = MF, Error = FusionAlgebraError>,
    ProductCategory<L, R>: FusionCategory<
            Object = (CatObj<L>, CatObj<R>),
            Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
        > + Send
        + Sync,
    P: crate::ProductSectorCodec + 'static + Sync,
    <ProductCategory<L, R> as FusionCategory>::Scalar: CategoricalScalar + Send + Sync,
    L: Sync,
    R: Sync,
{
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        violated(
            "MultiplicityFreeRigidSymbols::dim_scalar",
            forward::dimension(self, sector, FusionCategory::dim),
        )
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        violated(
            "MultiplicityFreeRigidSymbols::inv_dim_scalar",
            forward::dimension(self, sector, FusionCategory::inv_dim),
        )
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        violated(
            "MultiplicityFreeRigidSymbols::sqrt_dim_scalar",
            forward::dimension(self, sector, FusionCategory::sqrt_dim),
        )
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        violated(
            "MultiplicityFreeRigidSymbols::inv_sqrt_dim_scalar",
            forward::dimension(self, sector, FusionCategory::inv_sqrt_dim),
        )
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        require_braiding_or_panic(self, "twist");
        violated(
            "MultiplicityFreeRigidSymbols::twist_scalar",
            forward::twist(self, sector),
        )
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        violated(
            "MultiplicityFreeRigidSymbols::frobenius_schur_phase_scalar",
            forward::frobenius_schur_phase(self, sector),
        )
    }

    fn a_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar
    where
        Self::Scalar: Mul<Output = Self::Scalar>,
    {
        violated(
            "MultiplicityFreeRigidSymbols::a_symbol_scalar",
            forward::a_scalar(self, [left, right, coupled]),
        )
    }

    fn b_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar
    where
        Self::Scalar: Mul<Output = Self::Scalar>,
    {
        violated(
            "MultiplicityFreeRigidSymbols::b_symbol_scalar",
            forward::b_scalar(self, [left, right, coupled]),
        )
    }
}

impl<L, R, P> CanonicalUnitFusionRule for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error, Mode = MF, Error = FusionAlgebraError>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
}

impl<L, R, P> CheckedGenericFusion for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error, Mode = CG>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
    type Error = ErrorOf<L, R>;

    fn rule_identity(&self) -> RuleIdentity {
        self.identity().clone()
    }

    /// Always `Generic`: checked Generic admission executes every category on
    /// the multiplicity path, whatever its own fusion style.
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        forward::braiding_style_kind(AdapterAccess::category(self).braiding_style())
    }

    fn vacuum(&self) -> SectorId {
        self.vacuum_id()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        forward::dual(self, sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        forward::fusion_channels(self, left, right)
    }

    /// The complete channels: a category is unbounded, and an output its
    /// codec cannot represent is an error rather than a filtered channel.
    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        forward::fusion_channels(self, left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        forward::nsymbol(self, left, right, coupled)
    }

    fn sector_order_key(&self, sector: SectorId) -> SectorOrderKey {
        self.codec().sector_order_key(sector)
    }
}

impl<L, R, P> CheckedGenericRigidSymbols for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error, Mode = CG>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
    <ProductCategory<L, R> as FusionCategory>::Scalar: CategoricalScalar + Send + Sync,
{
    type Scalar = <ProductCategory<L, R> as FusionCategory>::Scalar;

    fn try_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        forward::dimension(self, sector, FusionCategory::dim)
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        forward::dimension(self, sector, FusionCategory::sqrt_dim)
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        forward::dimension(self, sector, FusionCategory::inv_sqrt_dim)
    }

    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        forward::frobenius_schur_phase(self, sector)
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
        forward::f_block(self, [a, b, c, d, e, f])
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        forward::r_block(self, [a, b, c])
    }

    fn try_a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        forward::a_block(self, [a, b, c])
    }

    fn try_b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        forward::b_block(self, [a, b, c])
    }
}

impl<L, R, P> CheckedGenericPivotal for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error, Mode = CG>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
    <ProductCategory<L, R> as FusionCategory>::Scalar: CategoricalScalar + Send + Sync,
{
    fn try_twist_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        forward::twist(self, sector)
    }
}

impl<L, R, P> CheckedCanonicalUnitFusionRule for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error, Mode = CG>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
}

// Nesting: a product is itself category-backed.
impl<L, R, P> CategoryProvider for ProductFusionRule<L, R, P>
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error>,
    ProductCategory<L, R>: FusionCategory<
        Object = (CatObj<L>, CatObj<R>),
        Error = fusion_symbols::product::ProductError<CatErr<L>, CatErr<R>>,
    >,
    P: crate::ProductSectorCodec + 'static,
{
    type Category = ProductCategory<L, R>;
    fn category(&self) -> &Self::Category {
        &self.category
    }
    fn object(
        &self,
        id: SectorId,
    ) -> Result<<Self::Category as FusionCategory>::Object, Self::Error> {
        self.codec.object(|| &*self.category, id)
    }
    fn sector_id(
        &self,
        object: &<Self::Category as FusionCategory>::Object,
    ) -> Result<SectorId, Self::Error> {
        self.codec
            .sector_id(|| &*self.category, &self.identity, object)
    }
    fn admit<const N: usize>(
        &self,
        operation: crate::CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[<Self::Category as FusionCategory>::Object; N], Self::Error> {
        self.codec
            .admit(|| &*self.category, &self.identity, operation, ids)
    }
    fn admit_r(
        &self,
        ids: [SectorId; 3],
    ) -> Result<Option<[<Self::Category as FusionCategory>::Object; 3]>, Self::Error> {
        self.codec.admit_r(|| &*self.category, &self.identity, ids)
    }
    fn map_error(
        &self,
        inputs: &[SectorId],
        error: crate::CategoryAdapterError<<Self::Category as FusionCategory>::Error>,
    ) -> Self::Error {
        self.codec.map_error(inputs, error)
    }
    fn sector_order_key(&self, id: SectorId) -> crate::SectorOrderKey {
        self.codec.sector_order_key(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    use crate::{
        BraidingStyleKind, CanonicalUnitFusionRule, CategoricalScalar, FermionParityFusionRule,
        FusionStyleKind, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, SectorId,
        TableRule, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
    };

    #[test]
    fn label_codec_resolves_by_method_syntax_and_component_ids_round_trip() {
        // What: `encode_sector`/`decode_sector` on a product rule are the label-level
        // `SectorCodec` methods; component ids use the `*_component_ids` pair.
        use crate::{product_sector, SectorCodec};

        let rule =
            ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
        let label = product_sector(Z2Irrep::ODD, U1Irrep::new(-3));
        let id = rule.encode_sector(&label).unwrap();
        assert_eq!(rule.decode_sector(id).unwrap(), label);
        let by_ids = rule.encode_component_ids(Z2Irrep::ODD.into(), U1Irrep::new(-3).into());
        assert_eq!(by_ids, id);
        assert_eq!(
            rule.decode_component_ids(id),
            Some((Z2Irrep::ODD.into(), U1Irrep::new(-3).into()))
        );
        assert_eq!(
            rule.try_encode_component_ids(Z2Irrep::ODD.into(), U1Irrep::new(-3).into()),
            Ok(id)
        );
    }

    #[test]
    fn product_of_canonical_unit_rules_has_canonical_unit() {
        // What: product construction preserves the unit law componentwise.
        fn accepts_canonical_unit<R: CanonicalUnitFusionRule>(_rule: &R) {}

        let rule =
            ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
        let vacuum = rule.vacuum();
        let sector = rule.encode_component_ids(Z2Irrep::ODD.into(), U1Irrep::new(-17).into());
        let fused = rule.fusion_channels(sector, sector)[0];
        accepts_canonical_unit(&rule);
        assert_eq!(rule.dual(vacuum), vacuum);
        assert_eq!(rule.fusion_channels(vacuum, sector).as_slice(), &[sector]);
        assert_eq!(rule.fusion_channels(sector, vacuum).as_slice(), &[sector]);
        assert_eq!(rule.nsymbol(vacuum, sector, sector), 1);
        assert_eq!(rule.nsymbol(sector, vacuum, sector), 1);
        assert_eq!(rule.nsymbol(vacuum, sector, vacuum), 0);
        assert_eq!(
            rule.f_symbol_scalar(vacuum, sector, sector, fused, sector, fused),
            1.0
        );
        assert_eq!(
            rule.f_symbol_scalar(sector, vacuum, sector, fused, sector, sector),
            1.0
        );
        assert_eq!(
            rule.f_symbol_scalar(sector, sector, vacuum, fused, fused, sector),
            1.0
        );
    }

    #[test]
    fn product_rule_composes_checked_symbols_rigidity_and_ordered_identity() {
        let rule =
            ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
        let odd_zero = rule.encode_component_ids(Z2Irrep::ODD.into(), U1Irrep::new(0).into());
        let vacuum = rule.vacuum();

        assert_eq!(
            rule.try_fusion_channels(odd_zero, odd_zero),
            Ok(std::iter::once(vacuum).collect())
        );
        assert_eq!(rule.r_symbol_scalar(odd_zero, odd_zero, vacuum), -1.0);
        assert_eq!(
            rule.f_symbol_scalar(vacuum, vacuum, vacuum, vacuum, vacuum, vacuum),
            1.0
        );
        assert_eq!(rule.dim_scalar(odd_zero), 1.0);
        let first = rule.rule_identity();
        assert_eq!(first, rule.rule_identity());
        assert_ne!(
            rule.rule_identity(),
            ProductFusionRule::<_, _>::try_new(U1FusionRule, FermionParityFusionRule)
                .unwrap()
                .rule_identity()
        );
    }

    // Oracle for the two tests below: TensorKitSectors 0.3.6 through
    // TensorKit 0.17, queried for `FibonacciAnyon ⊠ Z2Irrep` with
    // `t1 = FibonacciAnyon(:τ) ⊠ Z2Irrep(1)`. Recorded verbatim:
    //
    //   FusionStyle           = SimpleFusion()
    //   BraidingStyle         = Anyonic()
    //   sectorscalartype      = ComplexF64
    //   t1 ⊗ t1               = [(:I, 0), (:τ, 0)]
    //   dim(t1)               = 1.618033988749895
    //   twist(t1)             = -0.8090169943749473 - 0.5877852522924734im
    //   frobeniusschur(t1)    = 1.0
    //   R(t1,t1,(:I,0))       = -0.8090169943749476 + 0.587785252292473im
    //   R(t1,t1,(:τ,0))       = -0.30901699437494734 - 0.9510565162951536im
    //   F(t1,t1,t1,t1,e,f)    = 0.6180339887498948   (e,f) = ((:I,0),(:I,0))
    //                           0.7861513777574233   (e,f) = ((:I,0),(:τ,0))
    //                           0.7861513777574233   (e,f) = ((:τ,0),(:I,0))
    //                          -0.6180339887498948   (e,f) = ((:τ,0),(:τ,0))
    const TK_TOLERANCE: f64 = 1e-12;

    fn assert_close(actual: Complex64, expected: Complex64) {
        assert!(
            (actual - expected).norm() <= TK_TOLERANCE,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn product_promotes_a_complex_component_scalar() {
        // What: a multiplicity-free product whose components disagree on the
        // coefficient scalar is representable, and its topological data is the
        // componentwise product promoted to the wider scalar.
        let rule =
            ProductFusionRule::<_, _>::try_new(TableRule::fibonacci(), Z2FusionRule).unwrap();
        let tau = SectorId::new(1);
        let vacuum_fib = SectorId::new(0);
        let t1 = rule.encode_component_ids(tau, Z2Irrep::ODD.into());
        let i0 = rule.encode_component_ids(vacuum_fib, Z2Irrep::EVEN.into());
        let tau0 = rule.encode_component_ids(tau, Z2Irrep::EVEN.into());

        assert_eq!(rule.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Anyonic);
        assert_eq!(rule.vacuum(), i0);

        let mut channels = rule.fusion_channels(t1, t1).to_vec();
        channels.sort_unstable();
        assert_eq!(channels, vec![i0, tau0]);

        assert_close(
            rule.dim_scalar(t1),
            Complex64::new(std::f64::consts::GOLDEN_RATIO, 0.0),
        );
        assert_close(
            rule.twist_scalar(t1),
            Complex64::new(-0.809_016_994_374_947_3, -0.587_785_252_292_473_4),
        );
        assert_close(
            rule.frobenius_schur_phase_scalar(t1),
            Complex64::new(1.0, 0.0),
        );
        assert_close(
            rule.r_symbol_scalar(t1, t1, i0),
            Complex64::new(-0.809_016_994_374_947_6, 0.587_785_252_292_473),
        );
        assert_close(
            rule.r_symbol_scalar(t1, t1, tau0),
            Complex64::new(-0.309_016_994_374_947_34, -0.951_056_516_295_153_6),
        );
        for (left_coupled, right_coupled, expected) in [
            (i0, i0, 0.618_033_988_749_894_8),
            (i0, tau0, 0.786_151_377_757_423_3),
            (tau0, i0, 0.786_151_377_757_423_3),
            (tau0, tau0, -0.618_033_988_749_894_8),
        ] {
            assert_close(
                rule.f_symbol_scalar(t1, t1, t1, t1, left_coupled, right_coupled),
                Complex64::new(expected, 0.0),
            );
        }

        // The promoted unit and conjugation are the wider scalar's, not the
        // left component's.
        assert_eq!(Complex64::one(), Complex64::new(1.0, 0.0));
        assert_close(
            (rule.r_symbol_scalar(t1, t1, i0)).conj(),
            Complex64::new(-0.809_016_994_374_947_6, -0.587_785_252_292_473),
        );
    }

    #[test]
    fn product_promotion_is_component_order_independent() {
        // What: promotion does not depend on which side carries the wider
        // scalar; TensorKitSectors promotes rather than taking the left type.
        let forward =
            ProductFusionRule::<_, _>::try_new(TableRule::fibonacci(), Z2FusionRule).unwrap();
        let reversed =
            ProductFusionRule::<_, _>::try_new(Z2FusionRule, TableRule::fibonacci()).unwrap();
        let tau = SectorId::new(1);
        let vacuum_fib = SectorId::new(0);

        let forward_t1 = forward.encode_component_ids(tau, Z2Irrep::ODD.into());
        let forward_i0 = forward.encode_component_ids(vacuum_fib, Z2Irrep::EVEN.into());
        let reversed_t1 = reversed.encode_component_ids(Z2Irrep::ODD.into(), tau);
        let reversed_i0 = reversed.encode_component_ids(Z2Irrep::EVEN.into(), vacuum_fib);

        assert_close(
            reversed.r_symbol_scalar(reversed_t1, reversed_t1, reversed_i0),
            forward.r_symbol_scalar(forward_t1, forward_t1, forward_i0),
        );
        assert_close(
            reversed.dim_scalar(reversed_t1),
            forward.dim_scalar(forward_t1),
        );
    }
}
