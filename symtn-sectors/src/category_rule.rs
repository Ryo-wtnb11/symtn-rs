//! The fusion rule of a `fusion_symbols` category.
//!
//! [`CategoryRule`] is the production extension point: the category owns the
//! mathematics (fusion, N, F, R, A, B, dimensions, twists), and a
//! [`CategoryCodec`] owns what is symtn's alone — public labels, the
//! [`SectorId`] numbering, [`RuleIdentity`], sector order, operation-specific
//! representability, error translation, and the admission mode. The codec
//! contains no symbol formula; it only decides which ids name which objects
//! and whether an operation's inputs are admitted.

use core::fmt;
use core::marker::PhantomData;
use core::ops::Mul;
use std::sync::Arc;

use fusion_symbols::category::{
    BraidingStyle, EnumerationError, FusionCategory, FusionStyle, SymbolBlock,
};

use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CategoricalScalar, CheckedCanonicalUnitFusionRule,
    CheckedFusionAlgebra, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FusionAlgebraError, FusionRule, FusionStyleKind, GenericFArray,
    GenericRMatrix, MultiplicityFreeAdmissionMode, MultiplicityFreeFusionRule,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, RuleIdentity, SectorCodec,
    SectorId, SectorOrderKey, SectorVec, SymbolShapeError, TypedSectorAdmission,
};

/// The categorical operation whose inputs [`CategoryCodec::admit`] admits.
///
/// Arities are `Dual` and `Dimension` 1, `Fusion` 2, `N`, `R`, `A`, `B` 3,
/// and `F` 6, in TensorKitSectors argument order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CategoryOperation {
    Dual,
    Fusion,
    N,
    F,
    R,
    A,
    B,
    Dimension,
}

pub(crate) type AdmittedObjects<O, const N: usize> = ([O; N], Option<O>);

/// Requests constructor-validated canonical-unit admission: [`CategoryRule::try_new`]
/// requires the category's `has_canonical_unit` certificate, and the rule
/// then implements [`CanonicalUnitFusionRule`] or
/// [`CheckedCanonicalUnitFusionRule`].
pub struct CanonicalUnitAdmission;

/// Admits a category without the canonical-unit markers, for a category that
/// does not certify its unit gauge.
pub struct UncertifiedUnitAdmission;

mod private {
    /// Admission modes a codec may select, with the error each requires.
    pub trait AdmissionMode<E> {
        const MULTIPLICITY_FREE: bool;
    }

    /// Unit-admission selectors.
    pub trait UnitAdmission {
        const CANONICAL: bool;
    }
}

impl private::AdmissionMode<FusionAlgebraError> for MultiplicityFreeAdmissionMode {
    const MULTIPLICITY_FREE: bool = true;
}

impl<E: std::error::Error + Send + Sync + 'static> private::AdmissionMode<E>
    for CheckedGenericAdmissionMode
{
    const MULTIPLICITY_FREE: bool = false;
}

impl private::UnitAdmission for CanonicalUnitAdmission {
    const CANONICAL: bool = true;
}

impl private::UnitAdmission for UncertifiedUnitAdmission {
    const CANONICAL: bool = false;
}

/// Why [`CategoryRule::try_new`] refused a category.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CategoryAdmissionError<E> {
    /// The category's fusion style is not admitted: `Generic` under
    /// multiplicity-free admission, or a style symtn does not know.
    IncompatibleFusionStyle { actual: FusionStyle },
    /// The category reports a braiding style symtn does not know.
    UnsupportedBraidingStyle { actual: BraidingStyle },
    /// Canonical-unit admission was requested, but the category does not
    /// certify a canonical unit.
    MissingCanonicalUnit,
    /// The codec rejected the unit or its round trip.
    Codec(E),
}

impl<E: fmt::Display> fmt::Display for CategoryAdmissionError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleFusionStyle {
                actual: FusionStyle::Generic,
            } => formatter.write_str(
                "multiplicity-free admission requires a multiplicity-free category, got Generic fusion",
            ),
            Self::IncompatibleFusionStyle { actual } => {
                write!(formatter, "unsupported {actual:?} fusion style")
            }
            Self::UnsupportedBraidingStyle { actual } => {
                write!(formatter, "unsupported {actual:?} braiding style")
            }
            Self::MissingCanonicalUnit => {
                formatter.write_str("canonical-unit admission requires a canonical-unit category")
            }
            Self::Codec(_) => formatter.write_str("the category codec rejected the unit"),
        }
    }
}

/// Admission's style gate. Every style a rule later maps to a symtn kind
/// passes here first, so those maps never meet a style fusion-symbols adds
/// after this release.
pub(crate) fn admit_styles<E>(
    fusion: FusionStyle,
    braiding: BraidingStyle,
    multiplicity_free: bool,
) -> Result<(), CategoryAdmissionError<E>> {
    match fusion {
        FusionStyle::Unique | FusionStyle::Simple => {}
        FusionStyle::Generic if !multiplicity_free => {}
        actual => return Err(CategoryAdmissionError::IncompatibleFusionStyle { actual }),
    }
    match braiding {
        BraidingStyle::None
        | BraidingStyle::Bosonic
        | BraidingStyle::Fermionic
        | BraidingStyle::Anyonic => Ok(()),
        actual => Err(CategoryAdmissionError::UnsupportedBraidingStyle { actual }),
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CategoryAdmissionError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Codec(error) => Some(error),
            _ => None,
        }
    }
}

/// A failure between the category authority and a codec, translated by
/// [`CategoryCodec::map_error`] into the codec's public error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CategoryAdapterError<E> {
    /// The category returned an error.
    Provider(E),
    /// A codec or the category produced an object outside this category
    /// value; `sector` is the id it was decoded from, if any.
    ForeignObject { sector: Option<SectorId> },
    /// The dimension is not a finite positive `f64`.
    DimensionNotRepresentable { sector: SectorId },
    /// Returned symbol storage disagrees with its fusion multiplicities.
    MalformedSymbolData(SymbolShapeError),
    /// Returned symbol axes disagree with the fusion multiplicities, even
    /// when the storage length matches.
    MalformedSymbolAxes {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// The operation requires a braiding the category does not have.
    NoBraiding,
}

impl<E: fmt::Display> fmt::Display for CategoryAdapterError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(_) => formatter.write_str("the fusion category failed"),
            Self::ForeignObject {
                sector: Some(sector),
            } => {
                write!(formatter, "{sector:?} names an object outside the category")
            }
            Self::ForeignObject { sector: None } => {
                formatter.write_str("the category produced an object outside itself")
            }
            Self::DimensionNotRepresentable { sector } => write!(
                formatter,
                "the dimension of {sector:?} is not a finite positive f64"
            ),
            Self::MalformedSymbolData(_) => {
                formatter.write_str("symbol storage disagrees with its fusion multiplicities")
            }
            Self::MalformedSymbolAxes { expected, actual } => write!(
                formatter,
                "symbol axes {actual:?} disagree with the fusion multiplicities {expected:?}"
            ),
            Self::NoBraiding => formatter.write_str("the category has no braiding"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CategoryAdapterError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::MalformedSymbolData(error) => Some(error),
            _ => None,
        }
    }
}

/// An opaque, cloneable category failure that keeps its concrete source.
///
/// Equality is identity of the retained failure: clones compare equal, and
/// two independently created failures are not promised equal, because an
/// arbitrary source error has no equality.
#[derive(Clone)]
pub struct CategoryFailure(Arc<dyn std::error::Error + Send + Sync>);

impl CategoryFailure {
    /// Retains `source` without converting it to text.
    pub fn new<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self(Arc::new(source))
    }
}

impl fmt::Debug for CategoryFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CategoryFailure")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for CategoryFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl std::error::Error for CategoryFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.0)
    }
}

impl PartialEq for CategoryFailure {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for CategoryFailure {}

/// Labels, numbering, identity, admission, and errors of a [`CategoryRule`].
///
/// The category passed to every method is a thunk so that a codec can check
/// an id before an expensive or lazily built category is touched, and can
/// answer from its own state without building a public label.
///
/// Laws: `object` and `sector_id` are mutually inverse on the ids and member
/// objects they accept, as are `decode_label` and `encode_label`; distinct
/// objects never share an id; and every id the category generates from
/// admitted inputs (unit, duals, fusion outcomes) is either encoded or
/// reported as an error, never altered. [`Self::rule_identity`] must cover
/// every parameter of the category value and of the codec that can change an
/// answer.
///
/// [`EnumeratedCodec`] is the codec for every enumerable category; implement
/// this trait only for labels or numbering of your own.
pub trait CategoryCodec<C>: 'static
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
{
    /// The public sector label.
    type Sector: Clone + Eq + Ord + core::hash::Hash + fmt::Debug;
    /// The error of typed admission and of checked Generic queries.
    type Error: std::error::Error + Send + Sync + 'static;
    /// [`MultiplicityFreeAdmissionMode`] or [`CheckedGenericAdmissionMode`].
    type Mode;
    /// [`CanonicalUnitAdmission`] or [`UncertifiedUnitAdmission`].
    type Unit;

    /// Encodes a public label.
    fn encode_label<'a>(
        &self,
        category: impl Fn() -> &'a C + Copy,
        rule: &RuleIdentity,
        label: &Self::Sector,
    ) -> Result<SectorId, Self::Error>;

    /// Decodes an id to its public label.
    fn decode_label<'a>(
        &self,
        category: impl Fn() -> &'a C + Copy,
        id: SectorId,
    ) -> Result<Self::Sector, Self::Error>;

    /// The category object an id names.
    fn object<'a>(
        &self,
        category: impl Fn() -> &'a C + Copy,
        id: SectorId,
    ) -> Result<C::Object, Self::Error>;

    /// The id of a category object.
    fn sector_id<'a>(
        &self,
        category: impl Fn() -> &'a C + Copy,
        rule: &RuleIdentity,
        object: &C::Object,
    ) -> Result<SectorId, Self::Error>;

    /// A complete identity of this codec over `category`, or `None` for a
    /// fresh identity per constructed rule (shared by its clones).
    fn rule_identity(&self, category: &C) -> Option<RuleIdentity> {
        let _ = category;
        None
    }

    /// The id's position in TensorKit's sector order; the default is the id.
    fn sector_order_key(&self, id: SectorId) -> SectorOrderKey {
        SectorOrderKey::position(id.id() as u64)
    }

    /// Admits an operation's inputs and returns their objects.
    ///
    /// The default converts the ids in argument order, rejecting an object
    /// outside the category before any other category call, and for `A` and
    /// `B` also requires the id of `dual(a)` or `dual(b)`. Override it to
    /// keep a provider's own validation order or closure checks; it must not
    /// compute symbols.
    fn admit<'a, const N: usize>(
        &self,
        category: impl Fn() -> &'a C + Copy,
        rule: &RuleIdentity,
        operation: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[C::Object; N], Self::Error> {
        admit_in_order(self, category, rule, operation, ids)
    }

    /// Admits R's inputs; `None` admits an R block known to be zero without
    /// converting every input.
    fn admit_r<'a>(
        &self,
        category: impl Fn() -> &'a C + Copy,
        rule: &RuleIdentity,
        ids: [SectorId; 3],
    ) -> Result<Option<[C::Object; 3]>, Self::Error> {
        self.admit(category, rule, CategoryOperation::R, ids)
            .map(Some)
    }

    /// Encodes the complete fusion outcomes of `inputs`, in the order the
    /// codec promises; the default keeps the category's order.
    fn encode_channels(
        &self,
        category: &C,
        rule: &RuleIdentity,
        inputs: [SectorId; 2],
        outcomes: impl Iterator<Item = C::Object>,
    ) -> Result<SectorVec, Self::Error> {
        outcomes
            .map(|outcome| {
                if !category.contains(&outcome) {
                    return Err(self.map_error(
                        &inputs,
                        CategoryAdapterError::ForeignObject { sector: None },
                    ));
                }
                self.sector_id(|| category, rule, &outcome)
            })
            .collect()
    }

    /// Translates a category or adapter failure on `inputs` into this
    /// codec's error.
    fn map_error(&self, inputs: &[SectorId], error: CategoryAdapterError<C::Error>) -> Self::Error;
}

/// [`CategoryCodec::admit`]'s default: the ids in argument order, then for
/// `A` and `B` the id of the dualized input; a codec that checks one input
/// earlier keeps this for the rest.
pub(crate) fn admit_in_order<'a, C, K, const N: usize>(
    codec: &K,
    category: impl Fn() -> &'a C + Copy,
    rule: &RuleIdentity,
    operation: CategoryOperation,
    ids: [SectorId; N],
) -> Result<[C::Object; N], K::Error>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C> + ?Sized,
{
    admit_in_order_with_dual(codec, category, rule, operation, ids).map(|(objects, _)| objects)
}

pub(crate) fn admit_in_order_with_dual<'a, C, K, const N: usize>(
    codec: &K,
    category: impl Fn() -> &'a C + Copy,
    rule: &RuleIdentity,
    operation: CategoryOperation,
    ids: [SectorId; N],
) -> Result<AdmittedObjects<C::Object, N>, K::Error>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C> + ?Sized,
{
    let objects = try_map_array(ids, |id| member_object(codec, category, id))?;
    let dualized = match operation {
        CategoryOperation::A => objects.first(),
        CategoryOperation::B => objects.get(1),
        _ => None,
    };
    let dual = dualized.map(|object| category().dual(object));
    if let Some(dual) = &dual {
        codec.sector_id(category, rule, dual)?;
    }
    Ok((objects, dual))
}

/// `N`'s inputs decoded once each, the coupled (last) sector first and then
/// the others in argument order, returned in argument order.
pub(crate) fn admit_coupled_first<'a, C, K, const N: usize>(
    codec: &K,
    category: impl Fn() -> &'a C + Copy,
    ids: [SectorId; N],
) -> Result<[C::Object; N], K::Error>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C> + ?Sized,
{
    let mut objects: [Option<C::Object>; N] = core::array::from_fn(|_| None);
    for step in 0..N {
        let position = (step + N - 1) % N;
        objects[position] = Some(member_object(codec, category, ids[position])?);
    }
    // The loop visits every position once.
    Ok(objects.map(|object| object.expect("every input was decoded")))
}

/// The object `id` names, rejected if it is not a member of the category.
pub(crate) fn member_object<'a, C, K>(
    codec: &K,
    category: impl Fn() -> &'a C + Copy,
    id: SectorId,
) -> Result<C::Object, K::Error>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C> + ?Sized,
{
    let object = codec.object(category, id)?;
    if category().contains(&object) {
        Ok(object)
    } else {
        Err(codec.map_error(
            &[id],
            CategoryAdapterError::ForeignObject { sector: Some(id) },
        ))
    }
}

/// `items.map(f)`, stopping at the first error.
pub(crate) fn try_map_array<T, U, E, const N: usize>(
    items: [T; N],
    mut f: impl FnMut(T) -> Result<U, E>,
) -> Result<[U; N], E> {
    let mut failure = None;
    let mapped = items.map(|item| match failure {
        Some(_) => None,
        None => f(item).map_err(|error| failure = Some(error)).ok(),
    });
    match failure {
        Some(error) => Err(error),
        // Every slot is `Some` when no call failed.
        None => Ok(mapped.map(|slot| slot.expect("every element was mapped"))),
    }
}

/// The codec of an enumerable category: labels are the category's objects
/// and an id is the object's enumeration index.
///
/// `M` selects [`MultiplicityFreeAdmissionMode`] or
/// [`CheckedGenericAdmissionMode`]; `U` selects canonical-unit admission. The
/// error is [`FusionAlgebraError`] in both modes:
///
/// | failure | error |
/// | --- | --- |
/// | an id names no object, or its label overflows | `InvalidSector` |
/// | a label is foreign or its index does not fit an id | `UnrepresentableSectorLabel` |
/// | a fusion outcome does not fit an id | `FusionNotRepresentable` |
/// | the category is not enumerable, or the category failed | `CategoryFailure` |
///
/// Fusion channels keep the category's order. The sector order is the
/// enumeration order. Equal category values give equal rule identities, so
/// two independently constructed rules over equal categories share caches
/// while both are alive; the identity refers to the category weakly and
/// never keeps it alive after the last rule over it is dropped.
pub struct EnumeratedCodec<M, U = CanonicalUnitAdmission>(PhantomData<fn() -> (M, U)>);

#[allow(
    clippy::new_without_default,
    reason = "the approved surface of EnumeratedCodec is `new` alone"
)]
impl<M, U> EnumeratedCodec<M, U> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<M, U> Clone for EnumeratedCodec<M, U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M, U> Copy for EnumeratedCodec<M, U> {}

impl<M, U> fmt::Debug for EnumeratedCodec<M, U> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("EnumeratedCodec")
    }
}

fn enumeration_failure(error: EnumerationError) -> FusionAlgebraError {
    FusionAlgebraError::CategoryFailure(CategoryFailure::new(error))
}

/// The enumeration id of a member object; `unrepresentable` reports an index
/// past the id range.
fn enumeration_id<C: FusionCategory>(
    category: &C,
    object: &C::Object,
    unrepresentable: impl FnOnce() -> FusionAlgebraError,
) -> Result<SectorId, FusionAlgebraError> {
    match category.object_index(object) {
        Ok(index) => usize::try_from(index)
            .map(SectorId::new)
            .map_err(|_| unrepresentable()),
        Err(EnumerationError::Overflow) => Err(unrepresentable()),
        Err(error) => Err(enumeration_failure(error)),
    }
}

impl<C, M, U> CategoryCodec<C> for EnumeratedCodec<M, U>
where
    C: FusionCategory + Send + Sync + 'static,
    C::Error: Send + Sync + 'static,
    M: private::AdmissionMode<FusionAlgebraError> + 'static,
    U: private::UnitAdmission + 'static,
{
    type Sector = C::Object;
    type Error = FusionAlgebraError;
    type Mode = M;
    type Unit = U;

    fn encode_label<'a>(
        &self,
        category: impl Fn() -> &'a C + Copy,
        rule: &RuleIdentity,
        label: &Self::Sector,
    ) -> Result<SectorId, Self::Error> {
        <Self as CategoryCodec<C>>::sector_id(self, category, rule, label)
    }

    fn decode_label<'a>(
        &self,
        category: impl Fn() -> &'a C + Copy,
        id: SectorId,
    ) -> Result<Self::Sector, Self::Error> {
        <Self as CategoryCodec<C>>::object(self, category, id)
    }

    fn object<'a>(
        &self,
        category: impl Fn() -> &'a C + Copy,
        id: SectorId,
    ) -> Result<C::Object, Self::Error> {
        let invalid = FusionAlgebraError::InvalidSector { sector: id };
        let index = u64::try_from(id.id()).map_err(|_| invalid.clone())?;
        category().object_at(index).map_err(|error| match error {
            EnumerationError::OutOfRange | EnumerationError::Overflow => invalid,
            _ => enumeration_failure(error),
        })
    }

    fn sector_id<'a>(
        &self,
        category: impl Fn() -> &'a C + Copy,
        rule: &RuleIdentity,
        object: &C::Object,
    ) -> Result<SectorId, Self::Error> {
        let unrepresentable = || FusionAlgebraError::UnrepresentableSectorLabel {
            rule: rule.clone(),
            label: format!("{object:?}"),
        };
        let category = category();
        if !category.contains(object) {
            return Err(unrepresentable());
        }
        enumeration_id(category, object, unrepresentable)
    }

    /// The value identity of the rule's category. [`CategoryRule::try_new`]
    /// anchors it to the category the rule owns; observed anywhere else, it
    /// is unique to this call and its clones.
    fn rule_identity(&self, category: &C) -> Option<RuleIdentity> {
        let _ = category;
        Some(RuleIdentity::value_of::<(C, Self), C>())
    }

    fn encode_channels(
        &self,
        category: &C,
        _rule: &RuleIdentity,
        inputs: [SectorId; 2],
        outcomes: impl Iterator<Item = C::Object>,
    ) -> Result<SectorVec, Self::Error> {
        let [left, right] = inputs;
        outcomes
            .map(|outcome| {
                if !category.contains(&outcome) {
                    return Err(<Self as CategoryCodec<C>>::map_error(
                        self,
                        &inputs,
                        CategoryAdapterError::ForeignObject { sector: None },
                    ));
                }
                enumeration_id(category, &outcome, || {
                    FusionAlgebraError::FusionNotRepresentable { left, right }
                })
            })
            .collect()
    }

    fn map_error(
        &self,
        _inputs: &[SectorId],
        error: CategoryAdapterError<C::Error>,
    ) -> Self::Error {
        fusion_algebra_error(error)
    }
}

/// The multiplicity-free translation of an adapter failure: a foreign id is
/// `InvalidSector`, and everything else a `CategoryFailure` keeping its source.
pub(crate) fn fusion_algebra_error<E>(error: CategoryAdapterError<E>) -> FusionAlgebraError
where
    E: std::error::Error + Send + Sync + 'static,
{
    match error {
        CategoryAdapterError::Provider(error) => {
            FusionAlgebraError::CategoryFailure(CategoryFailure::new(error))
        }
        CategoryAdapterError::ForeignObject {
            sector: Some(sector),
        } => FusionAlgebraError::InvalidSector { sector },
        other => FusionAlgebraError::CategoryFailure(CategoryFailure::new(other)),
    }
}

/// The fusion rule of a `fusion_symbols` category under a [`CategoryCodec`].
///
/// ```
/// use fusion_symbols::finite::ZN;
/// use symtn_sectors::{CategoryRule, EnumeratedCodec, MultiplicityFreeAdmissionMode};
///
/// let z3 = CategoryRule::try_new(
///     ZN::new(3).unwrap(),
///     EnumeratedCodec::<MultiplicityFreeAdmissionMode>::new(),
/// )
/// .unwrap();
/// ```
///
/// The codec's `Mode` selects which provider traits the rule implements:
/// multiplicity-free admission gives [`FusionRule`], [`CheckedFusionAlgebra`],
/// [`SectorCodec`], and the multiplicity-free symbol traits; checked Generic
/// admission gives [`CheckedGenericFusion`], [`CheckedGenericRigidSymbols`],
/// and [`CheckedGenericPivotal`] and reports `Generic` fusion for every
/// category, including multiplicity-free ones.
///
/// [`MultiplicityFreeAdmissionMode`]: crate::MultiplicityFreeAdmissionMode
/// [`CheckedGenericAdmissionMode`]: crate::CheckedGenericAdmissionMode
///
/// An admission mode is chosen by the codec type, so an invalid combination
/// does not compile. A checked Generic rule is not a multiplicity-free rule:
///
/// ```compile_fail
/// use fusion_symbols::finite::ZN;
/// use symtn_sectors::{
///     CategoryRule, CheckedGenericAdmissionMode, EnumeratedCodec, FusionRule,
/// };
///
/// fn multiplicity_free<R: FusionRule>(_: &R) {}
/// let rule = CategoryRule::try_new(
///     ZN::new(3).unwrap(),
///     EnumeratedCodec::<CheckedGenericAdmissionMode>::new(),
/// )
/// .unwrap();
/// multiplicity_free(&rule);
/// ```
///
/// A mode outside the two admission modes is not admitted,
///
/// ```compile_fail
/// use fusion_symbols::finite::ZN;
/// use symtn_sectors::{CategoryRule, EnumeratedCodec};
///
/// struct OtherMode;
/// let _ = CategoryRule::try_new(ZN::new(3).unwrap(), EnumeratedCodec::<OtherMode>::new());
/// ```
///
/// nor is a multiplicity-free codec with another error:
///
/// ```compile_fail
/// use fusion_symbols::finite::ZN;
/// use symtn_sectors::{
///     CanonicalUnitAdmission, CategoryCodec, CategoryRule, MultiplicityFreeAdmissionMode,
/// };
///
/// fn admit<K>(codec: K)
/// where
///     K: CategoryCodec<
///         ZN,
///         Mode = MultiplicityFreeAdmissionMode,
///         Unit = CanonicalUnitAdmission,
///         Error = std::fmt::Error,
///     >,
/// {
///     let _ = CategoryRule::try_new(ZN::new(3).unwrap(), codec);
/// }
/// ```
///
/// Canonical-unit markers require canonical-unit admission:
///
/// ```compile_fail
/// use fusion_symbols::finite::ZN;
/// use symtn_sectors::{
///     CanonicalUnitFusionRule, CategoryRule, EnumeratedCodec, MultiplicityFreeAdmissionMode,
///     UncertifiedUnitAdmission,
/// };
///
/// fn canonical<R: CanonicalUnitFusionRule>(_: &R) {}
/// let rule = CategoryRule::try_new(
///     ZN::new(3).unwrap(),
///     EnumeratedCodec::<MultiplicityFreeAdmissionMode, UncertifiedUnitAdmission>::new(),
/// )
/// .unwrap();
/// canonical(&rule);
/// ```
///
/// The provider traits are sealed; a production type cannot implement them
/// natively:
///
/// ```compile_fail
/// use symtn_sectors::{
///     BraidingStyleKind, FusionRule, FusionStyleKind, RuleIdentity, SectorId, SectorVec,
/// };
///
/// struct Native;
/// impl FusionRule for Native {
///     fn rule_identity(&self) -> RuleIdentity {
///         RuleIdentity::of_type::<Self>()
///     }
///     fn fusion_style(&self) -> FusionStyleKind {
///         FusionStyleKind::Unique
///     }
///     fn braiding_style(&self) -> BraidingStyleKind {
///         BraidingStyleKind::Bosonic
///     }
///     fn vacuum(&self) -> SectorId {
///         SectorId::new(0)
///     }
///     fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
///         SectorVec::new()
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct CategoryRule<C, K> {
    // Why an `Arc`: a value identity refers to the category weakly, so that
    // a cache key never keeps it alive after the last clone of the rule.
    category: Arc<C>,
    codec: K,
    identity: RuleIdentity,
    vacuum: SectorId,
}

impl<C, K> CategoryRule<C, K> {
    /// The mathematical category.
    pub fn category(&self) -> &C {
        &self.category
    }

    /// The label, numbering, and admission policy.
    pub const fn codec(&self) -> &K {
        &self.codec
    }
}

impl<C, K> CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C>,
    K::Mode: private::AdmissionMode<K::Error>,
    K::Unit: private::UnitAdmission,
{
    /// Admits `category` under `codec`.
    ///
    /// # Errors
    ///
    /// Multiplicity-free admission of a category whose fusion has
    /// multiplicity, a fusion or braiding style symtn does not know,
    /// canonical-unit admission of a category without the
    /// canonical-unit certificate, and a unit the codec cannot encode or
    /// round-trip through its id and label.
    pub fn try_new(category: C, codec: K) -> Result<Self, CategoryAdmissionError<K::Error>> {
        admit_styles(
            category.fusion_style(),
            category.braiding_style(),
            <K::Mode as private::AdmissionMode<K::Error>>::MULTIPLICITY_FREE,
        )?;
        if <K::Unit as private::UnitAdmission>::CANONICAL && !category.has_canonical_unit() {
            return Err(CategoryAdmissionError::MissingCanonicalUnit);
        }
        let identity = codec.rule_identity(&category);
        let category = Arc::new(category);
        let identity = match identity {
            Some(identity) => identity.resolve_value(&category),
            None => RuleIdentity::new_unique::<Self>(),
        };
        let unit = category.unit();
        let category_ref = || &*category;
        let vacuum = codec
            .sector_id(category_ref, &identity, &unit)
            .map_err(CategoryAdmissionError::Codec)?;
        let not_round_trip = || {
            CategoryAdmissionError::Codec(codec.map_error(
                &[vacuum],
                CategoryAdapterError::ForeignObject {
                    sector: Some(vacuum),
                },
            ))
        };
        if member_object(&codec, category_ref, vacuum).map_err(CategoryAdmissionError::Codec)?
            != unit
        {
            return Err(not_round_trip());
        }
        let label = codec
            .decode_label(category_ref, vacuum)
            .map_err(CategoryAdmissionError::Codec)?;
        if codec
            .encode_label(category_ref, &identity, &label)
            .map_err(CategoryAdmissionError::Codec)?
            != vacuum
        {
            return Err(not_round_trip());
        }
        Ok(Self {
            category,
            codec,
            identity,
            vacuum,
        })
    }
}

/// Borrowed ingredients of a category-backed rule, so that one set of
/// provider implementations serves [`CategoryRule`] and the built-in rules
/// that keep their own public types (their identities, constructors, and
/// `Copy`/`const` forms).
pub(crate) trait AdapterAccess: crate::seal::Provider + 'static {
    type Category: FusionCategory<Error: Send + Sync + 'static> + 'static;
    type Codec: CategoryCodec<Self::Category>;

    fn category(&self) -> &Self::Category;
    fn codec(&self) -> &Self::Codec;
    fn identity(&self) -> &RuleIdentity;
    /// The id the codec gives the category's unit.
    fn vacuum_id(&self) -> SectorId;

    #[allow(clippy::type_complexity)]
    fn admit_vertex_move(
        &self,
        operation: CategoryOperation,
        ids: [SectorId; 3],
    ) -> Result<
        AdmittedObjects<<Self::Category as FusionCategory>::Object, 3>,
        <Self::Codec as CategoryCodec<Self::Category>>::Error,
    > {
        let objects = self
            .codec()
            .admit(|| self.category(), self.identity(), operation, ids)?;
        Ok((objects, None))
    }
}

impl<C, K> AdapterAccess for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C>,
{
    type Category = C;
    type Codec = K;

    fn category(&self) -> &C {
        &self.category
    }

    fn codec(&self) -> &K {
        &self.codec
    }

    fn identity(&self) -> &RuleIdentity {
        &self.identity
    }

    fn vacuum_id(&self) -> SectorId {
        self.vacuum
    }
}

impl<C, K> crate::seal::Provider for CategoryRule<C, K> {}

pub(crate) trait VertexMoveAdmission<C: FusionCategory, E> {
    fn admit_vertex_move(
        &self,
        operation: CategoryOperation,
        ids: [SectorId; 3],
    ) -> Result<AdmittedObjects<C::Object, 3>, E>;
}

impl<A: AdapterAccess>
    VertexMoveAdmission<A::Category, <A::Codec as CategoryCodec<A::Category>>::Error> for A
{
    fn admit_vertex_move(
        &self,
        operation: CategoryOperation,
        ids: [SectorId; 3],
    ) -> Result<
        AdmittedObjects<<A::Category as FusionCategory>::Object, 3>,
        <A::Codec as CategoryCodec<A::Category>>::Error,
    > {
        AdapterAccess::admit_vertex_move(self, operation, ids)
    }
}

/// A provider backed by a `fusion_symbols` category: every built-in rule,
/// every [`CategoryRule`], and every product of them.
///
/// Sealed. A product reaches its factors' categories and object conversion
/// through this trait; it never sees their codecs.
#[allow(private_bounds)]
pub trait CategoryProvider:
    TypedSectorAdmission + VertexMoveAdmission<Self::Category, Self::Error>
{
    /// The category this provider answers from.
    type Category: FusionCategory<Error: Send + Sync + 'static> + 'static;

    /// The category value.
    fn category(&self) -> &Self::Category;
    /// The category object an id names.
    fn object(
        &self,
        id: SectorId,
    ) -> Result<<Self::Category as FusionCategory>::Object, Self::Error>;
    /// The id of a category object.
    fn sector_id(
        &self,
        object: &<Self::Category as FusionCategory>::Object,
    ) -> Result<SectorId, Self::Error>;
    /// Admits an operation's inputs with this provider's own checks and
    /// precedence (e.g. U(1) fusion overflow), returning their objects.
    fn admit<const N: usize>(
        &self,
        operation: crate::CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[<Self::Category as FusionCategory>::Object; N], Self::Error>;
    /// Admits R's inputs; `None` is an R block known to be zero.
    #[allow(clippy::type_complexity)]
    fn admit_r(
        &self,
        ids: [SectorId; 3],
    ) -> Result<Option<[<Self::Category as FusionCategory>::Object; 3]>, Self::Error>;
    /// Translates a category or adapter failure into this provider's error.
    fn map_error(
        &self,
        inputs: &[SectorId],
        error: crate::CategoryAdapterError<<Self::Category as FusionCategory>::Error>,
    ) -> Self::Error;
    /// The id's position in TensorKit's sector order.
    fn sector_order_key(&self, id: SectorId) -> crate::SectorOrderKey;
}

// Concrete categories keep private adapter types out of the public boundary.
macro_rules! category_provider_via_adapter {
    ($($rule:ty => $cat:ty),+) => {$(
        impl CategoryProvider for $rule {
            type Category = $cat;
            fn category(&self) -> &Self::Category { AdapterAccess::category(self) }
            fn object(&self, id: SectorId)
                -> Result<<Self::Category as FusionCategory>::Object, Self::Error> {
                crate::category_rule::member_object(AdapterAccess::codec(self), || AdapterAccess::category(self), id)
            }
            fn sector_id(&self, object: &<Self::Category as FusionCategory>::Object)
                -> Result<SectorId, Self::Error> {
                AdapterAccess::codec(self).sector_id(|| AdapterAccess::category(self), AdapterAccess::identity(self), object)
            }
            fn admit<const N: usize>(&self, operation: crate::CategoryOperation, ids: [SectorId; N])
                -> Result<[<Self::Category as FusionCategory>::Object; N], Self::Error> {
                AdapterAccess::codec(self).admit(|| AdapterAccess::category(self), AdapterAccess::identity(self), operation, ids)
            }
            fn admit_r(&self, ids: [SectorId; 3])
                -> Result<Option<[<Self::Category as FusionCategory>::Object; 3]>, Self::Error> {
                AdapterAccess::codec(self).admit_r(|| AdapterAccess::category(self), AdapterAccess::identity(self), ids)
            }
            fn map_error(&self, inputs: &[SectorId],
                error: crate::CategoryAdapterError<<Self::Category as FusionCategory>::Error>) -> Self::Error {
                AdapterAccess::codec(self).map_error(inputs, error)
            }
            fn sector_order_key(&self, id: SectorId) -> crate::SectorOrderKey {
                AdapterAccess::codec(self).sector_order_key(id)
            }
        }
    )+};
}
category_provider_via_adapter!(
    crate::U1FusionRule => fusion_symbols::lie::U1,
    crate::Z2FusionRule => fusion_symbols::finite::ZN,
    crate::ZNFusionRule => fusion_symbols::finite::ZN,
    crate::FermionParityFusionRule => fusion_symbols::fermion::FermionParity,
    crate::CU1FusionRule => fusion_symbols::lie::CU1,
    crate::SU2FusionRule => fusion_symbols::lie::LieGroup,
    crate::TableRule => fusion_symbols::table::TableCategory
);

#[cfg(feature = "cgc-gen")]
category_provider_via_adapter!(crate::LieGroupFusionRule => fusion_symbols::lie::LieGroup);

impl<C, K> CategoryProvider for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C>,
{
    type Category = C;
    fn category(&self) -> &C {
        AdapterAccess::category(self)
    }
    fn object(&self, id: SectorId) -> Result<C::Object, K::Error> {
        crate::category_rule::member_object(
            AdapterAccess::codec(self),
            || AdapterAccess::category(self),
            id,
        )
    }
    fn sector_id(&self, object: &C::Object) -> Result<SectorId, K::Error> {
        AdapterAccess::codec(self).sector_id(
            || AdapterAccess::category(self),
            AdapterAccess::identity(self),
            object,
        )
    }
    fn admit<const N: usize>(
        &self,
        operation: crate::CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[C::Object; N], K::Error> {
        AdapterAccess::codec(self).admit(
            || AdapterAccess::category(self),
            AdapterAccess::identity(self),
            operation,
            ids,
        )
    }
    fn admit_r(&self, ids: [SectorId; 3]) -> Result<Option<[C::Object; 3]>, K::Error> {
        AdapterAccess::codec(self).admit_r(
            || AdapterAccess::category(self),
            AdapterAccess::identity(self),
            ids,
        )
    }
    fn map_error(
        &self,
        inputs: &[SectorId],
        error: crate::CategoryAdapterError<C::Error>,
    ) -> K::Error {
        AdapterAccess::codec(self).map_error(inputs, error)
    }
    fn sector_order_key(&self, id: SectorId) -> crate::SectorOrderKey {
        AdapterAccess::codec(self).sector_order_key(id)
    }
}

pub(crate) mod forward {
    //! The category operations of an [`AdapterAccess`] rule, each admitted by
    //! its codec before the category is asked.

    use super::*;
    use fusion_symbols::category::Scalar;

    type Category<A> = <A as AdapterAccess>::Category;
    type Object<A> = <Category<A> as FusionCategory>::Object;
    type Value<A> = <Category<A> as FusionCategory>::Scalar;
    pub(crate) type Error<A> = <<A as AdapterAccess>::Codec as CategoryCodec<Category<A>>>::Error;

    fn admit<A: AdapterAccess, const N: usize>(
        rule: &A,
        operation: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[Object<A>; N], Error<A>> {
        rule.codec()
            .admit(|| rule.category(), rule.identity(), operation, ids)
    }

    fn adapter<A: AdapterAccess>(
        rule: &A,
        inputs: &[SectorId],
        error: CategoryAdapterError<<Category<A> as FusionCategory>::Error>,
    ) -> Error<A> {
        rule.codec().map_error(inputs, error)
    }

    fn provider<'r, A: AdapterAccess, const N: usize>(
        rule: &'r A,
        inputs: [SectorId; N],
    ) -> impl FnOnce(<Category<A> as FusionCategory>::Error) -> Error<A> + 'r {
        move |error| adapter(rule, &inputs, CategoryAdapterError::Provider(error))
    }

    fn require_braiding<A: AdapterAccess>(rule: &A, inputs: &[SectorId]) -> Result<(), Error<A>> {
        if rule.category().braiding_style() == BraidingStyle::None {
            Err(adapter(rule, inputs, CategoryAdapterError::NoBraiding))
        } else {
            Ok(())
        }
    }

    /// The style of a multiplicity-free admitted category.
    pub(crate) fn multiplicity_free_style<A: AdapterAccess>(rule: &A) -> FusionStyleKind {
        match rule.category().fusion_style() {
            FusionStyle::Unique => FusionStyleKind::Unique,
            FusionStyle::Simple => FusionStyleKind::Simple,
            // `admit_styles` rejects Generic and unknown styles under
            // multiplicity-free admission; built-in rules wrap categories
            // whose styles are fixed by construction.
            style => unreachable!("multiplicity-free admission excludes {style:?} fusion"),
        }
    }

    pub(crate) fn braiding_style_kind(style: BraidingStyle) -> BraidingStyleKind {
        match style {
            BraidingStyle::None => BraidingStyleKind::NoBraiding,
            BraidingStyle::Bosonic => BraidingStyleKind::Bosonic,
            BraidingStyle::Fermionic => BraidingStyleKind::Fermionic,
            BraidingStyle::Anyonic => BraidingStyleKind::Anyonic,
            // `admit_styles` rejects unknown styles in CategoryRule and
            // ProductFusionRule admission; built-in rules wrap categories
            // whose styles are fixed by construction.
            style => unreachable!("admission excludes the {style:?} braiding style"),
        }
    }

    pub(crate) fn dual<A: AdapterAccess>(rule: &A, sector: SectorId) -> Result<SectorId, Error<A>> {
        let [object] = admit(rule, CategoryOperation::Dual, [sector])?;
        let dual = rule.category().dual(&object);
        rule.codec()
            .sector_id(|| rule.category(), rule.identity(), &dual)
    }

    pub(crate) fn fusion_channels<A: AdapterAccess>(
        rule: &A,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Error<A>> {
        let [a, b] = admit(rule, CategoryOperation::Fusion, [left, right])?;
        let outcomes = rule
            .category()
            .fusion(&a, &b)
            .map_err(provider(rule, [left, right]))?;
        rule.codec()
            .encode_channels(rule.category(), rule.identity(), [left, right], outcomes)
    }

    pub(crate) fn nsymbol<A: AdapterAccess>(
        rule: &A,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Error<A>> {
        let ids = [left, right, coupled];
        let [a, b, c] = admit(rule, CategoryOperation::N, ids)?;
        multiplicity(rule, ids, &a, &b, &c)
    }

    fn multiplicity<A: AdapterAccess, const N: usize>(
        rule: &A,
        inputs: [SectorId; N],
        a: &Object<A>,
        b: &Object<A>,
        c: &Object<A>,
    ) -> Result<usize, Error<A>> {
        let n = rule
            .category()
            .try_nsymbol(a, b, c)
            .map_err(provider(rule, inputs))?;
        // `u32` always fits the `usize` of a supported target.
        Ok(n as usize)
    }

    /// A dimension-like value of `sector`, rejected unless finite and positive.
    pub(crate) fn dimension<A: AdapterAccess>(
        rule: &A,
        sector: SectorId,
        value: impl FnOnce(&Category<A>, &Object<A>) -> f64,
    ) -> Result<Value<A>, Error<A>> {
        let [object] = admit(rule, CategoryOperation::Dimension, [sector])?;
        let value = value(rule.category(), &object);
        if value.is_finite() && value > 0.0 {
            Ok(Value::<A>::from_f64(value))
        } else {
            Err(adapter(
                rule,
                &[sector],
                CategoryAdapterError::DimensionNotRepresentable { sector },
            ))
        }
    }

    fn member<A: AdapterAccess>(rule: &A, sector: SectorId) -> Result<Object<A>, Error<A>> {
        member_object(rule.codec(), || rule.category(), sector)
    }

    pub(crate) fn frobenius_schur_phase<A: AdapterAccess>(
        rule: &A,
        sector: SectorId,
    ) -> Result<Value<A>, Error<A>> {
        let object = member(rule, sector)?;
        Ok(rule.category().frobenius_schur_phase(&object))
    }

    pub(crate) fn twist<A: AdapterAccess>(
        rule: &A,
        sector: SectorId,
    ) -> Result<Value<A>, Error<A>> {
        let object = member(rule, sector)?;
        require_braiding(rule, &[sector])?;
        Ok(rule.category().twist(&object))
    }

    /// The one logical entry of a multiplicity-free block, without allocating.
    fn single<A: AdapterAccess, const RANK: usize>(
        rule: &A,
        inputs: &[SectorId],
        block: &impl SymbolBlock<RANK, Scalar = Value<A>>,
        axes: [usize; RANK],
    ) -> Result<Value<A>, Error<A>> {
        match shaped(rule, inputs, block, axes)? {
            [] => Ok(Value::<A>::zero()),
            [value] => Ok(*value),
            data => Err(adapter(
                rule,
                inputs,
                CategoryAdapterError::MalformedSymbolData(SymbolShapeError {
                    actual_len: data.len(),
                    expected_len: Some(1),
                }),
            )),
        }
    }

    /// Validates borrowed fs storage against logical N axes. Only an empty
    /// slice or a bare scalar zero can represent a logically absent block.
    pub(crate) fn shaped<'b, A: AdapterAccess, const RANK: usize>(
        rule: &A,
        inputs: &[SectorId],
        block: &'b impl SymbolBlock<RANK, Scalar = Value<A>>,
        axes: [usize; RANK],
    ) -> Result<&'b [Value<A>], Error<A>> {
        let data = block.as_slice();
        let expected = axes.iter().try_fold(1usize, |n, &axis| n.checked_mul(axis));
        if expected == Some(0) && (data.is_empty() || matches!(data, [value] if value.abs() == 0.0))
        {
            return Ok(&data[..0]);
        }
        if expected != Some(data.len()) {
            return Err(adapter(
                rule,
                inputs,
                CategoryAdapterError::MalformedSymbolData(SymbolShapeError {
                    actual_len: data.len(),
                    expected_len: expected,
                }),
            ));
        }
        let actual = block.dims();
        if axes != actual {
            return Err(adapter(
                rule,
                inputs,
                CategoryAdapterError::MalformedSymbolAxes {
                    expected: axes.to_vec(),
                    actual: actual.to_vec(),
                },
            ));
        }
        Ok(data)
    }

    fn f_axes<A: AdapterAccess>(
        rule: &A,
        ids: [SectorId; 6],
        [a, b, c, d, e, f]: &[Object<A>; 6],
    ) -> Result<[usize; 4], Error<A>> {
        Ok([
            multiplicity(rule, ids, a, b, e)?,
            multiplicity(rule, ids, e, c, d)?,
            multiplicity(rule, ids, b, c, f)?,
            multiplicity(rule, ids, a, f, d)?,
        ])
    }

    pub(crate) fn f_scalar<A: AdapterAccess>(
        rule: &A,
        ids: [SectorId; 6],
    ) -> Result<Value<A>, Error<A>> {
        let objects = admit(rule, CategoryOperation::F, ids)?;
        let axes = f_axes(rule, ids, &objects)?;
        let [a, b, c, d, e, f] = objects;
        let block = rule
            .category()
            .f_symbol(&a, &b, &c, &d, &e, &f)
            .map_err(provider(rule, ids))?;
        single(rule, &ids, &block, axes)
    }

    pub(crate) fn r_scalar<A: AdapterAccess>(
        rule: &A,
        ids: [SectorId; 3],
    ) -> Result<Value<A>, Error<A>> {
        let Some([a, b, c]) = rule
            .codec()
            .admit_r(|| rule.category(), rule.identity(), ids)?
        else {
            return Ok(Value::<A>::zero());
        };
        require_braiding(rule, &ids)?;
        let axes = [
            multiplicity(rule, ids, &a, &b, &c)?,
            multiplicity(rule, ids, &b, &a, &c)?,
        ];
        let block = rule
            .category()
            .r_symbol(&a, &b, &c)
            .map_err(provider(rule, ids))?;
        single(rule, &ids, &block, axes)
    }

    pub(crate) fn f_block<A: AdapterAccess>(
        rule: &A,
        ids: [SectorId; 6],
    ) -> Result<GenericFArray<Value<A>>, Error<A>> {
        let objects = admit(rule, CategoryOperation::F, ids)?;
        let axes = f_axes(rule, ids, &objects)?;
        let [a, b, c, d, e, f] = objects;
        let block = rule
            .category()
            .f_symbol(&a, &b, &c, &d, &e, &f)
            .map_err(provider(rule, ids))?;
        let data = shaped(rule, &ids, &block, axes)?.to_vec();
        // `shaped` established the length.
        Ok(GenericFArray::new(
            data,
            (axes[0], axes[1], axes[2], axes[3]),
        ))
    }

    pub(crate) fn r_block<A: AdapterAccess>(
        rule: &A,
        ids: [SectorId; 3],
    ) -> Result<GenericRMatrix<Value<A>>, Error<A>> {
        let Some([a, b, c]) = rule
            .codec()
            .admit_r(|| rule.category(), rule.identity(), ids)?
        else {
            return Ok(GenericRMatrix::new(Vec::new(), 0, 0));
        };
        require_braiding(rule, &ids)?;
        let rows = multiplicity(rule, ids, &a, &b, &c)?;
        let cols = multiplicity(rule, ids, &b, &a, &c)?;
        let block = rule
            .category()
            .r_symbol(&a, &b, &c)
            .map_err(provider(rule, ids))?;
        let data = shaped(rule, &ids, &block, [rows, cols])?.to_vec();
        // `shaped` established the length.
        Ok(GenericRMatrix::new(data, rows, cols))
    }

    /// Admits A or B inputs and requires finite positive dimensions, which
    /// the `√d_a √d_b / √d_c` normalization of both symbols divides by.
    fn admit_vertex_move<A: AdapterAccess>(
        rule: &A,
        operation: CategoryOperation,
        ids: [SectorId; 3],
    ) -> Result<AdmittedObjects<Object<A>, 3>, Error<A>> {
        let (objects, dual) = AdapterAccess::admit_vertex_move(rule, operation, ids)?;
        for (object, sector) in objects.iter().zip(ids) {
            let dim = rule.category().dim(object);
            if !(dim.is_finite() && dim > 0.0) {
                return Err(adapter(
                    rule,
                    &ids,
                    CategoryAdapterError::DimensionNotRepresentable { sector },
                ));
            }
        }
        Ok((objects, dual))
    }

    /// An A or B block, shaped `[N(a,b,c), N(column vertex)]` from the
    /// checked multiplicities rather than from the returned storage.
    fn vertex_move_block<A: AdapterAccess>(
        rule: &A,
        ids: [SectorId; 3],
        objects: &[Object<A>; 3],
        column: [&Object<A>; 3],
        block: fusion_symbols::category::RBlock<Value<A>>,
    ) -> Result<GenericRMatrix<Value<A>>, Error<A>> {
        let [a, b, c] = objects;
        let rows = multiplicity(rule, ids, a, b, c)?;
        let cols = multiplicity(rule, ids, column[0], column[1], column[2])?;
        let data = shaped(rule, &ids, &block, [rows, cols])?.to_vec();
        // `shaped` established the length.
        Ok(GenericRMatrix::new(data, rows, cols))
    }

    pub(crate) fn a_block<A: AdapterAccess>(
        rule: &A,
        ids: [SectorId; 3],
    ) -> Result<GenericRMatrix<Value<A>>, Error<A>> {
        let (objects, a_dual) = admit_vertex_move(rule, CategoryOperation::A, ids)?;
        let [a, b, c] = &objects;
        let block = rule
            .category()
            .a_symbol(a, b, c)
            .map_err(provider(rule, ids))?;
        let a_dual = a_dual.unwrap_or_else(|| rule.category().dual(a));
        vertex_move_block(rule, ids, &objects, [&a_dual, c, b], block)
    }

    pub(crate) fn b_block<A: AdapterAccess>(
        rule: &A,
        ids: [SectorId; 3],
    ) -> Result<GenericRMatrix<Value<A>>, Error<A>> {
        let (objects, b_dual) = admit_vertex_move(rule, CategoryOperation::B, ids)?;
        let [a, b, c] = &objects;
        let block = rule
            .category()
            .b_symbol(a, b, c)
            .map_err(provider(rule, ids))?;
        let b_dual = b_dual.unwrap_or_else(|| rule.category().dual(b));
        vertex_move_block(rule, ids, &objects, [c, &b_dual, a], block)
    }

    pub(crate) fn a_scalar<A: AdapterAccess>(
        rule: &A,
        ids: [SectorId; 3],
    ) -> Result<Value<A>, Error<A>> {
        let ([a, b, c], a_dual) = admit_vertex_move(rule, CategoryOperation::A, ids)?;
        let block = rule
            .category()
            .a_symbol(&a, &b, &c)
            .map_err(provider(rule, ids))?;
        let rows = multiplicity(rule, ids, &a, &b, &c)?;
        let a_dual = a_dual.unwrap_or_else(|| rule.category().dual(&a));
        let cols = multiplicity(rule, ids, &a_dual, &c, &b)?;
        single(rule, &ids, &block, [rows, cols])
    }

    pub(crate) fn b_scalar<A: AdapterAccess>(
        rule: &A,
        ids: [SectorId; 3],
    ) -> Result<Value<A>, Error<A>> {
        let ([a, b, c], b_dual) = admit_vertex_move(rule, CategoryOperation::B, ids)?;
        let block = rule
            .category()
            .b_symbol(&a, &b, &c)
            .map_err(provider(rule, ids))?;
        let rows = multiplicity(rule, ids, &a, &b, &c)?;
        let b_dual = b_dual.unwrap_or_else(|| rule.category().dual(&b));
        let cols = multiplicity(rule, ids, &c, &b_dual, &a)?;
        single(rule, &ids, &block, [rows, cols])
    }
}

/// An infallible multiplicity-free query outside the rule's domain.
pub(crate) fn violated<T>(operation: &str, result: Result<T, FusionAlgebraError>) -> T {
    result.unwrap_or_else(|error| panic!("{operation} precondition violated: {error}"))
}

// `CategoryRule`'s provider traits. The built-in wrappers implement the
// multiplicity-free family through `multiplicity_free_adapter_impls!`, over
// the same `forward` helpers, so their impls name concrete types.

/// `R` or a twist on a category without braiding; checked before the inputs,
/// as a caller bug distinct from an out-of-domain label.
pub(crate) fn require_braiding_or_panic<A: AdapterAccess>(rule: &A, what: &str) {
    if rule.category().braiding_style() == BraidingStyle::None {
        std::panic::panic_any(format!(
            "{what} precondition violated: the category has no braiding"
        ));
    }
}

impl<C, K> TypedSectorAdmission for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C>,
{
    type Sector = K::Sector;
    type Error = K::Error;
    type Mode = K::Mode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        self.identity().clone()
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        self.codec()
            .encode_label(|| self.category(), self.identity(), sector)
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        self.codec().decode_label(|| self.category(), sector)
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        forward::dual(self, sector)
    }
}

impl<C, K> FusionRule for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C, Mode = MultiplicityFreeAdmissionMode, Error = FusionAlgebraError>,
{
    fn rule_identity(&self) -> RuleIdentity {
        self.identity().clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        forward::multiplicity_free_style(self)
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        forward::braiding_style_kind(self.category().braiding_style())
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

impl<C, K> CheckedFusionAlgebra for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C, Mode = MultiplicityFreeAdmissionMode, Error = FusionAlgebraError>,
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

impl<C, K> SectorCodec for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C, Mode = MultiplicityFreeAdmissionMode, Error = FusionAlgebraError>,
{
    type Sector = K::Sector;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        self.try_encode_label(value)
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        self.try_decode_label(id)
    }
}

impl<C, K> MultiplicityFreeFusionRule for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C, Mode = MultiplicityFreeAdmissionMode, Error = FusionAlgebraError>,
{
}

impl<C, K> MultiplicityFreeFusionSymbols for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    C::Scalar: CategoricalScalar + Send + Sync,
    K: CategoryCodec<C, Mode = MultiplicityFreeAdmissionMode, Error = FusionAlgebraError>,
{
    type Scalar = C::Scalar;

    fn has_trivial_associator_gauge(&self) -> bool {
        self.category().has_trivial_associator_gauge()
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

impl<C, K> MultiplicityFreeRigidSymbols for CategoryRule<C, K>
where
    C: FusionCategory + Send + Sync + 'static,
    C::Error: Send + Sync + 'static,
    K: Sync,
    C::Scalar: CategoricalScalar + Send + Sync,
    K: CategoryCodec<C, Mode = MultiplicityFreeAdmissionMode, Error = FusionAlgebraError>,
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

impl<C, K> CanonicalUnitFusionRule for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<
        C,
        Mode = MultiplicityFreeAdmissionMode,
        Error = FusionAlgebraError,
        Unit = CanonicalUnitAdmission,
    >,
{
}

impl<C, K> CheckedGenericFusion for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C, Mode = CheckedGenericAdmissionMode>,
{
    type Error = K::Error;

    fn rule_identity(&self) -> RuleIdentity {
        self.identity().clone()
    }

    /// Always `Generic`: checked Generic admission executes every category on
    /// the multiplicity path, whatever its own fusion style.
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        forward::braiding_style_kind(self.category().braiding_style())
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

impl<C, K> CheckedGenericRigidSymbols for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    C::Scalar: CategoricalScalar + Send + Sync,
    K: CategoryCodec<C, Mode = CheckedGenericAdmissionMode>,
{
    type Scalar = C::Scalar;

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

impl<C, K> CheckedGenericPivotal for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    C::Scalar: CategoricalScalar + Send + Sync,
    K: CategoryCodec<C, Mode = CheckedGenericAdmissionMode>,
{
    fn try_twist_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        forward::twist(self, sector)
    }
}

impl<C, K> CheckedCanonicalUnitFusionRule for CategoryRule<C, K>
where
    C: FusionCategory + 'static,
    C::Error: Send + Sync + 'static,
    K: CategoryCodec<C, Mode = CheckedGenericAdmissionMode, Unit = CanonicalUnitAdmission>,
{
}

/// The multiplicity-free provider traits of a built-in rule backed by the
/// category adapter: the same `forward` helpers as [`CategoryRule`]'s impls,
/// written per rule so that its impls name its concrete `Sector` and
/// `Scalar` (a blanket impl over [`AdapterAccess`] would leak that private
/// machinery into every public bound). The rule's codec must select
/// multiplicity-free, canonical-unit admission with [`FusionAlgebraError`].
macro_rules! multiplicity_free_adapter_impls {
    ($rule:ty, $sector:ty, $scalar:ty) => {
        const _: () = {
            use fusion_symbols::category::FusionCategory;
            use $crate::category_rule::{
                forward, require_braiding_or_panic, violated, AdapterAccess,
            };
            use $crate::{
                BraidingStyleKind, CategoryCodec, CheckedFusionAlgebra, FusionAlgebraError,
                FusionRule, FusionStyleKind, RuleIdentity, SectorId, SectorOrderKey, SectorVec,
            };

            impl $crate::TypedSectorAdmission for $rule {
                type Sector = $sector;
                type Error = FusionAlgebraError;
                type Mode = $crate::MultiplicityFreeAdmissionMode;

                fn typed_rule_identity(&self) -> RuleIdentity {
                    self.identity().clone()
                }
                fn try_encode_label(
                    &self,
                    sector: &$sector,
                ) -> Result<SectorId, FusionAlgebraError> {
                    self.codec()
                        .encode_label(|| self.category(), self.identity(), sector)
                }
                fn try_decode_label(
                    &self,
                    sector: SectorId,
                ) -> Result<$sector, FusionAlgebraError> {
                    self.codec().decode_label(|| self.category(), sector)
                }
                fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
                    forward::dual(self, sector)
                }
            }

            impl FusionRule for $rule {
                fn rule_identity(&self) -> RuleIdentity {
                    self.identity().clone()
                }
                fn fusion_style(&self) -> FusionStyleKind {
                    forward::multiplicity_free_style(self)
                }
                fn braiding_style(&self) -> BraidingStyleKind {
                    forward::braiding_style_kind(self.category().braiding_style())
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

            impl CheckedFusionAlgebra for $rule {
                fn try_dual_sector(
                    &self,
                    sector: SectorId,
                ) -> Result<SectorId, FusionAlgebraError> {
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

            impl $crate::SectorCodec for $rule {
                type Sector = $sector;

                fn encode_sector(&self, value: &$sector) -> Result<SectorId, FusionAlgebraError> {
                    $crate::TypedSectorAdmission::try_encode_label(self, value)
                }
                fn decode_sector(&self, id: SectorId) -> Result<$sector, FusionAlgebraError> {
                    $crate::TypedSectorAdmission::try_decode_label(self, id)
                }
            }

            impl $crate::MultiplicityFreeFusionRule for $rule {}
            impl $crate::CanonicalUnitFusionRule for $rule {}

            impl $crate::MultiplicityFreeFusionSymbols for $rule {
                type Scalar = $scalar;

                fn has_trivial_associator_gauge(&self) -> bool {
                    self.category().has_trivial_associator_gauge()
                }
                fn f_symbol_scalar(
                    &self,
                    left: SectorId,
                    middle: SectorId,
                    right: SectorId,
                    coupled: SectorId,
                    left_coupled: SectorId,
                    right_coupled: SectorId,
                ) -> $scalar {
                    violated(
                        "MultiplicityFreeFusionSymbols::f_symbol_scalar",
                        forward::f_scalar(
                            self,
                            [left, middle, right, coupled, left_coupled, right_coupled],
                        ),
                    )
                }
                fn r_symbol_scalar(
                    &self,
                    left: SectorId,
                    right: SectorId,
                    coupled: SectorId,
                ) -> $scalar {
                    require_braiding_or_panic(self, "r_symbol");
                    violated(
                        "MultiplicityFreeFusionSymbols::r_symbol_scalar",
                        forward::r_scalar(self, [left, right, coupled]),
                    )
                }
            }

            impl $crate::MultiplicityFreeRigidSymbols for $rule {
                fn dim_scalar(&self, sector: SectorId) -> $scalar {
                    violated(
                        "MultiplicityFreeRigidSymbols::dim_scalar",
                        forward::dimension(self, sector, FusionCategory::dim),
                    )
                }
                fn inv_dim_scalar(&self, sector: SectorId) -> $scalar {
                    violated(
                        "MultiplicityFreeRigidSymbols::inv_dim_scalar",
                        forward::dimension(self, sector, FusionCategory::inv_dim),
                    )
                }
                fn sqrt_dim_scalar(&self, sector: SectorId) -> $scalar {
                    violated(
                        "MultiplicityFreeRigidSymbols::sqrt_dim_scalar",
                        forward::dimension(self, sector, FusionCategory::sqrt_dim),
                    )
                }
                fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> $scalar {
                    violated(
                        "MultiplicityFreeRigidSymbols::inv_sqrt_dim_scalar",
                        forward::dimension(self, sector, FusionCategory::inv_sqrt_dim),
                    )
                }
                fn twist_scalar(&self, sector: SectorId) -> $scalar {
                    require_braiding_or_panic(self, "twist");
                    violated(
                        "MultiplicityFreeRigidSymbols::twist_scalar",
                        forward::twist(self, sector),
                    )
                }
                fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> $scalar {
                    violated(
                        "MultiplicityFreeRigidSymbols::frobenius_schur_phase_scalar",
                        forward::frobenius_schur_phase(self, sector),
                    )
                }
                fn a_symbol_scalar(
                    &self,
                    left: SectorId,
                    right: SectorId,
                    coupled: SectorId,
                ) -> $scalar {
                    violated(
                        "MultiplicityFreeRigidSymbols::a_symbol_scalar",
                        forward::a_scalar(self, [left, right, coupled]),
                    )
                }
                fn b_symbol_scalar(
                    &self,
                    left: SectorId,
                    right: SectorId,
                    coupled: SectorId,
                ) -> $scalar {
                    violated(
                        "MultiplicityFreeRigidSymbols::b_symbol_scalar",
                        forward::b_scalar(self, [left, right, coupled]),
                    )
                }
            }
        };
    };
}
pub(crate) use multiplicity_free_adapter_impls;

#[cfg(test)]
mod style_admission_tests {
    use super::*;

    type Result = core::result::Result<(), CategoryAdmissionError<FusionAlgebraError>>;

    // A style fusion-symbols does not define cannot be constructed, so the
    // rejecting wildcard arms are covered by construction, not by a value.
    #[test]
    fn admission_accepts_exactly_the_known_styles_for_each_mode() {
        let braidings = [
            BraidingStyle::None,
            BraidingStyle::Bosonic,
            BraidingStyle::Fermionic,
            BraidingStyle::Anyonic,
        ];
        for braiding in braidings {
            for fusion in [FusionStyle::Unique, FusionStyle::Simple] {
                for multiplicity_free in [false, true] {
                    let result: Result = admit_styles(fusion, braiding, multiplicity_free);
                    assert_eq!(result, Ok(()));
                }
            }
            let result: Result = admit_styles(FusionStyle::Generic, braiding, false);
            assert_eq!(result, Ok(()));
            let result: Result = admit_styles(FusionStyle::Generic, braiding, true);
            assert_eq!(
                result,
                Err(CategoryAdmissionError::IncompatibleFusionStyle {
                    actual: FusionStyle::Generic
                })
            );
        }
    }

    // The variant's own Display; no known style reaches it through admission.
    #[test]
    fn style_rejections_name_the_style() {
        let braiding = CategoryAdmissionError::<FusionAlgebraError>::UnsupportedBraidingStyle {
            actual: BraidingStyle::Anyonic,
        };
        assert_eq!(braiding.to_string(), "unsupported Anyonic braiding style");
        assert!(std::error::Error::source(&braiding).is_none());
    }
}
