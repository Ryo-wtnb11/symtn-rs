use core::marker::PhantomData;
use core::ops::Mul;
use std::sync::OnceLock;

use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError,
    FusionRule, FusionStyleKind, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, ProductSector, ProductSectorCodec, PromoteCoefficientScalar,
    RuleIdentity, SectorCodec, SectorId, SectorOrderKey, SectorVec, TensorKitProductCodec,
};

/// The coefficient scalar of `Left ⊠ Right`, promoted from its components'.
type ProductScalar<LeftRule, RightRule> =
    <<LeftRule as MultiplicityFreeFusionSymbols>::Scalar as PromoteCoefficientScalar<
        <RightRule as MultiplicityFreeFusionSymbols>::Scalar,
    >>::Output;

fn promote_left<LeftRule, RightRule>(value: LeftRule::Scalar) -> ProductScalar<LeftRule, RightRule>
where
    LeftRule: MultiplicityFreeFusionSymbols,
    RightRule: MultiplicityFreeFusionSymbols,
    LeftRule::Scalar: PromoteCoefficientScalar<RightRule::Scalar>,
{
    <LeftRule::Scalar as PromoteCoefficientScalar<RightRule::Scalar>>::promote_left(value)
}

fn promote_right<LeftRule, RightRule>(
    value: RightRule::Scalar,
) -> ProductScalar<LeftRule, RightRule>
where
    LeftRule: MultiplicityFreeFusionSymbols,
    RightRule: MultiplicityFreeFusionSymbols,
    LeftRule::Scalar: PromoteCoefficientScalar<RightRule::Scalar>,
{
    <LeftRule::Scalar as PromoteCoefficientScalar<RightRule::Scalar>>::promote_right(value)
}

/// The Deligne product `Left ⊠ Right` of two providers: one fusion rule over
/// [`ProductSector<L, R>`](ProductSector) labels whose fusion, dual, F/R and
/// dimension data are the componentwise combination of its factors'.
///
/// This is the generic product mechanism. Any ordered pair of providers is a
/// product, and because a `ProductFusionRule` is itself a provider, three or
/// more factors are the same construction applied again — no new type, enum
/// arm or dispatch branch per physical symmetry. Build one with
/// [`ProductFusionRuleExt::product`], which is where the worked example lives.
///
/// `Codec` decides how a component id pair is packed into one
/// [`SectorId`](crate::SectorId); the default
/// [`TensorKitProductCodec`](crate::TensorKitProductCodec) works for any pair
/// of providers, and a fixed-width
/// [`PackedProductCodec`](crate::PackedProductCodec) is available where the
/// component ranges are known. The codec is an id-packing choice only: two
/// rules that differ solely in `Codec` label the same category, but they are
/// different Rust types and report different
/// [`RuleIdentity`](crate::RuleIdentity)s, so tensors built on them do not mix.
///
/// TensorKit correspondence: `ProductSector` in TensorKitSectors 0.3.4
/// (`src/product.jl:245-294`), documented in TensorKit 0.17.0
/// `docs/src/man/sectors.md:468-506`. TensorKit's convenience aliases
/// `FermionNumber = U1Irrep ⊠ FermionParity` and
/// `FermionSpin = SU2Irrep ⊠ FermionParity` (`src/fermions.jl:71-102`) are
/// *not* a universal factor-order rule: each exists because it enforces an
/// extra invariant (parity tied to the charge, respectively to the spin), not
/// because that order is canonical.
///
/// One deliberate divergence from that correspondence: TensorKit's `⊠`
/// *flattens*, so `(A ⊠ B) ⊠ C` and `A ⊠ (B ⊠ C)` are the same
/// `ProductSector{Tuple{A,B,C}}` and association is unobservable there.
/// symtn keeps the nesting in the Rust type — `ProductFusionRule<ProductFusionRule<A, B>, C>`
/// over `ProductSector<ProductSector<A, B>, C>` labels — so the two
/// associations are distinct types with distinct
/// [`RuleIdentity`](crate::RuleIdentity)s, and converting between them is an
/// explicit relabel. Only the numeric ids can agree: a
/// [`PackedProductCodec`](crate::PackedProductCodec) layout is
/// association-independent by construction (see its own documentation), and
/// that numeric equality deliberately does not authorize mixing tensors across
/// the two providers.
#[derive(Clone, Debug)]
pub struct ProductFusionRule<LeftRule, RightRule, Codec = TensorKitProductCodec> {
    left: LeftRule,
    right: RightRule,
    _codec: PhantomData<Codec>,
    identity: OnceLock<RuleIdentity>,
}

impl<LeftRule, RightRule, Codec> ProductFusionRule<LeftRule, RightRule, Codec> {
    pub const fn new(left: LeftRule, right: RightRule) -> Self {
        Self {
            left,
            right,
            _codec: PhantomData,
            identity: OnceLock::new(),
        }
    }

    #[inline]
    pub const fn left_rule(&self) -> &LeftRule {
        &self.left
    }

    #[inline]
    pub const fn right_rule(&self) -> &RightRule {
        &self.right
    }

    pub fn encode_component_ids(&self, left: SectorId, right: SectorId) -> SectorId
    where
        Codec: ProductSectorCodec,
    {
        Codec::encode(left, right)
    }

    pub fn try_encode_component_ids(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorId, FusionAlgebraError>
    where
        Codec: ProductSectorCodec,
    {
        Codec::encode_checked(left, right).map_err(FusionAlgebraError::ProductCodec)
    }

    pub fn decode_component_ids(&self, sector: SectorId) -> Option<(SectorId, SectorId)>
    where
        Codec: ProductSectorCodec,
    {
        Codec::decode(sector)
    }

    fn decode_component_ids_or_panic(&self, sector: SectorId) -> (SectorId, SectorId)
    where
        Codec: ProductSectorCodec,
    {
        self.decode_component_ids(sector)
            .expect("product fusion rule received an invalid product sector")
    }
}

pub const fn product_fusion_rule<LeftRule, RightRule>(
    left: LeftRule,
    right: RightRule,
) -> ProductFusionRule<LeftRule, RightRule> {
    ProductFusionRule::new(left, right)
}

pub const fn product_fusion_rule_with_codec<LeftRule, RightRule, Codec>(
    left: LeftRule,
    right: RightRule,
) -> ProductFusionRule<LeftRule, RightRule, Codec> {
    ProductFusionRule::new(left, right)
}

/// `⊠` for providers: gives every [`FusionRule`] a `.product(other)` method.
///
/// Blanket-implemented, so it applies to a provider defined outside this
/// workspace exactly as it applies to a built-in one. Import the trait to use
/// the method (`use symtn::sector::ProductFusionRuleExt;`) — without it in
/// scope, `.product(…)` resolves to [`Iterator::product`] and the error
/// message is about iterators.
pub trait ProductFusionRuleExt: FusionRule + Sized {
    /// The ordered product `self ⊠ right`, with the default
    /// [`TensorKitProductCodec`](crate::TensorKitProductCodec).
    ///
    /// This is the canonical way to obtain a product symmetry. Labels are
    /// built the same way with
    /// [`product_sector`](crate::product_sector), and the two nest in step:
    ///
    /// ```
    /// use symtn_sectors::{
    ///     product_sector, FermionParityFusionRule, FusionRule, ProductFusionRuleExt,
    ///     SU2FusionRule, SU2Irrep, SectorCodec, U1FusionRule, U1Irrep, Z2Irrep,
    /// };
    ///
    /// // Two factors: fZ2 ⊠ U(1).
    /// let rule = FermionParityFusionRule.product(U1FusionRule);
    /// let odd = product_sector(Z2Irrep::ODD, U1Irrep::new(1));
    /// let id = rule.encode_sector(&odd)?;
    /// assert_eq!(rule.decode_sector(id)?, odd);
    ///
    /// // Three factors: the same call again, left-associated as
    /// // (fZ2 ⊠ U(1)) ⊠ SU(2). Nothing new is needed at the core.
    /// let rule = FermionParityFusionRule
    ///     .product(U1FusionRule)
    ///     .product(SU2FusionRule);
    /// let label = product_sector(odd, SU2Irrep::from_twice_spin(1));
    /// let id = rule.encode_sector(&label)?;
    /// assert_eq!(rule.decode_sector(id)?, label);
    ///
    /// // Factor order and association are structure, not an equivalence:
    /// // U(1) ⊠ fZ2 is a different Rust type with a different identity, and
    /// // converting between the two is an explicit component swap.
    /// let swapped = U1FusionRule.product(FermionParityFusionRule);
    /// assert_ne!(
    ///     FermionParityFusionRule.product(U1FusionRule).rule_identity(),
    ///     swapped.rule_identity(),
    /// );
    /// # Ok::<(), symtn_sectors::FusionAlgebraError>(())
    /// ```
    ///
    /// The `symtn::typed` module documentation carries the same product
    /// driving a `GradedSpace` and a `TensorMap`; that layer is where a
    /// product symmetry is actually used. It is not linked from here because
    /// `symtn-sectors` does not depend on `symtn` — the dependency runs the
    /// other way.
    fn product<RightRule>(self, right: RightRule) -> ProductFusionRule<Self, RightRule>
    where
        RightRule: FusionRule,
    {
        ProductFusionRule::new(self, right)
    }
}

impl<Rule> ProductFusionRuleExt for Rule where Rule: FusionRule + Sized {}

impl<LeftRule, RightRule, Codec> Default for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: Default,
    RightRule: Default,
{
    fn default() -> Self {
        Self::new(LeftRule::default(), RightRule::default())
    }
}

impl<LeftRule, RightRule, Codec> FusionRule for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: FusionRule,
    RightRule: FusionRule,
    Codec: ProductSectorCodec + 'static,
{
    fn rule_identity(&self) -> RuleIdentity {
        self.identity
            .get_or_init(|| {
                RuleIdentity::compose_with_codec::<Codec>(
                    self.left.rule_identity(),
                    self.right.rule_identity(),
                )
            })
            .clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.left
            .fusion_style()
            .combined_with(self.right.fusion_style())
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.left
            .braiding_style()
            .combined_with(self.right.braiding_style())
    }

    fn vacuum(&self) -> SectorId {
        self.encode_component_ids(self.left.vacuum(), self.right.vacuum())
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        let (left, right) = self.decode_component_ids_or_panic(sector);
        self.encode_component_ids(self.left.dual(left), self.right.dual(right))
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let (left_left, left_right) = self.decode_component_ids_or_panic(left);
        let (right_left, right_right) = self.decode_component_ids_or_panic(right);
        let left_channels = self.left.fusion_channels(left_left, right_left);
        let right_channels = self.right.fusion_channels(left_right, right_right);
        // Cartesian product of the two sub-rules' channels, matching TensorKit's
        // `⊗(p1,p2) = SectorSet(product(map(⊗, ...)))`. No dedup: each sub-rule
        // is multiplicity-free (distinct channels) and `encode_sector` is the
        // Cantor pairing (a bijection), so distinct (left,right) pairs always
        // encode to distinct ids — a `channels.contains()` guard would be
        // provably dead and make this O(k²) instead of O(k) in k = |L|·|R|.
        let mut channels = SectorVec::with_capacity(left_channels.len() * right_channels.len());
        for right_channel in right_channels {
            for &left_channel in &left_channels {
                channels.push(self.encode_component_ids(left_channel, right_channel));
            }
        }
        channels
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        let (left_left, left_right) = self.decode_component_ids_or_panic(left);
        let (right_left, right_right) = self.decode_component_ids_or_panic(right);
        let (coupled_left, coupled_right) = self.decode_component_ids_or_panic(coupled);
        self.left.nsymbol(left_left, right_left, coupled_left)
            * self.right.nsymbol(left_right, right_right, coupled_right)
    }

    /// Independent of `Codec`: the order is TensorKit's product order over the
    /// factors, whatever ids the codec packs them into.
    fn sector_order_key(&self, sector: SectorId) -> SectorOrderKey {
        let (left, right) = self.decode_component_ids_or_panic(sector);
        SectorOrderKey::product(
            self.left.sector_order_key(left),
            self.right.sector_order_key(right),
        )
    }
}

/// The product label is the component label pair; the id is whatever `Codec`
/// packs those component ids into.
///
/// The component-id level pair is
/// [`ProductFusionRule::encode_component_ids`] /
/// [`ProductFusionRule::decode_component_ids`].
impl<LeftRule, RightRule, Codec> SectorCodec for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: SectorCodec,
    RightRule: SectorCodec,
    Codec: ProductSectorCodec + 'static,
{
    type Sector = ProductSector<LeftRule::Sector, RightRule::Sector>;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        let left = self.left.encode_sector(value.left())?;
        let right = self.right.encode_sector(value.right())?;
        // Why not map this to `UnrepresentableSectorLabel`: the packed codec
        // already names the offending component and its bit budget, which a
        // label string would only blur.
        Ok(Codec::encode_checked(left, right)?)
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        let (left, right) = Codec::decode_checked(id)?;
        Ok(ProductSector::new(
            self.left.decode_sector(left)?,
            self.right.decode_sector(right)?,
        ))
    }
}

impl<LeftRule, RightRule, Codec> MultiplicityFreeFusionRule
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeFusionRule,
    RightRule: MultiplicityFreeFusionRule,
    Codec: ProductSectorCodec + 'static,
{
}

impl<LeftRule, RightRule, Codec> CanonicalUnitFusionRule
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: CanonicalUnitFusionRule,
    RightRule: CanonicalUnitFusionRule,
    Codec: ProductSectorCodec + 'static,
{
}

impl<LeftRule, RightRule, Codec> CheckedFusionAlgebra
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: CheckedFusionAlgebra,
    RightRule: CheckedFusionAlgebra,
    Codec: ProductSectorCodec + 'static,
{
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        let (left, right) =
            Codec::decode_checked(sector).map_err(FusionAlgebraError::ProductCodec)?;
        self.try_encode_component_ids(
            self.left.try_dual_sector(left)?,
            self.right.try_dual_sector(right)?,
        )
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let (left_left, left_right) =
            Codec::decode_checked(left).map_err(FusionAlgebraError::ProductCodec)?;
        let (right_left, right_right) =
            Codec::decode_checked(right).map_err(FusionAlgebraError::ProductCodec)?;
        let left_channels = self.left.try_fusion_channels(left_left, right_left)?;
        let right_channels = self.right.try_fusion_channels(left_right, right_right)?;
        let mut channels = SectorVec::new();
        for right_channel in right_channels {
            for &left_channel in &left_channels {
                channels.push(self.try_encode_component_ids(left_channel, right_channel)?);
            }
        }
        Ok(channels)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let (left_left, left_right) =
            Codec::decode_checked(left).map_err(FusionAlgebraError::ProductCodec)?;
        let (right_left, right_right) =
            Codec::decode_checked(right).map_err(FusionAlgebraError::ProductCodec)?;
        let (coupled_left, coupled_right) =
            Codec::decode_checked(coupled).map_err(FusionAlgebraError::ProductCodec)?;
        let left_multiplicity = self.left.try_nsymbol(left_left, right_left, coupled_left)?;
        let right_multiplicity = self
            .right
            .try_nsymbol(left_right, right_right, coupled_right)?;
        match left_multiplicity.checked_mul(right_multiplicity) {
            Some(multiplicity) => Ok(multiplicity),
            None => Err(FusionAlgebraError::MultiplicityOverflow {
                left,
                right,
                coupled,
            }),
        }
    }
}

/// The product's coefficient scalar is the promotion of its components', not a
/// fixed type: a component with complex F/R data (an anyon model) composes with
/// a real-coefficient group provider. See [`PromoteCoefficientScalar`].
impl<LeftRule, RightRule, Codec> MultiplicityFreeFusionSymbols
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeFusionSymbols,
    RightRule: MultiplicityFreeFusionSymbols,
    LeftRule::Scalar: PromoteCoefficientScalar<RightRule::Scalar>,
    Codec: ProductSectorCodec + 'static,
{
    type Scalar = ProductScalar<LeftRule, RightRule>;

    fn has_trivial_associator_gauge(&self) -> bool {
        self.left.has_trivial_associator_gauge() && self.right.has_trivial_associator_gauge()
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
        let (left_l, left_r) = self.decode_component_ids_or_panic(left);
        let (middle_l, middle_r) = self.decode_component_ids_or_panic(middle);
        let (right_l, right_r) = self.decode_component_ids_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_component_ids_or_panic(coupled);
        let (left_coupled_l, left_coupled_r) = self.decode_component_ids_or_panic(left_coupled);
        let (right_coupled_l, right_coupled_r) = self.decode_component_ids_or_panic(right_coupled);
        promote_left::<LeftRule, RightRule>(self.left.f_symbol_scalar(
            left_l,
            middle_l,
            right_l,
            coupled_l,
            left_coupled_l,
            right_coupled_l,
        )) * promote_right::<LeftRule, RightRule>(self.right.f_symbol_scalar(
            left_r,
            middle_r,
            right_r,
            coupled_r,
            left_coupled_r,
            right_coupled_r,
        ))
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_component_ids_or_panic(left);
        let (right_l, right_r) = self.decode_component_ids_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_component_ids_or_panic(coupled);
        promote_left::<LeftRule, RightRule>(self.left.r_symbol_scalar(left_l, right_l, coupled_l))
            * promote_right::<LeftRule, RightRule>(
                self.right.r_symbol_scalar(left_r, right_r, coupled_r),
            )
    }
}

impl<LeftRule, RightRule, Codec> MultiplicityFreeRigidSymbols
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeRigidSymbols,
    RightRule: MultiplicityFreeRigidSymbols,
    LeftRule::Scalar: PromoteCoefficientScalar<RightRule::Scalar>,
    // The A/B overrides below delegate to each component's own default body,
    // which is stated over that component's scalar.
    LeftRule::Scalar: Mul<Output = LeftRule::Scalar>,
    RightRule::Scalar: Mul<Output = RightRule::Scalar>,
    // Sync via the trait's supertrait; the codec is a PhantomData marker.
    Codec: ProductSectorCodec + Sync + 'static,
{
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_component_ids_or_panic(sector);
        promote_left::<LeftRule, RightRule>(self.left.dim_scalar(left))
            * promote_right::<LeftRule, RightRule>(self.right.dim_scalar(right))
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_component_ids_or_panic(sector);
        promote_left::<LeftRule, RightRule>(self.left.inv_dim_scalar(left))
            * promote_right::<LeftRule, RightRule>(self.right.inv_dim_scalar(right))
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_component_ids_or_panic(sector);
        promote_left::<LeftRule, RightRule>(self.left.sqrt_dim_scalar(left))
            * promote_right::<LeftRule, RightRule>(self.right.sqrt_dim_scalar(right))
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_component_ids_or_panic(sector);
        promote_left::<LeftRule, RightRule>(self.left.inv_sqrt_dim_scalar(left))
            * promote_right::<LeftRule, RightRule>(self.right.inv_sqrt_dim_scalar(right))
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_component_ids_or_panic(sector);
        promote_left::<LeftRule, RightRule>(self.left.twist_scalar(left))
            * promote_right::<LeftRule, RightRule>(self.right.twist_scalar(right))
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_component_ids_or_panic(sector);
        promote_left::<LeftRule, RightRule>(self.left.frobenius_schur_phase_scalar(left))
            * promote_right::<LeftRule, RightRule>(self.right.frobenius_schur_phase_scalar(right))
    }

    fn a_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_component_ids_or_panic(left);
        let (right_l, right_r) = self.decode_component_ids_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_component_ids_or_panic(coupled);
        promote_left::<LeftRule, RightRule>(self.left.a_symbol_scalar(left_l, right_l, coupled_l))
            * promote_right::<LeftRule, RightRule>(
                self.right.a_symbol_scalar(left_r, right_r, coupled_r),
            )
    }

    fn b_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_component_ids_or_panic(left);
        let (right_l, right_r) = self.decode_component_ids_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_component_ids_or_panic(coupled);
        promote_left::<LeftRule, RightRule>(self.left.b_symbol_scalar(left_l, right_l, coupled_l))
            * promote_right::<LeftRule, RightRule>(
                self.right.b_symbol_scalar(left_r, right_r, coupled_r),
            )
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

        let rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
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

        let rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
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
        let rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
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
        assert!(rule.identity.get().is_none());
        let first = rule.rule_identity();
        let cached = rule.identity.get().unwrap() as *const RuleIdentity;
        assert_eq!(first, rule.rule_identity());
        assert_eq!(cached, rule.identity.get().unwrap() as *const RuleIdentity);
        assert_ne!(
            rule.rule_identity(),
            product_fusion_rule(U1FusionRule, FermionParityFusionRule).rule_identity()
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
        let rule = product_fusion_rule(TableRule::fibonacci(), Z2FusionRule);
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
        let forward = product_fusion_rule(TableRule::fibonacci(), Z2FusionRule);
        let reversed = product_fusion_rule(Z2FusionRule, TableRule::fibonacci());
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

    #[test]
    fn product_of_real_components_keeps_the_real_scalar() {
        // What: promotion must not widen a product whose components are both
        // real; the existing providers keep f64 coefficients and values.
        let rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
        let odd_zero = rule.encode_component_ids(Z2Irrep::ODD.into(), U1Irrep::new(0).into());
        let vacuum = rule.vacuum();
        let one: f64 = CategoricalScalar::one();

        assert_eq!(one, 1.0);
        assert_eq!(rule.r_symbol_scalar(odd_zero, odd_zero, vacuum), -1.0);
        assert_eq!(rule.dim_scalar(odd_zero), 1.0);
    }
}
