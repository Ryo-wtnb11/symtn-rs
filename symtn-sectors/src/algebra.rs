use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul};
use std::convert::Infallible;

use num_complex::Complex64;
use smallvec::SmallVec;

use crate::{BraidingStyleKind, FusionStyleKind, RuleIdentity, SectorId, SectorVec};

/// Classification of a leg tuple's coupled-sector candidates for a possibly
/// catalog-bounded rule.
///
/// A candidate is *clean* when every complete fusion tree ending in it stays
/// inside the provider's table, so enumeration is exactly the provider's
/// complete tree set. It is *tainted* when some complete tree passes through an
/// out-of-table inner line: constructing that sector must be an error, NEVER a
/// silently truncated block. `out_of_table` carries display labels for
/// candidates that escaped the table entirely and cannot even be named by a
/// dense [`SectorId`].
///
/// Why an enum rather than four fields: the conventions between them are not
/// independent — losing classification precision means no candidate may be
/// reported clean, and a sector cannot be both clean and tainted. As separate
/// public fields those invariants had to be re-established by every producer
/// and re-checked by every consumer.
///
/// The variants make the first one unrepresentable: an [`Self::Unknown`] fold
/// has no clean list to contradict. The second is normalized rather than
/// forbidden, because enum fields are public too — a hand-written
/// [`Self::Partial`] can still name a sector on both sides.
/// [`CoupledSectorFoldBuilder::seal`] resolves it the conservative way, and a
/// provider that hands the engine an inconsistent fold directly loses a
/// candidate it could have kept rather than gaining one it should not have.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoupledSectorFold {
    /// Every candidate is clean; nothing escaped the table.
    Complete { clean: Vec<SectorId> },
    /// Some candidates escaped. `clean` is the remainder that stays provably
    /// complete.
    Partial {
        clean: Vec<SectorId>,
        tainted: Vec<SectorId>,
        out_of_table: Vec<String>,
    },
    /// The fold left the one-hop frontier shell, so the clean/tainted split is
    /// unknown and every candidate is treated as tainted.
    Unknown {
        tainted: Vec<SectorId>,
        out_of_table: Vec<String>,
    },
}

impl CoupledSectorFold {
    /// The fold of an unbounded provider: every candidate is clean.
    pub fn complete(clean: Vec<SectorId>) -> Self {
        Self::Complete { clean }
    }

    /// Whether the fold proved the candidate set complete and in-table.
    pub fn is_fully_clean(&self) -> bool {
        matches!(self, Self::Complete { .. })
    }

    /// Candidates whose complete tree set is enumerable. Empty when the
    /// classification itself is unknown.
    pub fn clean(&self) -> &[SectorId] {
        match self {
            Self::Complete { clean } | Self::Partial { clean, .. } => clean,
            Self::Unknown { .. } => &[],
        }
    }

    /// Candidates that must not be constructed.
    pub fn tainted(&self) -> &[SectorId] {
        match self {
            Self::Complete { .. } => &[],
            Self::Partial { tainted, .. } | Self::Unknown { tainted, .. } => tainted,
        }
    }

    /// Display labels of candidates that escaped the table entirely.
    pub fn out_of_table(&self) -> &[String] {
        match self {
            Self::Complete { .. } => &[],
            Self::Partial { out_of_table, .. } | Self::Unknown { out_of_table, .. } => out_of_table,
        }
    }

    /// Whether the clean/tainted split could not be determined.
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown { .. })
    }
}

/// Accumulates per-tuple folds into one classification for a whole space.
///
/// Sealing applies the two cross-tuple conventions once, where they can be
/// stated instead of repeated: tainted-anywhere beats clean-somewhere, and an
/// unknown split anywhere demotes every candidate.
#[derive(Clone, Debug, Default)]
pub struct CoupledSectorFoldBuilder {
    clean: Vec<SectorId>,
    tainted: Vec<SectorId>,
    out_of_table: Vec<String>,
    unknown: bool,
}

impl CoupledSectorFoldBuilder {
    pub fn absorb(&mut self, fold: CoupledSectorFold) {
        match fold {
            CoupledSectorFold::Complete { clean } => self.clean.extend(clean),
            CoupledSectorFold::Partial {
                clean,
                tainted,
                out_of_table,
            } => {
                self.clean.extend(clean);
                self.tainted.extend(tainted);
                self.out_of_table.extend(out_of_table);
            }
            CoupledSectorFold::Unknown {
                tainted,
                out_of_table,
            } => {
                self.unknown = true;
                self.tainted.extend(tainted);
                self.out_of_table.extend(out_of_table);
            }
        }
    }

    pub fn seal(mut self) -> CoupledSectorFold {
        self.tainted.sort_unstable();
        self.tainted.dedup();
        self.out_of_table.sort();
        self.out_of_table.dedup();
        self.clean.sort_unstable();
        self.clean.dedup();
        self.clean.retain(|sector| !self.tainted.contains(sector));
        if self.unknown {
            self.tainted.append(&mut self.clean);
            self.tainted.sort_unstable();
            self.tainted.dedup();
            return CoupledSectorFold::Unknown {
                tainted: self.tainted,
                out_of_table: self.out_of_table,
            };
        }
        if self.tainted.is_empty() && self.out_of_table.is_empty() {
            return CoupledSectorFold::Complete { clean: self.clean };
        }
        CoupledSectorFold::Partial {
            clean: self.clean,
            tainted: self.tainted,
            out_of_table: self.out_of_table,
        }
    }
}

pub trait FusionRule: 'static {
    /// Stable semantic identity of this provider.
    ///
    /// Equal identities require immutable equivalence of every operation that
    /// can affect fusion-space structure or numerical recoupling, including
    /// fusion/braiding styles, vacuum, fusion channels, duals, F/R symbols,
    /// dimensions, twists, and bends.
    /// Returning one identity for semantically different providers violates
    /// this trait contract.
    ///
    /// Why not `TypeId` or the provider address: one Rust type may hold
    /// different tables, while distinct allocations may represent the same
    /// immutable rule and must be safe to share in semantic caches.
    fn rule_identity(&self) -> RuleIdentity;

    fn fusion_style(&self) -> FusionStyleKind;

    fn braiding_style(&self) -> BraidingStyleKind;

    fn vacuum(&self) -> SectorId;

    /// The default is the identity, which is correct only for a provider whose
    /// every sector is self-dual.
    ///
    /// A provider that also implements [`CheckedFusionAlgebra`] must override
    /// this with a forward to [`CheckedFusionAlgebra::dual_or_panic`]. Keeping
    /// the default there is the #971 defect: the checked entry point rejects an
    /// out-of-domain sector while this one silently accepts it.
    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    /// Fusion channels of `left ⊗ right`. Returns a `SectorVec` so the common
    /// small result stays stack-inline — this is called millions of times in
    /// the cold recoupling build, and a heap `Vec` per call was ~5% of all
    /// cold-path allocations.
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec;

    /// The default answers from [`Self::fusion_channels`], which never inspects
    /// `coupled`'s domain: an unrepresentable `coupled` reads as "not a
    /// channel" rather than as an error.
    ///
    /// A provider that also implements [`CheckedFusionAlgebra`] must override
    /// this with a forward to [`CheckedFusionAlgebra::nsymbol_or_panic`], so
    /// that the two entry points agree on which sectors exist (#971).
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        usize::from(self.fusion_channels(left, right).contains(&coupled))
    }

    /// This sector's position in TensorKit's sector order; see
    /// [`SectorOrderKey`] for the contract.
    ///
    /// The default is the id itself, which is correct only when ids ascend in
    /// TensorKit `findindex` order. A provider whose codec is not monotone in
    /// that order (U(1)'s zigzag ids) or that has several factors (products)
    /// must override it.
    fn sector_order_key(&self, sector: SectorId) -> SectorOrderKey {
        SectorOrderKey::position(sector.id() as u64)
    }
}

/// A sector's position in TensorKit's sector order, used where a symtn result
/// depends on sector order and must agree with TensorKit's, such as which
/// sector a truncation keeps at an exact cross-sector tie or which fusion
/// trees a partial identity selects.
///
/// `SectorId` stays the storage and interning key; this is only a sort key.
///
/// A key is the list of TensorKitSectors `findindex(values(I), c) - 1`
/// positions of the sector's factors, one position for a simple sector.
/// Keys order by the sum of the positions, then lexicographically. For one
/// position that is the plain position order, which is `isless` for every
/// TensorKitSectors irrep type. For several positions it is TensorKitSectors
/// `isless(::ProductSector, ::ProductSector)` (`src/product.jl`), which
/// compares exactly this degree-then-lexicographic order over the factors'
/// `findindex` values. Because TensorKit's `⊠` flattens and symtn's product
/// nests, a product's key is its factors' keys concatenated.
///
/// Why positions rather than a comparison hook: a product's order needs its
/// factors' positions, not only their mutual order, and a key lets a caller
/// compute each sector's rank once instead of inside a sort comparator.
///
/// Implementations must give distinct sectors of one rule distinct keys of
/// one length.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SectorOrderKey(SmallVec<[u64; 4]>);

impl SectorOrderKey {
    /// The key of a simple sector at `findindex - 1 = position`.
    pub fn position(position: u64) -> Self {
        Self(SmallVec::from_slice(&[position]))
    }

    /// The key of `left ⊠ right`, flattened as TensorKit's `⊠` flattens.
    pub fn product(mut left: Self, right: Self) -> Self {
        left.0.extend_from_slice(&right.0);
        left
    }

    fn degree(&self) -> u128 {
        self.0.iter().map(|&position| u128::from(position)).sum()
    }
}

impl Ord for SectorOrderKey {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.degree()
            .cmp(&other.degree())
            .then_with(|| self.0.cmp(&other.0))
    }
}

impl PartialOrd for SectorOrderKey {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Marks a fusion rule whose vacuum obeys symtn's canonical unit conventions.
///
/// Implementors promise that the vacuum is self-dual, `1 x a -> a` and
/// `a x 1 -> a` are the only multiplicity-one channels, and unitors and
/// associators act as the identity on every multiplicity space.
/// The marker is intentionally open: external sector providers can certify
/// the same law without depending on symtn's execution crates.
pub trait CanonicalUnitFusionRule: FusionRule {}

/// Marks a checked Generic provider with the same canonical vacuum laws.
///
/// This is the fallible-provider counterpart of [`CanonicalUnitFusionRule`];
/// it intentionally does not require the infallible [`FusionRule`].
pub trait CheckedCanonicalUnitFusionRule: CheckedGenericFusion {}

#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FusionAlgebraError {
    /// An input ID does not name a sector in the rule's representable domain.
    InvalidSector {
        sector: SectorId,
    },
    /// A semantic sector label has no ID in the rule's representable domain.
    ///
    /// This is the encode-side counterpart of [`Self::InvalidSector`]: no
    /// `SectorId` names the rejected label, so the label is reported as the
    /// rule's own display form together with the rejecting rule identity.
    UnrepresentableSectorLabel {
        /// Identity of the rule whose codec rejected the label.
        rule: RuleIdentity,
        /// The rejected label in the rule's display form.
        label: String,
    },
    U1FusionOverflow {
        left: i32,
        right: i32,
    },
    /// Both inputs are valid, but at least one output channel cannot be represented.
    FusionNotRepresentable {
        left: SectorId,
        right: SectorId,
    },
    ProductCodec(ProductSectorCodecError),
    /// The rule's dual maps two distinct sectors onto one, so it is not the
    /// involution rigidity requires. Reported with the shared image.
    DualNotInjective {
        dual: SectorId,
    },
    MultiplicityOverflow {
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    },
}

impl fmt::Display for FusionAlgebraError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSector { sector } => write!(formatter, "invalid fusion sector {sector:?}"),
            // Why not include `rule`: it stays for programmatic matching, but
            // `RuleIdentity` has no display form worth showing a user, and
            // giving it one would widen a public surface for a message.
            Self::UnrepresentableSectorLabel { label, .. } => write!(
                formatter,
                "sector label {label} is not representable by its fusion rule"
            ),
            Self::U1FusionOverflow { left, right } => write!(
                formatter,
                "U(1) fusion charge {left} + {right} is not representable"
            ),
            Self::FusionNotRepresentable { left, right } => write!(
                formatter,
                "fusion output for {left:?} x {right:?} is not representable"
            ),
            Self::DualNotInjective { dual } => write!(
                formatter,
                "fusion rule dual is not injective: two sectors share the dual {dual:?}"
            ),
            Self::ProductCodec(_) => formatter.write_str("product sector codec failed"),
            Self::MultiplicityOverflow {
                left,
                right,
                coupled,
            } => write!(
                formatter,
                "fusion multiplicity overflows usize for {left:?} x {right:?} -> {coupled:?}"
            ),
        }
    }
}

impl std::error::Error for FusionAlgebraError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ProductCodec(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ProductSectorCodecError> for FusionAlgebraError {
    fn from(error: ProductSectorCodecError) -> Self {
        Self::ProductCodec(error)
    }
}

/// Checked companion for finite or encoded fusion algebras.
///
/// This is the sole authority for a provider's representable-domain
/// precondition. [`Self::dual_or_panic`], [`Self::fusion_channels_or_panic`],
/// and [`Self::nsymbol_or_panic`] apply that precondition exactly once, so a
/// provider's [`FusionRule::dual`] / [`FusionRule::fusion_channels`] /
/// [`FusionRule::nsymbol`] implementation forwards to them instead of
/// restating the check next to an `expect` (issue #971: the two entry points
/// must not carry two independent copies of the same algebra).
pub trait CheckedFusionAlgebra: FusionRule {
    /// Return the exact dual when both the input and output are representable
    /// by this provider; otherwise return the exact representation failure.
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError>;

    /// Return the complete mathematical channel set for two representable
    /// inputs.
    ///
    /// On success this is exactly [`FusionRule::fusion_channels`], and every
    /// returned sector is representable by this provider. Failure preserves
    /// the exact cause for an invalid input or an unrepresentable generated
    /// channel.
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError>;

    /// Return the exact multiplicity for three representable sectors.
    ///
    /// Zero means the coupled sector is mathematically absent from the fusion
    /// product. Failure preserves the exact representation error for any
    /// input sector.
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError>;

    /// [`FusionRule::dual`]'s single panic site: succeeds exactly where
    /// [`Self::try_dual_sector`] returns `Ok`, and panics with a message
    /// derived from its `Err` otherwise.
    fn dual_or_panic(&self, sector: SectorId) -> SectorId {
        self.try_dual_sector(sector)
            .unwrap_or_else(|error| panic!("FusionRule::dual precondition violated: {error}"))
    }

    /// [`FusionRule::fusion_channels`]'s single panic site: succeeds exactly
    /// where [`Self::try_fusion_channels`] returns `Ok`, and panics with a
    /// message derived from its `Err` otherwise.
    fn fusion_channels_or_panic(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.try_fusion_channels(left, right)
            .unwrap_or_else(|error| {
                panic!("FusionRule::fusion_channels precondition violated: {error}")
            })
    }

    /// [`FusionRule::nsymbol`]'s single panic site: succeeds exactly where
    /// [`Self::try_nsymbol`] returns `Ok`, and panics with a message derived
    /// from its `Err` otherwise.
    fn nsymbol_or_panic(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.try_nsymbol(left, right, coupled)
            .unwrap_or_else(|error| panic!("FusionRule::nsymbol precondition violated: {error}"))
    }
}

/// Translation between a provider's semantic sector labels and the opaque
/// [`SectorId`] keys the fusion engine runs on.
///
/// `SectorId` stays the canonical internal key: the engine never needs
/// labels, and this trait is the only place a label enters or leaves it.
/// Providers whose label domain is wider than their id domain (or vice
/// versa) report the rejected value instead of wrapping or truncating it:
/// [`FusionAlgebraError::UnrepresentableSectorLabel`] on the encode side and
/// [`FusionAlgebraError::InvalidSector`] on the decode side.
///
/// Implementors promise that `decode_sector(encode_sector(x)) == x` for every
/// representable label, that `encode_sector(decode_sector(id)) == id` for
/// every decodable id, and that distinct labels never encode to one id.
/// Equal [`FusionRule::rule_identity`] values therefore require an equal
/// codec, not only equal fusion and recoupling semantics.
///
/// The fourth law is decode totality over the reachable algebra:
/// [`FusionRule::vacuum`] must decode, and so must every id the provider
/// itself generates from decodable inputs through [`FusionRule::dual`] /
/// [`CheckedFusionAlgebra::try_dual_sector`] and
/// [`FusionRule::fusion_channels`] / [`CheckedFusionAlgebra::try_fusion_channels`].
/// A decode domain narrower than that closure is a codec bug, not a user
/// error: the engine hands the codec exactly the ids its own fusion and duals
/// produced, so a facade inspecting a legitimately built tensor would fail on
/// its own output. Encoding stays free to reject labels outside the provider's
/// representable range — that is a user-supplied value, not an engine output.
pub trait SectorCodec: FusionRule {
    /// The provider's public sector label.
    ///
    /// `Ord` and `Hash` exist so callers can key ordered and hashed
    /// containers on labels; the order is the label order the provider
    /// chooses and carries no fusion-algebra meaning.
    type Sector: Clone + Eq + Ord + core::hash::Hash + core::fmt::Debug;

    /// Returns the id naming `value`.
    ///
    /// # Errors
    ///
    /// Returns [`FusionAlgebraError::UnrepresentableSectorLabel`] (or the
    /// provider's more specific representation error) when no id names the
    /// label.
    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError>;

    /// Returns the label named by `id`.
    ///
    /// # Errors
    ///
    /// Returns [`FusionAlgebraError::InvalidSector`] (or the provider's more
    /// specific representation error) when the id is outside the provider's
    /// domain.
    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError>;
}

/// Error shared by checked physical carrier-basis providers.
///
/// `Coefficient` remains provider-defined so an implementation can preserve
/// its coefficient authority's typed error instead of flattening it to text.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PhysicalBasisError<Coefficient> {
    FusionAlgebra(FusionAlgebraError),
    CarrierIndexOutOfBounds {
        sector: SectorId,
        index: usize,
        dimension: usize,
    },
    FusionMultiplicityOutOfBounds {
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
        multiplicity: usize,
        dimension: usize,
    },
    Coefficient(Coefficient),
}

impl<Coefficient: fmt::Display> fmt::Display for PhysicalBasisError<Coefficient> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FusionAlgebra(_) => formatter.write_str("fusion algebra error"),
            Self::CarrierIndexOutOfBounds {
                sector,
                index,
                dimension,
            } => write!(
                formatter,
                "carrier-basis index {index} is out of bounds for sector {sector:?} of dimension {dimension}"
            ),
            Self::FusionMultiplicityOutOfBounds {
                left,
                right,
                coupled,
                multiplicity,
                dimension,
            } => write!(
                formatter,
                "fusion multiplicity index {multiplicity} is out of bounds for ({left:?}, {right:?}) -> {coupled:?} of dimension {dimension}"
            ),
            Self::Coefficient(_) => formatter.write_str("basis coefficient provider failed"),
        }
    }
}

impl<Coefficient> std::error::Error for PhysicalBasisError<Coefficient>
where
    Coefficient: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FusionAlgebra(error) => Some(error),
            Self::Coefficient(error) => Some(error),
            _ => None,
        }
    }
}

impl<Coefficient> From<FusionAlgebraError> for PhysicalBasisError<Coefficient> {
    fn from(error: FusionAlgebraError) -> Self {
        Self::FusionAlgebra(error)
    }
}

/// Checked local embedding of a coupled irrep into physical carrier bases.
///
/// The returned element is
/// `C[left_basis, right_basis, coupled_basis, multiplicity]` for the local
/// fusion `left x right -> coupled`. All indices are zero-based. Their order is
/// part of the provider's executable convention and must agree with any dense
/// data exchanged with another library.
///
/// This trait describes primal carrier bases only. A tensor engine is
/// responsible for mapping dual legs through the provider's checked duality
/// and rigidity data; there is deliberately no independent dual-basis
/// convention here.
/// Implementations must use the same sector domain and fusion multiplicities
/// as their structural fusion authority. In particular, an element exists
/// exactly for `multiplicity < N(left, right, coupled)`, and the local tensors
/// are orthonormal isometries whose sum over coupled sectors and
/// multiplicities is complete on the product carrier space.
///
/// There is intentionally no `FusionRule` supertrait: checked Generic
/// providers do not implement that infallible interface. Tensor engines
/// compose this capability with the provider's applicable structural trait.
pub trait PhysicalFusionBasis {
    type Scalar: CategoricalScalar + Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Dimension of the physical carrier basis of `sector`.
    fn try_carrier_dimension(&self, sector: SectorId) -> Result<usize, Self::Error>;

    /// One checked local fusion-tensor element.
    #[allow(clippy::too_many_arguments)]
    fn try_fusion_tensor_element(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_basis: usize,
        right_basis: usize,
        coupled_basis: usize,
        multiplicity: usize,
    ) -> Result<Self::Scalar, Self::Error>;
}

pub trait MultiplicityFreeFusionRule: FusionRule {}

pub trait MultiplicityFreeFusionSymbols: MultiplicityFreeFusionRule {
    // Send + Sync because cached recoupling coefficients are shared across
    // tree-transform replay workers (TensorKit sectorscalartype parity: the
    // concrete scalar is a plain number type). `CategoricalScalar` supplies
    // the unit and conjugation every provider needs (issue #972: those two
    // operations never depend on the provider, so they are not provider
    // methods).
    type Scalar: CategoricalScalar + Send + Sync;

    /// Whether every allowed associator coefficient in this provider's
    /// current gauge is exactly the scalar unit.
    ///
    /// TensorKit's direct Unique + SymmetricBraiding permutation lowering
    /// relies on this stronger property, not symmetric braiding alone.
    /// Defaulting to false keeps custom providers on the gauge-general Artin
    /// path unless they explicitly certify the invariant.
    fn has_trivial_associator_gauge(&self) -> bool {
        false
    }

    /// `F^{left middle right}_{coupled}[left_coupled, right_coupled]`, in
    /// TensorKitSectors' `Fsymbol(a, b, c, d, e, f)` argument order.
    ///
    /// # Admissible labels only
    ///
    /// The value is defined only when all four vertices exist:
    /// `N(left, middle, left_coupled)`, `N(left_coupled, right, coupled)`,
    /// `N(middle, right, right_coupled)` and `N(left, right_coupled, coupled)`
    /// are all nonzero, i.e. the labels come from valid fusion trees. For any
    /// other labels the value is unspecified: an implementation may return
    /// any scalar, `0`, or panic. Callers must gate with
    /// [`FusionRule::nsymbol`] first; the tree engine does (its trees are
    /// validated, and its multi-associator gates the cross channel
    /// explicitly), and no engine path relies on a zero.
    ///
    /// TensorKitSectors differs only in that most of its providers multiply
    /// by the `Nsymbol`s (Abelian irreps, `FermionParity`, `CU1Irrep`) or
    /// gate on them (`FibonacciAnyon`, `IsingAnyon`), so their off-shell value
    /// is `0`; `SU2Irrep` does not gate and defers to `WignerSymbols.racahW`.
    /// Every shipped symtn provider returns `0` off-shell as well: the
    /// Abelian rules and `CU1FusionRule` return the `fusion-symbols` value,
    /// `SU2FusionRule` too (`0` for an inadmissible 6j), and `TableRule`
    /// its table entry (`0`). A label outside a provider's sector domain
    /// panics for every shipped provider.
    /// `ProductFusionRule` multiplies its components' values, so it inherits
    /// each component's off-shell value and panics if a component panics or
    /// an id does not decode into the product domain.
    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar;

    /// `R^{left right}_{coupled}`, in TensorKitSectors'
    /// `Rsymbol(a, b, c)` argument order.
    ///
    /// # Admissible labels only
    ///
    /// The value is defined only when `N(left, right, coupled)` is nonzero.
    /// For any other labels it is unspecified: an implementation may return
    /// any scalar, `0`, or panic, and callers must gate with
    /// [`FusionRule::nsymbol`]. TensorKitSectors multiplies by or gates on
    /// `Nsymbol(a, b, c)` (`irreps.jl` `Rsymbol`, `fermions.jl`, `cu1irrep.jl`,
    /// `su2irrep.jl`, `anyons.jl`), so its off-shell value is `0`. Every
    /// shipped symtn provider returns `0` as well (the Abelian, CU(1) and
    /// SU(2) rules through `fusion-symbols`, `TableRule` from its table).
    /// Out-of-domain labels panic for every shipped provider, and
    /// `SU2FusionRule` also panics when the closure of `left x right` leaves
    /// its supported spin range, even for an admissible `coupled`.
    /// `ProductFusionRule` multiplies its components' values and inherits
    /// their off-shell behaviour and panics.
    ///
    /// # Braided rules only
    ///
    /// Defined only when [`FusionRule::braiding_style`] has a braiding.
    /// Callers gate on it before asking (TensorKit's `artin_braid` and
    /// `braid` do the same); a `NoBraiding` provider may panic here
    /// (`TableRule` does for a fusion-only category).
    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar;
}

// `Sync` because the tree-transform plan compile computes recoupling rows
// for independent source trees in parallel, sharing the rule across workers
// (TensorKit threaded transformer construction parity: sector types are
// plain shareable data). All rule implementations are plain data.
pub trait MultiplicityFreeRigidSymbols: MultiplicityFreeFusionSymbols + Sync {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    /// The twist `θ`. Defined only for a braided rule, as
    /// [`MultiplicityFreeFusionSymbols::r_symbol_scalar`]: callers gate on
    /// [`FusionRule::braiding_style`] first, and a `NoBraiding` provider may
    /// panic (TensorKit's twist of a unit leg of a `NoBraiding` sector is
    /// the identity and never asks for `θ`).
    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar;

    /// Defined only for `N(left, right, coupled) != 0`; the default body
    /// is an `F` evaluation, so see [`MultiplicityFreeFusionSymbols::f_symbol_scalar`].
    fn a_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar
    where
        Self::Scalar: Mul<Output = Self::Scalar>,
    {
        let factor = self.sqrt_dim_scalar(left)
            * self.sqrt_dim_scalar(right)
            * self.inv_sqrt_dim_scalar(coupled);
        let symbol = self.frobenius_schur_phase_scalar(left)
            * self.f_symbol_scalar(self.dual(left), left, right, right, self.vacuum(), coupled);
        factor * symbol.conj()
    }

    /// Defined only for `N(left, right, coupled) != 0`; the default body
    /// is an `F` evaluation, so see [`MultiplicityFreeFusionSymbols::f_symbol_scalar`].
    fn b_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar
    where
        Self::Scalar: Mul<Output = Self::Scalar>,
    {
        self.sqrt_dim_scalar(left)
            * self.sqrt_dim_scalar(right)
            * self.inv_sqrt_dim_scalar(coupled)
            * self.f_symbol_scalar(left, right, self.dual(right), left, coupled, self.vacuum())
    }
}

/// Dense rank-4 F-symbol block for `FusionStyleKind::Generic` (outer
/// multiplicity) rules: `F(a,b,c,d,e,f)[mu,nu,kappa,lambda]` with shape
/// `(N(a,b,e), N(e,c,d), N(b,c,f), N(a,f,d))` — TensorKit `GenericFusion`
/// convention (`sectors.jl` Fsymbol). Row-major over `[mu,nu,kappa,lambda]`.
///
/// Production type: the F-symbol block returned by checked Generic providers,
/// including the SU(N) rule. It is a plain `Vec<Scalar>` + shape tuple rather
/// than an ndarray-style type because consumers only read it element-wise
/// while assembling recoupling matrices, so an N-d array crate would add a
/// dependency without removing any work.
#[derive(Clone, Debug)]
pub struct GenericFArray<Scalar> {
    data: Vec<Scalar>,
    shape: (usize, usize, usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymbolShapeError {
    pub actual_len: usize,
    pub expected_len: Option<usize>,
}

impl fmt::Display for SymbolShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.expected_len {
            Some(expected) => write!(
                f,
                "symbol data length {} does not match shape product {expected}",
                self.actual_len
            ),
            None => write!(f, "symbol shape product overflows usize"),
        }
    }
}

impl std::error::Error for SymbolShapeError {}

impl<Scalar> GenericFArray<Scalar> {
    pub fn new(data: Vec<Scalar>, shape: (usize, usize, usize, usize)) -> Self {
        Self::try_new(data, shape).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_new(
        data: Vec<Scalar>,
        shape: (usize, usize, usize, usize),
    ) -> Result<Self, SymbolShapeError> {
        let (n_mu, n_nu, n_kappa, n_lambda) = shape;
        let expected_len = n_mu
            .checked_mul(n_nu)
            .and_then(|n| n.checked_mul(n_kappa))
            .and_then(|n| n.checked_mul(n_lambda));
        if expected_len != Some(data.len()) {
            return Err(SymbolShapeError {
                actual_len: data.len(),
                expected_len,
            });
        }
        Ok(Self { data, shape })
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        self.shape
    }

    #[inline]
    pub fn data(&self) -> &[Scalar] {
        &self.data
    }

    /// `F[mu,nu,kappa,lambda]`, row-major over the shape tuple.
    ///
    /// Only the flat index is bounds-checked; an out-of-axis label reads a
    /// neighbouring entry. Callers compare labels with the categorical shape
    /// first, which a per-axis check here could not report as a typed error.
    pub fn get(&self, mu: usize, nu: usize, kappa: usize, lambda: usize) -> &Scalar {
        let (_, n_nu, n_kappa, n_lambda) = self.shape;
        let idx = ((mu * n_nu + nu) * n_kappa + kappa) * n_lambda + lambda;
        &self.data[idx]
    }
}

/// Dense R-symbol matrix for `FusionStyleKind::Generic` rules:
/// `R(a,b,c)` is `N(a,b,c) x N(b,a,c)`, row-major.
#[derive(Clone, Debug)]
pub struct GenericRMatrix<Scalar> {
    data: Vec<Scalar>,
    rows: usize,
    cols: usize,
}

impl<Scalar> GenericRMatrix<Scalar> {
    pub fn new(data: Vec<Scalar>, rows: usize, cols: usize) -> Self {
        Self::try_new(data, rows, cols).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_new(data: Vec<Scalar>, rows: usize, cols: usize) -> Result<Self, SymbolShapeError> {
        let expected_len = rows.checked_mul(cols);
        if expected_len != Some(data.len()) {
            return Err(SymbolShapeError {
                actual_len: data.len(),
                expected_len,
            });
        }
        Ok(Self { data, rows, cols })
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn data(&self) -> &[Scalar] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> &Scalar {
        &self.data[row * self.cols + col]
    }
}

/// Fallible categorical input for Generic-fusion structural construction.
///
/// This trait is intentionally independent of [`FusionRule`]. Generated
/// providers can therefore reject an unavailable sector or bounded-table
/// lookup without first manufacturing a value for the infallible path.
/// Symbol and rigidity queries are deliberately absent: structural admission
/// does not inspect F/R/A/B data.
///
/// For a fixed [`RuleIdentity`], all mathematical answers must be logically
/// immutable and independent of query order or concurrent interleaving.
/// Interior warm caches may change performance, but never catalog/gauge
/// semantics. This permits one operation to use shared `&self` queries and
/// equal-identity provider allocations as the same mathematical authority.
pub trait CheckedGenericFusion {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Stable semantic provider identity; the contract is the same as
    /// [`FusionRule::rule_identity`].
    fn rule_identity(&self) -> RuleIdentity;
    fn fusion_style(&self) -> FusionStyleKind;
    fn braiding_style(&self) -> BraidingStyleKind;
    fn vacuum(&self) -> SectorId;
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error>;
    /// Return the complete mathematical channel set. A bounded provider may
    /// return an error when its finite table cannot represent the full
    /// product. This differs from [`Self::try_fusion_channels_in_table`],
    /// which reports only representable table channels.
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error>;
    /// Representable table channels only; this is distinct from the complete
    /// channel query above for bounded providers.
    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error>;
    fn try_coupled_sector_fold(
        &self,
        effective: &[SectorId],
    ) -> Result<CoupledSectorFold, Self::Error> {
        let mut acc = match effective.first() {
            None => vec![self.vacuum()],
            Some(&first) => vec![first],
        };
        for &last in effective.iter().skip(1) {
            let mut next = Vec::new();
            for front in acc {
                next.extend(self.try_fusion_channels(front, last)?);
            }
            next.sort_unstable();
            next.dedup();
            acc = next;
        }
        acc.sort_unstable();
        acc.dedup();
        Ok(CoupledSectorFold::complete(acc))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error>;

    /// The checked-Generic counterpart of [`FusionRule::sector_order_key`],
    /// with the same default and contract.
    fn sector_order_key(&self, sector: SectorId) -> SectorOrderKey {
        SectorOrderKey::position(sector.id() as u64)
    }
}

/// Provider-side labels and checked duality used by ordinary typed tensors.
///
/// The provider owns this capability; tensor crates only select layout
/// admission from [`Self::Mode`].
pub trait TypedSectorAdmission: 'static {
    /// Public semantic sector label.
    type Sector: Clone + Eq + Ord + core::hash::Hash + core::fmt::Debug;
    /// Provider error preserved by the typed facade.
    type Error: std::error::Error + Send + Sync + 'static;
    /// Layout-admission mode.
    type Mode;

    /// Stable categorical authority identity.
    fn typed_rule_identity(&self) -> RuleIdentity;
    /// Encodes a public label as an engine sector.
    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error>;
    /// Decodes an engine sector as a public label.
    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error>;
    /// Returns the exact dual engine sector.
    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error>;
}

/// Layout mode for existing multiplicity-free providers.
#[doc(hidden)]
pub struct MultiplicityFreeAdmissionMode;

/// Layout mode for checked Generic providers.
#[doc(hidden)]
pub struct CheckedGenericAdmissionMode;

impl<R> TypedSectorAdmission for R
where
    R: CheckedFusionAlgebra + SectorCodec,
{
    type Sector = R::Sector;
    type Error = FusionAlgebraError;
    type Mode = MultiplicityFreeAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        FusionRule::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        SectorCodec::encode_sector(self, sector)
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        SectorCodec::decode_sector(self, sector)
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        CheckedFusionAlgebra::try_dual_sector(self, sector)
    }
}

/// Fallible categorical symbol access for Generic-fusion plan lowering.
///
/// This extends [`CheckedGenericFusion`] with only the data needed to compile
/// Generic permute, repartition, and braid rows.  The provider's error type is
/// the structural provider error, so a failed symbol lookup keeps its typed
/// source all the way to the plan boundary.  A/B moves are deliberately not
/// provider methods: their matrices are derived from these primitives after
/// symtn validates the categorical F/R shapes.
///
/// Implementations inherit [`CheckedGenericFusion`]'s logical-immutability
/// contract: F/R and rigidity answers for one identity are stable across
/// query order and concurrency even when internal warm state changes.
pub trait CheckedGenericRigidSymbols: CheckedGenericFusion {
    type Scalar: CategoricalScalar + Send + Sync;

    /// The quantum dimension `dim(sector)` (TensorKitSectors `dim`), the
    /// weight of every quantum-dimension-weighted reduction and truncation
    /// (TensorKit `findtruncated(::SectorVector, ...)`, `truncation.jl:187`).
    ///
    /// The default squares [`try_sqrt_dim_scalar`](Self::try_sqrt_dim_scalar),
    /// which can be one rounding off an integer dimension. A provider that
    /// knows `dim` exactly overrides this so that weighted budgets and exact
    /// cross-sector ties compare exactly, as TensorKit's `Int` dimensions do.
    ///
    /// A wrapper that forwards to another checked provider must forward this
    /// method too: left to the default, it silently degrades the wrapped
    /// provider's exact `dim` to `sqrt(dim)^2`.
    fn try_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        let sqrt_dim = self.try_sqrt_dim_scalar(sector)?;
        Ok(sqrt_dim.clone() * sqrt_dim)
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error>;
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error>;
    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error>;
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error>;
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error>;
}

/// Checked provider-owned pivotal/twist data for categorical traces.
///
/// This is deliberately separate from [`CheckedGenericRigidSymbols`]: a
/// Frobenius--Schur phase and a ribbon twist are distinct categorical data,
/// and existing Generic providers must not silently acquire a default twist.
pub trait CheckedGenericPivotal: CheckedGenericRigidSymbols {
    fn try_twist_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error>;
}

/// Checked structural view of an existing infallible Generic provider.
///
/// A wrapper, rather than a blanket implementation on `R`, leaves a future
/// fallible provider free to implement [`CheckedGenericFusion`] without a
/// coherence conflict with the infallible traits.
pub struct InfallibleGeneric<'a, R>(&'a R);

impl<'a, R> InfallibleGeneric<'a, R> {
    pub fn new(rule: &'a R) -> Self {
        Self(rule)
    }
}

impl<R: FusionRule> CheckedGenericFusion for InfallibleGeneric<'_, R> {
    type Error = Infallible;

    fn rule_identity(&self) -> RuleIdentity {
        self.0.rule_identity()
    }

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

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.0.fusion_channels(left, right))
    }

    /// An infallible provider is unbounded by definition of [`FusionRule`], so
    /// its representable channels are its complete channels and the inherited
    /// default fold classifies every candidate clean.
    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.0.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(self.0.nsymbol(left, right, coupled))
    }

    fn sector_order_key(&self, sector: SectorId) -> SectorOrderKey {
        self.0.sector_order_key(sector)
    }
}

#[cfg(any(test, feature = "testing"))]
impl<R> CheckedGenericRigidSymbols for InfallibleGeneric<'_, R>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar + Send + Sync,
{
    type Scalar = R::Scalar;

    fn try_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.0.dim_scalar(sector))
    }

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

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        Ok(self.0.r_symbol_generic(a, b, c))
    }
}

/// Outer-multiplicity ("Generic" fusion, TensorKit `FusionStyle` = `GenericFusion`)
/// sibling of [`MultiplicityFreeFusionSymbols`]. Where the multiplicity-free
/// trait returns a bare `Scalar` per (a,b,c,...) because `nsymbol` is always
/// 0 or 1, this trait returns a dense rank-4 array / matrix because
/// `nsymbol` can exceed 1.
///
/// Test-only (`testing` feature): production Generic is checked only.
#[cfg(any(test, feature = "testing"))]
pub trait GenericFusionSymbols: FusionRule {
    type Scalar: CategoricalScalar + Send + Sync;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar>;

    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar>;
}

/// The capability every structural-coefficient scalar carries: a unit, a
/// conjugation, and — for the Generic-fusion accumulation and pruning paths —
/// an additive identity and an exact-zero test.
///
/// [`MultiplicityFreeFusionSymbols::Scalar`] and `CheckedGenericRigidSymbols::Scalar`
/// both bound on this one trait (issue #972). The unit and conjugate live
/// here rather than on the provider because they cannot depend on the
/// provider; `zero`/`is_zero` serve the Generic Artin braid's `coeff += Rmat1[ν,ρ] * conj(Fmat[κ,λ,μ,ρ]) *
/// conj(Rmat2[σ,κ])` accumulation and its `iszero(...) && continue` prune
/// (`braiding_manipulations.jl:179-184`). `zero`/`is_zero` stay on this one
/// trait rather than splitting into a base/extension pair: the multiplicity-
/// free path simply does not name them, and a trait member with a real
/// consumer does not need every bound site to use it.
///
/// `Add`/`Mul` are supertraits so the braid can use the `+`/`*` operators
/// exactly as TensorKit writes them. `Div` is one for the trace channel
/// factor `dim(c) / dim(first)` (TensorKit `_trace_permute!`,
/// `tensoroperations.jl:252`), which divides the provider's exact
/// dimensions once instead of squaring two rounded square roots.
pub trait CategoricalScalar:
    Clone + Add<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    /// Additive identity — the `coeff = zero(oneT)` accumulator seed at
    /// `braiding_manipulations.jl:179`.
    fn zero() -> Self;

    /// Multiplicative identity — the `oneT` unit-braid / seed coefficient
    /// (`braiding_manipulations.jl:96`, `:117`), and TensorKit
    /// `sectorscalartype`'s unit for the multiplicity-free F/R symbols.
    fn one() -> Self;

    /// Complex conjugation — TensorKit's `conj(...)` and the matrix adjoint
    /// `'` (`braiding_manipulations.jl:139`, `:172-173`, `:181-182`). Real for
    /// real scalars.
    fn conj(&self) -> Self;

    /// Whether this coefficient is exactly zero — the `iszero(R) && continue`
    /// / `iszero(coeff) && continue` prune (`braiding_manipulations.jl:142`,
    /// `:184`). Exact compare mirrors Julia's `iszero`.
    fn is_zero(&self) -> bool;
}

/// The coefficient scalar a product category carries, given its components'.
///
/// TensorKitSectors derives it by promotion rather than by requiring the
/// components to agree: `fusionscalartype(ProductSector{T}) =
/// typeof(prod(zero ∘ fusionscalartype, _sectors(T)))` (0.3.6,
/// `src/product.jl:218-219`, and `:230-231` for `sectorscalartype`), because
/// the product's symbols are the componentwise products of the components'
/// (`:95-100`, `:126-131`). Julia gets that from `promote_type`; Rust has no
/// such operation, so the promotion is spelled out here.
///
/// Why a closed set of implementations rather than a numeric-tower
/// abstraction: the admitted coefficient scalars are `f64` and [`Complex64`],
/// the promotion between them is total, and a fifth case would be a new
/// categorical decision rather than a new arithmetic one.
pub trait PromoteCoefficientScalar<Rhs> {
    /// The promoted scalar. `CategoricalScalar` supplies the unit,
    /// conjugation and product the promoted symbols are combined with.
    type Output: CategoricalScalar + Clone + Send + Sync;

    fn promote_left(value: Self) -> Self::Output;

    fn promote_right(value: Rhs) -> Self::Output;
}

impl PromoteCoefficientScalar<f64> for f64 {
    type Output = f64;

    fn promote_left(value: Self) -> Self::Output {
        value
    }

    fn promote_right(value: f64) -> Self::Output {
        value
    }
}

impl PromoteCoefficientScalar<Complex64> for f64 {
    type Output = Complex64;

    fn promote_left(value: Self) -> Self::Output {
        Complex64::new(value, 0.0)
    }

    fn promote_right(value: Complex64) -> Self::Output {
        value
    }
}

impl PromoteCoefficientScalar<f64> for Complex64 {
    type Output = Complex64;

    fn promote_left(value: Self) -> Self::Output {
        value
    }

    fn promote_right(value: f64) -> Self::Output {
        Complex64::new(value, 0.0)
    }
}

impl PromoteCoefficientScalar<Complex64> for Complex64 {
    type Output = Complex64;

    fn promote_left(value: Self) -> Self::Output {
        value
    }

    fn promote_right(value: Complex64) -> Self::Output {
        value
    }
}

impl CategoricalScalar for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn conj(&self) -> Self {
        *self
    }

    fn is_zero(&self) -> bool {
        *self == 0.0
    }
}

impl CategoricalScalar for Complex64 {
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }

    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }

    fn conj(&self) -> Self {
        Complex64::conj(self)
    }

    fn is_zero(&self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
}

/// Outer-multiplicity ("Generic" fusion) sibling of
/// [`MultiplicityFreeRigidSymbols`]: the rigidity/pivotal data (quantum
/// dimensions and Frobenius–Schur phase) for a rule whose `nsymbol` can
/// exceed 1.
///
/// The Generic A/B matrices are deliberately not provider methods: symtn
/// derives them once from these primitives and the validated F-symbol, as
/// TensorKitSectors `Asymbol_from_Fsymbol` / `Bsymbol_from_Fsymbol` do
/// (`sectors.jl:501-552`, v0.3.6), so every engine reads the same
/// coefficients (see [`CheckedGenericRigidSymbols`]).
///
/// `frobenius_schur_phase_scalar` stays a bare `Scalar` (not a matrix): the
/// FS phase is `sign(F^{a ā a}_a[1])` and for *any* fusion style the relevant
/// `F` block has all its `N`-labels forced to 1 by the pivotal axioms, so it
/// is a single number even in the Generic case
/// (TensorKitSectors `sectors.jl:463-468`, `frobenius_schur_phase_from_Fsymbol`).
///
/// Test-only (`testing` feature): production Generic is checked only.
#[cfg(any(test, feature = "testing"))]
pub trait GenericRigidSymbols: GenericFusionSymbols
where
    Self::Scalar: CategoricalScalar,
{
    /// The quantum dimension `dim(sector)` — TensorKitSectors `dim`, an exact
    /// categorical dimension (`Int` for group-like sectors), not a square.
    ///
    /// The default squares [`sqrt_dim_scalar`](Self::sqrt_dim_scalar), which
    /// is exact only where `√dim` is (dims 1, 4, 9, …; an irrational dim such
    /// as `1 + √2` has no better `Scalar`). A provider with any other integer
    /// dimension overrides this, and a wrapper of another provider forwards it:
    /// [`InfallibleGeneric`] reports it as `try_dim_scalar`, the exact weight
    /// of rank budgets and the trace channel factor.
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let sqrt_dim = self.sqrt_dim_scalar(sector);
        sqrt_dim.clone() * sqrt_dim
    }

    /// `√dim(sector)` — TensorKitSectors `sqrtdim` (`sectors.jl:440`).
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    /// `1/√dim(sector)` — TensorKitSectors `invsqrtdim` (`sectors.jl:441`).
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    /// Frobenius–Schur phase `κ_sector` — bare scalar (`sectors.jl:463-468`).
    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar;
}

/// Label of a product sector, ordered lexicographically as `(left, right)`.
///
/// The derived order follows the component declaration order. It is a label
/// order for ordered containers only and carries no fusion-algebra meaning.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ProductSector<Left, Right> {
    left: Left,
    right: Right,
}

impl<Left, Right> ProductSector<Left, Right> {
    pub const fn new(left: Left, right: Right) -> Self {
        Self { left, right }
    }

    #[inline]
    pub const fn left(&self) -> &Left {
        &self.left
    }

    #[inline]
    pub const fn right(&self) -> &Right {
        &self.right
    }

    pub fn sector_id_with<C>(self) -> SectorId
    where
        C: ProductSectorCodec,
        Left: Into<SectorId>,
        Right: Into<SectorId>,
    {
        C::encode(self.left.into(), self.right.into())
    }
}

pub const fn product_sector<Left, Right>(left: Left, right: Right) -> ProductSector<Left, Right> {
    ProductSector::new(left, right)
}

/// Reversible encoding for product-sector IDs.
///
/// For every representable pair, `decode(encode(left, right))` must return
/// that pair; every decodable ID must round-trip through `encode`; and
/// distinct representable pairs must not alias. Violating these laws breaks
/// `ProductFusionRule`'s fusion, dual, and symbol semantics.
pub trait ProductSectorCodec {
    fn try_encode(left: SectorId, right: SectorId) -> Option<SectorId>;

    fn encode(left: SectorId, right: SectorId) -> SectorId {
        Self::try_encode(left, right).expect("product sector id overflow")
    }

    fn decode(sector: SectorId) -> Option<(SectorId, SectorId)>;

    fn encode_checked(
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorId, ProductSectorCodecError> {
        Self::try_encode(left, right).ok_or(ProductSectorCodecError::CodecRejected)
    }

    fn decode_checked(sector: SectorId) -> Result<(SectorId, SectorId), ProductSectorCodecError> {
        Self::decode(sector).ok_or(ProductSectorCodecError::CodecRejected)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductSectorComponent {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductSectorCodecError {
    CodecRejected,
    WidthOverflow {
        left_bits: u32,
        right_bits: u32,
        available_bits: u32,
    },
    ComponentOutOfRange {
        component: ProductSectorComponent,
        sector: SectorId,
        bits: u32,
    },
    InvalidHighBits {
        sector: SectorId,
        total_bits: u32,
    },
}

impl core::fmt::Display for ProductSectorCodecError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CodecRejected => write!(formatter, "product sector codec rejected the value"),
            Self::WidthOverflow {
                left_bits,
                right_bits,
                available_bits,
            } => write!(
                formatter,
                "product sector needs {} bits but this target provides {available_bits}",
                u64::from(*left_bits) + u64::from(*right_bits)
            ),
            Self::ComponentOutOfRange {
                component,
                sector,
                bits,
            } => write!(
                formatter,
                "{component:?} product component {sector:?} does not fit its {bits}-bit layout"
            ),
            Self::InvalidHighBits { sector, total_bits } => write!(
                formatter,
                "packed product sector {sector:?} has bits above its {total_bits}-bit layout"
            ),
        }
    }
}

impl std::error::Error for ProductSectorCodecError {}

/// Cantor-pairing product codec.
///
/// This is the default codec parameter of core's `ProductFusionRule`.
/// Built-in user-layer product spaces use [`PackedProductCodec`] instead.
/// Nested Cantor decoding is
/// magnitude-dependent and its representable component domain shrinks as
/// pairing is nested.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct TensorKitProductCodec;

impl ProductSectorCodec for TensorKitProductCodec {
    fn try_encode(left: SectorId, right: SectorId) -> Option<SectorId> {
        let left = left.id() as u128;
        let right = right.id() as u128;
        let sum = left.checked_add(right)?;
        let paired = sum
            .checked_mul(sum + 1)
            .and_then(|value| value.checked_div(2))
            .and_then(|value| value.checked_add(left))
            .and_then(|value| usize::try_from(value).ok())?;
        Some(SectorId::new(paired))
    }

    fn decode(sector: SectorId) -> Option<(SectorId, SectorId)> {
        let paired = sector.id() as u128;
        let sum = tensor_kit_product_pairing_sum(paired);
        let triangular = sum.checked_mul(sum + 1)?.checked_div(2)?;
        let left = paired.checked_sub(triangular)?;
        let right = sum.checked_sub(left)?;
        Some((
            SectorId::new(usize::try_from(left).ok()?),
            SectorId::new(usize::try_from(right).ok()?),
        ))
    }
}

fn tensor_kit_product_pairing_sum(paired: u128) -> u128 {
    let mut low = 0u128;
    let mut high = 1u128;
    while triangular_number(high) <= paired {
        high *= 2;
    }
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if triangular_number(mid) <= paired {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

fn triangular_number(value: u128) -> u128 {
    value * (value + 1) / 2
}

/// Fixed-width leaf or recursively packed product layout.
///
/// Packed products place the left component in the low bits and the right
/// component in the high bits. A zero-bit layout can represent only ID 0;
/// a full-`usize` layout accepts every raw ID but cannot be combined with any
/// positive-width sibling. The combined width must not exceed `usize::BITS`.
///
/// [`PackedProductCodec`] independently enforces the declared raw bit width
/// before calling [`Self::validate`], so a custom validator cannot permit
/// overlapping component bits.
pub trait PackedSectorLayout {
    const BITS: u32;

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Fz2SectorLayout;

impl PackedSectorLayout for Fz2SectorLayout {
    const BITS: u32 = 1;

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError> {
        if sector.id() <= 1 {
            Ok(())
        } else {
            Err(ProductSectorCodecError::InvalidHighBits {
                sector,
                total_bits: Self::BITS,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct U1SectorLayout;

impl PackedSectorLayout for U1SectorLayout {
    const BITS: u32 = 32;

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError> {
        if u32::try_from(sector.id()).is_ok() {
            Ok(())
        } else {
            Err(ProductSectorCodecError::InvalidHighBits {
                sector,
                total_bits: Self::BITS,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Su2SectorLayout;

impl PackedSectorLayout for Su2SectorLayout {
    const BITS: u32 = 8;

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError> {
        if sector.id() <= 254 {
            Ok(())
        } else {
            Err(ProductSectorCodecError::InvalidHighBits {
                sector,
                total_bits: Self::BITS,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ProductSectorLayout<Left, Right>(PhantomData<(Left, Right)>);

impl<Left, Right> PackedSectorLayout for ProductSectorLayout<Left, Right>
where
    Left: PackedSectorLayout,
    Right: PackedSectorLayout,
{
    const BITS: u32 = Left::BITS.saturating_add(Right::BITS);

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError> {
        let (left, right) = decode_packed_components::<Left, Right>(sector)?;
        validate_packed_component::<Left>(left, ProductSectorComponent::Left)?;
        validate_packed_component::<Right>(right, ProductSectorComponent::Right)
    }
}

/// Association-independent fixed-width product encoding.
///
/// `left` occupies bits `0..Left::BITS`; `right` occupies the next
/// `Right::BITS`. Recursive products therefore flatten to the same numeric ID
/// for `(A x B) x C` and `A x (B x C)` when the ordered leaf layouts agree.
/// The codec types, and therefore `RuleIdentity`/cache identities, remain
/// distinct across those source-level associations; numeric equality does not
/// authorize cross-provider cache reuse.
///
/// Capacity is target-dependent because the final ID is a `usize`. A layout
/// wider than `usize::BITS` is rejected with
/// [`ProductSectorCodecError::WidthOverflow`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct PackedProductCodec<Left, Right>(PhantomData<(Left, Right)>);

impl<Left, Right> ProductSectorCodec for PackedProductCodec<Left, Right>
where
    Left: PackedSectorLayout,
    Right: PackedSectorLayout,
{
    fn try_encode(left: SectorId, right: SectorId) -> Option<SectorId> {
        Self::encode_checked(left, right).ok()
    }

    fn decode(sector: SectorId) -> Option<(SectorId, SectorId)> {
        Self::decode_checked(sector).ok()
    }

    fn encode_checked(
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorId, ProductSectorCodecError> {
        validate_packed_width::<Left, Right>()?;
        validate_packed_component::<Left>(left, ProductSectorComponent::Left)?;
        validate_packed_component::<Right>(right, ProductSectorComponent::Right)?;
        let shifted_right = if Left::BITS == usize::BITS {
            0
        } else {
            right.id() << Left::BITS
        };
        Ok(SectorId::new(left.id() | shifted_right))
    }

    fn decode_checked(sector: SectorId) -> Result<(SectorId, SectorId), ProductSectorCodecError> {
        let (left, right) = decode_packed_components::<Left, Right>(sector)?;
        validate_packed_component::<Left>(left, ProductSectorComponent::Left)?;
        validate_packed_component::<Right>(right, ProductSectorComponent::Right)?;
        Ok((left, right))
    }
}

fn validate_packed_component<Layout>(
    sector: SectorId,
    component: ProductSectorComponent,
) -> Result<(), ProductSectorCodecError>
where
    Layout: PackedSectorLayout,
{
    let fits_declared_width = match Layout::BITS.cmp(&usize::BITS) {
        core::cmp::Ordering::Greater => false,
        core::cmp::Ordering::Equal => true,
        core::cmp::Ordering::Less if Layout::BITS == 0 => sector.id() == 0,
        core::cmp::Ordering::Less => sector.id() < (1usize << Layout::BITS),
    };
    if !fits_declared_width {
        return Err(ProductSectorCodecError::ComponentOutOfRange {
            component,
            sector,
            bits: Layout::BITS,
        });
    }
    Layout::validate(sector).map_err(|_| ProductSectorCodecError::ComponentOutOfRange {
        component,
        sector,
        bits: Layout::BITS,
    })
}

fn validate_packed_width<Left, Right>() -> Result<u32, ProductSectorCodecError>
where
    Left: PackedSectorLayout,
    Right: PackedSectorLayout,
{
    let total_bits =
        Left::BITS
            .checked_add(Right::BITS)
            .ok_or(ProductSectorCodecError::WidthOverflow {
                left_bits: Left::BITS,
                right_bits: Right::BITS,
                available_bits: usize::BITS,
            })?;
    if total_bits > usize::BITS {
        return Err(ProductSectorCodecError::WidthOverflow {
            left_bits: Left::BITS,
            right_bits: Right::BITS,
            available_bits: usize::BITS,
        });
    }
    Ok(total_bits)
}

fn decode_packed_components<Left, Right>(
    sector: SectorId,
) -> Result<(SectorId, SectorId), ProductSectorCodecError>
where
    Left: PackedSectorLayout,
    Right: PackedSectorLayout,
{
    let total_bits = validate_packed_width::<Left, Right>()?;
    if total_bits < usize::BITS && sector.id() >> total_bits != 0 {
        return Err(ProductSectorCodecError::InvalidHighBits { sector, total_bits });
    }
    let left_mask = if Left::BITS == usize::BITS {
        usize::MAX
    } else {
        (1usize << Left::BITS) - 1
    };
    let right = if Left::BITS == usize::BITS {
        0
    } else {
        sector.id() >> Left::BITS
    };
    Ok((SectorId::new(sector.id() & left_mask), SectorId::new(right)))
}

#[cfg(test)]
mod coupled_sector_fold_tests {
    use super::*;

    fn sector(id: usize) -> SectorId {
        SectorId::new(id)
    }

    #[test]
    fn sealing_promotes_an_all_clean_accumulation_to_complete() {
        // What: the two conventions are stated once, at seal, so an
        // accumulation with nothing escaping is indistinguishable from a
        // provider that reported `Complete` directly.
        let mut builder = CoupledSectorFoldBuilder::default();
        builder.absorb(CoupledSectorFold::complete(vec![sector(2), sector(1)]));
        builder.absorb(CoupledSectorFold::complete(vec![sector(1)]));

        let sealed = builder.seal();
        assert!(sealed.is_fully_clean());
        assert_eq!(sealed.clean(), &[sector(1), sector(2)]);
        assert!(sealed.tainted().is_empty());
    }

    #[test]
    fn tainted_anywhere_beats_clean_somewhere() {
        // What: a sector reported clean by one leg tuple and tainted by
        // another must not survive as clean — the taint is a statement about
        // trees that exist, the cleanliness only about the tuple that was
        // examined.
        let mut builder = CoupledSectorFoldBuilder::default();
        builder.absorb(CoupledSectorFold::complete(vec![sector(1), sector(3)]));
        builder.absorb(CoupledSectorFold::Partial {
            clean: vec![sector(3)],
            tainted: vec![sector(1)],
            out_of_table: vec!["27".to_owned()],
        });

        let sealed = builder.seal();
        assert!(!sealed.is_fully_clean());
        assert!(!sealed.is_unknown());
        assert_eq!(sealed.clean(), &[sector(3)]);
        assert_eq!(sealed.tainted(), &[sector(1)]);
        assert_eq!(sealed.out_of_table(), &["27".to_owned()]);
    }

    #[test]
    fn an_unknown_split_anywhere_demotes_every_candidate() {
        // What: once the fold leaves the frontier shell the clean/tainted
        // split is not knowable, so no candidate may be reported clean —
        // including ones an earlier tuple certified.
        let mut builder = CoupledSectorFoldBuilder::default();
        builder.absorb(CoupledSectorFold::complete(vec![sector(1), sector(2)]));
        builder.absorb(CoupledSectorFold::Unknown {
            tainted: vec![sector(4)],
            out_of_table: Vec::new(),
        });

        let sealed = builder.seal();
        assert!(sealed.is_unknown());
        assert!(!sealed.is_fully_clean());
        assert!(sealed.clean().is_empty());
        assert_eq!(sealed.tainted(), &[sector(1), sector(2), sector(4)]);
    }
}
