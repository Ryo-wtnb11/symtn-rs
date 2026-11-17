use super::*;

use std::cell::RefCell;
use std::hash::Hash;

/// The checked provider as one cold Generic build reads it: each successful
/// answer is kept for the rest of the build, so a provider query runs once
/// per distinct argument tuple instead of once per basis row.
///
/// The rows of one fusion-tree block share their uncoupled sectors, so the
/// row-wise moves of TensorKit's `artin_braid`/`bendright`/`foldright` on a
/// `FusionTreeBlock` repeat the same `N`, channel, tree and symbol queries.
/// QSpace has no such tree-step path: `QSpace::Permute` permutes each stored
/// CGC reference once (`CRef::Permute`). Keeping a build's answers is a
/// symtn-specific optimization, not a port of either reference.
///
/// Errors are never kept: a failing query fails where it would have failed
/// without the lookup, so the first error is unchanged. Why not keep the
/// answers beyond one build: that would be an implicit provider cache under
/// an eager operation; completed builds are cached by their own keys.
#[doc(hidden)]
pub struct CheckedGenericLookup<'p, P: CheckedGenericRigidSymbols> {
    provider: &'p P,
    memo: RefCell<LookupMemo<P::Scalar>>,
}

/// A tree enumeration's arguments: uncoupled sectors, duality, coupled sector.
type TreeShape = (SectorVec, SmallVec<[bool; 8]>, SectorId);

struct LookupMemo<S> {
    n: FxHashMap<[SectorId; 3], usize>,
    channels: FxHashMap<[SectorId; 2], SectorVec>,
    trees: FxHashMap<TreeShape, Vec<FusionTreeKey>>,
    f: FxHashMap<[SectorId; 6], GenericFArray<S>>,
    r: FxHashMap<[SectorId; 3], GenericRMatrix<S>>,
    a: FxHashMap<[SectorId; 3], GenericRMatrix<S>>,
    b: FxHashMap<[SectorId; 3], GenericRMatrix<S>>,
}

impl<'p, P: CheckedGenericRigidSymbols> CheckedGenericLookup<'p, P> {
    pub fn new(provider: &'p P) -> Self {
        Self {
            provider,
            memo: RefCell::new(LookupMemo {
                n: FxHashMap::default(),
                channels: FxHashMap::default(),
                trees: FxHashMap::default(),
                f: FxHashMap::default(),
                r: FxHashMap::default(),
                a: FxHashMap::default(),
                b: FxHashMap::default(),
            }),
        }
    }

    #[inline]
    pub fn provider(&self) -> &'p P {
        self.provider
    }

    /// The kept answer for `key`, or `fetch`'s, kept on success. The borrow
    /// ends before `fetch` runs, which may query this lookup again.
    fn kept<K, V, E>(
        &self,
        map: fn(&mut LookupMemo<P::Scalar>) -> &mut FxHashMap<K, V>,
        key: K,
        fetch: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E>
    where
        K: Hash + Eq,
        V: Clone,
    {
        if let Some(value) = map(&mut self.memo.borrow_mut()).get(&key) {
            return Ok(value.clone());
        }
        let value = fetch()?;
        map(&mut self.memo.borrow_mut()).insert(key, value.clone());
        Ok(value)
    }
}

impl<P: CheckedGenericRigidSymbols> GenericFusionAccess for CheckedGenericLookup<'_, P> {
    type Error = P::Error;
    fn vacuum(&self) -> SectorId {
        CheckedGenericFusion::vacuum(self.provider)
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        CheckedGenericFusion::try_dual(self.provider, sector)
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        self.kept(
            |memo| &mut memo.n,
            [a, b, c],
            || CheckedGenericFusion::try_nsymbol(self.provider, a, b, c),
        )
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.kept(
            |memo| &mut memo.channels,
            [a, b],
            || CheckedGenericFusion::try_fusion_channels_in_table(self.provider, a, b),
        )
    }
}

impl<P: CheckedGenericRigidSymbols> GenericFRAccess for CheckedGenericLookup<'_, P> {
    type Scalar = P::Scalar;
    fn fusion_style(&self) -> FusionStyleKind {
        CheckedGenericFusion::fusion_style(self.provider)
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        CheckedGenericFusion::braiding_style(self.provider)
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
        self.kept(
            |memo| &mut memo.f,
            [a, b, c, d, e, f],
            || CheckedGenericRigidSymbols::try_f_symbol_generic(self.provider, a, b, c, d, e, f),
        )
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
        self.kept(
            |memo| &mut memo.r,
            [a, b, c],
            || CheckedGenericRigidSymbols::try_r_symbol_generic(self.provider, a, b, c),
        )
    }
    fn try_fusion_trees(
        &self,
        uncoupled: &[SectorId],
        is_dual: &[bool],
        coupled: SectorId,
    ) -> Result<Vec<FusionTreeKey>, CheckedGenericSymbolError<Self::Error>> {
        self.kept(
            |memo| &mut memo.trees,
            (uncoupled.into(), is_dual.into(), coupled),
            || {
                collect_generic_fusion_trees_for_coupled_frozen_checked(
                    self, uncoupled, is_dual, uncoupled, coupled,
                )
                .map_err(map_checked_generic_structure_error)
            },
        )
    }
}

impl<P: CheckedGenericRigidSymbols> GenericRigidAccess for CheckedGenericLookup<'_, P> {
    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        CheckedGenericRigidSymbols::try_sqrt_dim_scalar(self.provider, sector)
    }
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        CheckedGenericRigidSymbols::try_inv_sqrt_dim_scalar(self.provider, sector)
    }
    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        CheckedGenericRigidSymbols::try_frobenius_schur_phase_scalar(self.provider, sector)
    }
    fn try_b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        self.kept(
            |memo| &mut memo.b,
            [a, b, c],
            || {
                checked_vertex_move(self, VertexMove::B, [a, b, c], || {
                    CheckedGenericRigidSymbols::try_b_symbol_generic(self.provider, a, b, c)
                })
            },
        )
    }
    fn try_a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, CheckedGenericSymbolError<Self::Error>> {
        self.kept(
            |memo| &mut memo.a,
            [a, b, c],
            || {
                checked_vertex_move(self, VertexMove::A, [a, b, c], || {
                    CheckedGenericRigidSymbols::try_a_symbol_generic(self.provider, a, b, c)
                })
            },
        )
    }
}
