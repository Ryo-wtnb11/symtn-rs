//! The shared operands of the complex-symbol (#2281) test targets: the
//! tree-hash entries on which tests hardcode values checked against
//! TensorKit 0.17.1, and a real-symbol rule embedded in `Complex64`
//! (optionally with a non-real quantum dimension).
//!
//! Test targets include this file with `#[macro_use] #[path = "support/table_fixture.rs"] mod table_fixture;`.

#![allow(dead_code)]

use num_complex::Complex64;
use symtn::sector::{
    BraidingStyleKind, CheckedFusionAlgebra, FusionRule, FusionStyleKind,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    RuleIdentity, SU2Irrep, SectorCodec, SectorId, SectorOrderKey, SectorVec, U1Irrep,
};
use symtn::typed::{BlockFusionTrees, FusionAlgebraError};
use symtn_sectors::TableSector;

pub fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub fn unit_interval(hash: u64) -> f64 {
    (hash >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

pub fn labels(sectors: &[TableSector]) -> String {
    sectors
        .iter()
        .map(|sector| sector.label().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// The fixture's `entry`: AnyonWiki labels, zero-based indices.
pub fn entry(tag: &str, trees: &BlockFusionTrees<TableSector>, index: &[usize]) -> Complex64 {
    let key = format!(
        "{tag}|{}|{}|{}|{}|{}|{}",
        trees.coupled().label(),
        labels(trees.codomain_uncoupled()),
        labels(trees.codomain_innerlines()),
        labels(trees.domain_uncoupled()),
        labels(trees.domain_innerlines()),
        index
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    Complex64::new(
        unit_interval(fnv1a(&format!("{key}|re"))),
        unit_interval(fnv1a(&format!("{key}|im"))),
    )
}

/// A few dozen products per accumulated value, of entries at most 1/2.
pub fn assert_close(what: &str, got: Complex64, want: Complex64) {
    let bound = 64.0 * f64::EPSILON * want.norm().max(1.0) * 64.0;
    assert!(
        (got - want).norm() <= bound,
        "{what}: {got:?} != {want:?} (residual {:e}, bound {bound:e})",
        (got - want).norm()
    );
}

/// A real-symbol rule with every symbol embedded in `Complex64`; `TILTED`
/// adds `i/1024` to the quantum dimension of every non-unit sector, which no
/// unitary category has.
#[derive(Clone, Copy)]
pub struct Embedded<B, const TILTED: bool>(pub B);

pub trait Base:
    Copy
    + MultiplicityFreeRigidSymbols<Scalar = f64>
    + CheckedFusionAlgebra
    + SectorCodec
    + Send
    + Sync
    + 'static
{
}
impl<B> Base for B where
    B: Copy
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync
        + 'static
{
}

impl<B: Base, const TILTED: bool> FusionRule for Embedded<B, TILTED> {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
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
    fn dual(&self, sector: SectorId) -> SectorId {
        self.0.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.0.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.0.nsymbol(left, right, coupled)
    }
    fn sector_order_key(&self, sector: SectorId) -> SectorOrderKey {
        self.0.sector_order_key(sector)
    }
}

impl<B: Base, const TILTED: bool> MultiplicityFreeFusionRule for Embedded<B, TILTED> {}

impl<B: Base, const TILTED: bool> MultiplicityFreeFusionSymbols for Embedded<B, TILTED> {
    type Scalar = Complex64;
    fn has_trivial_associator_gauge(&self) -> bool {
        self.0.has_trivial_associator_gauge()
    }
    fn f_symbol_scalar(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Complex64 {
        self.0.f_symbol_scalar(a, b, c, d, e, f).into()
    }
    fn r_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> Complex64 {
        self.0.r_symbol_scalar(a, b, c).into()
    }
}

impl<B: Base, const TILTED: bool> MultiplicityFreeRigidSymbols for Embedded<B, TILTED> {
    fn dim_scalar(&self, s: SectorId) -> Complex64 {
        let tilt = if TILTED && s != self.0.vacuum() {
            1.0 / 1024.0
        } else {
            0.0
        };
        Complex64::new(self.0.dim_scalar(s), tilt)
    }
    fn inv_dim_scalar(&self, s: SectorId) -> Complex64 {
        self.0.inv_dim_scalar(s).into()
    }
    fn sqrt_dim_scalar(&self, s: SectorId) -> Complex64 {
        self.0.sqrt_dim_scalar(s).into()
    }
    fn inv_sqrt_dim_scalar(&self, s: SectorId) -> Complex64 {
        self.0.inv_sqrt_dim_scalar(s).into()
    }
    fn twist_scalar(&self, s: SectorId) -> Complex64 {
        self.0.twist_scalar(s).into()
    }
    fn frobenius_schur_phase_scalar(&self, s: SectorId) -> Complex64 {
        self.0.frobenius_schur_phase_scalar(s).into()
    }
}

impl<B: Base, const TILTED: bool> CheckedFusionAlgebra for Embedded<B, TILTED> {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        self.0.try_dual_sector(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        self.0.try_fusion_channels(left, right)
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        self.0.try_nsymbol(left, right, coupled)
    }
}

impl<B: Base, const TILTED: bool> SectorCodec for Embedded<B, TILTED> {
    type Sector = B::Sector;
    fn encode_sector(&self, value: &B::Sector) -> Result<SectorId, FusionAlgebraError> {
        self.0.encode_sector(value)
    }
    fn decode_sector(&self, id: SectorId) -> Result<B::Sector, FusionAlgebraError> {
        self.0.decode_sector(id)
    }
}

/// The same operand for every rule sharing one label type: entries are a
/// hash of the trees' and index's `Debug` text.
macro_rules! operands {
    ($runtime:expr, $rule:expr, $sectors:expr) => {{
        let v = GradedSpace::try_new(Arc::new($rule), $sectors).unwrap();
        let t: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn($runtime, [&v, &v], [&v, &v], |trees, index| {
                let key = format!("{trees:?}{index:?}");
                Complex64::new(
                    unit_interval(fnv1a(&key)),
                    unit_interval(fnv1a(&(key + "i"))),
                )
            })
            .unwrap();
        (v, t)
    }};
}

pub fn su2_sectors() -> [(SU2Irrep, usize); 3] {
    [0, 1, 2].map(|twice| (SU2Irrep::from_twice_spin(twice), 2))
}

pub fn u1_sectors() -> [(U1Irrep, usize); 3] {
    [(-1, 2), (0, 1), (1, 2)].map(|(charge, n)| (U1Irrep::new(charge), n))
}

pub fn is_non_real_dimension(error: &symtn::typed::Error) -> bool {
    matches!(
        error,
        symtn::typed::Error::Operation(operation)
            if matches!(
                **operation,
                symtn::typed::OperationError::InvalidArgument { message }
                    if message.starts_with("quantum dimension is not real")
            )
    )
}
