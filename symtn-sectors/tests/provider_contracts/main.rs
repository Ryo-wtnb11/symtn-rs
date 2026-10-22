//! Frozen observable contracts of every built-in fusion-rule provider, pinned
//! with hand-written literals before the providers are re-backed by the
//! fusion-symbols category adapter (#2016, option C).
//!
//! The adapter migration must leave each value here unchanged: sector-id
//! encodings and their boundaries, rule identities, fusion-channel order,
//! TensorKit sector-order keys, checked errors and their precedence, and the
//! reported capability classification. Contracts already pinned with literals
//! by another test are referenced in a comment instead of being repeated.

mod capabilities;
mod channels;
mod errors;
mod identity;
mod ids;
mod order;

use std::sync::Arc;

use fusion_symbols::category::FusionCategory;
use fusion_symbols::table::TableCategory;
use symtn_sectors::{RuleIdentity, SectorId, TableRule};

fn s(id: usize) -> SectorId {
    SectorId::new(id)
}

/// The legacy `TableRule` identity built independently of `TableRule`: the
/// table's fusion-symbols authority fingerprint as canonical bytes, under the
/// `TableRule` type tag, with an FNV-1a 64 prehash (offset basis
/// 0xcbf29ce484222325, prime 0x100000001b3).
fn table_identity(category: &TableCategory) -> RuleIdentity {
    let bytes = category.authority_fingerprint();
    let prehash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    RuleIdentity::from_canonical_bytes::<TableRule>(prehash, Arc::from(bytes))
}
