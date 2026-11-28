//! Frozen observable contracts of every built-in fusion-rule provider, pinned
//! with hand-written literals before the providers are re-backed by the
//! fusion-symbols category adapter (option C).
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

use symtn_sectors::SectorId;

fn s(id: usize) -> SectorId {
    SectorId::new(id)
}
