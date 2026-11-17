use std::hash::Hash;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct RuleIdentity(RuleIdentityNode);

#[derive(Clone, Debug)]
enum RuleIdentityNode {
    Type(std::any::TypeId),
    Unique(std::any::TypeId, u64),
    Content {
        rule_type: std::any::TypeId,
        prehash: u64,
        bytes: Arc<[u8]>,
    },
    Product(Arc<ProductRuleIdentity>),
}

#[derive(Debug, Eq, Hash, PartialEq)]
struct ProductRuleIdentity {
    codec: std::any::TypeId,
    left: RuleIdentity,
    right: RuleIdentity,
}

/// Records the exact byte sequence a `Hash` impl writes.
///
/// Why the input and not a reference to the value: an identity must keep its
/// equality for as long as a cache key or admitted structure stores it, and
/// it must not keep an uncharged category alive. The recorded input is owned,
/// charged by length, and outlives every rule over the value.
struct HashInput(Vec<u8>);

impl std::hash::Hasher for HashInput {
    fn write(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn finish(&self) -> u64 {
        fnv1a64(&self.0)
    }
}

impl PartialEq for RuleIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for RuleIdentity {}

impl std::hash::Hash for RuleIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialEq for RuleIdentityNode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Type(left), Self::Type(right)) => left == right,
            (Self::Unique(left_type, left), Self::Unique(right_type, right)) => {
                left_type == right_type && left == right
            }
            (
                Self::Content {
                    bytes: left,
                    prehash: left_hash,
                    rule_type: left_type,
                },
                Self::Content {
                    bytes: right,
                    prehash: right_hash,
                    rule_type: right_type,
                },
            ) => {
                left_type == right_type
                    && left_hash == right_hash
                    && (Arc::ptr_eq(left, right) || left.as_ref() == right.as_ref())
            }
            (Self::Product(left), Self::Product(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for RuleIdentityNode {}

impl std::hash::Hash for RuleIdentityNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Type(rule_type) => rule_type.hash(state),
            Self::Unique(rule_type, instance) => {
                rule_type.hash(state);
                instance.hash(state);
            }
            Self::Content {
                rule_type, prehash, ..
            } => {
                rule_type.hash(state);
                prehash.hash(state);
            }
            Self::Product(identity) => identity.hash(state),
        }
    }
}

impl RuleIdentity {
    pub fn of_type<R: 'static + ?Sized>() -> Self {
        Self(RuleIdentityNode::Type(std::any::TypeId::of::<R>()))
    }

    pub fn new_unique<R: 'static>() -> Self {
        Self::unique_of(std::any::TypeId::of::<R>())
    }

    fn unique_of(rule_type: std::any::TypeId) -> Self {
        Self(RuleIdentityNode::Unique(rule_type, next_instance()))
    }

    pub fn from_canonical_bytes<R: 'static>(prehash: u64, bytes: Arc<[u8]>) -> Self {
        Self(RuleIdentityNode::Content {
            rule_type: std::any::TypeId::of::<R>(),
            prehash,
            bytes,
        })
    }

    /// The identity, under rule type `R`, of a value whose `Hash` input is
    /// an injective encoding of it, as fusion-symbols' `FusionCategory`
    /// contract requires of every category ("category equality and hashing
    /// use full mathematical identity").
    ///
    /// Equality compares the complete recorded input, never a digest of it.
    pub(crate) fn of_value<R: 'static, V: std::hash::Hash + ?Sized>(value: &V) -> Self {
        let mut input = HashInput(Vec::new());
        value.hash(&mut input);
        let bytes: Arc<[u8]> = input.0.into();
        Self::from_canonical_bytes::<R>(fnv1a64(&bytes), bytes)
    }

    /// Composes two semantic category identities under one product codec type.
    #[doc(hidden)]
    pub fn compose_with_codec<Codec: 'static>(left: Self, right: Self) -> Self {
        Self(RuleIdentityNode::Product(Arc::new(ProductRuleIdentity {
            codec: std::any::TypeId::of::<Codec>(),
            left,
            right,
        })))
    }

    #[doc(hidden)]
    pub fn charged_retained_bytes(&self) -> usize {
        const ARC_HEADER_BYTES: usize = 2 * std::mem::size_of::<usize>();

        // Why not charge only `size_of::<RuleIdentity>()`: content and product
        // identities retain heap allocations whose size can dominate the key.
        // Shared descendants are deliberately charged recursively per entry;
        // the admission budget is conservative rather than allocator-exact.
        match &self.0 {
            RuleIdentityNode::Type(_) | RuleIdentityNode::Unique(_, _) => 0,
            // The `Arc<[u8]>` allocation is padded to its counters' alignment.
            RuleIdentityNode::Content { bytes, .. } => ARC_HEADER_BYTES
                .saturating_add(bytes.len())
                .next_multiple_of(std::mem::align_of::<usize>()),
            RuleIdentityNode::Product(identity) => ARC_HEADER_BYTES
                .saturating_add(std::mem::size_of::<ProductRuleIdentity>())
                .saturating_add(identity.left.charged_retained_bytes())
                .saturating_add(identity.right.charged_retained_bytes()),
        }
    }
}

/// FNV-1a over 64 bits, the prehash a content identity carries beside its
/// bytes.
///
/// Why not `DefaultHasher`: its output is not stable across Rust releases.
/// The prehash is only a bucket key; equality compares the full bytes.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A process-unique instance for identities that compare by allocation.
fn next_instance() -> u64 {
    static NEXT_INSTANCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT_INSTANCE
        .try_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |current| current.checked_add(1),
        )
        .expect("fusion-rule identity space exhausted")
}

#[cfg(test)]
mod tests {
    use super::fnv1a64;

    #[test]
    fn fnv1a64_matches_the_published_reference_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }
}
