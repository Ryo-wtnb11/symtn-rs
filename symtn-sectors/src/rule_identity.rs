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
    Value(Arc<ValueRuleIdentity>),
    /// A codec's unresolved request. `CategoryRule::try_new` resolves it;
    /// observed anywhere else (the public codec hook, or composed into a
    /// product identity), it is a unique identity shared only by its clones.
    PendingValue(PendingValue),
}

#[derive(Debug, Eq, Hash, PartialEq)]
struct ProductRuleIdentity {
    codec: std::any::TypeId,
    left: RuleIdentity,
    right: RuleIdentity,
}

/// Identity of a rule determined by one category value under one rule type.
///
/// Equality is the value's full `Eq`, never its hash or fingerprint: a
/// `fusion_symbols` fingerprint is shared across parameters such as the ZN
/// modulus, and a hash collision must not merge two categories.
///
/// Why the value is held weakly: a cache key is charged by
/// [`RuleIdentity::charged_retained_bytes`], and an erased category has no
/// size query, so a key that owned the category could keep unbounded memory
/// alive (a table's derived data, a Lie group's generated catalog) beyond its
/// charge. The rule owns the category; the identity only refers to it. Once
/// every clone of the rule is dropped, the identity can no longer equal any
/// other identity, so entries keyed on it are unreachable and leave the cache
/// through its ordinary eviction. That loses sharing with a later rule over
/// an equal category, never correctness.
struct ValueRuleIdentity {
    rule_type: std::any::TypeId,
    prehash: u64,
    /// `size_of` the category: a `Weak` keeps the allocation, not the value.
    value_bytes: usize,
    value: std::sync::Weak<dyn std::any::Any + Send + Sync>,
    equals: fn(&dyn std::any::Any, &dyn std::any::Any) -> bool,
}

/// A codec's request for a value identity, resolved against the `Arc` the
/// rule owns: `anchor` is monomorphized for the category type.
///
/// Why `instance`: the request does not hold its category, so it cannot
/// compare categories; without its own instance, requests for Z3 and Z5 would
/// be equal wherever they escape resolution.
#[derive(Clone, Copy, Debug)]
struct PendingValue {
    rule_type: std::any::TypeId,
    instance: u64,
    anchor: fn(std::any::TypeId, &dyn std::any::Any) -> Option<ValueRuleIdentity>,
}

impl std::fmt::Debug for ValueRuleIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ValueRuleIdentity")
            .field("rule_type", &self.rule_type)
            .field("prehash", &self.prehash)
            .finish_non_exhaustive()
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
            (Self::PendingValue(left), Self::PendingValue(right)) => {
                left.rule_type == right.rule_type && left.instance == right.instance
            }
            (Self::Value(left), Self::Value(right)) => {
                if Arc::ptr_eq(left, right) {
                    return true;
                }
                if left.rule_type != right.rule_type || left.prehash != right.prehash {
                    return false;
                }
                // A dropped category equals nothing; see `ValueRuleIdentity`.
                match (left.value.upgrade(), right.value.upgrade()) {
                    (Some(left_value), Some(right_value)) => {
                        Arc::ptr_eq(&left_value, &right_value)
                            || (left.equals)(&*left_value, &*right_value)
                    }
                    _ => false,
                }
            }
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
            Self::Value(identity) => {
                identity.rule_type.hash(state);
                identity.prehash.hash(state);
            }
            Self::PendingValue(pending) => {
                pending.rule_type.hash(state);
                pending.instance.hash(state);
            }
        }
    }
}

impl RuleIdentity {
    #[cfg(all(test, feature = "cgc-gen"))]
    pub(crate) fn same_content_allocation(&self, other: &Self) -> bool {
        matches!((&self.0, &other.0), (RuleIdentityNode::Content { bytes: a, .. }, RuleIdentityNode::Content { bytes: b, .. }) if Arc::ptr_eq(a, b))
    }

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

    /// A codec's request for the value identity, under rule type `R`, of
    /// the category `V` its rule will own; [`Self::resolve_value`] anchors it.
    pub(crate) fn value_of<R: 'static, V>() -> Self
    where
        V: Eq + std::hash::Hash + Send + Sync + 'static,
    {
        fn equals<V: Eq + 'static>(left: &dyn std::any::Any, right: &dyn std::any::Any) -> bool {
            matches!(
                (left.downcast_ref::<V>(), right.downcast_ref::<V>()),
                (Some(left), Some(right)) if left == right
            )
        }

        fn anchor<V>(
            rule_type: std::any::TypeId,
            owner: &dyn std::any::Any,
        ) -> Option<ValueRuleIdentity>
        where
            V: Eq + std::hash::Hash + Send + Sync + 'static,
        {
            use std::hash::BuildHasher;

            let value = owner.downcast_ref::<Arc<V>>()?;
            let weak: std::sync::Weak<V> = Arc::downgrade(value);
            Some(
                ValueRuleIdentity {
                    rule_type,
                    prehash: std::hash::BuildHasherDefault::<
                        std::collections::hash_map::DefaultHasher,
                    >::default()
                    .hash_one(&**value),
                    value_bytes: std::mem::size_of::<V>(),
                    value: weak,
                    equals: equals::<V>,
                },
            )
        }

        Self(RuleIdentityNode::PendingValue(PendingValue {
            rule_type: std::any::TypeId::of::<R>(),
            instance: next_instance(),
            anchor: anchor::<V>,
        }))
    }

    /// Anchors a [`Self::value_of`] request to `owner`, the `Arc` holding the
    /// rule's category; any other identity is returned unchanged. A request
    /// for another category type than `owner` holds becomes a fresh identity,
    /// which can share nothing but cannot alias.
    pub(crate) fn resolve_value(self, owner: &dyn std::any::Any) -> Self {
        let RuleIdentityNode::PendingValue(pending) = self.0 else {
            return self;
        };
        match (pending.anchor)(pending.rule_type, owner) {
            Some(identity) => Self(RuleIdentityNode::Value(Arc::new(identity))),
            None => Self::unique_of(pending.rule_type),
        }
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
            RuleIdentityNode::Content { bytes, .. } => ARC_HEADER_BYTES
                .saturating_add(bytes.len().saturating_mul(std::mem::size_of::<u8>())),
            RuleIdentityNode::Product(identity) => ARC_HEADER_BYTES
                .saturating_add(std::mem::size_of::<ProductRuleIdentity>())
                .saturating_add(identity.left.charged_retained_bytes())
                .saturating_add(identity.right.charged_retained_bytes()),
            // The category is held weakly and owned by the rule: the key
            // retains the node and, through the `Weak`, the category's
            // allocation, whose contents drop with the last rule.
            RuleIdentityNode::Value(identity) => ARC_HEADER_BYTES
                .saturating_add(std::mem::size_of::<ValueRuleIdentity>())
                .saturating_add(ARC_HEADER_BYTES)
                .saturating_add(identity.value_bytes),
            RuleIdentityNode::PendingValue(_) => 0,
        }
    }
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
