//! Export firewall for the infallible Generic symbol tier.
//!
//! Production Generic is checked only: the infallible symbol traits and the
//! adapter's symbol impl exist only behind `testing`. The adapter itself and
//! its structural `CheckedGenericFusion` impl stay in production, because they
//! carry fusion structure for any `FusionRule`, not Generic symbols.

const GATE: &str = "#[cfg(any(test, feature = \"testing\"))]";

/// Whether the item that starts with `header` carries `GATE` among the
/// attribute lines directly above it.
fn gated(source: &str, header: &str) -> bool {
    let at = source
        .find(header)
        .unwrap_or_else(|| panic!("missing `{header}`"));
    source[..at]
        .lines()
        .rev()
        .map(str::trim)
        .take_while(|line| line.starts_with("#[") || line.starts_with("///"))
        .any(|line| line == GATE)
}

#[test]
fn infallible_symbol_tier_is_testing_only() {
    let source = include_str!("../src/algebra.rs");
    for header in [
        "pub trait GenericFusionSymbols",
        "pub trait GenericRigidSymbols",
        "impl<R> CheckedGenericRigidSymbols for InfallibleGeneric<'_, R>",
    ] {
        assert!(gated(source, header), "`{header}` is not testing-gated");
    }
    for header in [
        "pub struct InfallibleGeneric<'a, R>",
        "impl<R: FusionRule> CheckedGenericFusion for InfallibleGeneric<'_, R>",
    ] {
        assert!(!gated(source, header), "`{header}` must stay in production");
    }
}
