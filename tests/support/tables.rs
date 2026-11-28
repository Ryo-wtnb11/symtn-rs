//! AnyonWiki table fixtures, admitted through the one table path:
//! `CategoryRule<TableCategory, EnumeratedCodec<_>>`.
#![allow(dead_code)]

use std::sync::OnceLock;

use crate::sector_types::{CategoryRule, EnumeratedCodec, MultiplicityFreeAdmissionMode};
pub use fusion_symbols::table::{TableCategory, TableObject};

pub type TableCategoryRule =
    CategoryRule<TableCategory, EnumeratedCodec<MultiplicityFreeAdmissionMode>>;

pub fn table_rule(category: TableCategory) -> TableCategoryRule {
    CategoryRule::try_new(category, EnumeratedCodec::new()).unwrap()
}

pub fn table_fibonacci() -> TableCategoryRule {
    table_rule(TableCategory::fibonacci())
}

pub fn table_ising() -> TableCategoryRule {
    table_rule(TableCategory::ising())
}

pub fn table_by_uuid(uuid: &str) -> TableCategoryRule {
    table_rule(TableCategory::by_uuid(uuid).unwrap())
}

/// The object with one-based AnyonWiki label `label` of `rule`'s table.
pub fn table_object(rule: &TableCategoryRule, label: u16) -> TableObject {
    rule.category().object(label).unwrap()
}

/// Fibonacci's object `label`: `1` = 𝟙, `2` = τ.
pub fn fib_object(label: u16) -> TableObject {
    static FIB: OnceLock<TableCategory> = OnceLock::new();
    FIB.get_or_init(TableCategory::fibonacci)
        .object(label)
        .unwrap()
}

/// Ising's object `label`: `1` = 𝟙, `2` = ψ, `3` = σ.
pub fn ising_object(label: u16) -> TableObject {
    static ISING: OnceLock<TableCategory> = OnceLock::new();
    ISING
        .get_or_init(TableCategory::ising)
        .object(label)
        .unwrap()
}
