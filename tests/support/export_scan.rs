//! Source scanner behind the export-firewall tests.
//!
//! The workspace has no parser dependency, and an import list proves presence
//! but not absence, so each firewall test reads `lib.rs` (or a module file)
//! as text. This module owns what every scan shares: blanking comments and
//! literals, reading `pub use` trees, and recognising the `testing` gate. A
//! firewall test keeps only its pinned name list and the rule for its crate.
//!
//! Test targets include this file with
//! `#[path = ".../tests/support/export_scan.rs"] mod export_scan;`.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

/// Removes comments and blanks string and char literals, so that brace depth
/// and item keywords are read from code alone and neither a comment nor a
/// literal can hide or fake an item.
pub fn strip(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    while i < bytes.len() {
        let rest = &source[i..];
        if rest.starts_with("//") {
            i += rest.find('\n').unwrap_or(rest.len());
        } else if rest.starts_with("/*") {
            i += rest.find("*/").expect("unterminated block comment") + 2;
        } else if let Some((open, hashes)) = raw_string_open(rest).filter(|_| !ends_in_word(&out)) {
            let close = format!("\"{}", "#".repeat(hashes));
            i += open + rest[open..].find(&close).expect("unterminated raw string") + close.len();
            out.push_str("\"\"");
        } else if bytes[i] == b'"' {
            i += 1;
            while bytes[i] != b'"' {
                i += if bytes[i] == b'\\' { 2 } else { 1 };
            }
            i += 1;
            out.push_str("\"\"");
        } else if let Some(len) = char_literal_len(rest) {
            i += len;
            out.push_str("' '");
        } else {
            let ch = rest.chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

pub fn ends_in_word(text: &str) -> bool {
    text.ends_with(|c: char| c.is_alphanumeric() || c == '_')
}

/// Byte length and hash count of a raw-string opener (`r"`, `r#"`,
/// `br##"`, ...) at the start of `rest`.
fn raw_string_open(rest: &str) -> Option<(usize, usize)> {
    let tail = rest.strip_prefix("br").or_else(|| rest.strip_prefix('r'))?;
    let hashes = tail.len() - tail.trim_start_matches('#').len();
    tail[hashes..]
        .starts_with('"')
        .then_some((rest.len() - tail.len() + hashes + 1, hashes))
}

/// Byte length of a char literal at the start of `rest`; `None` for a
/// lifetime or anything else.
fn char_literal_len(rest: &str) -> Option<usize> {
    let body = rest.strip_prefix('\'')?;
    if body.starts_with('\\') {
        return body[2..].find('\'').map(|end| end + 4);
    }
    let first = body.chars().next()?;
    body[first.len_utf8()..]
        .starts_with('\'')
        .then_some(first.len_utf8() + 2)
}

/// The `testing` gate after [`strip`] blanks its string literal. No other
/// feature uses the `any(test, feature = ..)` shape
/// (`only_testing_uses_the_test_or_feature_gate` in `symtn-core`).
pub const TESTING_GATE: &str = "#[cfg(any(test, feature = \"\"))]";

/// Whether the item starting at `at` in stripped `code` carries
/// [`TESTING_GATE`] among the attribute lines directly above it.
pub fn testing_gated(code: &str, at: usize) -> bool {
    code[..at]
        .lines()
        .rev()
        .map(str::trim)
        .skip_while(|line| line.is_empty())
        .take_while(|line| line.starts_with("#["))
        .any(|line| line == TESTING_GATE)
}

/// Whether the `pub` at `at` is a restricted `pub(..)` or part of a longer
/// identifier, and so not an unrestricted export.
pub fn not_unrestricted_pub(code: &str, at: usize) -> bool {
    let before = code[..at].chars().next_back();
    let after = &code[at + 3..];
    before.is_some_and(|c| c.is_alphanumeric() || c == '_')
        || after.starts_with('(')
        || after.starts_with(|c: char| c.is_alphanumeric() || c == '_')
}

/// Leaf names of one flat `pub use` tree (`a::B`, `a::{B, c::D as E}`): the
/// name after `as`, else the last path segment. A glob keeps its whole path
/// (whitespace removed). A nested group is an error.
pub fn use_leaves(tree: &str) -> Result<Vec<String>, String> {
    let leaves = match tree.find('{') {
        Some(open) => {
            let inner = tree[open + 1..]
                .trim_end()
                .strip_suffix('}')
                .ok_or("bad use tree")?;
            if inner.contains('{') {
                return Err(format!("nested use tree: {tree}"));
            }
            inner.split(',').collect::<Vec<_>>()
        }
        None => vec![tree],
    };
    Ok(leaves
        .iter()
        .map(|leaf| leaf.trim())
        .filter(|leaf| !leaf.is_empty())
        .map(|leaf| {
            if leaf.ends_with('*') {
                return leaf.split_whitespace().collect();
            }
            let name = leaf.rsplit(" as ").next().unwrap();
            name.rsplit("::").next().unwrap().trim().to_owned()
        })
        .collect())
}

/// What a crate root re-exports.
pub struct Exports {
    /// Leaf of every `pub use`, with whether the `use` is behind the
    /// `testing` gate.
    pub names: BTreeMap<String, bool>,
    /// Out-of-line `pub mod m;` declarations.
    pub modules: BTreeSet<String>,
}

impl Exports {
    /// The ungated `pub use` names.
    pub fn production(&self) -> BTreeSet<String> {
        self.names
            .iter()
            .filter(|&(_, &gated)| !gated)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Fails if the root declares a `pub mod`, for crates whose root only
    /// re-exports.
    pub fn without_modules(self) -> Result<Self, String> {
        match self.modules.first() {
            Some(module) => Err(format!("lib.rs gained a `pub mod {module}`")),
            None => Ok(self),
        }
    }
}

/// The `pub use` leaves and `pub mod m;` declarations of `source`, or an
/// error for any other public item (`pub fn`, `pub enum`, `pub const`, an
/// inline `pub mod m { .. }`, ...) or `#[macro_export]`, which the firewall
/// would otherwise not see.
pub fn reexports(source: &str) -> Result<Exports, String> {
    let code = strip(source);
    if code.contains("macro_export") {
        return Err("lib.rs exports a macro_rules! macro".into());
    }
    let mut names = BTreeMap::new();
    let mut modules = BTreeSet::new();
    for (at, _) in code.match_indices("pub") {
        if not_unrestricted_pub(&code, at) {
            continue;
        }
        let gated = testing_gated(&code, at);
        let item = code[at + 3..].trim_start();
        if let Some(module) = item
            .strip_prefix("mod")
            .filter(|t| t.starts_with(char::is_whitespace))
        {
            let end = module.find([';', '{']).ok_or("unterminated pub mod")?;
            if module[end..].starts_with('{') {
                return Err("lib.rs gained an inline `pub mod` body".into());
            }
            modules.insert(module[..end].trim().to_owned());
            continue;
        }
        let Some(tree) = item
            .strip_prefix("use")
            .filter(|t| t.starts_with(char::is_whitespace))
        else {
            let keyword = item.split_whitespace().next().unwrap_or("");
            return Err(format!("lib.rs gained an inline `pub {keyword}` item"));
        };
        let tree = tree[..tree.find(';').ok_or("unterminated pub use")?].trim();
        for name in use_leaves(tree)? {
            names.insert(name, gated);
        }
    }
    Ok(Exports { names, modules })
}
