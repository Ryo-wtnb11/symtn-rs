//! Export firewall for the `symtn` facade.
//!
//! Each typed operation has exactly one public path, the method on
//! `symtn::typed::TensorMap`; lower-layer free functions and whole-crate
//! re-exports are not reachable from `symtn`. The exact names each module
//! exports are recorded by the API snapshot (`api/`); this test checks that
//! no name is exported by two public modules.
//!
//! Why a source scan and not a compile-time import list: an import list only
//! proves presence, not absence, and the workspace has no parser dependency.

use std::collections::BTreeSet;
use std::path::Path;

use crate::export_scan::{ends_in_word, strip};

/// Leaf names of a `use` tree such as `a::{b, c::{d, e as f}}`.
fn use_leaves(tree: &str, out: &mut BTreeSet<String>) {
    let tree = tree.trim();
    if let Some(inner) = tree.strip_suffix('}') {
        let open = inner.find('{').expect("use group");
        let mut depth = 0;
        let mut start = open + 1;
        for (index, ch) in inner.char_indices().skip(open + 1) {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                ',' if depth == 0 => {
                    use_leaves(&inner[start..index], out);
                    start = index + 1;
                }
                _ => {}
            }
        }
        use_leaves(&inner[start..], out);
    } else if !tree.is_empty() {
        let leaf = tree.rsplit(" as ").next().unwrap();
        let leaf = leaf.rsplit("::").next().unwrap().trim();
        assert_ne!(leaf, "*", "glob re-export in the facade: {tree}");
        out.insert(leaf.to_owned());
    }
}

/// Public item names declared at depth zero of `body`, and the bodies of its
/// inline `pub mod` blocks.
fn exports(body: &str) -> (BTreeSet<String>, Vec<(String, String)>) {
    let mut names = BTreeSet::new();
    let mut inline_modules = Vec::new();
    let mut depth = 0usize;
    let mut i = 0;
    while i < body.len() {
        let rest = &body[i..];
        let ch = rest.chars().next().unwrap();
        match ch {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            _ => {}
        }
        let at_word = !ends_in_word(&body[..i]);
        if depth == 0 && at_word && rest.starts_with("pub ") {
            let decl = rest["pub ".len()..].trim_start();
            let (mut keyword, mut tail) = decl.split_once(char::is_whitespace).unwrap();
            if keyword == "use" {
                use_leaves(&tail[..tail.find(';').unwrap()], &mut names);
                i += rest.find(';').unwrap() + 1;
                continue;
            }
            if keyword == "unsafe" || (keyword == "const" && tail.starts_with("fn ")) {
                (keyword, tail) = tail.trim_start().split_once(char::is_whitespace).unwrap();
            }
            assert!(
                matches!(
                    keyword,
                    "fn" | "struct" | "enum" | "trait" | "type" | "const" | "static" | "mod"
                ),
                "unrecognized public item `pub {keyword}` in the facade"
            );
            let tail = tail.trim_start();
            let name: String = tail
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            let after = tail[name.len()..].trim_start();
            if keyword == "mod" && after.starts_with('{') {
                let open = rest.len() - after.len();
                let close = open + matching_brace(&rest[open..]);
                inline_modules.push((name.clone(), rest[open + 1..close].to_owned()));
                names.insert(name);
                i += close + 1;
                continue;
            }
            names.insert(name);
        }
        i += ch.len_utf8();
    }
    (names, inline_modules)
}

/// Byte index of the brace closing the one that opens `text`.
fn matching_brace(text: &str) -> usize {
    let mut depth = 0;
    for (index, ch) in text.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return index;
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces")
}

fn module_exports(file: &str) -> (BTreeSet<String>, Vec<(String, String)>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(file);
    exports(&strip(&std::fs::read_to_string(path).unwrap()))
}

#[test]
fn each_name_has_one_path() {
    let modules = [
        ("symtn", "lib.rs"),
        ("typed", "typed.rs"),
        ("sector", "sector.rs"),
        ("expert", "expert.rs"),
        ("cache", "cache.rs"),
        ("scratch", "scratch.rs"),
        ("typed::__network", "typed/__network.rs"),
    ]
    .map(|(module, file)| (module, module_exports(file).0));
    for (index, (left, left_names)) in modules.iter().enumerate() {
        for (right, right_names) in &modules[index + 1..] {
            let shared: Vec<_> = left_names.intersection(right_names).collect();
            assert!(
                shared.is_empty(),
                "names exported by both `{left}` and `{right}`: {shared:?}"
            );
        }
    }
}

#[test]
fn scanner_reads_nested_groups_renames_and_inline_modules() {
    let source = strip(
        "pub use a::{b, c::{d, e as f}};\n\
         // pub fn commented() {}\n\
         pub(crate) fn private() { let _ = \"pub fn in_string\"; let _ = '{'; }\n\
         pub const fn g<'a>() {}\n\
         pub mod m { pub struct Inner; }\n\
         impl X { pub fn method(&self) {} }\n",
    );
    let (names, inline) = exports(&source);
    let expected: BTreeSet<String> = ["b", "d", "f", "g", "m"].map(String::from).into();
    assert_eq!(names, expected);
    assert_eq!(
        exports(&inline[0].1).0,
        BTreeSet::from(["Inner".to_owned()])
    );
}
