//! Malformed serialized plans and plan-cache blobs have a defined outcome.
//!
//! `ContractionPlan::from_text` rejects text it cannot restore with
//! `ContractError`; `load_plan_cache` drops entries it cannot parse and keeps
//! the rest; a persisted order is replayed rather than re-searched, and one
//! that does not describe its network is rejected when it is used.

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Complex32, Complex64};
use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};
use symtn_network::{
    load_plan_cache, plan_cache_stats, save_plan_cache, ContractError, ContractionPlan, Network,
};

#[path = "../../tests/support/network.rs"]
mod network_support;
use network_support::{net, op};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

// `[i; l] = A[i; j] B[j; k] C[k; l]` contracted as `A (B C)`: B and C first
// (result 3, open legs `j l`), then A with 3 (result 4, open legs `i l`).
const B_C_FIRST: &str = r#"symtn-contract-plan-v2
tensor_count 3
output "i" "l"
step 1 2 3 32 "j" "l"
step 0 3 4 32 "i" "l"
"#;

fn chain() -> Network {
    net(
        &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
        &["i"],
        &["l"],
    )
}

fn chain_tensors(runtime: &Runtime) -> [TensorMap<U1FusionRule, f64>; 3] {
    let provider = Arc::new(U1FusionRule);
    let space =
        |dim| GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), dim)]).unwrap();
    let (two, eight) = (space(2), space(8));
    [
        TensorMap::rand_with_seed(runtime, [&two], [&two], 1).unwrap(),
        TensorMap::rand_with_seed(runtime, [&two], [&two], 2).unwrap(),
        TensorMap::rand_with_seed(runtime, [&two], [&eight], 3).unwrap(),
    ]
}

/// The version line this build writes, taken from an empty save.
fn file_header() -> String {
    save_plan_cache(&Runtime::builder().build().unwrap())
}

fn cache_blob(entries: &[(&str, &str)]) -> String {
    let mut blob = file_header();
    for (topology, plan) in entries {
        blob.push_str(&format!(
            "TOPO {topology}\nPLAN {}\n{plan}",
            plan.lines().count()
        ));
    }
    blob
}

#[test]
fn from_text_restores_a_hand_written_plan_exactly() {
    let plan = ContractionPlan::from_text(B_C_FIRST).unwrap();
    assert_eq!(plan.tensor_count(), 3);
    assert_eq!(plan.steps().len(), 2);
    assert_eq!(plan.to_text(), B_C_FIRST);
}

#[test]
fn from_text_rejects_malformed_text_with_a_contract_error() {
    let replace = |from: &str, to: &str| B_C_FIRST.replacen(from, to, 1);
    // ponytail: the reasons are only distinguishable by message text; a typed
    // reason per case belongs to the error-taxonomy work (#2313).
    let cases = [
        ("empty text", String::new(), "missing header"),
        (
            "foreign header",
            replace("symtn-contract-plan-v2", "symtn-contract-plan-v1"),
            "unsupported plan header",
        ),
        (
            "header only",
            "symtn-contract-plan-v2\n".to_string(),
            "missing tensor_count line",
        ),
        (
            "tensor_count keyword",
            replace("tensor_count 3", "tensors 3"),
            "invalid tensor_count line",
        ),
        (
            "tensor_count value",
            replace("tensor_count 3", "tensor_count three"),
            "invalid tensor_count value",
        ),
        (
            "no output line",
            "symtn-contract-plan-v2\ntensor_count 3\n".to_string(),
            "missing output line",
        ),
        (
            "output keyword",
            replace("output \"i\"", "outputs \"i\""),
            "invalid output line",
        ),
        (
            "short step line",
            replace("step 1 2 3 32 \"j\" \"l\"", "step 1 2 3"),
            "invalid step line",
        ),
        (
            "step operand id",
            replace("step 1 2 3", "step one 2 3"),
            "invalid lhs tensor id",
        ),
        (
            "step cost",
            replace("3 32 \"j\"", "3 -1 \"j\""),
            "invalid step cost",
        ),
        (
            "missing step",
            replace("step 0 3 4 32 \"i\" \"l\"\n", ""),
            "needs 2 steps, got 1",
        ),
        (
            "operand consumed twice",
            replace("step 0 3 4", "step 1 3 4"),
            "missing lhs tensor 1",
        ),
        (
            "final labels differ from output",
            replace("output \"i\" \"l\"", "output \"l\" \"i\""),
            "final step labels do not match plan output labels",
        ),
    ];
    for (case, text, reason) in cases {
        match ContractionPlan::from_text(&text) {
            Err(ContractError::InvalidContractionPlan(message)) => {
                assert!(message.contains(reason), "{case}: {message}");
            }
            other => panic!("{case}: expected InvalidContractionPlan, got {other:?}"),
        }
    }
    assert_eq!(
        ContractionPlan::from_text(&replace("tensor_count 3", "tensor_count 0")),
        Err(ContractError::NotEnoughTensors)
    );
}

#[test]
fn load_plan_cache_drops_unparsable_entries_and_keeps_the_rest() {
    let corrupt = B_C_FIRST.replacen("step 1 2 3", "step 1 2 x", 1);
    let blob = cache_blob(&[("bad", &corrupt), ("good", B_C_FIRST)]);
    let runtime = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&runtime, &blob), 1);
    assert_eq!(plan_cache_stats(&runtime).persisted_orders, 1);
    assert_eq!(
        save_plan_cache(&runtime),
        cache_blob(&[("good", B_C_FIRST)])
    );

    // A blob from another file version is ignored and does not turn
    // persistence on: a later search is not recorded.
    let header = file_header();
    let stale_header = format!("{} stale\n", header.trim_end());
    let stale = cache_blob(&[("good", B_C_FIRST)]).replacen(&header, &stale_header, 1);
    assert_ne!(stale_header, header);
    let runtime = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&runtime, &stale), 0);
    let tensors = chain_tensors(&runtime);
    chain().contract(&tensors.each_ref()).unwrap();
    assert_eq!(plan_cache_stats(&runtime).persisted_orders, 0);
}

#[test]
fn load_plan_cache_resumes_at_the_next_topo_line_after_a_corrupt_entry() {
    let header = file_header();
    let expected = cache_blob(&[("a", B_C_FIRST), ("c", B_C_FIRST)]);
    let entries = expected.strip_prefix(&header).unwrap();
    let second = entries.find("TOPO c\n").unwrap();
    let (first, last) = entries.split_at(second);
    for corrupt in [
        // No numeric count.
        "TOPO b\nPLAN ?\ntensor_count 3\n",
        // No PLAN line at all: the next line is already the next entry.
        "TOPO b\n",
        // A count larger than the plan: must not swallow the next entry.
        "TOPO b\nPLAN 999\ntensor_count 3\n",
    ] {
        let blob = format!("{header}{first}{corrupt}{last}");
        let runtime = Runtime::builder().build().unwrap();
        assert_eq!(load_plan_cache(&runtime, &blob), 2, "{corrupt:?}");
        assert_eq!(plan_cache_stats(&runtime).persisted_orders, 2);
        assert_eq!(save_plan_cache(&runtime), expected, "{corrupt:?}");
    }
}

#[test]
fn loaded_non_greedy_order_is_replayed_instead_of_searched() {
    // Record the searched order to learn this network's persisted key.
    let searched = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&searched, ""), 0);
    let tensors = chain_tensors(&searched);
    let expected = chain().contract(&tensors.each_ref()).unwrap();
    let saved = save_plan_cache(&searched);
    let topology = saved.lines().nth(1).unwrap().strip_prefix("TOPO ").unwrap();
    let blob = cache_blob(&[(topology, B_C_FIRST)]);
    assert_ne!(
        saved, blob,
        "the search must pick another order than A (B C)"
    );

    let replayed = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&replayed, &blob), 1);
    let tensors = chain_tensors(&replayed);
    let actual = chain().contract(&tensors.each_ref()).unwrap();
    // A fresh search would have recorded its own order over the loaded one.
    assert_eq!(save_plan_cache(&replayed), blob);
    assert_eq!(plan_cache_stats(&replayed).misses, 1);
    // Both orders compute the same product; each entry sums 2 x 2 terms.
    numerics::assert_slices_close(
        "A (B C) against the searched order",
        actual.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        4,
    );
}

#[test]
fn loaded_order_for_another_network_is_rejected_when_used() {
    let searched = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&searched, ""), 0);
    let tensors = chain_tensors(&searched);
    chain().contract(&tensors.each_ref()).unwrap();
    let saved = save_plan_cache(&searched);
    let topology = saved.lines().nth(1).unwrap().strip_prefix("TOPO ").unwrap();

    // Parses, but its first step claims labels the chain never produces:
    // B C contracted over `k` leaves `j l`, not `k l`.
    let wrong_labels = B_C_FIRST.replacen("3 32 \"j\"", "3 32 \"k\"", 1);
    let runtime = Runtime::builder().build().unwrap();
    assert_eq!(
        load_plan_cache(&runtime, &cache_blob(&[(topology, &wrong_labels)])),
        1
    );
    let tensors = chain_tensors(&runtime);
    assert!(matches!(
        chain().contract(&tensors.each_ref()),
        Err(Error::InvalidArgument(_))
    ));
}

// `chain` with labels a bare-label format cannot carry: a space, a tab and the
// escape character, a line break followed by a `TOPO ` line, and a `TOPO `
// prefix with a quote, a single quote and a combining mark (written `\u{301}`).
const ODD: [&str; 4] = ["i j", "j\tk\\", "k\nTOPO x", "TOPO \"l\u{301}'"];

fn odd_chain() -> Network {
    let [i, j, k, l] = ODD;
    net(
        &[op(&[i], &[j]), op(&[j], &[k]), op(&[k], &[l])],
        &[i],
        &[l],
    )
}

/// `B_C_FIRST` for `odd_chain`, labels escaped by hand.
fn odd_b_c_first() -> String {
    B_C_FIRST
        .replace("\"i\"", r#""i j""#)
        .replace("\"j\"", r#""j\tk\\""#)
        .replace("\"l\"", r#""TOPO \"l\u{301}'""#)
}

#[test]
fn plan_text_round_trips_labels_with_whitespace_line_breaks_and_escapes() {
    let text = odd_b_c_first();
    let plan = ContractionPlan::from_text(&text).unwrap();
    let [i, j, _, l] = ODD;
    let labels = |step: usize| {
        plan.steps()[step]
            .result_labels()
            .iter()
            .map(|label| label.as_str())
            .collect::<Vec<_>>()
    };
    assert_eq!(labels(0), [j, l]);
    assert_eq!(labels(1), [i, l]);
    assert_eq!(plan.to_text(), text);
    assert_eq!(ContractionPlan::from_text(&plan.to_text()), Ok(plan));
    // Every label stays on its own line, whatever it contains.
    assert_eq!(text.lines().count(), 5);

    for (case, text, reason) in [
        (
            "bare label",
            B_C_FIRST.replacen(r#""i" "l""#, "i l", 1),
            "label is not a quoted string",
        ),
        (
            "unknown escape",
            B_C_FIRST.replacen(r#""j""#, r#""j\q""#, 1),
            "invalid quoted label",
        ),
        (
            "unterminated label",
            B_C_FIRST.replacen(r#""j" "l""#, r#""j"#, 1),
            "invalid quoted label",
        ),
    ] {
        match ContractionPlan::from_text(&text) {
            Err(ContractError::InvalidContractionPlan(message)) => {
                assert!(message.contains(reason), "{case}: {message}");
            }
            other => panic!("{case}: expected InvalidContractionPlan, got {other:?}"),
        }
    }
}

#[test]
fn plan_cache_persists_and_replays_an_order_for_labels_needing_escapes() {
    // A searched entry saves and reloads byte-identically.
    let searched = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&searched, ""), 0);
    let tensors = chain_tensors(&searched);
    let expected = odd_chain().contract(&tensors.each_ref()).unwrap();
    let saved = save_plan_cache(&searched);
    assert_eq!(saved.lines().filter(|l| l.starts_with("TOPO ")).count(), 1);
    let reloaded = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&reloaded, &saved), 1);
    assert_eq!(save_plan_cache(&reloaded), saved);

    // A non-searched order under that key is replayed, not re-searched.
    let topology = saved.lines().nth(1).unwrap().strip_prefix("TOPO ").unwrap();
    let blob = cache_blob(&[(topology, &odd_b_c_first())]);
    assert_ne!(saved, blob);
    let replayed = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&replayed, &blob), 1);
    let tensors = chain_tensors(&replayed);
    let actual = odd_chain().contract(&tensors.each_ref()).unwrap();
    assert_eq!(save_plan_cache(&replayed), blob);
    numerics::assert_slices_close(
        "A (B C) against the searched order",
        actual.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        4,
    );
}
