//! Malformed networks are rejected at construction with a typed error.
//!
//! The pairwise executor supports exactly: every output label on one operand,
//! every contracted label on exactly two distinct operands, and output labels
//! present and unique. Each row below breaks one of those rules (numpy-einsum
//! notation in the comment) and names the variant it must produce.

use symtn_network::{ContractError, Network, NetworkIR, TemporaryLabel};

type Case = (
    &'static str,
    &'static [&'static [&'static str]],
    &'static [&'static str],
    ContractError,
);

fn labels(labels: &[&str]) -> Vec<TemporaryLabel> {
    labels.iter().map(|&label| label.into()).collect()
}

fn cases() -> Vec<Case> {
    let s = String::from;
    vec![
        ("no operands", &[], &[], ContractError::EmptyInput),
        (
            "ij,jk->ii",
            &[&["i", "j"], &["j", "k"]],
            &["i", "i"],
            ContractError::DuplicateOutputLabel(s("i")),
        ),
        (
            "ij,jk->iz",
            &[&["i", "j"], &["j", "k"]],
            &["i", "z"],
            ContractError::UnknownOutputLabel(s("z")),
        ),
        (
            "iij,j->i",
            &[&["i", "i", "j"], &["j"]],
            &["i"],
            ContractError::UnsupportedDiagonal {
                label: s("i"),
                tensor: 0,
            },
        ),
        (
            "a,a,a->",
            &[&["a"], &["a"], &["a"]],
            &[],
            ContractError::UnsupportedHyperedge {
                label: s("a"),
                operand_count: 3,
            },
        ),
        (
            "ab,ab->ab",
            &[&["a", "b"], &["a", "b"]],
            &["a", "b"],
            ContractError::UnsupportedBatchLabel {
                label: s("a"),
                operand_count: 2,
            },
        ),
        (
            "ab->a",
            &[&["a", "b"]],
            &["a"],
            ContractError::UnsupportedReduction { label: s("b") },
        ),
    ]
}

#[test]
fn ir_rejects_each_unsupported_label_structure_with_its_variant() {
    for (case, inputs, output, expected) in cases() {
        let result =
            NetworkIR::from_labels(inputs.iter().map(|l| labels(l)).collect(), labels(output));
        assert_eq!(result, Err(expected), "{case}");
    }
}

#[test]
fn network_new_reports_the_same_structural_rejection() {
    for (case, inputs, output, expected) in cases() {
        let n = inputs.len();
        let result = Network::new(
            inputs.iter().map(|l| l.iter().copied()),
            vec![false; n],
            vec![0; n],
            output.iter().copied(),
            None,
        )
        .err();
        assert_eq!(result, Some(expected), "{case}");
    }
}

#[test]
fn network_new_rejects_short_conj_list() {
    let result = Network::new([["i", "j"], ["j", "k"]], [false], [1, 1], ["i", "k"], None);
    assert_eq!(
        result.err(),
        Some(ContractError::MarkerCountMismatch {
            operands: 2,
            conj: 1,
            codomain_splits: 2,
        })
    );
}

#[test]
fn network_new_rejects_long_codomain_splits_list() {
    let result = Network::new(
        [["i", "j"], ["j", "k"]],
        [false, false],
        [1, 1, 1],
        ["i", "k"],
        None,
    );
    assert_eq!(
        result.err(),
        Some(ContractError::MarkerCountMismatch {
            operands: 2,
            conj: 2,
            codomain_splits: 3,
        })
    );
}

#[test]
fn network_new_rejects_oversized_output_codomain() {
    let inputs = [["i", "j"], ["j", "k"]];
    let result = Network::new(inputs, [false; 2], [1; 2], ["i", "k"], Some(3));
    assert_eq!(
        result.err(),
        Some(ContractError::OutputCodomainRankOutOfRange {
            codomain_rank: 3,
            output_rank: 2,
        })
    );
    // The boundary itself (all-codomain output written as `ik;`) is valid.
    assert!(Network::new(inputs, [false; 2], [1; 2], ["i", "k"], Some(2)).is_ok());
}

#[test]
fn supported_pairwise_networks_are_accepted() {
    // Matrix product, full scalar contraction, outer product, transpose.
    let ok: [(&[&[&str]], &[&str]); 4] = [
        (&[&["i", "j"], &["j", "k"]], &["i", "k"]),
        (&[&["a", "b"], &["b", "a"]], &[]),
        (&[&["a"], &["b"]], &["a", "b"]),
        (&[&["a", "b"]], &["b", "a"]),
    ];
    for (inputs, output) in ok {
        assert!(
            NetworkIR::from_labels(inputs.iter().map(|l| labels(l)).collect(), labels(output))
                .is_ok(),
            "{inputs:?} -> {output:?}"
        );
    }
}
