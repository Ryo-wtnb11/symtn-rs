//! Labeled networks for tests and examples, written with the explicit
//! `symtn_network::Network` API.
//!
//! Include with `#[path = ".../tests/support/network.rs"] mod network;`.
//! An operand is its codomain and domain labels, as `t[i, j; k]` writes them:
//! `op(&["i", "j"], &["k"])`; `conj(op(..))` reads it through its adjoint.
//! `net(&[op(..), ..], &[codomain..], &[domain..])` builds the network,
//! and `.contract(&[&t, ..])` runs it through the Runtime's plan cache.

#![allow(dead_code)]

use symtn_network::Network;

/// One operand's written labels, codomain then domain.
#[derive(Clone, Copy)]
pub struct Operand<'a> {
    codomain: &'a [&'a str],
    domain: &'a [&'a str],
    conj: bool,
}

/// The operand `t[codomain; domain]`.
pub fn op<'a>(codomain: &'a [&'a str], domain: &'a [&'a str]) -> Operand<'a> {
    Operand {
        codomain,
        domain,
        conj: false,
    }
}

/// The operand read through its adjoint, `conj(t)[codomain; domain]`, with
/// the labels written for the stored tensor's legs.
pub fn conj(operand: Operand<'_>) -> Operand<'_> {
    Operand {
        conj: true,
        ..operand
    }
}

/// The network `[output_codomain; output_domain] = operands[0] * ...`.
///
/// # Panics
///
/// When the labels are not a valid network (a label on one operand only and
/// not in the output, a label written twice on one operand, ...).
pub fn net(operands: &[Operand<'_>], output_codomain: &[&str], output_domain: &[&str]) -> Network {
    Network::new(
        operands
            .iter()
            .map(|operand| operand.codomain.iter().chain(operand.domain).copied()),
        operands.iter().map(|operand| operand.conj),
        operands.iter().map(|operand| operand.codomain.len()),
        output_codomain.iter().chain(output_domain).copied(),
        Some(output_codomain.len()),
    )
    .expect("valid test network")
}
