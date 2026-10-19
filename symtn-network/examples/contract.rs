//! Compose diagonal U(1) maps with index notation and an explicit Network.

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};
use symtn_network::{contract, Network};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::builder().build()?;
    let space = GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )?;

    // Fill each allowed charge block. Unequal indices stay zero, so both maps
    // are diagonal.
    let a: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&space], [&space], |trees, indices| {
            if indices[0] == indices[1] {
                f64::from(2 + trees.coupled().charge())
            } else {
                0.0
            }
        })?;
    let b: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&space], [&space], |trees, indices| {
            if indices[0] == indices[1] {
                f64::from(2 - trees.coupled().charge())
            } else {
                0.0
            }
        })?;

    let c = contract!([i; k] := a[i; j] * b[j; k])?;
    assert_eq!((c.codomain_rank(), c.domain_rank()), (1, 1));
    // A tensor's inner product with itself is its squared norm.
    let inner = c.inner(&c)?;
    assert!((inner - 50.0).abs() < 1e-12);
    let network = Network::new(
        [["i", "j"], ["j", "k"]],
        [false, false],
        [1, 1],
        ["i", "k"],
        Some(1),
    )?;
    let same = network.contract(&[&a, &b])?;
    assert!(same.axpby(1.0, &c, -1.0)?.norm(2.0)? < 1e-12);

    println!(
        "result: codomain rank = {}, domain rank = {}, squared norm = {inner}",
        c.codomain_rank(),
        c.domain_rank()
    );
    Ok(())
}
