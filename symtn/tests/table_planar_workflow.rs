//! The public planar workflow (#633) through `TableRule`: Fibonacci against
//! a TensorKit golden-chain oracle, and Ising against TensorKit's
//! `IsingAnyon` on gauge-invariant quantities plus the exact vertex gauge
//! between AnyonWiki's and TensorKit's Ising data.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};
use symtn_sectors::{
    CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec, TableRule, TableSector,
};

/// One fixed bound for the planar workflow. At most three unit-norm 2x2
/// transforms and two 2x2 products contribute to any checked coordinate, so
/// `256 * f64::EPSILON` leaves margin for the imported decimals and BLAS
/// arithmetic without making the bound provider-tunable.
const PLANAR_BOUND: f64 = 256.0 * f64::EPSILON;

type Matrix2 = [[Complex64; 2]; 2];

struct PlanarOracle {
    f: Matrix2,
    sigma1: Matrix2,
    sigma2: Matrix2,
    yang_baxter: Matrix2,
    hamiltonian: Matrix2,
}

fn matrix_product(lhs: Matrix2, rhs: Matrix2) -> Matrix2 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..2)
                .map(|inner| lhs[row][inner] * rhs[inner][column])
                .sum()
        })
    })
}

fn matrix_adjoint(matrix: Matrix2) -> Matrix2 {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row].conj()))
}

fn assert_matrix_close(actual: Matrix2, expected: Matrix2, what: &str) {
    for row in 0..2 {
        for column in 0..2 {
            let residual = (actual[row][column] - expected[row][column]).norm();
            assert!(
                residual <= PLANAR_BOUND,
                "{what}[{row},{column}] residual {residual:e}: {:?} != {:?}",
                actual[row][column],
                expected[row][column]
            );
        }
    }
}

fn coordinates<R>(state: &TensorMap<R, Complex64>, channels: [R::Sector; 2]) -> [Complex64; 2]
where
    R: MultiplicityFreeRigidSymbols<Scalar = Complex64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Copy,
{
    let mut values = [Complex64::new(0.0, 0.0); 2];
    for (trees, block) in state.subblocks().unwrap() {
        let slot = match trees.codomain_innerlines() {
            [channel] if *channel == channels[0] => 0,
            [channel] if *channel == channels[1] => 1,
            inner => panic!("unexpected fusion path {inner:?}"),
        };
        values[slot] = *block.get(&[0, 0, 0, 0]).unwrap();
    }
    values
}

fn sigma<R>(
    state: &TensorMap<R, Complex64>,
    generator: usize,
    inverse: bool,
) -> TensorMap<R, Complex64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = Complex64> + CheckedFusionAlgebra + SectorCodec,
{
    let codomain = if generator == 1 { [1, 0, 2] } else { [0, 2, 1] };
    // Source axes are [tau_1, tau_2, tau_3 | total tau]. Increasing levels
    // define positive Artin generators; reversing them selects the inverse.
    // Why not `permute`: an anyonic swap without source levels loses the
    // over/under convention and is not this braid word.
    let levels = if inverse { [3, 2, 1, 0] } else { [0, 1, 2, 3] };
    state.braid(&codomain, &[3], &levels).unwrap()
}

/// The workflow on `leg ⊗ leg ⊗ leg ← leg`, whose two states are the inner
/// channels `channels` of the codomain tree; `counts` are the stored lengths
/// of `leg^n ← leg` for `n = 2, 3, 4`.
fn planar_workflow<R>(
    provider: R,
    leg: R::Sector,
    channels: [R::Sector; 2],
    counts: [usize; 3],
) -> PlanarOracle
where
    R: MultiplicityFreeRigidSymbols<Scalar = Complex64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Copy,
{
    let runtime = Runtime::builder().build().unwrap();
    let tau = GradedSpace::try_new(Arc::new(provider), [(leg, 1)]).unwrap();
    let dual_tau = tau.try_dual().unwrap();

    for (rank, expected) in [2, 3, 4].into_iter().zip(counts) {
        let legs = vec![&tau; rank];
        let space: TensorMap<_, Complex64> = TensorMap::zeros(&runtime, legs, [&tau]).unwrap();
        assert_eq!(
            space.dense_data().unwrap().len(),
            expected,
            "all-leg rank {rank}"
        );
    }

    let basis: [TensorMap<_, Complex64>; 2] = channels.map(|channel| {
        TensorMap::from_subblock_fn(&runtime, [&tau, &tau, &tau], [&tau], |trees, _| {
            Complex64::new((trees.codomain_innerlines() == [channel]) as u8 as f64, 0.0)
        })
        .unwrap()
    });
    let ordinary = basis[0]
        .contract(
            &basis[0],
            &ContractSpec {
                lhs: &[3],
                rhs: &[0],
                codomain: &[0, 1, 2],
                domain: &[3, 4, 5],
            },
        )
        .unwrap_err();
    assert!(matches!(
        ordinary,
        symtn::typed::Error::Operation(operation)
            if matches!(*operation, symtn::typed::OperationError::UnsupportedTensorContractScope { .. })
    ));
    let dual_basis: [TensorMap<_, Complex64>; 2] = channels.map(|channel| {
        TensorMap::from_subblock_fn(
            &runtime,
            [&dual_tau, &dual_tau, &dual_tau],
            [&dual_tau],
            |trees, _| Complex64::new((trees.codomain_innerlines() == [channel]) as u8 as f64, 0.0),
        )
        .unwrap()
    });

    // `transpose` is the crossing-free boundary move. Its two basis columns
    // reconstruct F; applying the same planar move again is its inverse.
    let f = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let transformed = basis[column].transpose(&[3], &[2, 1, 0]).unwrap();
            assert_eq!(transformed.dense_data().unwrap().len(), 2);
            let roundtrip = transformed.transpose(&[3, 2, 1], &[0]).unwrap();
            for (slot, value) in coordinates(&roundtrip, channels).into_iter().enumerate() {
                let expected = Complex64::new((slot == column) as u8 as f64, 0.0);
                assert!((value - expected).norm() <= PLANAR_BOUND);
            }
            // Exact-boundary composition only: (tau <- tau^3) o
            // (tau^3 <- tau). Why not cross providers: their RuleIdentity
            // values intentionally differ, so each workflow stays bound to
            // its own provider allocation.
            transformed
                .compose(&dual_basis[row])
                .unwrap()
                .dense_data()
                .unwrap()[0]
        })
    });

    let generator_matrix = |generator| {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                coordinates(&sigma(&basis[column], generator, false), channels)[row]
            })
        })
    };
    let inverse_matrix = |generator| {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                coordinates(&sigma(&basis[column], generator, true), channels)[row]
            })
        })
    };
    let sigma1 = generator_matrix(1);
    let sigma2 = generator_matrix(2);
    let identity = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    for (generator, matrix) in [(1, sigma1), (2, sigma2)] {
        let inverse = inverse_matrix(generator);
        assert_matrix_close(inverse, matrix_adjoint(matrix), "braid inverse");
        assert_matrix_close(
            matrix_product(inverse, matrix),
            identity,
            "generator unitarity",
        );
        assert_matrix_close(
            matrix_product(matrix, inverse),
            identity,
            "generator inverse",
        );
    }

    let braid_word = |word: [usize; 3]| {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                let state = word
                    .into_iter()
                    .fold(basis[column].clone(), |state, generator| {
                        sigma(&state, generator, false)
                    });
                coordinates(&state, channels)[row]
            })
        })
    };
    let yang_baxter = braid_word([1, 2, 1]);
    assert_matrix_close(yang_baxter, braid_word([2, 1, 2]), "Yang--Baxter");

    let projector = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let second_projector = matrix_product(matrix_product(f, projector), matrix_adjoint(f));
    let hamiltonian = std::array::from_fn(|row| {
        std::array::from_fn(|column| -(projector[row][column] + second_projector[row][column]))
    });

    PlanarOracle {
        f,
        sigma1,
        sigma2,
        yang_baxter,
        hamiltonian,
    }
}

#[test]
fn fibonacci_public_planar_workflow_matches_tensorkit() {
    let fibonacci = planar_workflow(
        TableRule::fibonacci(),
        TableSector::new(2),
        [TableSector::new(1), TableSector::new(2)],
        [1, 2, 3],
    );

    // TensorKit 0.17 at cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91 with
    // TensorKitSectors + CategoryData, pinned in the
    // vacuum-first left-associated basis [(tau tau)->1, (tau tau)->tau],
    // source order [tau_1,tau_2,tau_3 | total tau], and the TensorKitSectors
    // `FibonacciAnyon` gauge written out in
    // `symtn-sectors/tests/fibonacci_providers.rs`. The braid oracle uses the
    // positive words sigma_1 sigma_2 sigma_1 and sigma_2 sigma_1 sigma_2 with
    // increasing source levels [0,1,2,3] (reversed for the inverse). This is
    // H = -(P_12 + F P_12 F^dagger), not a non-planar ordinary contraction.
    let tensorkit_hamiltonian = [
        [
            Complex64::new(-1.381966011250105, 0.0),
            Complex64::new(-0.48586827175664565, 0.0),
        ],
        [
            Complex64::new(-0.48586827175664565, 0.0),
            Complex64::new(-0.6180339887498948, 0.0),
        ],
    ];
    assert_matrix_close(fibonacci.hamiltonian, tensorkit_hamiltonian, "golden chain");
}

/// TensorKit 0.17.1 / TensorKitSectors 0.3.9 `IsingAnyon`
/// (`src/anyons.jl`: `Fsymbol`, `Rsymbol`), from
/// `fixtures/tensorkit_table_planar.jl`.
const TENSORKIT_ISING: &str = include_str!("fixtures/tensorkit-0.17.1-table-planar.txt");

fn tensorkit_ising(tag: &str) -> Matrix2 {
    let mut matrix = [[Complex64::new(f64::NAN, f64::NAN); 2]; 2];
    let mut seen = 0;
    for fields in TENSORKIT_ISING
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .filter(|fields| fields[0] == tag && fields.len() == 5)
    {
        let (row, column): (usize, usize) =
            (fields[1].parse().unwrap(), fields[2].parse().unwrap());
        matrix[row][column] =
            Complex64::new(fields[3].parse().unwrap(), fields[4].parse().unwrap());
        seen += 1;
    }
    assert_eq!(seen, 4, "fixture rows for {tag}");
    matrix
}

fn hamiltonian(f: Matrix2) -> Matrix2 {
    let projector = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let second = matrix_product(matrix_product(f, projector), matrix_adjoint(f));
    std::array::from_fn(|row| {
        std::array::from_fn(|column| -(projector[row][column] + second[row][column]))
    })
}

/// The eigenvalues of a 2x2 matrix, ordered by (re, im).
fn spectrum(m: Matrix2) -> [Complex64; 2] {
    let half_trace = (m[0][0] + m[1][1]) / 2.0;
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let root = (half_trace * half_trace - det).sqrt();
    let mut values = [half_trace - root, half_trace + root];
    values.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    values
}

fn assert_spectrum_close(actual: Matrix2, expected: Matrix2, what: &str) {
    for (a, e) in spectrum(actual).into_iter().zip(spectrum(expected)) {
        // Eigenvalues of a nondegenerate unitary 2x2 matrix are
        // well conditioned; the square root halves the available digits only
        // at a degeneracy, which none of these spectra has.
        assert!(
            (a - e).norm() <= PLANAR_BOUND,
            "{what} eigenvalue {a} != {e}"
        );
    }
}

/// `D M D` for the vertex gauge `D = diag(u^{σσ}_𝟙, u^{σσ}_ψ)`.
fn gauge(m: Matrix2, d: [f64; 2]) -> Matrix2 {
    std::array::from_fn(|row| std::array::from_fn(|column| m[row][column] * d[row] * d[column]))
}

fn max_residual(a: Matrix2, b: Matrix2) -> f64 {
    (0..4)
        .map(|i| (a[i / 2][i % 2] - b[i / 2][i % 2]).norm())
        .fold(0.0, f64::max)
}

#[test]
fn ising_public_planar_workflow_matches_tensorkit_isinganyon() {
    // AnyonWiki order (𝟙, ψ, σ); the states of σσσ ← σ are the inner
    // channels (σσ) → 𝟙 and (σσ) → ψ, TensorKit's basis order too.
    let ising = planar_workflow(
        TableRule::ising(),
        TableSector::new(3),
        [TableSector::new(1), TableSector::new(2)],
        [0, 2, 0],
    );
    let (sigma1, sigma2, f) = (
        tensorkit_ising("sigma1"),
        tensorkit_ising("sigma2"),
        tensorkit_ising("f"),
    );

    // Gauge-invariant: braid-generator and golden-chain spectra, and the
    // trace of the braid word σ₁σ₂σ₁.
    assert_spectrum_close(ising.sigma1, sigma1, "sigma1");
    assert_spectrum_close(ising.sigma2, sigma2, "sigma2");
    assert_spectrum_close(ising.hamiltonian, hamiltonian(f), "H");
    let word = matrix_product(matrix_product(sigma1, sigma2), sigma1);
    let trace = |m: Matrix2| m[0][0] + m[1][1];
    assert!((trace(ising.yang_baxter) - trace(word)).norm() <= PLANAR_BOUND);
    for (generator, tensorkit) in [(1, sigma1), (2, sigma2)] {
        let inverse = tensorkit_ising(&format!("sigma_inv{generator}"));
        assert_matrix_close(
            matrix_product(inverse, tensorkit),
            [
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            ],
            "TensorKit generator inverse",
        );
    }

    // The exact gauge witness: AnyonWiki's Ising is TensorKit's under the
    // vertex gauge u^{σσ}_ψ = −1, which fixes σ₁ (diagonal R) and conjugates
    // σ₂ and the planar F matrix by D = diag(1, −1).
    let witness = [1.0, -1.0];
    assert_matrix_close(
        ising.sigma1,
        gauge(sigma1, witness),
        "sigma1 under the witness",
    );
    assert_matrix_close(
        ising.sigma2,
        gauge(sigma2, witness),
        "sigma2 under the witness",
    );
    assert_matrix_close(ising.f, gauge(f, witness), "F under the witness");
    // Negative control: without the gauge the F data disagree, so the
    // workflow ran in AnyonWiki's gauge, not TensorKit's.
    for (actual, tensorkit) in [(ising.sigma2, sigma2), (ising.f, f)] {
        assert!(max_residual(actual, tensorkit) > 0.5);
    }
}

fn tensorkit_semion(tag: &str) -> Complex64 {
    let fields: Vec<&str> = TENSORKIT_ISING
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .find(|fields| fields.first() == Some(&tag))
        .unwrap_or_else(|| panic!("fixture is missing {tag}"));
    Complex64::new(fields[1].parse().unwrap(), fields[2].parse().unwrap())
}

/// The semion `[2,1,0,1,1,1,1]` is pointed, so `TableRule` reports `Unique`
/// fusion with a nontrivial associator `F^{sss}_s = −1` and `R^{ss}_𝟙 = i`.
/// Hand oracle on the one state of `s ⊗ s ⊗ s ← s`: the braid generators act
/// by `R^{ss}_𝟙 = i` (σ₂ = F† R F = R, `|F| = 1`) and the inverses by `−i`;
/// the crossing-free transpose then compose is `F^{sss}_s[𝟙, 𝟙]` times the
/// Frobenius–Schur phase `κ_s = −1` of the bend, `+1`. An engine that took
/// `F = 1` on the Unique path but kept `κ_s` from the table would give `−1`.
/// TensorKit's `Z2Element{1}`, the same symbols, agrees on every value.
#[test]
fn semion_unique_fusion_uses_its_associator() {
    use symtn_sectors::{FusionRule, FusionStyleKind};
    let rule = TableRule::by_uuid("37fa4d16-97c6-11f1-9c13-c70bad3f9e94").unwrap();
    assert_eq!(rule.fusion_style(), FusionStyleKind::Unique);
    let runtime = Runtime::builder().build().unwrap();
    let s = GradedSpace::try_new(Arc::new(rule), [(TableSector::new(2), 1)]).unwrap();
    let dual_s = s.try_dual().unwrap();
    let ones = |leg: &GradedSpace<TableRule>| -> TensorMap<TableRule, Complex64> {
        TensorMap::from_subblock_fn(&runtime, [leg, leg, leg], [leg], |_, _| {
            Complex64::new(1.0, 0.0)
        })
        .unwrap()
    };
    let (state, dual_state) = (ones(&s), ones(&dual_s));
    assert_eq!(state.dense_data().unwrap().len(), 1);

    let value = |t: &TensorMap<TableRule, Complex64>| t.dense_data().unwrap()[0];
    let close = |what: &str, actual: Complex64, hand: Complex64| {
        assert!(
            (actual - hand).norm() <= PLANAR_BOUND,
            "{what}: {actual} != {hand}"
        );
        let tensorkit = tensorkit_semion(what);
        assert!(
            (actual - tensorkit).norm() <= PLANAR_BOUND,
            "{what}: {actual} != TensorKit {tensorkit}"
        );
    };
    let i = Complex64::new(0.0, 1.0);
    for generator in [1, 2] {
        close(
            &format!("semion_sigma{generator}"),
            value(&sigma(&state, generator, false)),
            i,
        );
        close(
            &format!("semion_sigma_inv{generator}"),
            value(&sigma(&state, generator, true)),
            -i,
        );
    }
    let f = state
        .transpose(&[3], &[2, 1, 0])
        .unwrap()
        .compose(&dual_state)
        .unwrap();
    close("semion_f", value(&f), Complex64::new(1.0, 0.0));
}
