//! Contraction with a lazy adjoint (conjugated) operand on the left, on the
//! right, and on both sides is correct for non-self-dual sectors, where a
//! charge and its dual are distinct labels, so a dualization or conjugation
//! slip changes the result rather than hiding behind `c = c̄`.
//!
//! TensorKit `cfaa073e` `blas_contract!` (`src/tensors/tensoroperations.jl`)
//! contracts `A'` as `block(parent, c)'`; `block(t', c) = block(t, c)'`.
//! Oracles never pass through the contraction under test:
//!
//! - U(1), mixed-side contracted legs (both operands bend): the physical
//!   dense expansion, with `Array(t') = conj(Array(t))ᵀ` taken as the matrix
//!   conjugate transpose of the parent's expansion, then an index sum over
//!   the contracted legs. TensorKit 0.17.1
//!   `src/tensors/adjoint.jl:subblock(::AdjointTensorMap)` is
//!   `permutedims(conj(subblock(parent, (f2, f1))), (domainind..., codomainind...))`,
//!   and `convert(Array, ·)` multiplies each subblock by a unit U(1) tree.
//! - SU(3) (no physical basis is provided): the coupled-block identity
//!   `block(X ∘ Y, c) = block(X, c) · block(Y, c)` (TensorKit `mul!`), with
//!   each adjoint block written as the conjugate transpose of the parent's
//!   block. The leg holds `3` and `3̄` at different degeneracies plus `8`,
//!   whose `8 ⊗ 8 → 8` has outer multiplicity two.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{BlockFusionTrees, ContractSpec, GradedSpace, Runtime, TensorMap};

/// Distinct complex entries per `salt`, so conjugation is visible.
fn fill<S>(salt: usize) -> impl FnMut(&BlockFusionTrees<S>, &[usize]) -> Complex64 {
    let mut next = salt * 101;
    move |_, _| {
        next += 1;
        Complex64::new(
            ((next * 37) % 17) as f64 / 8.0 - 1.0,
            ((next * 11) % 13) as f64 / 16.0 - 0.375,
        )
    }
}

fn assert_close(what: &str, actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    assert!(
        expected.iter().any(|value| value.norm() > 0.5),
        "{what}: trivial oracle"
    );
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).norm() <= 1e-12 * (1.0 + e.norm()),
            "{what}[{index}]: {a} vs {e}"
        );
    }
}

mod u1 {
    use super::*;
    use symtn::typed::Engine;
    use symtn::typed::PhysicalDense;

    fn leg(charges: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
        GradedSpace::try_new(
            Arc::new(U1FusionRule),
            charges.iter().map(|&(q, n)| (U1Irrep::new(q), n)),
        )
        .unwrap()
    }

    /// `Array(t')` from `Array(t)`: the conjugate transpose of the
    /// `prod(codomain) × prod(domain)` column-major matrix, codomain and
    /// domain axes swapped.
    fn adjoint_dense(
        t: &PhysicalDense<Complex64>,
        codomain_rank: usize,
    ) -> PhysicalDense<Complex64> {
        let rows: usize = t.shape[..codomain_rank].iter().product();
        let cols: usize = t.shape[codomain_rank..].iter().product();
        let mut data = vec![Complex64::new(0.0, 0.0); rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[c + cols * r] = t.data[r + rows * c].conj();
            }
        }
        let shape = [&t.shape[codomain_rank..], &t.shape[..codomain_rank]].concat();
        PhysicalDense { shape, data }
    }

    fn strides(shape: &[usize]) -> Vec<usize> {
        let mut stride = 1;
        shape
            .iter()
            .map(|&n| {
                let s = stride;
                stride *= n;
                s
            })
            .collect()
    }

    /// Column-major multi-indices of `shape`.
    fn indices(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
        (0..shape.iter().product()).map(move |mut linear: usize| {
            shape
                .iter()
                .map(|&n| {
                    let i = linear % n;
                    linear /= n;
                    i
                })
                .collect()
        })
    }

    /// `Σ_k a[open_a, k] b[open_b, k]`, output axes `spec.codomain ++
    /// spec.domain` over the joint open legs `(open_a, open_b)`.
    fn contract_dense(
        a: &PhysicalDense<Complex64>,
        b: &PhysicalDense<Complex64>,
        spec: &ContractSpec<'_>,
    ) -> Vec<Complex64> {
        let (sa, sb) = (strides(&a.shape), strides(&b.shape));
        let open = |shape: &[usize], contracted: &[usize]| {
            (0..shape.len())
                .filter(|axis| !contracted.contains(axis))
                .collect::<Vec<_>>()
        };
        let (open_a, open_b) = (open(&a.shape, spec.lhs), open(&b.shape, spec.rhs));
        let joint: Vec<(usize, usize)> = open_a
            .iter()
            .map(|&axis| (a.shape[axis], sa[axis]))
            .chain(open_b.iter().map(|&axis| (b.shape[axis], 0)))
            .collect();
        let output_axes = [spec.codomain, spec.domain].concat();
        let output_shape: Vec<usize> = output_axes.iter().map(|&j| joint[j].0).collect();
        let output_strides = strides(&output_shape);
        let summed: Vec<usize> = spec.lhs.iter().map(|&axis| a.shape[axis]).collect();
        let joint_shape: Vec<usize> = joint.iter().map(|&(n, _)| n).collect();
        let mut out = vec![Complex64::new(0.0, 0.0); output_shape.iter().product()];
        for j in indices(&joint_shape) {
            let base_a: usize = open_a.iter().zip(&j).map(|(&x, &i)| i * sa[x]).sum();
            let base_b: usize = open_b
                .iter()
                .zip(&j[open_a.len()..])
                .map(|(&x, &i)| i * sb[x])
                .sum();
            let mut total = Complex64::new(0.0, 0.0);
            for k in indices(&summed) {
                let (mut ia, mut ib) = (base_a, base_b);
                for (p, (&xa, &xb)) in spec.lhs.iter().zip(spec.rhs).enumerate() {
                    ia += k[p] * sa[xa];
                    ib += k[p] * sb[xb];
                }
                total += a.data[ia] * b.data[ib];
            }
            let o: usize = output_axes
                .iter()
                .zip(&output_strides)
                .map(|(&axis, &s)| j[axis] * s)
                .sum();
            out[o] = total;
        }
        out
    }

    /// `A[W <- V, V]` against `B[V, V <- W]`: `A`'s domain leg 2 with `B`'s
    /// codomain leg 0 and `A`'s codomain leg 0 with `B`'s domain leg 2, so
    /// both operands bend a leg; the open legs come out swapped.
    #[test]
    fn u1_conjugated_contraction_matches_dense_expansion() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        // `-1` and `1` at different degeneracies: `V ≠ V'`.
        let v = leg(&[(-1, 2), (0, 1), (1, 3)]);
        let w = leg(&[(0, 1), (1, 2), (2, 1)]);
        let ket = |engine: &mut dyn Engine, salt| {
            TensorMap::from_subblock_fn(engine, &runtime, [&v, &v], [&w], fill(salt)).unwrap()
        };
        let bra = |engine: &mut dyn Engine, salt| {
            TensorMap::from_subblock_fn(engine, &runtime, [&w], [&v, &v], fill(salt)).unwrap()
        };
        let (p, q, s, r) = (
            ket(engine, 1),
            ket(engine, 2),
            bra(engine, 3),
            bra(engine, 4),
        );
        let spec = ContractSpec {
            lhs: &[2, 0],
            rhs: &[0, 2],
            codomain: &[1],
            domain: &[0],
        };
        let dense = |engine: &mut dyn Engine, t: &TensorMap<U1FusionRule, Complex64>| {
            t.to_physical_dense(engine).unwrap()
        };
        let (p_adj, r_adj) = (p.adjoint(engine).unwrap(), r.adjoint(engine).unwrap());
        let cases = [
            (
                "lhs adjoint",
                &p_adj,
                &q,
                adjoint_dense(&dense(engine, &p), 2),
                dense(engine, &q),
            ),
            (
                "rhs adjoint",
                &s,
                &r_adj,
                dense(engine, &s),
                adjoint_dense(&dense(engine, &r), 1),
            ),
            (
                "both adjoint",
                &p_adj,
                &r_adj,
                adjoint_dense(&dense(engine, &p), 2),
                adjoint_dense(&dense(engine, &r), 1),
            ),
        ];
        for (what, a, b, a_dense, b_dense) in cases {
            let actual = a
                .contract(engine, b, &spec)
                .unwrap()
                .to_physical_dense(engine)
                .unwrap();
            let expected = contract_dense(&a_dense, &b_dense, &spec);
            assert_eq!(actual.shape, [6, 6], "{what}: shape (dim V = 2 + 1 + 3)");
            assert_close(what, &actual.data, &expected);
        }
    }
}

#[cfg(feature = "cgc-gen")]
mod su3 {
    use super::*;
    use symtn::sector::LieGroupFusionRule;
    use symtn::typed::Engine;

    type Matrix = Vec<Vec<Complex64>>;

    fn block(t: &TensorMap<LieGroupFusionRule, Complex64>, c: &Vec<i64>) -> Matrix {
        let b = t.block(c).unwrap();
        (0..b.rows())
            .map(|i| (0..b.cols()).map(|j| b.get(i, j).unwrap()).collect())
            .collect()
    }

    fn adjoint(m: &Matrix) -> Matrix {
        let cols = m.first().map_or(0, Vec::len);
        (0..cols)
            .map(|j| m.iter().map(|row| row[j].conj()).collect())
            .collect()
    }

    fn product(x: &Matrix, y: &Matrix) -> Matrix {
        let cols = y.first().map_or(0, Vec::len);
        x.iter()
            .map(|row| {
                (0..cols)
                    .map(|j| row.iter().zip(y).map(|(&a, y_row)| a * y_row[j]).sum())
                    .collect()
            })
            .collect()
    }

    /// `A[V <- V, V]` composed with `B[V, V <- V]` through `contract`, so
    /// each coupled block of the result is one matrix product.
    #[test]
    fn su3_conjugated_contraction_matches_coupled_block_products() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let rule = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        );
        // Dynkin labels: `3 = [1, 0]` and `3̄ = [0, 1]` at different
        // degeneracies, so `V ≠ V'`; `8 = [1, 1]`.
        let fundamental = vec![1i64, 0];
        let v = GradedSpace::try_new(
            rule,
            [
                (vec![0i64, 0], 1),
                (fundamental.clone(), 2),
                (vec![0, 1], 1),
                (vec![1, 1], 2),
            ],
        )
        .unwrap();
        let ket = |engine: &mut dyn Engine, salt| {
            TensorMap::from_subblock_fn(engine, &runtime, [&v, &v], [&v], fill(salt)).unwrap()
        };
        let bra = |engine: &mut dyn Engine, salt| {
            TensorMap::from_subblock_fn(engine, &runtime, [&v], [&v, &v], fill(salt)).unwrap()
        };
        let (p, q, s, r) = (
            ket(engine, 1),
            ket(engine, 2),
            bra(engine, 3),
            bra(engine, 4),
        );
        let spec = ContractSpec {
            lhs: &[1, 2],
            rhs: &[0, 1],
            codomain: &[0],
            domain: &[1],
        };
        let (p_adj, r_adj) = (p.adjoint(engine).unwrap(), r.adjoint(engine).unwrap());
        type Oracle<'a> = &'a dyn Fn(&Vec<i64>) -> Matrix;
        let cases: [(&str, _, _, Oracle); 3] = [
            ("lhs adjoint", &p_adj, &q, &|c| {
                product(&adjoint(&block(&p, c)), &block(&q, c))
            }),
            ("rhs adjoint", &s, &r_adj, &|c| {
                product(&block(&s, c), &adjoint(&block(&r, c)))
            }),
            ("both adjoint", &p_adj, &r_adj, &|c| {
                product(&adjoint(&block(&p, c)), &adjoint(&block(&r, c)))
            }),
        ];
        for (what, a, b, oracle) in cases {
            let result = a.contract(engine, b, &spec).unwrap();
            let sectors: Vec<Vec<i64>> = p.blocks().unwrap().map(|(c, _)| c).collect();
            assert_eq!(result.blocks().unwrap().len(), sectors.len(), "{what}");
            assert!(sectors.contains(&fundamental), "{what}: 3 is coupled");
            let (mut actual, mut expected) = (Vec::new(), Vec::new());
            for c in &sectors {
                let (got, want) = (block(&result, c), oracle(c));
                assert_eq!(got.len(), want.len(), "{what} {c:?}: rows");
                actual.extend(got.concat());
                expected.extend(want.concat());
            }
            assert_close(what, &actual, &expected);
        }
    }
}
