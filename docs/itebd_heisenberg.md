# U(1) iTEBD for the Heisenberg Chain

This guide follows the runnable
[`itebd_heisenberg.rs`](../symtn-network/examples/itebd_heisenberg.rs) example. It shows
how symtn represents the physics and how one infinite time-evolving block
decimation (iTEBD) update is assembled. The algorithm lives in the example;
the small fragments below only connect its code to the equations.

## 1. Model and target energy

The example studies the infinite spin-1/2 antiferromagnetic Heisenberg chain
with coupling $J=1$,

$$
H=\sum_i h_{i,i+1}, \qquad
h_{i,i+1}=S_i^xS_{i+1}^x+S_i^yS_{i+1}^y+S_i^zS_{i+1}^z.
$$

Its exact ground-state energy per bond is
$e_0=1/4-\ln 2\simeq-0.44314718$. The example stores this comparison value in
[`E_EXACT`](../symtn-network/examples/itebd_heisenberg.rs). iTEBD tries to reach the
ground state by repeatedly applying imaginary-time evolution and truncating
the resulting bond.

## 2. Integer U(1) charges

The conserved quantity is $S^z$, but the physical leg uses the integer label

$$
q=2S^z.
$$

Therefore spin up has $q=+1$ and spin down has $q=-1$. This is only a
rescaling of the charge label; it does not change the spin operators. The
physical space is constructed in [`run`](../symtn-network/examples/itebd_heisenberg.rs).

symtn writes every tensor as `codomain <- domain`. For example,
`Gamma: [left, physical] <- [right]` has two codomain legs and one domain leg.
In the contractions below, every repeated label joins a codomain leg to a
domain leg made from the same `GradedSpace`. Those opposite orientations are
already the required dual pairing, so no explicit `try_dual()` call is needed.
Call `try_dual()` when an explicit dual space is needed, such as when
contracting two domain legs.

## 3. Two-site Hamiltonian blocks

Charge conservation makes $h$ block diagonal in
$q_{\mathrm{total}}=q_1+q_2$. The $q_{\mathrm{total}}=+2$ state
$\lvert\uparrow\uparrow\rangle$ and the $q_{\mathrm{total}}=-2$ state
$\lvert\downarrow\downarrow\rangle$ each form a scalar block with value
$1/4$. In the ordered charge-zero basis
$\{\lvert\uparrow\downarrow\rangle,\lvert\downarrow\uparrow\rangle\}$,

$$
h_{q=0}=\begin{pmatrix}-1/4&1/2\\1/2&-1/4\end{pmatrix}.
$$

The diagonal terms come from $S_i^zS_{i+1}^z$ and the off-diagonal $1/2$
terms flip the two spins. [`heisenberg_two_site`](../symtn-network/examples/itebd_heisenberg.rs)
builds exactly these allowed blocks with `TensorMap::from_subblock_fn`.

## 4. The two-site Vidal state

The infinite state has a repeating A-B unit cell in Vidal form,

$$
\cdots\lambda_b\Gamma_A\lambda_a\Gamma_B\lambda_b\cdots.
$$

Here each $\Gamma$ has shape `[left, physical] <- [right]`, while each
$\lambda$ is a diagonal map `bond <- bond` containing Schmidt weights. The two
different bond tensors are needed because an A-B bond and a B-A bond need not
be identical during the update. [`State`](../symtn-network/examples/itebd_heisenberg.rs)
stores these four tensors and constructs a charge-balanced entangled initial
state. The plus-sign state used there is not the two-spin singlet.

## 5. Imaginary-time gates and alternating bonds

For one time step $dt$, the local gate is

$$
G(dt)=\exp(-dt\,h).
$$

[`run`](../symtn-network/examples/itebd_heisenberg.rs) constructs this gate
with `h.scale(-dt)?.exp(&[0, 1], &[2, 3])`. [`State::step`](../symtn-network/examples/itebd_heisenberg.rs)
first updates the A-B bond and then the B-A bond. Repeating these two updates
projects the initial state toward low-energy states while preserving the
two-site unit cell.

## 6. Contract the local wavefunction

[`bond_update`](../symtn-network/examples/itebd_heisenberg.rs) absorbs the two
outer weights, the two-site tensors, the middle weight, and the gate into one
two-site tensor. This is the complete network from the example:

```rust
let theta = Network::new(
    vec![
        vec!["l", "x"],               // l_out[l; x]
        vec!["x", "qa", "y"],         // g1[x, qa; y]
        vec!["y", "z"],               // l_mid[y; z]
        vec!["z", "qb", "w"],         // g2[z, qb; w]
        vec!["w", "r"],               // l_out[w; r]
        vec!["pa", "pb", "qa", "qb"], // gate[pa, pb; qa, qb]
    ],
    vec![false; 6],
    vec![1, 2, 1, 2, 1, 2],
    vec!["l", "pa", "pb", "r"],
    Some(2),
)
.expect("fixed network labels are valid")
.contract(&[l_out, g1, l_mid, g2, l_out, gate])?;
```

Each operand's labels list its codomain legs then its domain legs, and the
matching entry of `[1, 2, 1, 2, 1, 2]` is how many of them are codomain legs;
`Network::contract` checks it against the tensor before planning. The output
`[l, pa; pb, r]` (two codomain legs) sets the bipartition for the following
SVD: the first site and left bond are separated from the second site and right
bond. Labels that appear twice are summed; labels that remain once become
output legs. `Network::contract` runs the network through the Runtime's plan
cache, so the repeated update reuses one contraction order.

## 7. Truncate the new bond

A truncated SVD is four primitive steps: `theta.svd_compact(&[0, 1], &[2, 3])` returns $U$,
$S$, $V^\dagger$; `diagview` reads the singular values off $S$;
`find_truncated` on the bond leg decides which to keep and reports the
discarded weight as `error`; and `restrict_leg` cuts the bond leg of $U$ and
$V^\dagger$, and both legs of $S$ in one call, to that selection.
[`run`](../symtn-network/examples/itebd_heisenberg.rs) combines a bond budget
`Truncation::rank(chi)` with `Truncation::relative_cutoff(rtol)`. The latter
keeps singular values satisfying
$\sigma_i \ge r_{\mathrm{tol}}\lVert\sigma\rVert_{2,w}$, where
$\lVert\sigma\rVert_{2,w}^2=\sum_{q,i}d_q\sigma_{q,i}^2$. `chi` bounds the kept
**quantum-dimension-weighted** bond dimension,
$\sum_q d_q n_q$. For U(1), every quantum dimension is $d_q=1$, so this is the
ordinary sum of the kept sector degeneracies.

That `error` is the absolute quantum-dimension-weighted 2-norm of the discarded
singular values for this one local `theta`. It is not a relative energy error.
The printed `max trunc err` is the largest such local error among the A-B and
B-A updates in one schedule stage; it is not an accumulated error and does not
by itself establish convergence.

## 8. Restore Vidal form

The SVD factors still include the outer $\lambda_{\mathrm{out}}$ weights. The
new tensors must divide those weights back out:

$$
\Gamma'_1=\lambda_{\mathrm{out}}^+U,\qquad
\Gamma'_2=V^\dagger\lambda_{\mathrm{out}}^+,
$$

where $+$ denotes the Moore-Penrose pseudo-inverse. The middle factor is
normalized and stored as the new $\lambda_{\mathrm{mid}}$. These operations
are the final part of [`bond_update`](../symtn-network/examples/itebd_heisenberg.rs).
The pseudo-inverse uses a different relative cutoff: it keeps values strictly
greater than `rcond` times the largest singular value across all sectors. A
tiny Schmidt value would otherwise produce a very large, unstable factor.
Directions at or below the cutoff are set to zero.

## 9. Energy, schedule, and convergence

[`bond_energy`](../symtn-network/examples/itebd_heisenberg.rs) evaluates the
normalized local expectation value
$\langle\theta\rvert h\lvert\theta\rangle/\langle\theta\vert\theta\rangle$.
[`State::energy`](../symtn-network/examples/itebd_heisenberg.rs) averages the
A-B and B-A bond energies. The schedule in [`main`](../symtn-network/examples/itebd_heisenberg.rs)
uses progressively smaller $dt$: large steps project quickly, while small
steps reduce imaginary-time discretization error.

Three diagnostics answer different questions:

- The truncation `error` measures the discarded quantum-dimension-weighted 2-norm in one
  local SVD.
- An energy plateau, or the change in energy between decreasing-$dt$ stages,
  measures whether this finite schedule has stabilized.
- `E/bond - (1/4 - ln 2)` compares the result with the exact ground-state
  energy.

A short run at fixed `chi` cannot separate finite-time-step error,
finite-entanglement error, and incomplete imaginary-time projection. For a
careful calculation, increase `chi`, extend each stage, reduce $dt$, and check
that the energy is stable under each change.

## 10. Run and read the example

Run the full application with the same pure-Rust CPU backend used by the
documented example:

```sh
cargo run --release -p symtn-network --example itebd_heisenberg --no-default-features --features cpu-faer
```

A stage line has this form (rates and times depend on the machine):

```text
dt = ... steps = ... chi = ... E/bond = ... (err vs exact ..., max trunc err ..., ... steps/s, ... s)
```

`dt` and `steps` identify the schedule stage. `chi` is the largest current
weighted bond dimension. `E/bond` is the A-B/B-A average, and `err vs exact`
is its signed difference from $1/4-\ln 2$. `max trunc err` has the local meaning
described above. `steps/s` and seconds are measurements of that run, not fixed
performance guarantees. The final lines repeat the computed and exact
energies, their signed difference, and total wall time.

Use the focused checks when changing the example or this guide:

```sh
# Fast smoke tests used by normal test runs.
cargo test -p symtn-network --test itebd_smoke

# Shorter release-mode convergence test; ignored by default.
cargo test -p symtn-network --release --test itebd_smoke -- --ignored itebd

# Compiling tutorial examples and warning-free crate documentation.
cargo test -p symtn --doc
RUSTDOCFLAGS="-D warnings" cargo doc -p symtn --no-deps
```

The smoke test checks the same physics with a small bond dimension and short
schedule. The ignored test is slower and checks a tighter energy tolerance.
Neither replaces convergence checks for a production calculation.
