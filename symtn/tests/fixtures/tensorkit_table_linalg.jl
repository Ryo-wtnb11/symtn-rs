# Offline generator for tensorkit-0.17.1-table-linalg.txt. Nothing in the
# shipped crates runs this; it is a manual step.
#
# Reproduce in a scratch project (NOT this directory):
#
#   julia --project=/tmp/fixture -e 'using Pkg; Pkg.add(name="TensorKit", version="0.17.1"); Pkg.add(name="TensorKitSectors", version="0.3.9")'
#   julia --project=/tmp/fixture path/to/tensorkit_table_linalg.jl > path/to/tensorkit-0.17.1-table-linalg.txt
#
# TensorKit's own factorizations, truncation and planar contraction on
# Fibonacci and Ising tensors whose every entry is a function of its
# fusion-tree pair and degeneracy index alone (`entry`, the FNV-1a hash
# below; `support/table_fixture.rs` computes the same function). Sectors
# are written with AnyonWiki's labels, which `TableRule` uses: Fibonacci
# 1 = I, 2 = τ; Ising 1 = I, 2 = ψ, 3 = σ.
#
# AnyonWiki's Ising gauge differs from TensorKit's by the vertex sign
# u^{σσ}_ψ = -1 (#2277). Each Ising entry is therefore multiplied by
# `gauge_sign` of its two trees (-1 per (σσ)→ψ vertex), so the TensorKit
# tensor is the same morphism as the symtn one and every planar result is
# comparable; the `planar_ungauged` rows repeat the planar contraction without
# that sign, as the negative control. Blockwise spectra are invariant under
# the sign anyway (it is a diagonal ±1 on rows and columns).
#
# The golden-chain (`chain` rows, Fibonacci τ^⊗L) and Ising σ-chain rows are
# the spectra of H = -Σ_i P^{(1)}_{i,i+1}, built with `⊗` and `+` from the
# vacuum-channel projector P^{(1)} on two sites.
#
# Rows: category quantity sector values... (AnyonWiki sector label, or `-`
# for a scalar; `eig` values are re im pairs). Values use Julia's shortest
# round-trip `repr`.
using Pkg
using TensorKit

dep(uuid) = Pkg.dependencies()[Base.UUID(uuid)]
for (name, uuid) in (("TensorKit", "07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"),
                     ("TensorKitSectors", "13a9c161-d5da-41f0-bcbd-e1a08ae0647f"),
                     ("MatrixAlgebraKit", "6c742aac-3347-4629-af66-fc926824e5e4"))
    d = dep(uuid)
    println("# ", name, " v", d.version, " tree ", d.tree_hash, " (", basename(d.source), ")")
end
println("# rows: category quantity sector values...")

function fnv1a(s::AbstractString)
    h = 0xcbf29ce484222325
    for b in codeunits(s)
        h = (h ⊻ UInt64(b)) * 0x00000100000001b3
    end
    return h
end
unit_interval(h::UInt64) = Float64(h >> 11) / 2.0^53 - 0.5

join_labels(label, xs) = join((label(x) for x in xs), ",")

function entry(label, tag, f₁, f₂, index)
    key = string(tag, "|", label(f₁.coupled), "|",
                 join_labels(label, f₁.uncoupled), "|", join_labels(label, f₁.innerlines), "|",
                 join_labels(label, f₂.uncoupled), "|", join_labels(label, f₂.innerlines), "|",
                 join((i - 1 for i in Tuple(index)), ","))
    return complex(unit_interval(fnv1a(key * "|re")), unit_interval(fnv1a(key * "|im")))
end

# The vertices of a tree: (u₁, u₂) → i₁, (i₁, u₃) → i₂, …, (·, u_N) → coupled.
function vertices(f)
    n = length(f.uncoupled)
    n < 2 && return ()
    outs = (f.innerlines..., f.coupled)
    ins = (f.uncoupled[1], f.innerlines...)
    return ((ins[k], f.uncoupled[k + 1], outs[k]) for k in 1:(n - 1))
end

const σ = IsingAnyon(:σ)
const ψ = IsingAnyon(:ψ)
gauge_sign(f) = prod((a == σ && b == σ && c == ψ ? -1 : 1 for (a, b, c) in vertices(f)); init = 1)
no_sign(f) = 1

function filled(label, sign, tag, space)
    t = zeros(ComplexF64, space)
    for (f₁, f₂) in fusiontrees(t)
        block = t[f₁, f₂]
        s = sign(f₁) * sign(f₂)
        for index in CartesianIndices(block)
            block[index] = s * entry(label, tag, f₁, f₂, index)
        end
    end
    return t
end

fmt(x) = repr(x)
emit(category, quantity, sector, values) =
    println(category, " ", quantity, " ", sector, " ", join((fmt(v) for v in values), " "))
emit_complex(category, quantity, sector, values) =
    emit(category, quantity, sector, Iterators.flatten((real(z), imag(z)) for z in values))

function emit_spectra(category, label, quantity, spectra; complex = false, sort_by = identity)
    for c in sort(collect(keys(spectra)); by = label)
        values = sort(collect(spectra[c]); by = sort_by)
        complex ? emit_complex(category, quantity, label(c), values) :
        emit(category, quantity, label(c), values)
    end
end

function emit_dims(category, label, quantity, space)
    for c in sort(collect(sectors(space)); by = label)
        emit(category, quantity, label(c), (dim(space, c),))
    end
end

lexicographic(z) = (real(z), imag(z))

function report(category, label, sign, V)
    t = filled(label, sign, "t", V ⊗ V ← V ⊗ V)
    emit_spectra(category, label, "svd", svd_vals(t); sort_by = -)
    emit_spectra(category, label, "eigh", eigh_vals(t + t'))
    emit_spectra(category, label, "eig", eig_vals(t); complex = true, sort_by = lexicographic)

    # `truncerror` takes `atol`: this TensorKit's `SectorVector` method scales
    # `rtol^p` by `norm(values, p)` rather than MatrixAlgebraKit's `norm^p`
    # (`src/factorizations/truncation.jl:findtruncated`), so only the
    # absolute budget is the same policy in both libraries.
    for (tag, trunc) in (("rank7", truncrank(7)), ("tol", trunctol(; atol = 1.0)),
                         ("discard", truncerror(; atol = 1.5)))
        U, S, Vh, ϵ = svd_trunc(t; trunc)
        emit_dims(category, label, "trunc_" * tag, space(S, 1))
        emit(category, "trunc_" * tag * "_error", "-", (ϵ,))
        emit(category, "trunc_" * tag * "_norm", "-", (norm(U * S * Vh),))
    end

    x = filled(label, sign, "x", V ⊗ V ← V)
    emit_dims(category, label, "left_null", domain(left_null(x))[1])
    y = filled(label, sign, "y", V ← V ⊗ V)
    emit_dims(category, label, "right_null", codomain(right_null(y))[1])

    # The planar contraction C[b d; h f] = A[a b; c d] B[c f; a h], as
    # `planarcontract!` runs it: transpose A to ((2,4),(1,3)), B to
    # ((3,1),(4,2)), then compose.
    for (quantity, s) in (("planar", sign), ("planar_ungauged", no_sign))
        A = filled(label, s, "a", V ⊗ V ← V ⊗ V)
        B = filled(label, s, "b", V ⊗ V ← V ⊗ V)
        C = transpose(A, ((2, 4), (1, 3))) * transpose(B, ((3, 1), (4, 2)))
        @planar C′[b d; h f] := A[a b; c d] * B[c f; a h]
        @assert C ≈ C′
        emit_spectra(category, label, quantity * "_svd", svd_vals(C); sort_by = -)
        quantity == "planar" || continue
        emit_spectra(category, label, quantity * "_eigh", eigh_vals(C + C'))
        emit_complex(category, quantity * "_tr", "-", (tr(C),))
        emit(category, quantity * "_norm", "-", (norm(C),))
    end
end

function chain(category, label, W, Ls)
    P = zeros(ComplexF64, W ⊗ W ← W ⊗ W)
    block(P, one(sectortype(W))) .= 1
    for L in Ls
        H = zeros(ComplexF64, W^L ← W^L)
        for i in 1:(L - 1)
            term = P
            i > 1 && (term = id(ComplexF64, W^(i - 1)) ⊗ term)
            i + 1 < L && (term = term ⊗ id(ComplexF64, W^(L - i - 1)))
            H -= term
        end
        emit_spectra(category, label, "chain" * string(L), eigh_vals(H))
    end
end

# `planar_contract` (#2284): TensorKit `@planar` binary contractions, each
# also asserted equal to `planarcontract!`'s explicit steps where the step
# order is fixed. Every result is reported by `norm` and by `dot(P, C)` with a
# probe `P` filled from the same entry hash ("p" ++ geometry) on C's space;
# both are invariant under nothing but the recorded witness sign.
function emit_planar(category, quantity, label, sign, C)
    P = filled(label, sign, "p" * quantity, space(C))
    emit(category, "planar_" * quantity * "_norm", "-", (norm(C),))
    emit_complex(category, "planar_" * quantity * "_inner", "-", (dot(P, C),))
end

planar_rejects(ex) =
    try
        eval(ex)
        false
    catch err
        err isa LoadError && err.error isa ArgumentError
    end

function planar_report(category, label, sign, V)
    T(tag, cod, dom) = filled(label, sign, tag, cod ← dom)
    # G1: the composition geometry, `A * B`.
    A = T("a1", V ⊗ V, V ⊗ V); B = T("b1", V ⊗ V, V ⊗ V)
    @planar C1[a b; e f] := A[a b; c d] * B[c d; e f]
    @assert C1 ≈ A * B
    emit_planar(category, "g1", label, sign, C1)
    # G2: both operands transposed (`planarcontract!` + nothing rotated).
    A = T("a2", V ⊗ V, V ⊗ V); B = T("b2", V ⊗ V, V ⊗ V)
    @planar C2[b d; h f] := A[a b; c d] * B[c f; a h]
    @assert C2 ≈ transpose(A, ((2, 4), (1, 3))) * transpose(B, ((3, 1), (4, 2)))
    emit_planar(category, "g2", label, sign, C2)
    # G3: the golden-chain H_eff step with a rotated output.
    H = T("h3", V ⊗ V, V ⊗ V); θ = T("t3", V ⊗ V ⊗ V, V)
    @planar C3[l s1 s2; r] := H[s1 s2; t1 t2] * θ[l t1 t2; r]
    @assert C3 ≈ transpose(H * transpose(θ, ((2, 3), (1, 4))), ((3, 1, 2), (4,)))
    emit_planar(category, "g3", label, sign, C3)
    # G4: a full contraction to a scalar.
    A = T("a4", V ⊗ V, V ⊗ V); B = T("b4", V ⊗ V, V ⊗ V)
    s4 = @planar A[a b; c d] * B[c d; a b]
    emit_complex(category, "planar_g4_scalar", "-", (s4,))
    # G5: nothing contracted, the output a rotation of `A`'s boundary.
    A = T("a5", V, V); B = T("b5", V, V)
    @planar C5[a c; b d] := A[a; b] * B[c; d]
    emit_planar(category, "g5", label, sign, C5)
    # G6: a conjugated operand, read as the adjoint.
    θ = T("t6", V ⊗ V ⊗ V, V)
    @planar C6[a; b] := θ[l s1 s2; a] * conj(θ[l s1 s2; b])
    @assert C6 ≈ transpose(θ' * θ, ((2,), (1,)))
    emit_planar(category, "g6", label, sign, C6)
    # `planarcontract!` into an existing destination, complex α and β ≠ 0:
    # G1 directly, G3 as `@planar` then `α C + β D`.
    α, β = 0.5 - 0.25im, 1.5 + 0.75im
    A = T("a1", V ⊗ V, V ⊗ V); B = T("b1", V ⊗ V, V ⊗ V)
    D1 = T("d1", V ⊗ V, V ⊗ V)
    TensorKit.planarcontract!(D1, A, ((1, 2), (3, 4)), B, ((1, 2), (3, 4)), ((1, 2), (3, 4)), α, β)
    emit_planar(category, "into_g1", label, sign, D1)
    D3 = filled(label, sign, "d3", space(C3))
    emit_planar(category, "into_g3", label, sign, α * C3 + β * D3)
    # Negative controls: `@planar` rejects each at expansion.
    A = T("a2", V ⊗ V, V ⊗ V); B = T("b2", V ⊗ V, V ⊗ V)
    for (name, ex) in (
            ("same_orientation", :(@planar X[b d; f h] := $A[a b; c d] * $B[a f; c h])),
            ("not_an_arc", :(@planar X[b c; e f] := $A[a b; c d] * $B[a d; e f])),
            ("not_a_rotation", :(@planar X[d b; h f] := $A[a b; c d] * $B[c f; a h])),
        )
        emit(category, "planar_reject_" * name, "-", (planar_rejects(ex) ? 1 : 0,))
    end
end

# The fermionic planar contraction is the twist-free one (#2284 §1.4): a
# fℤ₂ row in the cyclic geometry of the design probe, whose `@tensor`
# differs. Parity labels 0, 1.
function fermion_report()
    label(a) = a.isodd ? 1 : 0
    V = Vect[FermionParity](0 => 2, 1 => 2)
    X = filled(label, no_sign, "fx", V ⊗ V ← V ⊗ V)
    Y = filled(label, no_sign, "fy", V ⊗ V ← V ⊗ V)
    @planar E[a z; c x] := X[a b; c d] * Y[x d; z b]
    @tensor F[a z; c x] := X[a b; c d] * Y[x d; z b]
    @assert norm(E - F) > 1.0e-3 * norm(E)
    emit_planar("fz2", "cyclic", label, no_sign, E)
    emit("fz2", "tensor_cyclic_norm", "-", (norm(F),))
end

const fib_labels = Dict(FibonacciAnyon(:I) => 1, FibonacciAnyon(:τ) => 2)
fib_label(a) = fib_labels[a]
report("fibonacci", fib_label, no_sign,
       Vect[FibonacciAnyon](FibonacciAnyon(:I) => 1, FibonacciAnyon(:τ) => 2))
chain("fibonacci", fib_label, Vect[FibonacciAnyon](FibonacciAnyon(:τ) => 1), 4:8)

const ising_labels = Dict(IsingAnyon(:I) => 1, ψ => 2, σ => 3)
ising_label(a) = ising_labels[a]
report("ising", ising_label, gauge_sign, Vect[IsingAnyon](IsingAnyon(:I) => 1, ψ => 1, σ => 2))
chain("ising", ising_label, Vect[IsingAnyon](σ => 1), 4:8)
planar_report("fibonacci", fib_label, no_sign,
              Vect[FibonacciAnyon](FibonacciAnyon(:I) => 1, FibonacciAnyon(:τ) => 2))
planar_report("ising", ising_label, gauge_sign, Vect[IsingAnyon](IsingAnyon(:I) => 1, ψ => 1, σ => 2))
fermion_report()
