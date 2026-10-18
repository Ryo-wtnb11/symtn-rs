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
# fusion-tree pair and degeneracy index alone (`entry`, the hash of
# `tensorkit_table_reductions.jl`; `table_factorizations.rs` computes the same
# function). Sectors are written with AnyonWiki's labels, which `TableRule`
# uses: Fibonacci 1 = I, 2 = τ; Ising 1 = I, 2 = ψ, 3 = σ.
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

const fib_labels = Dict(FibonacciAnyon(:I) => 1, FibonacciAnyon(:τ) => 2)
fib_label(a) = fib_labels[a]
report("fibonacci", fib_label, no_sign,
       Vect[FibonacciAnyon](FibonacciAnyon(:I) => 1, FibonacciAnyon(:τ) => 2))
chain("fibonacci", fib_label, Vect[FibonacciAnyon](FibonacciAnyon(:τ) => 1), 4:8)

const ising_labels = Dict(IsingAnyon(:I) => 1, ψ => 2, σ => 3)
ising_label(a) = ising_labels[a]
report("ising", ising_label, gauge_sign, Vect[IsingAnyon](IsingAnyon(:I) => 1, ψ => 1, σ => 2))
chain("ising", ising_label, Vect[IsingAnyon](σ => 1), 4:8)
