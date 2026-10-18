# TensorKit `convert(Array, t)` fixtures for `symtn/tests/physical_dense.rs`
# (issue #1464).
#
# Every symmetry-allowed reduced element is set by `value` below, which
# `physical_dense.rs::fixture_value` mirrors. The output lists each nonzero
# dense entry as `linear_index re im` (column-major, zero-based) after a
# `# name shape...` header.
#
# Run (Julia 1.11.6, pinned TensorKit main `f87ca7f` = 0.17.1):
#   julia --project=<environment with TensorKit 0.17.1> \
#     benchmarks/tensorkit_physical_dense_oracle.jl \
#     > symtn/tests/fixtures/physical_dense/tensorkit_dense.txt

using TensorKit
using Printf

label(c::U1Irrep) = Int(c.charge)
label(c::SU2Irrep) = Int(2 * c.j)

function value(f₁, f₂, index, complexvalued)
    labels = label.((f₁.uncoupled..., f₂.uncoupled...))
    inner = sum(label.(f₁.innerlines); init = 0) + sum(label.(f₂.innerlines); init = 0)
    v = 1.0 + sum(k * (labels[k] + 3) / 7 for k in eachindex(labels)) +
        sum((index[k] - 1) * (k + 1) / 3 for k in eachindex(labels)) +
        label(f₁.coupled) / 5 + inner / 11
    return complexvalued ? complex(v, v / 2 - sum(labels) / 13) : v
end

function fixture(T, space)
    t = zeros(T, space)
    for (f₁, f₂) in fusiontrees(t)
        block = t[f₁, f₂]
        for index in CartesianIndices(block)
            block[index] = value(f₁, f₂, Tuple(index), T <: Complex)
        end
    end
    return t
end

function emit(name, t)
    A = convert(Array, t)
    # Bending or braiding a leg is a plain axis permutation of this array.
    for p in (Tuple(1:numind(t)), Tuple(numind(t):-1:1))
        q = (p[1:1], p[2:end])
        @assert maximum(abs, convert(Array, permute(t, q)) - permutedims(A, p)) < 1e-12
    end
    println("# ", name, " ", join(size(A), " "))
    for (i, x) in enumerate(A)
        iszero(x) || @printf("%d %.17g %.17g\n", i - 1, real(x), imag(x))
    end
end

V = Vect[U1Irrep](-1 => 1, 0 => 2, 1 => 1)
W = Vect[SU2Irrep](0 => 2, 1 // 2 => 1, 1 => 1)
for T in (Float64, ComplexF64)
    emit("u1_$(T <: Complex ? "complex" : "real")", fixture(T, V ⊗ V' ← V' ⊗ V))
    emit("su2_$(T <: Complex ? "complex" : "real")", fixture(T, W ⊗ W' ⊗ W ← W'))
end
