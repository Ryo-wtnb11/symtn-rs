# Offline generator for tensorkit-0.17.1-table-twist-flip.txt. Nothing in the
# shipped crates runs this; it is a manual step.
#
# Reproduce in a scratch project (NOT this directory):
#
#   julia --project=/tmp/fixture -e 'using Pkg; Pkg.add(name="TensorKit", version="0.17.1"); Pkg.add(name="TensorKitSectors", version="0.3.9")'
#   julia --project=/tmp/fixture path/to/tensorkit_table_twist_flip.jl > path/to/tensorkit-0.17.1-table-twist-flip.txt
#
# TensorKit's own `twist` (`src/tensors/indexmanipulations.jl:twist`, which
# copies and calls `twist!`) and `flip` (`:flip`, per tree pair
# `src/fusiontrees/braiding_manipulations.jl:flip`) on Fibonacci and Ising
# tensors whose every entry is a function of its fusion-tree pair and
# degeneracy index alone (`entry`, the hash of `tensorkit_table_reductions.jl`;
# `table_twist_flip.rs` computes the same function). Sectors are written with
# AnyonWiki's labels, which `TableRule` uses: Fibonacci 1 = I, 2 = τ; Ising
# 1 = I, 2 = ψ, 3 = σ.
#
# Each row is `inner(p, r)` for the result `r` and a probe `p` filled by the
# same hash on `space(r)`. Both operations scale each tree pair by a product
# of twists and Frobenius-Schur phases of its uncoupled sectors, which no
# vertex gauge changes, and `inner` pairs entries of equal trees; so the rows
# are invariant under the vertex sign u^{σσ}_ψ = -1 by which AnyonWiki's
# Ising differs from TensorKit's (#2277), and no gauge witness is needed.
#
# Rows: category operation inv legs re im; `legs` are TensorKit's one-based
# flat indices, codomain first. Values use Julia's shortest round-trip `repr`.
using Pkg
using TensorKit

dep(uuid) = Pkg.dependencies()[Base.UUID(uuid)]
for (name, uuid) in (("TensorKit", "07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"),
                     ("TensorKitSectors", "13a9c161-d5da-41f0-bcbd-e1a08ae0647f"))
    d = dep(uuid)
    println("# ", name, " v", d.version, " tree ", d.tree_hash, " (", basename(d.source), ")")
end
println("# rows: category operation inv legs re im")

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

function filled(label, tag, space)
    t = zeros(ComplexF64, space)
    for (f₁, f₂) in fusiontrees(t)
        block = t[f₁, f₂]
        for index in CartesianIndices(block)
            block[index] = entry(label, tag, f₁, f₂, index)
        end
    end
    return t
end

const CASES = (
    (twist, false, [2]), (twist, true, [2]), (twist, false, [1, 3]),
    (twist, true, [2, 4]), (twist, false, [1, 2, 3, 4]),
    (flip, false, [1]), (flip, false, [2]), (flip, false, [3]), (flip, false, [4]),
    (flip, true, [1, 4]), (flip, true, [2, 3]), (flip, false, [1, 2, 3, 4]),
)

function report(category, label, V)
    t = filled(label, "t", V ⊗ V' ← V' ⊗ V)
    for (op, inv, legs) in CASES
        r = op(t, legs; inv)
        z = inner(filled(label, "p", space(r)), r)
        println(category, " ", op, " ", inv, " ", join(legs, ","), " ",
                repr(real(z)), " ", repr(imag(z)))
    end
end

const fib_labels = Dict(FibonacciAnyon(:I) => 1, FibonacciAnyon(:τ) => 2)
report("fibonacci", a -> fib_labels[a],
       Vect[FibonacciAnyon](FibonacciAnyon(:I) => 2, FibonacciAnyon(:τ) => 2))

const ising_labels = Dict(IsingAnyon(:I) => 1, IsingAnyon(:ψ) => 2, IsingAnyon(:σ) => 3)
report("ising", a -> ising_labels[a],
       Vect[IsingAnyon](IsingAnyon(:I) => 1, IsingAnyon(:ψ) => 2, IsingAnyon(:σ) => 2))
