# TensorKit reference for symtn/benches/eager_overhead_ledger.rs (#1313):
# the same cases, operations, timing rule, and CSV columns.
# Usage:
#   julia -t 1 --project=benchmarks/tensorkit_benchmark \
#       benchmarks/tensorkit_eager_overhead.jl [filter ...]
#
# Timing: warm, batches of >= 10 us, 250 ms per row, per-call minimum and
# median over the batches (BenchmarkTools is not in the pinned environment;
# this is the same estimator applied to both libraries). Allocations: one warm
# call through `@timed`, as allocation count and bytes.

using LinearAlgebra
using TensorKit

BLAS.set_num_threads(1)
const TO = TensorKit.TO

const CASES = [
    ("r2_s8_d2", 1, 1, 8, 2),
    ("r2_s8_d16", 1, 1, 8, 16),
    ("r3_s4_d4", 2, 1, 4, 4),
    ("r4_s3_d4", 2, 2, 3, 4),
    ("r5_s2_d2", 3, 2, 2, 2),
]

centered(n) = (-((n - 1) ÷ 2)) .+ (0:(n - 1))
legspace(::Val{:U1}, n, d) = U1Space((q => d for q in centered(n))...)
legspace(::Val{:fZ2xU1}, n, d) =
    Vect[FermionParity ⊠ U1Irrep]((((q & 1), q) => d for q in centered(n))...)
legspace(::Val{:SU2}, n, d) = SU2Space(((j2 // 2) => d for j2 in 0:(n - 1))...)

function time_calls(f)
    warm_start = time_ns()
    warm_calls = 0
    while time_ns() - warm_start < 30_000_000
        f()
        warm_calls += 1
    end
    per_call = 30_000_000 / max(warm_calls, 1)
    batch = max(1, ceil(Int, 10_000 / per_call))
    samples = Float64[]
    start = time_ns()
    while length(samples) < 4000 && (length(samples) < 50 || time_ns() - start < 250_000_000)
        t = time_ns()
        for _ in 1:batch
            f()
        end
        push!(samples, (time_ns() - t) / batch)
    end
    sort!(samples)
    return length(samples) * batch, samples[1], samples[div(length(samples), 2) + 1]
end

function run_op(filters, prefix, op, f)
    key = "$prefix,$op"
    all(filter -> filter in split(key, ","), filters) || return
    calls, min_ns, median_ns = time_calls(f)
    f()
    stats = @timed f()
    count = Base.gc_alloc_count(stats.gcstats)
    println("tensorkit,$key,$calls,$(round(min_ns; digits = 1)),$(round(median_ns; digits = 1)),$count,$(stats.bytes)")
    return
end

function ledger(filters, symmetry, T, dname)
    for (name, nc, nd, sectors, deg) in CASES
        V = legspace(Val(symmetry), sectors, deg)
        W = legspace(Val(symmetry), sectors, cld(deg, 2))
        cod = nc == 1 ? V : ⊗(ntuple(_ -> V, nc)...)
        dom = nd == 1 ? V : ⊗(ntuple(_ -> V, nd)...)
        A = randn(T, cod ← dom)
        A2 = randn(T, cod ← dom)
        S = randn(T, dom ← dom)
        M = randn(T, V ← V)
        P = copy(adjoint(isometry(T, V, W)))
        r = nc + nd
        rotated = ((2:r)..., 1)
        perm = (rotated[1:nc], rotated[(nc + 1):r])
        N = nd >= 2 ? nc + 1 : nc - 1
        open_a = (Tuple(2:r), (1,))
        pab = (Tuple(1:(r - 1)), (r,))
        restrict_pab = (Tuple(1:nc), Tuple((nc + 1):r))
        blocks = length(fusiontrees(A))
        coupled = length(blocksectors(A))
        prefix = "$symmetry,$dname,$name,$r,$blocks,$coupled,$(dim(space(A)))"
        run_op(filters, prefix, "compose", () -> A * S)
        run_op(filters, prefix, "contract",
            () -> TO.tensorcontract(A, open_a, false, M, ((2,), (1,)), false, pab))
        run_op(filters, prefix, "permute", () -> permute(A, perm))
        run_op(filters, prefix, "repartition", () -> repartition(A, N))
        run_op(filters, prefix, "qr_compact", () -> qr_compact(A))
        run_op(filters, prefix, "restrict_leg",
            () -> TO.tensorcontract(P, ((1,), (2,)), false, A, ((1,), Tuple(2:r)), false, restrict_pab))
        run_op(filters, prefix, "scale", () -> scale(A, 2))
        run_op(filters, prefix, "add", () -> A + A2)
        run_op(filters, prefix, "norm", () -> norm(A))
    end
end

function main(filters)
    println("threads,symmetry,dtype,case,rank,blocks,coupled,elements,op,iterations,min_ns,median_ns,alloc_calls,alloc_bytes")
    for symmetry in (:U1, :fZ2xU1, :SU2), (T, dname) in ((Float64, "f64"), (ComplexF64, "c64"))
        ledger(filters, symmetry, T, dname)
    end
end

main(ARGS)
