using TensorKit

struct BenchConfig
    reps::Int
    warmup::Int
end

function parse_config()
    reps = 10_000
    warmup = 1_000
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--reps"
            i += 1
            reps = parse(Int, ARGS[i])
        elseif ARGS[i] == "--warmup"
            i += 1
            warmup = parse(Int, ARGS[i])
        else
            error("unknown argument: $(ARGS[i])")
        end
        i += 1
    end
    return BenchConfig(reps, warmup)
end

function run_bench(f, config::BenchConfig)
    for _ in 1:config.warmup
        f()
    end
    GC.gc()
    start = time_ns()
    for _ in 1:config.reps
        f()
    end
    return time_ns() - start
end

function run_single(f)
    f()
    GC.gc()
    start = time_ns()
    result = f()
    return (time_ns() - start, result)
end

function allocated_bytes(f)
    f()
    GC.gc()
    return @allocated f()
end

function print_result(fixture, metric, ns, reps, alloc_bytes, checksum)
    avg_ns = ns / reps
    println(
        "$fixture,$metric,$reps,$(round(avg_ns; digits=3))," *
        "$alloc_bytes,$(round(real(checksum); digits=12)),$(round(imag(checksum); digits=12))",
    )
end

function fill_sequence!(t, start, step)
    for i in eachindex(t.data)
        t.data[i] = start + step * (i - 1)
    end
    return t
end

function fill_data!(t, values)
    length(t.data) == length(values) || error("data length mismatch: $(length(t.data)) != $(length(values))")
    t.data .= values
    return t
end

function checksum_real(t)
    return sum(t.data)
end

function checksum_symtn_complex(t)
    return sum(real(x) + 0.5 * imag(x) for x in t.data)
end

function bench_su2_noncanonical_source(config::BenchConfig)
    V = SU2Space(1//2 => 2)
    A = fill_sequence!(zeros(Float64, (V ⊗ V ⊗ V) ← V), 1.0, 0.125)
    B = fill_sequence!(zeros(Float64, V ← (V ⊗ V ⊗ V)), -3.0, 0.25)
    C = zeros(Float64, V ← V)
    initial = [2.0, -1.0, 4.0, -3.0]
    alpha = -1.5
    beta = 0.25

    into! = () -> begin
        C.data .= initial
        @tensor C[b; a] = alpha * A[c d e; a] * B[b; c d e] + beta * C[b; a]
        return checksum_real(C)
    end

    single_ns, single_chk = run_single(into!)
    elapsed = run_bench(into!, config)
    print_result("tensorkit_su2_noncanonical_source_degeneracy", "single_after_warmup_into", single_ns, 1, allocated_bytes(into!), single_chk)
    print_result("tensorkit_su2_noncanonical_source_degeneracy", "warm_after_warmup_into", elapsed, config.reps, allocated_bytes(into!), into!())
end

function bench_su2_output_transform_canonical_dst_scratch(config::BenchConfig)
    V = SU2Space(1//2 => 1)
    A = fill_sequence!(zeros(Float64, (V ⊗ V) ← (V ⊗ V)), 1.0, 0.25)
    # Build the destination from TensorKit's own transform so dual flags match.
    C = permute(A, ((1, 3, 2, 4), ()))
    initial = [0.5 + (i - 1) for i in eachindex(C.data)]
    alpha = -0.75
    rhs_scalar = 2.0
    beta = 0.5

    into! = () -> begin
        C.data .= initial
        TensorKit.add_permute!(C, A, ((1, 3, 2, 4), ()), alpha * rhs_scalar, beta)
        return checksum_real(C)
    end

    single_ns, single_chk = run_single(into!)
    elapsed = run_bench(into!, config)
    print_result("tensorkit_su2_output_transform_canonical_dst_scratch", "single_after_warmup_add_permute", single_ns, 1, allocated_bytes(into!), single_chk)
    print_result("tensorkit_su2_output_transform_canonical_dst_scratch", "warm_after_warmup_add_permute", elapsed, config.reps, allocated_bytes(into!), into!())
end

function bench_product_fz2_u1_su2_complex(config::BenchConfig)
    I = FermionParity ⊠ U1Irrep ⊠ Irrep[SU₂]
    Va = Vect[I]((1, 1, 1//2) => 1)
    Vb = Vect[I]((1, -1, 1//2) => 1)
    Vc = Vect[I]((0, 0, 0) => 1, (0, 0, 1) => 1)
    A = fill_data!(zeros(ComplexF64, (Va ⊗ Vb) ← Vc), ComplexF64[1 + 2im, 3 - 1im])
    C = zeros(ComplexF64, (Vb ⊗ Va) ← Vc)
    initial = ComplexF64[5 + 1im, -2 + 4im]
    alpha = 2.0 + 0.0im
    rhs_scalar = 2.0 + 0.5im
    beta = 3.0 + 0.0im

    into! = () -> begin
        C.data .= initial
        TensorKit.add_permute!(C, A, ((2, 1), (3,)), alpha * rhs_scalar, beta)
        return checksum_symtn_complex(C)
    end

    single_ns, single_chk = run_single(into!)
    elapsed = run_bench(into!, config)
    print_result("tensorkit_product_fz2_u1_su2_complex", "single_after_warmup_add_permute", single_ns, 1, allocated_bytes(into!), single_chk)
    print_result("tensorkit_product_fz2_u1_su2_complex", "warm_after_warmup_add_permute", elapsed, config.reps, allocated_bytes(into!), into!())
end

config = parse_config()
println("fixture,metric,reps,avg_ns,allocated_bytes,checksum_re,checksum_im")
bench_su2_noncanonical_source(config)
bench_su2_output_transform_canonical_dst_scratch(config)
bench_product_fz2_u1_su2_complex(config)
