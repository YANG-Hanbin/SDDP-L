const _GEP_ADAPTIVE_PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const _GEP_ADAPTIVE_JULIA11_BIN = joinpath(homedir(), ".juliaup", "bin", "julia")

function ensure_adaptive_cut_julia_11!(project_root::AbstractString)::Nothing
    if VERSION >= v"1.12" &&
       get(ENV, "SDDP_L_SKIP_JULIA_REEXEC", "0") != "1" &&
       !isempty(PROGRAM_FILE) &&
       isfile(PROGRAM_FILE) &&
       isfile(_GEP_ADAPTIVE_JULIA11_BIN)

        script_path = abspath(PROGRAM_FILE)
        cmd = `$_GEP_ADAPTIVE_JULIA11_BIN +1.11 --project=$project_root $script_path $(ARGS...)`
        run(setenv(cmd, merge(copy(ENV), Dict("SDDP_L_SKIP_JULIA_REEXEC" => "1"))))
        exit(0)
    end
    return
end

ensure_adaptive_cut_julia_11!(_GEP_ADAPTIVE_PROJECT_ROOT)

include(joinpath(@__DIR__, "loadMod.jl"))

"""
    run_generation_expansion_adaptive_cut_test(; kwargs...)

Run a compact GEP regression for the adaptive-level PLC and SMC variants.
The defaults execute two SDDP-L iterations on the T10/R5 benchmark without
writing result files. Additional keyword arguments are forwarded to
`run_generation_expansion_experiments`.
"""
function run_generation_expansion_adaptive_cut_test(; kwargs...)
    defaults = (
        algorithms = [:SDDPL],
        cutTypes = [:AdaptivePLC, :AdaptiveSMC],
        T_list = [10],
        num_list = [5],
        timeSDDP = 600.0,
        gapSDDP = 1e-6,
        iterSDDP = 2,
        levelMethodMaxIter = 5,
        sample_size_SDDP = 2,
        solverGap = 1e-6,
        solverTime = 20.0,
        ε = 1e-4,
        discreteZ = true,
        cutSparsity = true,
        partitionRule = :Bisection,
        branchingStart = 1,
        M = 1,
        verbose = false,
        ℓ1 = 0.0,
        ℓ2 = 0.0,
        nxt_bound = 1e5,
        logger_save = false,
        cutDiagnostics = false,
    )
    options = merge(defaults, (; kwargs...))
    options.iterSDDP >= 2 || throw(ArgumentError("iterSDDP must be at least 2"))
    results = run_generation_expansion_experiments(; options...)

    for algorithm in options.algorithms,
        cut in options.cutTypes,
        T in options.T_list,
        num in options.num_list

        is_supported_configuration(algorithm, cut) || continue
        key = (algorithm, cut, T, num)
        haskey(results, key) || error("Missing adaptive-cut result for $key.")

        history = results[key][:solHistory]
        nrow(history) == options.iterSDDP || error(
            "$key stopped after $(nrow(history)) iterations; expected $(options.iterSDDP).",
        )
        maximum(history.LM_iter) > 0 || error(
            "$key did not report any adaptive level-method iterations.",
        )
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_generation_expansion_adaptive_cut_test()
end
