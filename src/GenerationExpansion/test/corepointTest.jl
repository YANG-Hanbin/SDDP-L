include(joinpath(@__DIR__, "loadMod.jl"))

const GEP_CORE_POINT_CONFIG = (
    T = 15,
    num = 10,
    algorithms = [:SDDPL],
    cutTypes = [:PLC, :LNC],
    corePointStrategies = [:Eps, :Conv, :Mid],
    experiment_kwargs = (
        timeSDDP = 3600.0,
        gapSDDP = 1e-3,
        iterSDDP = 300,
        levelMethodMaxIter = 200,
        sample_size_SDDP = 500,
        solverGap = 1e-6,
        solverTime = 20.0,
        ε = 1e-4,
        discreteZ = true,
        cutSparsity = true,
        branchingStart = 10,
        M = 1,
        verbose = false,
        ℓ1 = 0.0,
        ℓ2 = 0.0,
        nxt_bound = 1e5,
        logger_save = true,
        corePointWeight = 0.5,
        corePointEpsilon = 1e-2,
    ),
)

function summarize_core_point_results(results::Dict)::DataFrame
    rows = NamedTuple[]

    for (algorithm, cut, T, num, core) in sort(collect(keys(results)))
        last_row = results[(algorithm, cut, T, num, core)][:solHistory][end, :]
        push!(
            rows,
            (
                problem = "GEP",
                algorithm = algorithm,
                cut = cut,
                T = T,
                num = num,
                core = core,
                LB = last_row.LB,
                UB = last_row.UB,
                gap = last_row.gap,
                Time = last_row.Time,
            ),
        )
    end

    return DataFrame(rows)
end

function run_core_point_test()
    cfg = GEP_CORE_POINT_CONFIG
    results = run_generation_expansion_core_point_sensitivity(
        ;
        algorithms = cfg.algorithms,
        cutTypes = cfg.cutTypes,
        T = cfg.T,
        num = cfg.num,
        corePointStrategies = cfg.corePointStrategies,
        cfg.experiment_kwargs...,
    )

    summary = summarize_core_point_results(results)
    output_path = joinpath(@__DIR__, "core_point_summary_gep_T$(cfg.T)_R$(cfg.num).csv")
    CSV.write(output_path, summary)

    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    summary = run_core_point_test()
    show(summary; allcols = true, allrows = true)
    println()
end
