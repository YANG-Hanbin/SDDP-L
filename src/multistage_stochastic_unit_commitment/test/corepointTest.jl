include(joinpath(@__DIR__, "loadMod.jl"))

const MSUC_CORE_POINT_CONFIG = (
    case = "case30",
    T = 12,
    num = 10,
    core_point_strategies = ["Eps", "Conv", "Mid"],
    cuts = [:PLC, :LNC],
    experiment_kwargs = (
        numScenarios = 500,
        M = 1,
        logger_save = true,
        terminate_time = 3600,
        TimeLimit = 50,
        MaxIter = 3000,
        ε = 1 / 2^8,
        δ = 1e-2,
        levelMethodMaxIter = 200,
        core_point_weight = 0.5,
        core_point_epsilon = 1e-2,
        sparse_cut = :sparse,
        tightness = false,
        branch_variable = :ALL,
        LiftIterThreshold = 2,
    ),
)

function run_core_point_test()
    cfg = MSUC_CORE_POINT_CONFIG
    summary = run_msuc_core_point_sensitivity(
        ;
        case = cfg.case,
        T = cfg.T,
        num = cfg.num,
        core_point_strategies = cfg.core_point_strategies,
        cuts = cfg.cuts,
        cfg.experiment_kwargs...,
    )

    output_path = joinpath(@__DIR__, "core_point_summary_msuc_T$(cfg.T)_R$(cfg.num).csv")
    CSV.write(output_path, summary)

    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    summary = run_core_point_test()
    show(summary; allcols = true, allrows = true)
    println()
end
