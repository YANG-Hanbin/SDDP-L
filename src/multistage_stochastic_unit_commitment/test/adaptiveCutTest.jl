include(joinpath(@__DIR__, "loadMod.jl"))

"""
    run_msuc_adaptive_cut_smoke_tests(; ...)

Run short SDDP-L checks for `AdaptivePLC` and `AdaptiveSMC` on the published
case30, six-stage, five-realization fixture. Results are returned in a
`DataFrame`; no experiment files are written.
"""
function run_msuc_adaptive_cut_smoke_tests(;
    iterations::Int = 2,
    num_scenarios::Int = 2,
    level_method_iterations::Int = 4,
    level_method_verbose::Bool = false,
    cut_diagnostics::Bool = false,
    seed::Int = 20260629,
)::DataFrame
    iterations >= 2 || throw(ArgumentError("iterations must be at least 2"))
    rows = NamedTuple[]

    for cut in (:AdaptivePLC, :AdaptiveSMC)
        Random.seed!(seed)
        result = run_single_experiment(
            :SDDPL,
            cut,
            6,
            5;
            case = "case30",
            numScenarios = num_scenarios,
            M = 1,
            logger_save = false,
            terminate_time = 600,
            terminate_threshold = -1.0,
            TimeLimit = 30,
            MIPGap = 1e-4,
            MaxIter = iterations,
            partitionRule = :Bisection,
            LiftIterThreshold = 1,
            δ = 1e-2,
            levelMethodMaxIter = level_method_iterations,
            levelMethodVerbose = level_method_verbose,
            cutDiagnostics = cut_diagnostics,
        )

        history = result.sddpResults[:solHistory]
        nrow(history) == iterations || error(
            "$cut stopped after $(nrow(history)) iterations; expected $iterations.",
        )
        maximum(history.LM_iter) > 0 || error(
            "$cut did not report any adaptive level-method iterations.",
        )
        push!(rows, merge(result.summary, (adaptive_level_method_entered = true,)))
    end

    return DataFrame(rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    display(run_msuc_adaptive_cut_smoke_tests())
end
