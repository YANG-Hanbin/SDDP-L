using Pkg

const PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
@info "Project root detected as: $PROJECT_ROOT"

Pkg.activate(PROJECT_ROOT)
using Distributed
addprocs(5)

const UC_SRC = abspath(joinpath(@__DIR__, ".."))
@info "UC source dir: $UC_SRC"
@everywhere const PROJECT_ROOT = $PROJECT_ROOT
@everywhere const UC_SRC       = $UC_SRC

@everywhere begin
    using JuMP, Gurobi, PowerModels
    using Statistics, StatsBase, Random, Dates, Distributions
    using Distributed, ParallelDataTransfer
    using CSV, DataFrames, Printf
    using JLD2, FileIO
    using Base.Filesystem: mkpath, dirname, isdir

    const GRB_ENV = Gurobi.Env()
end

@everywhere begin
    include(joinpath(UC_SRC, "utilities", "structs.jl"))
    include(joinpath(UC_SRC, "utilities", "auxiliary.jl"))
    include(joinpath(UC_SRC, "utilities", "adaptive_level_method.jl"))
    include(joinpath(UC_SRC, "utilities", "level_method_regular_subproblem.jl"))
    include(joinpath(UC_SRC, "utilities", "level_method_normalized_subproblem.jl"))
    include(joinpath(UC_SRC, "utilities", "cut_variants.jl"))

    include(joinpath(UC_SRC, "utils.jl"))
    include(joinpath(UC_SRC, "forward_pass.jl"))
    include(joinpath(UC_SRC, "backward_pass.jl"))
    include(joinpath(UC_SRC, "partition_tree.jl"))
    include(joinpath(UC_SRC, "sddp.jl"))
    # include(joinpath(UC_SRC, "utilities", "extForm.jl"))
end


"""
    experiment_dir(case::AbstractString, T::Integer, num::Integer)

Return the directory that stores the serialized experiment instance for the
given case, horizon length, and number of realizations.
"""
function experiment_dir(case::AbstractString, T::Integer, num::Integer)
    return joinpath(
        PROJECT_ROOT,
        "src", "multistage_stochastic_unit_commitment",
        "experiment_$case",
        "stage($T)real($num)",
    )
end

"""
Load all serialized data associated with one `(case, T, num)` experiment
instance.
"""
function load_experiment_data(case::AbstractString, T::Integer, num::Integer)
    dir = experiment_dir(case, T, num)

    indexSets        = load(joinpath(dir, "indexSets.jld2"))["indexSets"]
    paramOPF         = load(joinpath(dir, "paramOPF.jld2"))["paramOPF"]
    paramDemand      = load(joinpath(dir, "paramDemand.jld2"))["paramDemand"]
    scenarioTree     = load(joinpath(dir, "scenarioTree.jld2"))["scenarioTree"]

    # `initialStateInfo` is shared across all `(T, num)` instances of the case.
    initialStateInfo = load(
        joinpath(
            PROJECT_ROOT,
            "src", "multistage_stochastic_unit_commitment",
            "experiment_$case",
            "initialStateInfo.jld2",
        )
    )["initialStateInfo"]

    return (
        indexSets        = indexSets,
        paramOPF         = paramOPF,
        paramDemand      = paramDemand,
        scenarioTree     = scenarioTree,
        initialStateInfo = initialStateInfo,
    )
end

"""
    run_single_experiment(algorithm, cut, T, num; ...)

Build the parameters, load the experiment instance, run
`stochastic_dual_dynamic_programming_algorithm`, and return
`(config = ..., summary = ..., sddpResults = ...)`.
"""
function run_single_experiment(
    algorithm::Symbol,
    cut::Symbol,
    T::Integer,
    num::Integer;
    # Global experiment configuration exposed as keyword arguments.
    case::AbstractString      = "case30",
    numScenarios::Int         = 500,
    M::Int                    = 1,
    logger_save::Bool         = true,
    terminate_time            = 3600,
    TimeLimit                 = 50,
    terminate_threshold::Float64 = 1e-3,
    MIPGap::Float64           = 1e-4,
    MaxIter::Int              = 3000,
    partitionRule::Symbol     = :Bisection,
    ε::Float64                = 1 / 2^8,
    δ::Float64                = 1e-2,
    levelMethodMaxIter::Int   = 200,
    levelMethodVerbose::Bool  = false,
    core_point_strategy::AbstractString = "Mid",
    ℓ::Union{Nothing, Float64} = nothing,
    core_point_weight::Float64 = ℓ === nothing ? 0.75 : ℓ,
    core_point_epsilon::Float64 = 1e-2,
    sparse_cut::Symbol        = :sparse,
    tightness::Bool           = false,
    branch_variable::Symbol   = :ALL,
    LiftIterThreshold::Int    = 2,
    lncMinScale::Float64      = 1e-3,
    lncCoreThetaMargin::Float64 = 1e-4,
    cutDiagnostics::Bool      = false,
)
    @info "Running experiment: case=$case, alg=$algorithm, cut=$cut, T=$T, num=$num"

    # 1. Build the main algorithm parameter bundle.
    param = param_setup(
        terminate_time      = terminate_time,
        TimeLimit           = TimeLimit,
        terminate_threshold = terminate_threshold,
        ε                   = ε,
        verbose             = false,
        MIPGap              = MIPGap,
        MaxIter             = MaxIter,
        tightness           = tightness,
        numScenarios        = numScenarios,
        M                   = M,
        LiftIterThreshold   = LiftIterThreshold,
        branch_threshold    = 1e-6,
        branch_variable     = branch_variable,
        sparse_cut          = sparse_cut,
        cutSelection        = cut,
        algorithm           = algorithm,
        T                   = T,
        num                 = num,
        partitionRule       = partitionRule,
        case                = case,
        logger_save         = logger_save,
        lncMinScale         = lncMinScale,
        lncCoreThetaMargin  = lncCoreThetaMargin,
        cutDiagnostics      = cutDiagnostics,
    )

    # 2. Build cut-specific and level-set parameters.
    param_cut = param_cut_setup(
        core_point_strategy = String(core_point_strategy),
        δ                   = δ,
        core_point_weight   = core_point_weight,
        core_point_epsilon  = core_point_epsilon,
    )

    param_levelsetmethod = param_levelsetmethod_setup(
        μ         = 0.9,
        λ         = 0.5,
        threshold = 1e-4,
        nxt_bound = 1e10,
        MaxIter   = levelMethodMaxIter,
        verbose   = levelMethodVerbose,
    )

    # 3. Load the serialized experiment data.
    data = load_experiment_data(case, T, num)

    # 4. Broadcast data and parameters to all workers.
    @everywhere begin
        indexSets        = $data.indexSets
        paramOPF         = $data.paramOPF
        paramDemand      = $data.paramDemand
        scenarioTree     = $data.scenarioTree
        initialStateInfo = $data.initialStateInfo
        param_cut        = $param_cut
        param_levelsetmethod = $param_levelsetmethod
        param            = $param
    end

    # 5. Run the selected algorithm.
    t_start = now()
    sddpResults = stochastic_dual_dynamic_programming_algorithm(
        data.scenarioTree,
        data.indexSets,
        data.paramDemand,
        data.paramOPF;
        initialStateInfo     = data.initialStateInfo,
        param_cut            = param_cut,
        param_levelsetmethod = param_levelsetmethod,
        param                = param,
    )
    t_end = now()
    elapsed = (t_end - t_start).value / 1000

    # 6. Extract a compact summary row for downstream tables.
    solHistory = sddpResults[:solHistory]
    last_row   = solHistory[end, :]

    summary = (
        case      = case,
        algorithm = algorithm,
        cut       = cut,
        T         = T,
        num       = num,
        LB        = last_row.LB,
        UB        = last_row.UB,
        gap_str   = last_row.gap,
        time      = last_row.Time,
        runtime   = elapsed,
        core_point_strategy = String(core_point_strategy),
        core_point_weight = core_point_weight,
        core_point_epsilon = core_point_epsilon,
    )

    # 7. Run garbage collection on the main process.
    GC.gc()

    return (config = param, summary = summary, sddpResults = sddpResults)
end

"""
    run_experiment_grid()

Run an entire experiment grid and return a summary `DataFrame`.
"""
function run_experiment_grid(;
    case = "case30",
    algorithms   = [:SDDPL, :SDDP, :SDDiP],
    cuts         = [:PLC, :SMC, :LC, :LNC, :SBC, :SBCLC, :SBCSMC, :SBCPLC, :ReLUC, :NormalizedReLUC],
    nums         = [5, 10],
    Ts           = [6, 8, 12],
    numScenarios = 500,
    M            = 1,
    logger_save  = true,
    terminate_time = 3600,
    TimeLimit    = 50,
    terminate_threshold = 1e-3,
    MIPGap       = 1e-4,
    MaxIter      = 3000,
    partitionRule= :Bisection,
    ε            = 1 / 2^8,
    δ            = 1e-2,
    levelMethodMaxIter = 200,
    levelMethodVerbose = false,
    core_point_strategies = ["Mid"],
    ℓ::Union{Nothing, Float64} = nothing,
    core_point_weight = ℓ === nothing ? 0.75 : ℓ,
    core_point_epsilon = 1e-2,
    sparse_cut   = :sparse,
    tightness    = false,
    branch_variable   = :ALL,
    LiftIterThreshold = 2,
    lncMinScale = 1e-3,
    lncCoreThetaMargin = 1e-4,
    cutDiagnostics = false,
    task_ids   = nothing
)::DataFrame

    is_supported_configuration(algorithm::Symbol, cut::Symbol) =
        algorithm != :SDDiP ||
        cut ∉ (:ReLUC, :NormalizedReLUC, :SBCReLUC, :SBCNormalizedReLUC)

    # Build the complete experiment list. Each task is a
    # `(algorithm, cut, num, T, core_point_strategy)` tuple.
    all_tasks = [(a, c, n, T, cps)
                 for a in algorithms
                 for c in cuts
                 for n in nums
                 for T in Ts
                 for cps in core_point_strategies
                 if is_supported_configuration(a, c)]

    # Optionally keep only the requested task subset.
    if task_ids !== nothing
        all_tasks = all_tasks[task_ids]
    end

    @info "Total experiment configs: $(length(all_tasks))"

    # Collect one summary row per experiment configuration.
    summary_df = DataFrame(
        case      = String[],
        algorithm = Symbol[],
        cut       = Symbol[],
        T         = Int[],
        num       = Int[],
        LB        = Float64[],
        UB        = Float64[],
        gap_str   = String[],
        time      = Float64[],
        runtime   = Float64[],
        core_point_strategy = String[],
        core_point_weight = Float64[],
        core_point_epsilon = Float64[],
    )

    # Execute the tasks sequentially.
    for (algorithm, cut, num, T, core_point_strategy) in all_tasks
        println()
        @info "=================================================================="
        @info "Start: case=$case, alg=$algorithm, cut=$cut, T=$T, num=$num, core=$core_point_strategy"
        @info "=================================================================="

        result = run_single_experiment(
            algorithm, cut, T, num;
            case              = case,
            numScenarios      = numScenarios,
            M                 = M,
            logger_save       = logger_save,
            terminate_time    = terminate_time,
            TimeLimit         = TimeLimit,
            terminate_threshold = terminate_threshold,
            MIPGap            = MIPGap,
            MaxIter           = MaxIter,
            partitionRule     = partitionRule,
            ε                 = ε,
            δ                 = δ,
            levelMethodMaxIter = levelMethodMaxIter,
            levelMethodVerbose = levelMethodVerbose,
            core_point_strategy = core_point_strategy,
            core_point_weight = core_point_weight,
            core_point_epsilon = core_point_epsilon,
            sparse_cut        = sparse_cut,
            tightness         = tightness,
            branch_variable   = branch_variable,
            LiftIterThreshold = LiftIterThreshold,
            lncMinScale       = lncMinScale,
            lncCoreThetaMargin = lncCoreThetaMargin,
            cutDiagnostics    = cutDiagnostics,
        )

        s = result.summary
        push!(summary_df, (
            s.case,
            s.algorithm,
            s.cut,
            s.T,
            s.num,
            s.LB,
            s.UB,
            s.gap_str,
            s.time,
            s.runtime,
            s.core_point_strategy,
            s.core_point_weight,
            s.core_point_epsilon,
        ))

        @everywhere GC.gc()
    end

    return summary_df
end

"""
    run_msuc_core_point_sensitivity(; ...)

Run the core-point sensitivity study for MSUC. The default configuration
uses the largest case30 instance and compares `Eps` and `Conv` for PLC/LNC
under SDDP-L with bisection branching.
"""
function run_msuc_core_point_sensitivity(;
    case::AbstractString = "case30",
    T::Int = 12,
    num::Int = 10,
    core_point_strategies = ["Eps", "Conv"],
    kwargs...
)::DataFrame
    return run_experiment_grid(;
        case = case,
        algorithms = [:SDDPL],
        cuts = [:PLC, :LNC],
        nums = [num],
        Ts = [T],
        partitionRule = :Bisection,
        core_point_strategies = core_point_strategies,
        kwargs...,
    )
end
