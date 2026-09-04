const RESULTS_ROOT = joinpath(@__DIR__, "..", "results")

"""
    sample_scenarios(;
        numScenarios::Int64 = 10,
        scenarioTree::ScenarioTree = scenarioTree,
    )

Sample `numScenarios` paths independently from the scenario tree and return
them in the dictionary format expected by the SDDP routines.
"""
function sample_scenarios(; 
    numScenarios::Int64 = 10, 
    scenarioTree::ScenarioTree = scenarioTree
)
    Ξ = Dict{Int64, Dict{Int64, RandomVariables}}()
    for ω in 1:numScenarios
        ξ = Dict{Int64, RandomVariables}()
        ξ[1] = scenarioTree.tree[1].nodes[1]
        first_stage_nodes = sort(collect(keys(scenarioTree.tree[1].prob)))
        n = wsample(
            first_stage_nodes,
            [scenarioTree.tree[1].prob[node] for node in first_stage_nodes],
            1
        )[1]
        for t in 2:length(keys(scenarioTree.tree))
            ξ[t] = scenarioTree.tree[t].nodes[n]
            next_stage_nodes = sort(collect(keys(scenarioTree.tree[t].prob)))
            n = wsample(
                next_stage_nodes,
                [scenarioTree.tree[t].prob[node] for node in next_stage_nodes],
            1)[1]
        end
        Ξ[ω] = ξ
    end
    return Ξ
end

"""
    print_iteration_info(
        i::Int64,
        LB::Float64,
        UB::Float64,
        gap::Float64,
        iter_time::Float64,
        LM_iter::Int,
        total_Time::Float64,
    )::Nothing

Print one formatted row of the experiment progress table.
"""
function print_iteration_info(
    i::Int64, 
    LB::Float64, 
    UB::Float64,
    gap::Float64, 
    iter_time::Float64, 
    LM_iter::Int, 
    total_Time::Float64
)::Nothing
    @printf("%4d | %12.2f     | %12.2f     | %9.2f%%     | %9.2f s     | %6d     | %10.2f s     \n", 
                i, LB, UB, gap, iter_time, LM_iter, total_Time); 
    return 
end

function print_iteration_info_bar()::Nothing
    println("------------------------------------------ Iteration Info ------------------------------------------------")
    println("Iter |        LB        |        UB        |       Gap      |      i-time     |    #LM     |     T-Time")
    println("----------------------------------------------------------------------------------------------------------")
    return 
end

"""
    build_results_path(param, param_cut; root=RESULTS_ROOT, run_id=nothing)
"""
function build_results_path(
    param::NamedTuple,
    param_cut::NamedTuple;
    root::AbstractString = RESULTS_ROOT,
    run_id = nothing,
)::String
    case      = param.case
    algorithm = param.algorithm
    T         = param.T
    num       = param.num

    # Directory hierarchy.
    dir = joinpath(
        root,
        "case=$(case)",
        "alg=$(algorithm)",
        "T=$(T)",
        "Real=$(num)",
    )

    # Filename tags.
    tags = String[]

    cutSelection = param.cutSelection
    push!(tags, "cut=$(cutSelection)")

    # Partition rule, when available.
    if haskey(param, :partitionRule)
        partitionRule = param.partitionRule
        push!(tags, "med=$(partitionRule)")
    end

    # Epsilon tag.
    if haskey(param, :ε)
        ε = param.ε
        eps_int = Int(round(1 / ε))
        push!(tags, "eps=$(eps_int)")
    end

    if haskey(param_cut, :core_point_strategy)
        strategy = param_cut.core_point_strategy
        push!(tags, "core=$(strategy)")

        if strategy == "Conv"
            weight = haskey(param_cut, :core_point_weight) ? param_cut.core_point_weight : param_cut.ℓ
            push!(tags, "cpw=$(weight)")
        elseif strategy == "Eps" && haskey(param_cut, :core_point_epsilon)
            push!(tags, "cpeps=$(param_cut.core_point_epsilon)")
        end
    end

    # Sparsity tag.
    if haskey(param, :sparse_cut)
        sparse_cut = param.sparse_cut
        push!(tags, "sparsity=$(sparse_cut)")
    end

    # Repetition counter or experiment multiplier.
    if haskey(param, :M)
        M = param.M
        push!(tags, "M=$(M)")
    end

    # Default run identifier: current date.
    if run_id === nothing
        ts = Dates.format(now(), "yyyymmdd")
        run_id = ts
    end
    push!(tags, "run=$(run_id)")

    filename = join(tags, "__") * ".jld2"

    return joinpath(dir, filename)
end

function save_results_info(
    param::NamedTuple,
    param_cut::NamedTuple,
    sddpResults::Dict,
)::Nothing
    param.logger_save || return nothing

    filepath = build_results_path(
        param,
        param_cut;
        root   = param.results_root,
        run_id = param.run_id,
    )

    dir = dirname(filepath)
    isdir(dir) || mkpath(dir)

    # @info "Saving results to $filepath"
    @save filepath sddpResults

    return nothing
end

const DEFAULT_RESULTS_ROOT = RESULTS_ROOT

function param_setup(;
    terminate_time::Any = 3600,
    TimeLimit::Any = 10,
    terminate_threshold::Float64 = 1e-3,
    ε::Float64 = 0.125,
    verbose::Bool = false,
    MIPGap::Float64 = 1e-4,
    MaxIter::Int64 = 3000,
    tightness::Bool = true,
    numScenarios::Int64 = 3,
    M::Int64 = 1,
    LiftIterThreshold::Int64 = 10,
    branch_threshold::Float64 = 1e-3,
    branch_variable::Symbol = :ALL, # :ALL, :MFV
    sparse_cut::Symbol = :sparse, # :sparse, :dense
    cutSelection::Symbol = :PLC, 
    algorithm::Symbol = :SDDPL,
    T::Int64 = 12,
    num::Int64 = 10,
    partitionRule::Symbol = :Bisection,
    case::String = "case30",
    logger_save::Bool = true,
    lncMinScale::Float64 = 1e-3,
    lncCoreThetaMargin::Float64 = 1e-4,
    cutDiagnostics::Bool = false,
    # Experiment-output controls.
    results_root::AbstractString = DEFAULT_RESULTS_ROOT,
    run_id::Union{Nothing,String} = nothing,
)::NamedTuple

    return (
        verbose             = verbose,
        MIPGap              = MIPGap,
        TimeLimit           = TimeLimit,
        terminate_time      = terminate_time,
        terminate_threshold = terminate_threshold,
        MaxIter             = MaxIter,
        θ̲                   = 0.0,
        # No exact benchmark value is loaded by the standard experiment path.
        # Keep the output column for compatibility, but represent an unavailable
        # reference optimum explicitly instead of reporting a misleading zero.
        OPT                 = nothing,
        ε                   = ε,
        κ                   = Dict{Int64, Int64}(),
        tightness           = tightness,
        numScenarios        = numScenarios,
        M                   = M,
        LiftIterThreshold   = LiftIterThreshold,
        branch_threshold    = branch_threshold,
        branch_variable     = branch_variable,
        sparse_cut          = sparse_cut,
        partitionRule       = partitionRule,
        cutSelection        = cutSelection,
        algorithm           = algorithm,
        T                   = T,
        num                 = num,
        case                = case,
        logger_save         = logger_save,
        lncMinScale         = lncMinScale,
        lncCoreThetaMargin  = lncCoreThetaMargin,
        cutDiagnostics      = cutDiagnostics,
        results_root        = results_root,
        run_id              = run_id,
    )
end

function param_levelsetmethod_setup(;
    μ::Float64 = 0.9,
    λ::Float64 = 0.5,
    threshold::Float64 = 1e-4,
    nxt_bound::Float64 = 1e10,
    MaxIter::Int64 = 200,
    verbose::Bool = false
)::NamedTuple
    return (
        μ             = μ,
        λ             = λ,
        threshold     = threshold,
        nxt_bound     = nxt_bound,
        MaxIter       = MaxIter,
        verbose       = verbose,
    )
end


function param_cut_setup(;
    core_point_strategy::String = "Mid",
    δ::Float64 = 1e-3,
    ℓ::Float64 = 0.75,
    core_point_weight::Union{Nothing, Float64} = nothing,
    core_point_epsilon::Float64 = 1e-2,
)::NamedTuple
    weight = core_point_weight === nothing ? ℓ : core_point_weight
    return (
        core_point_strategy = core_point_strategy,
        δ                   = δ,
        ℓ                   = weight,
        core_point_weight   = weight,
        core_point_epsilon  = core_point_epsilon,
    )
end
