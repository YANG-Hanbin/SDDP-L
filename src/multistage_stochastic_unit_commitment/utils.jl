"""
function sample_scenarios(; 
    numScenarios::Int64 = 10, 
    scenarioTree::ScenarioTree = scenarioTree
)

# Arguments

    1. `numScenarios`: The number of scenarios will be sampled
    2. `scenarioTree`: A scenario tree

# Returns
    1. `Ξ`: A subset of scenarios.
"""
function sample_scenarios(; 
    numScenarios::Int64 = 10, 
    scenarioTree::ScenarioTree = scenarioTree
)
    Ξ = Dict{Int64, Dict{Int64, RandomVariables}}()
    for ω in 1:numScenarios
        ξ = Dict{Int64, RandomVariables}()
        ξ[1] = scenarioTree.tree[1].nodes[1]
        n = wsample(
            collect(keys(scenarioTree.tree[1].prob)), 
            collect(values(scenarioTree.tree[1].prob)), 
            1
        )[1]
        for t in 2:length(keys(scenarioTree.tree))
            ξ[t] = scenarioTree.tree[t].nodes[n]
            n = wsample(
                collect(keys(scenarioTree.tree[t].prob)), 
                collect(values(scenarioTree.tree[t].prob)), 
            1)[1]
        end
        Ξ[ω] = ξ
    end
    return Ξ
end

"""

function print_iteration_info(
    i::Int, 
    LB::Float64, 
    UB::Float64,
    gap::Float64, 
    iter_time::Float64, 
    LM_iter::Int, 
    total_time::Float64
)

# Arguments

    1. `i`: The current iteration number
    2. `LB`: The lower bound
    3. `UB`: The upper bound
    4. `gap`: The gap between the lower and upper bounds
    5. `iter_time`: The time spent on the current iteration
    6. `LM_iter`: The number of Lagrangian multipliers updated in the current iteration
    7. `total_time`: The total time spent on the algorithm

# Prints

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

    # --- 目录层级 ---
    dir = joinpath(
        root,
        "case=$(case)",
        "alg=$(algorithm)",
        "T=$(T)",
        "Real=$(num)",
    )

    # --- 文件名标签 ---
    tags = String[]

    cutSelection = param.cutSelection
    push!(tags, "cut=$(cutSelection)")

    # 有 med_method 就加
    if haskey(param, :med_method)
        med_method = param.med_method
        push!(tags, "med=$(med_method)")
    end

    # 有 ε 就加 eps 标签
    if haskey(param, :ε)
        ε = param.ε
        eps_int = Int(round(1 / ε))
        push!(tags, "eps=$(eps_int)")
    end

    # 有 ℓ 就加 ell 标签
    if haskey(param_cut, :ℓ)
        ℓ = param_cut.ℓ
        push!(tags, "ell=$(ℓ)")
    end

    # 有 sparsity 就加 sparsity 标签
    if haskey(param, :sparse_cut)
        sparse_cut = param.sparse_cut
        push!(tags, "sparsity=$(sparse_cut)")
    end

    # 有 M 就加 M 标签
    if haskey(param, :M)
        M = param.M
        push!(tags, "M=$(M)")
    end

    # run_id：如果外面没指定，用时间戳
    if run_id === nothing
        # ts = Dates.format(now(), "yyyymmdd\\THHMMSS")
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

const DEFAULT_RESULTS_ROOT = joinpath(@__DIR__, "..", "results")

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
    med_method::Symbol = :ExactPoint,
    case::String = "case30pwl",
    logger_save::Bool = true,
    # 新增两个实验相关参数：
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
        OPT                 = 0.0,
        ε                   = ε,
        κ                   = Dict{Int64, Int64}(),
        tightness           = tightness,
        numScenarios        = numScenarios,
        M                   = M,
        LiftIterThreshold   = LiftIterThreshold,
        branch_threshold    = branch_threshold,
        branch_variable     = branch_variable,
        sparse_cut          = sparse_cut,
        med_method          = med_method,
        cutSelection        = cutSelection,
        algorithm           = algorithm,
        T                   = T,
        num                 = num,
        case                = case,
        logger_save         = logger_save,
        # 新增
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
    core_point_strategy::String = "Eps", # "Mid", "Eps"
    δ::Float64 = 1e-3,
    ℓ::Float64 = .0,
)::NamedTuple
    return (
        core_point_strategy = core_point_strategy, # "Mid", "Eps"
        δ                   = δ,
        ℓ                   = ℓ,
    )
end