include(joinpath(@__DIR__, "loadMod.jl"))

function experiment_dir(case::AbstractString, T::Integer, num::Integer)
    return joinpath(
        PROJECT_ROOT,
        "src", "multistage_stochastic_unit_commitment",
        "experiment_$case",
        "stage($T)real($num)",
    )
end

"""
加载一个 (case, T, num) 对应的所有数据
"""
function load_experiment_data(case::AbstractString, T::Integer, num::Integer)
    dir = experiment_dir(case, T, num)

    indexSets        = load(joinpath(dir, "indexSets.jld2"))["indexSets"]
    paramOPF         = load(joinpath(dir, "paramOPF.jld2"))["paramOPF"]
    paramDemand      = load(joinpath(dir, "paramDemand.jld2"))["paramDemand"]
    scenarioTree     = load(joinpath(dir, "scenarioTree.jld2"))["scenarioTree"]

    # initialStateInfo 是不随 (T,num) 变的，仍然用你之前的路径
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

给定一组 (algorithm, cutSelection, T, num)，构造参数、加载数据、调用
`stochastic_dual_dynamic_programming_algorithm`，并返回一个 NamedTuple：
    (config = ..., summary = ..., sddpResults = ...)
"""
function run_single_experiment(
    algorithm::Symbol,
    cut::Symbol,
    T::Integer,
    num::Integer;
    # 下面这些是你在最前面写的“全局配置”，做成 keyword 更方便调
    case::AbstractString      = "case30",
    numScenarios::Int         = 500,
    M::Int                    = 1,
    logger_save::Bool         = true,
    med_method::Symbol        = :IntervalMed,
    ε::Float64                = 1 / 2^8,
    ℓ::Float64                = 0.5,
    δ::Float64                = 1e-2,
    sparse_cut::Symbol        = :sparse,
    tightness::Bool           = false,
    branch_variable::Symbol   = :ALL,
    LiftIterThreshold::Int    = 2,
)
    @info "Running experiment: case=$case, alg=$algorithm, cut=$cut, T=$T, num=$num"

    # 1. 构造 param
    param = param_setup(
        terminate_time      = 3600,
        TimeLimit           = 10,
        terminate_threshold = 1e-3,
        ε                   = ε,
        verbose             = false,
        MIPGap              = 1e-4,
        MaxIter             = 3000,
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
        med_method          = med_method,
        case                = case,
        logger_save         = logger_save,
    )

    # 2. 构造 param_cut / param_levelsetmethod
    param_cut = param_cut_setup(
        core_point_strategy = "Eps",  # 你原来的设定
        δ                   = δ,
        ℓ                   = ℓ,
    )

    param_levelsetmethod = param_levelsetmethod_setup(
        μ         = 0.9,
        λ         = 0.5,
        threshold = 1e-4,
        nxt_bound = 1e10,
        MaxIter   = 200,
        verbose   = false,
    )

    # 3. 加载数据
    data = load_experiment_data(case, T, num)

    # 4. 把参数 / 数据广播到 worker（如果你仍然需要）
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

    # 5. 跑算法
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
    elapsed = (t_end - t_start).value / 1000  # 秒

    # 6. 从结果里抽一点 summary 出来（方便做表）
    solHistory = sddpResults[:solHistory]
    last_row   = solHistory[end, :]   # 最后一行

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
    )

    # 7. 主进程做一次 GC，worker 的你原来就有 @everywhere GC.gc()
    GC.gc()

    return (config = param, summary = summary, sddpResults = sddpResults)
end

"""
    run_experiment_grid()

跑一整个实验网格，并返回一个 DataFrame 的 summary。
"""
function run_experiment_grid(;
    case         = "case30",
    algorithms   = [:SDDPL, :SDDP, :SDDiP],
    cuts         = [:PLC, :SMC, :LC, :SBC, :SBCLC, :SBCSMC, :SBCPLC, :NormalizedCut],
    nums         = [5, 10],
    Ts           = [6, 8, 12],
    numScenarios = 500,
    M            = 1,
    logger_save  = true,
    med_method   = :IntervalMed,
    ε            = 1 / 2^8,
    ℓ            = 0.5,
    δ            = 1e-2,
    sparse_cut   = :sparse,
    tightness    = false,
    branch_variable   = :ALL,
    LiftIterThreshold = 2,
    task_ids   = nothing
)::DataFrame

    # ======== 先生成所有实验组合的列表 ========
    # 每个元素是一个 (algorithm, cut, num, T) 的 tuple
    all_tasks = [(a, c, n, T)
                 for a in algorithms
                 for c in cuts
                 for n in nums
                 for T in Ts]

    # 如果指定了 task_ids，就只保留那一部分
    if task_ids !== nothing
        all_tasks = all_tasks[task_ids]
    end

    @info "Total experiment configs: $(length(all_tasks))"

    # ======== 用一个 DataFrame 收集 summary ========
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
    )

    # ======== 按任务列表依次跑 ========
    for (algorithm, cut, num, T) in all_tasks
        println()
        @info "=================================================================="
        @info "Start: case=$case, alg=$algorithm, cut=$cut, T=$T, num=$num"
        @info "=================================================================="

        result = run_single_experiment(
            algorithm, cut, T, num;
            case              = case,
            numScenarios      = numScenarios,
            M                 = M,
            logger_save       = logger_save,
            med_method        = med_method,
            ε                 = ε,
            ℓ                 = ℓ,
            δ                 = δ,
            sparse_cut        = sparse_cut,
            tightness         = tightness,
            branch_variable   = branch_variable,
            LiftIterThreshold = LiftIterThreshold,
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
        ))

        @everywhere GC.gc()
    end

    return summary_df
end


if abspath(PROGRAM_FILE) == @__FILE__
    ## 用法示例：julia cutTest.jl 1 20   # 跑第 1 到第 20 个组合
    # julia src/multistage_stochastic_unit_commitment/test/cutTest.jl 1 50
    # julia src/multistage_stochastic_unit_commitment/test/cutTest.jl 51 100
    # julia src/multistage_stochastic_unit_commitment/test/cutTest.jl 101 144


    if length(ARGS) == 2
        start_id = parse(Int, ARGS[1])
        end_id   = parse(Int, ARGS[2])
        @info "Running experiments $start_id:$end_id ..."
        summary = run_experiment_grid(task_ids = start_id:end_id)
        outname = "results_summary_$(start_id)_$(end_id).csv"
    else
        @info "Running full experiment grid..."
        summary = run_experiment_grid(
            case         = "case30",
            algorithms   = [:SDDPL, :SDDP, :SDDiP],
            # cuts         = [:PLC, :SMC, :LC, :SBC, :SBCLC, :SBCSMC, :SBCPLC, :NormalizedCut],
            cuts         = [:SBCNormalizedCut],
            nums         = [5, 10],
            Ts           = [6, 8, 12],
            numScenarios = 500,
            M            = 1,
            logger_save  = true,
            med_method   = :IntervalMed,
            ε            = 1 / 2^8,
            ℓ            = 0.5,
            δ            = 1e-2,
            sparse_cut   = :sparse,
            tightness    = false,
            branch_variable   = :ALL,
            LiftIterThreshold = 2,
        )
        outname = "results_summary_full.csv"
    end

    @show summary
    CSV.write(joinpath(project_root, outname), summary)
end