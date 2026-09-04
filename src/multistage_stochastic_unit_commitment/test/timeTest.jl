include(joinpath(@__DIR__, "loadMod.jl"))

const TIME_RESULTS_DIR = joinpath(@__DIR__, "time_results")

const MSUC_TIME_CONFIG = (
    case = "case30",
    T = 12,
    num = 10,
    algorithms = [:SDDiP],
    cuts = [:LC, :SMC, :PLC, :LNC],
    experiment_kwargs = (
        numScenarios = 500,
        M = 1,
        logger_save = true,
        terminate_time = 3600,
        TimeLimit = 10,
        terminate_threshold = 1e-3,
        MIPGap = 1e-4,
        MaxIter = 3000,
        partitionRule = :Bisection,
        ε = 1 / 2^8,
        δ = 1e-2,
        levelMethodMaxIter = 200,
        levelMethodVerbose = false,
        core_point_strategy = "Mid",
        core_point_weight = 0.5,
        core_point_epsilon = 1e-2,
        sparse_cut = :sparse,
        tightness = false,
        branch_variable = :ALL,
        LiftIterThreshold = 2,
        lncMinScale = 1e-3,
        lncCoreThetaMargin = 1e-4,
        cutDiagnostics = false,
    ),
)

std_or_zero(values) = length(values) <= 1 ? 0.0 : std(values)

function add_configuration_columns(
    runtimeHistory::DataFrame;
    problem::AbstractString,
    algorithm::Symbol,
    cut::Symbol,
    T::Int,
    num::Int,
)::DataFrame
    df = deepcopy(runtimeHistory)
    df[!, :problem] = fill(String(problem), nrow(df))
    df[!, :algorithm] = fill(algorithm, nrow(df))
    df[!, :cut_family] = fill(cut, nrow(df))
    df[!, :T] = fill(T, nrow(df))
    df[!, :num] = fill(num, nrow(df))
    select!(
        df,
        :problem,
        :algorithm,
        :cut_family,
        :T,
        :num,
        Not([:problem, :algorithm, :cut_family, :T, :num]),
    )
    return df
end

function summarize_runtime_history(
    runtimeHistory::DataFrame;
    problem::AbstractString,
    algorithm::Symbol,
    cut::Symbol,
    T::Int,
    num::Int,
)::NamedTuple
    completed = runtimeHistory[runtimeHistory.completed_backward .== true, :]

    avg_iteration_time = mean(completed.iteration_time)
    avg_forward_time = mean(completed.forward_time)
    avg_lifting_time = mean(completed.lifting_time)
    avg_backward_time = mean(completed.backward_time)
    avg_other_time = mean(completed.other_time)
    share(component_time) =
        avg_iteration_time <= 0.0 ? 0.0 : 100.0 * component_time / avg_iteration_time

    return (
        problem = problem,
        algorithm = algorithm,
        cut = cut,
        T = T,
        num = num,
        completed_iterations = nrow(completed),
        recorded_iterations = nrow(runtimeHistory),
        avg_iteration_time = avg_iteration_time,
        std_iteration_time = std_or_zero(completed.iteration_time),
        avg_forward_time = avg_forward_time,
        avg_lifting_time = avg_lifting_time,
        avg_backward_time = avg_backward_time,
        avg_other_time = avg_other_time,
        forward_share = share(avg_forward_time),
        lifting_share = share(avg_lifting_time),
        backward_share = share(avg_backward_time),
        other_share = share(avg_other_time),
        final_recorded_time = runtimeHistory.total_time[end],
    )
end

function write_runtime_tables(
    results::Dict;
    problem::AbstractString,
    detail_path::AbstractString,
    summary_path::AbstractString,
)
    detail_frames = DataFrame[]
    summary_rows = NamedTuple[]

    for (algorithm, cut, T, num) in sort(collect(keys(results)))
        runtimeHistory = results[(algorithm, cut, T, num)][:runtimeHistory]
        push!(
            detail_frames,
            add_configuration_columns(
                runtimeHistory;
                problem = problem,
                algorithm = algorithm,
                cut = cut,
                T = T,
                num = num,
            ),
        )
        push!(
            summary_rows,
            summarize_runtime_history(
                runtimeHistory;
                problem = problem,
                algorithm = algorithm,
                cut = cut,
                T = T,
                num = num,
            ),
        )
    end

    mkpath(dirname(detail_path))
    detail = vcat(detail_frames...)
    summary = DataFrame(summary_rows)
    CSV.write(detail_path, detail)
    CSV.write(summary_path, summary)

    return (detail = detail, summary = summary)
end

function run_msuc_time_test()
    cfg = MSUC_TIME_CONFIG
    results = Dict{Tuple{Symbol, Symbol, Int, Int}, Dict}()

    for algorithm in cfg.algorithms, cut in cfg.cuts
        output = run_single_experiment(
            algorithm,
            cut,
            cfg.T,
            cfg.num;
            case = cfg.case,
            cfg.experiment_kwargs...,
        )
        results[(algorithm, cut, cfg.T, cfg.num)] = output.sddpResults
        @everywhere GC.gc()
    end

    return write_runtime_tables(
        results;
        problem = "MSUC",
        detail_path = joinpath(
            TIME_RESULTS_DIR,
            "msuc_runtime_detail_largest_T$(cfg.T)_R$(cfg.num).csv",
        ),
        summary_path = joinpath(
            TIME_RESULTS_DIR,
            "msuc_runtime_summary_largest_T$(cfg.T)_R$(cfg.num).csv",
        ),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    output = run_msuc_time_test()
    show(output.summary; allcols = true, allrows = true)
    println()
end
