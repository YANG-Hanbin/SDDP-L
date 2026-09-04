using CSV
using DataFrames
using Dates
using JLD2
using Statistics

const TABLE_GAP_THRESHOLD = 0.1

std_or_zero(values) = length(values) <= 1 ? 0.0 : std(values)

function parse_gap_percent(value)::Float64
    value === missing && return NaN
    text = replace(String(value), "%" => "")
    parsed = tryparse(Float64, strip(text))
    return parsed === nothing ? NaN : parsed
end

function row_value(row, names; default = missing)
    props = propertynames(row)
    for name in names
        name in props && return getproperty(row, name)
    end
    return default
end

function first_time_to_gap(solHistory::DataFrame, threshold::Float64 = TABLE_GAP_THRESHOLD)::Float64
    for row in eachrow(solHistory)
        gap = parse_gap_percent(row_value(row, (:gap, :gap_str), default = "NaN"))
        if isfinite(gap) && gap <= threshold
            time_value = row_value(row, (:Time, :time), default = NaN)
            return Float64(time_value)
        end
    end
    return NaN
end

function time_at_best_lb(solHistory::DataFrame)::Float64
    nrow(solHistory) == 0 && return NaN
    idx = argmax(solHistory.LB)
    row = solHistory[idx, :]
    return Float64(row_value(row, (:Time, :time), default = NaN))
end

function runtime_stats(runtimeHistory::DataFrame)
    if !(:completed_backward in propertynames(runtimeHistory))
        return (completed_iterations = 0, avg_iteration_time = NaN, std_iteration_time = NaN)
    end

    completed = runtimeHistory[runtimeHistory.completed_backward .== true, :]
    if nrow(completed) == 0
        return (completed_iterations = 0, avg_iteration_time = NaN, std_iteration_time = NaN)
    end

    return (
        completed_iterations = nrow(completed),
        avg_iteration_time = mean(completed.iteration_time),
        std_iteration_time = std_or_zero(completed.iteration_time),
    )
end

function save_runtime_detail!(
    out_path::AbstractString,
    runtimeHistory::DataFrame;
    problem::AbstractString,
    algorithm::Symbol,
    cut::Symbol,
    partition_rule::Symbol,
    T::Int,
    num::Int,
)
    detail = deepcopy(runtimeHistory)
    detail[!, :problem] = fill(String(problem), nrow(detail))
    detail[!, :algorithm] = fill(algorithm, nrow(detail))
    detail[!, :cut] = fill(cut, nrow(detail))
    detail[!, :partition_rule] = fill(partition_rule, nrow(detail))
    detail[!, :T] = fill(T, nrow(detail))
    detail[!, :num] = fill(num, nrow(detail))
    select!(
        detail,
        :problem,
        :algorithm,
        :cut,
        :partition_rule,
        :T,
        :num,
        Not([:problem, :algorithm, :cut, :partition_rule, :T, :num]),
    )
    CSV.write(out_path, detail)
    return detail
end

function summarize_result(;
    problem::AbstractString,
    algorithm::Symbol,
    cut::Symbol,
    partition_rule::Symbol,
    T::Int,
    num::Int,
    result,
    elapsed_wall_time::Float64,
    status::AbstractString = "ok",
    error_message::AbstractString = "",
)
    if result === nothing
        return (
            problem = String(problem),
            algorithm = algorithm,
            cut = cut,
            partition_rule = partition_rule,
            T = T,
            num = num,
            status = String(status),
            LB = NaN,
            UB = NaN,
            gap_percent = NaN,
            iterations = 0,
            completed_iterations = 0,
            avg_iteration_time = NaN,
            std_iteration_time = NaN,
            final_time = NaN,
            time_to_gap_0p1 = NaN,
            time_at_best_lb = NaN,
            elapsed_wall_time = elapsed_wall_time,
            error_message = String(error_message),
        )
    end

    solHistory = result[:solHistory]
    runtimeHistory = result[:runtimeHistory]
    last_row = solHistory[end, :]
    stats = runtime_stats(runtimeHistory)
    gap_percent = parse_gap_percent(row_value(last_row, (:gap, :gap_str), default = "NaN"))

    return (
        problem = String(problem),
        algorithm = algorithm,
        cut = cut,
        partition_rule = partition_rule,
        T = T,
        num = num,
        status = String(status),
        LB = Float64(last_row.LB),
        UB = Float64(last_row.UB),
        gap_percent = gap_percent,
        iterations = Int(row_value(last_row, (:iter, :Iter), default = nrow(solHistory))),
        completed_iterations = stats.completed_iterations,
        avg_iteration_time = stats.avg_iteration_time,
        std_iteration_time = stats.std_iteration_time,
        final_time = Float64(row_value(last_row, (:Time, :time), default = NaN)),
        time_to_gap_0p1 = first_time_to_gap(solHistory),
        time_at_best_lb = time_at_best_lb(solHistory),
        elapsed_wall_time = elapsed_wall_time,
        error_message = String(error_message),
    )
end

function write_summary(path::AbstractString, rows::Vector)
    CSV.write(path, DataFrame(rows))
end

function compact_error(err, bt)::String
    text = sprint(showerror, err, bt)
    return replace(text, '\n' => " | ")
end
