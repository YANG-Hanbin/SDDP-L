using Pkg
Pkg.activate(".")
using JuMP, Gurobi, PowerModels
using Statistics, StatsBase, Random, Dates, Distributions
using Distributed, ParallelDataTransfer
using CSV, DataFrames, Printf
using JLD2, FileIO
using StatsPlots, PlotThemes
using PrettyTables;
using VegaLite, VegaDatasets;

project_root = @__DIR__;
include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "utilities", "structs.jl"))
theme(:default)

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## --------------------------------------------------------------------------------- TO GENERATE SDDP-L ----------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
algorithm = :SDDPL;
sparsity = :sparse;
med = :IntervalMed; # :IntervalMed, :ExactPoint
result_df = DataFrame(
    cut=Symbol[], 
    T=Int[], 
    num=Int[], 
    LB=Float64[],         
    Gap=Float64[], 
    Iter=Int[], 
    Time=String[],         
    # best_LB_time=Float64[], 
    # best_LB_iter=Int[],
    Time1=Union{Missing, Float64}[],
    Iter1=Union{Missing, Int}[],
    totTime=Union{Missing, Float64}[]
);

for cut in [:LC, :LNC, :PLC, :SMC]
    for T in [6, 8, 12]
        for num in [5, 10]
            try
                file_path = "/Users/aaron/SDDiP_with_EnhancedCut/src/results/case=case30/alg=$algorithm/T=$T/Real=$num/cut=$(cut)__med=$(med)__eps=256__ell=0.5__sparsity=$(sparsity).jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                # 计算所需的统计数据
                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                total_iter = solHistory.Iter[end]  # 总迭代数
                iter_times = diff(solHistory.Time)  # 计算每次迭代的时间间隔
                avg_time = mean(iter_times)  # 计算平均迭代时间
                std_time = std(iter_times)   # 计算标准差
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time  # 格式化字符串
                best_LB_time = solHistory.Time[best_LB_idx]  # 到达best LB的时间
                best_LB_iter = solHistory.Iter[best_LB_idx]  # 到达best LB的迭代数

                # 将 gap 列（字符串）转换为 Float64 含义的百分数
                gap_vals = parse.(Float64, replace.(solHistory.gap, "%" => ""))

                # 找到 gap 第一次小于 1.0 的位置
                below1_idx = findfirst(<(0.1), gap_vals)

                # 初始化默认值
                gap_under_1_iter = missing
                gap_under_1_time = missing

                if below1_idx !== nothing
                    gap_under_1_iter = solHistory.Iter[below1_idx]
                    gap_under_1_time = solHistory.Time[below1_idx]
                end

                # 添加到DataFrame
                push!(result_df, (
                    cut, T, num, best_LB, final_gap, total_iter, 
                    avg_iter_time, 
                    # best_LB_time, best_LB_iter,
                    gap_under_1_time, gap_under_1_iter, solHistory.Time[end]
                    )
                );
            catch e
                @warn "Error processing file: $file_path" exception=(e, catch_backtrace())
            end
        end
    end
end

# 定义格式化函数，保留一位小数
column_formatter = function(x, i, j)
    if x isa Float64
        return @sprintf("%.1f", x)  # 保留一位小数
    elseif x isa Tuple  # 处理 iter_range 之类的元组数据
        return "$(x[1])--$(x[2])"
    else
        return string(x)  # 其他数据类型转换为字符串
    end
end

# 生成 LaTeX 表格
latex_table = pretty_table(
    String, 
    result_df, 
    backend=Val(:latex),
    formatters=(column_formatter,)
)

# 输出 LaTeX 代码
println(latex_table)

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ------------------------------------------------------------------------------- TO GENERATE SDDiP/SDDP --------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
algorithm = :SDDiP;
sparsity = :sparse;
med = :IntervalMed; # :IntervalMed, :ExactPoint
result_df = DataFrame(
    cut=Symbol[], 
    T=Int[], 
    num=Int[], 
    LB=Float64[],         
    Gap=Float64[], 
    Iter=Int[], 
    Time=String[],         
    # best_LB_time=Float64[], 
    # best_LB_iter=Int[],
    Time1=Union{Missing, Float64}[],
    Iter1=Union{Missing, Int}[],
    totTime=Union{Missing, Float64}[]
);

for cut in [:LC, :LNC, :PLC, :SMC]
    for T in [6, 8, 12]
        for num in [5, 10]
            try
                file_path = "/Users/aaron/SDDiP_with_EnhancedCut/src/results/case=case30/alg=$algorithm/T=$T/Real=$num/cut=$(cut)__med=$(med)__eps=256__ell=0.5__sparsity=$(sparsity).jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                # 计算所需的统计数据
                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                total_iter = solHistory.Iter[end]  # 总迭代数
                iter_times = diff(solHistory.Time)  # 计算每次迭代的时间间隔
                avg_time = mean(iter_times)  # 计算平均迭代时间
                std_time = std(iter_times)   # 计算标准差
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time  # 格式化字符串
                best_LB_time = solHistory.Time[best_LB_idx]  # 到达best LB的时间
                best_LB_iter = solHistory.Iter[best_LB_idx]  # 到达best LB的迭代数

                # 将 gap 列（字符串）转换为 Float64 含义的百分数
                gap_vals = parse.(Float64, replace.(solHistory.gap, "%" => ""))

                # 找到 gap 第一次小于 1.0 的位置
                below1_idx = findfirst(<(1.0), gap_vals)

                # 初始化默认值
                gap_under_1_iter = missing
                gap_under_1_time = missing

                if below1_idx !== nothing
                    gap_under_1_iter = solHistory.Iter[below1_idx]
                    gap_under_1_time = solHistory.Time[below1_idx]
                end

                # 添加到DataFrame
                push!(result_df, (
                    cut, T, num, best_LB, final_gap, total_iter, 
                    avg_iter_time, 
                    # best_LB_time, best_LB_iter,
                    gap_under_1_time, gap_under_1_iter, solHistory.Time[end]
                    )
                );
            catch e
                @warn "Error processing file: $file_path" exception=(e, catch_backtrace())
            end
        end
    end
end

# 定义格式化函数，保留一位小数
column_formatter = function(x, i, j)
    if x isa Float64
        return @sprintf("%.1f", x)  # 保留一位小数
    elseif x isa Tuple  # 处理 iter_range 之类的元组数据
        return "$(x[1])--$(x[2])"
    else
        return string(x)  # 其他数据类型转换为字符串
    end
end

# 生成 LaTeX 表格
latex_table = pretty_table(
    String, 
    result_df, 
    backend=Val(:latex),
    formatters=(column_formatter,)
)

# 输出 LaTeX 代码
println(latex_table)

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------  SDDP-L with SBC ------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
algorithm = :SDDPL;
sparsity = :sparse;
med = :IntervalMed; # :IntervalMed, :ExactPoint
result_df = DataFrame(
    cut=Symbol[], 
    T=Int[], 
    num=Int[], 
    LB=Float64[],         
    Gap=Float64[], 
    Iter=Int[], 
    Time=String[],         
    # best_LB_time=Float64[], 
    # best_LB_iter=Int[],
    Time1=Union{Missing, Float64}[],
    Iter1=Union{Missing, Int}[],
    totTime=Union{Missing, Float64}[]
);

for cut in [:SBCLC, :SBCLNC, :SBCPLC, :SBCSMC]
    for T in [6, 8, 12]
        for num in [5, 10]
            try
                file_path = "/Users/aaron/SDDiP_with_EnhancedCut/src/results/case=case30/alg=$algorithm/T=$T/Real=$num/cut=$(cut)__med=$(med)__eps=256__ell=0.5__sparsity=$(sparsity).jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                # 计算所需的统计数据
                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                total_iter = solHistory.Iter[end]  # 总迭代数
                iter_times = diff(solHistory.Time)  # 计算每次迭代的时间间隔
                avg_time = mean(iter_times)  # 计算平均迭代时间
                std_time = std(iter_times)   # 计算标准差
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time  # 格式化字符串
                best_LB_time = solHistory.Time[best_LB_idx]  # 到达best LB的时间
                best_LB_iter = solHistory.Iter[best_LB_idx]  # 到达best LB的迭代数

                # 将 gap 列（字符串）转换为 Float64 含义的百分数
                gap_vals = parse.(Float64, replace.(solHistory.gap, "%" => ""))

                # 找到 gap 第一次小于 1.0 的位置
                below1_idx = findfirst(<(0.1), gap_vals)

                # 初始化默认值
                gap_under_1_iter = missing
                gap_under_1_time = missing

                if below1_idx !== nothing
                    gap_under_1_iter = solHistory.Iter[below1_idx]
                    gap_under_1_time = solHistory.Time[below1_idx]
                end

                # 添加到DataFrame
                push!(result_df, (
                    cut, T, num, best_LB, final_gap, total_iter, 
                    avg_iter_time, 
                    # best_LB_time, best_LB_iter,
                    gap_under_1_time, gap_under_1_iter, solHistory.Time[end]
                    )
                );
            catch e
                @warn "Error processing file: $file_path" exception=(e, catch_backtrace())
            end
        end
    end
end

# 定义格式化函数，保留一位小数
column_formatter = function(x, i, j)
    if x isa Float64
        return @sprintf("%.1f", x)  # 保留一位小数
    elseif x isa Tuple  # 处理 iter_range 之类的元组数据
        return "$(x[1])--$(x[2])"
    else
        return string(x)  # 其他数据类型转换为字符串
    end
end

# 生成 LaTeX 表格
latex_table = pretty_table(
    String, 
    result_df, 
    backend=Val(:latex),
    formatters=(column_formatter,)
)

# 输出 LaTeX 代码
println(latex_table)

## ---------------------------------------------------------------------------------------------------------------------------------------- ##
## ------------------------------------------------------------  Sparsity  ---------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------- ##
algorithm = :SDDPL
med = :IntervalMed  # :IntervalMed, :ExactPoint

result_df = DataFrame(
    cut      = Symbol[],
    T        = Int[],
    num      = Int[],
    LB       = Float64[],
    Gap      = Float64[],
    Iter     = Int[],
    TimeMean = Float64[],               # 平均迭代时间（数值）
    TimeStd  = Float64[],               # 迭代时间标准差（数值）
    TimeStr  = String[],                # "0.3 (0.1)" 这种字符串
    Time1    = Union{Missing, Float64}[],  # gap<1% 时的时间
    Iter1    = Union{Missing, Int}[],      # gap<1% 时的迭代数
    sparsity = Symbol[],
)

for sparsity in [:sparse, :dense]
    for cut in [:LC, :LNC, :PLC, :SMC]
        for T in [6, 8, 12]
            for num in [5, 10]
                file_path = "/Users/aaron/SDDiP_with_EnhancedCut/src/results/case=case30/alg=$algorithm/T=$T/Real=$num/cut=$(cut)__med=$(med)__eps=256__ell=0.5__sparsity=$(sparsity).jld2"
                try
                    solHistory = load(file_path)["sddpResults"][:solHistory]

                    # 最优 LB 及其索引
                    best_LB, best_LB_idx = findmax(solHistory.LB)

                    # 最终 gap（百分数字符串转成 Float64）
                    final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))

                    # 总迭代数
                    total_iter = solHistory.Iter[end]

                    # 每次迭代时间间隔
                    iter_times = diff(solHistory.Time)
                    avg_time = mean(iter_times)
                    std_time = std(iter_times)

                    # 字符串形式（比如 "0.3 (0.1)"）
                    avg_iter_time_str = @sprintf "%.1f (%.1f)" avg_time std_time

                    # gap 曲线转成 Float64
                    gap_vals = parse.(Float64, replace.(solHistory.gap, "%" => ""))

                    below1_idx = findfirst(<(1.0), gap_vals)

                    gap_under_1_iter = missing
                    gap_under_1_time = missing
                    if below1_idx !== nothing
                        gap_under_1_iter = solHistory.Iter[below1_idx]
                        gap_under_1_time = solHistory.Time[below1_idx]
                    end

                    push!(result_df, (
                        cut,
                        T,
                        num,
                        best_LB,
                        final_gap,
                        total_iter,
                        avg_time,
                        std_time,
                        avg_iter_time_str,
                        gap_under_1_time,
                        gap_under_1_iter,
                        sparsity,
                    ))
                catch e
                    @warn "Error processing file: $file_path" exception=(e, catch_backtrace())
                end
            end
        end
    end
end

results = result_df
# 颜色：给四种 cut 四个颜色
colors = ["#1f77b4", "#808080", "#ff7f0e", "#2ca02c"]  # LC, LNC, PLC, SMC

# 只取 T = 12, num = 10 的子集来画
T = 12; num = 5;
df_plot = filter(row -> row.T == T && row.num == num, results)

df_plot.sparsity_str = ifelse.(df_plot.sparsity .== :dense, "Nominal", "Sparse")

plt = df_plot |>
@vlplot(
    :bar,
    x = {
        "sparsity_str:n",
        title = nothing,
        axis = {labelFont="Times New Roman", labelFontSize=18, titleFontSize=18, labelAngle=0}
    },
    xOffset = {"cut:n", title="Cut"},
    y = {
        "TimeMean:q",
        title = "Average Iteration Time",
        axis = {labelFontSize=18, titleFontSize=18}
    },
    color = {
        "cut:n",
        scale = {range=colors},
        title = nothing
    },
    column = {
        "T:n",
        header = {
            title = nothing,   # 不要标题
            labels = false     # 不要每个 facet 的 label（比如 6, 8, 12）
        }
    },
    row = {
        "num:n",
        header = {
            title = nothing,
            labels = false
        }
    },
    tooltip = [
        {"sparsity_str:n"},
        {"cut:n"},
        {"TimeMean:q"},
        {"TimeStd:q"},
        {"TimeStr:n"},
    ],
    width = 250,
    height = 250,
    config = {
        axis = {labelFont="Times New Roman", titleFont="Times New Roman"},
        legend = {
            labelFont="Times New Roman", titleFont="Times New Roman",
            labelFontSize=15, symbolSize=150, symbolStrokeWidth=4
        },
        title = {font="Times New Roman"},
        bar = {width=20}
    }
) +
@vlplot(
    :errorbar,
    x = {"sparsity_str:n"},
    xOffset = {"cut:n"},
    y = {"TimeMean:q"},
    yError = {"TimeStd:q"},
)|> save("$(homedir())/SDDiP_with_EnhancedCut/src/results/case=case30/alg=$algorithm/T=$(T)/Real=$(num)/avg_iter_time_T=$(T)_R=$(num).pdf")