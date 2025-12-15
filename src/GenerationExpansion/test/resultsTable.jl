using Pkg;
Pkg.activate(".");
using JuMP, Gurobi, ParallelDataTransfer;
using Distributions, Statistics, StatsBase, Distributed, Random;
using Test, Dates, Printf;
using CSV, DataFrames;
using JLD2, FileIO;
using PrettyTables;
using VegaLite, VegaDatasets;

const GRB_ENV = Gurobi.Env();
project_root = @__DIR__;
include(joinpath(project_root, "src", "GenerationExpansion", "utilities", "structs.jl"));


## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- comparison with algorithms/cuts ----------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
# 初始化DataFrame
algorithm = :SDDPL;
sparsity = true
result_df = DataFrame(
    cut=Symbol[], 
    T=Int[], 
    num=Int[], 
    best_LB=Float64[],         
    final_gap=Float64[], 
    total_iter=Int[], 
    avg_iter_time=String[],         
    # best_LB_time=Float64[], 
    # best_LB_iter=Int[],
    gap_under_1_time=Union{Missing, Float64}[],
    gap_under_1_iter=Union{Missing, Int}[]
);

for cut in [:LC, :LNC, :PLC, :SMC]
    for T in [10, 15]
        for num in [5, 10]
            try
                # 加载数据
                file_path = "/Users/aaron/SDDiP_with_EnhancedCut/src/GenerationExpansion/logger/case=GenerationExpansion/alg=$algorithm/T=$T/Real=$num/cut=$(cut)__sparsity=$(sparsity)__discZ=true.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                LB = solHistory.LB
                UB = solHistory.UB
                n  = length(LB)

                # 1) 逐步更新“到目前为止最好的 UB”
                best_UB = similar(UB)
                best_UB[1] = UB[1]
                for i in 2:n
                    best_UB[i] = min(best_UB[i-1], UB[i])
                end

                # 2) 用 best_UB 来重算每一轮的 gap
                gap_best = (best_UB .- LB) ./ best_UB .* 100.0

                # 3) 你要的统计量：
                #    - best_LB 仍然用 LB 最大值
                best_LB, best_LB_idx = findmax(LB)

                #    - 最终 gap 用“best UB 定义”的 gap
                final_gap = gap_best[end]

                #    - 总迭代数
                total_iter = solHistory.iter[end]

                #    - 平均迭代时间（还是用 Time 列）
                iter_times = diff(solHistory.Time)
                avg_time   = mean(iter_times)
                std_time   = std(iter_times)
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time

                #    - 到 best LB 的时间 / 迭代数
                best_LB_time = solHistory.Time[best_LB_idx]
                best_LB_iter = solHistory.iter[best_LB_idx]

                # 4) 第一次 gap < 1% 的位置（用新 gap_best）
                below1_idx = findfirst(<(0.1), gap_best)

                gap_under_1_iter = missing
                gap_under_1_time = missing
                if below1_idx !== nothing
                    gap_under_1_iter = solHistory.iter[below1_idx]
                    gap_under_1_time = solHistory.Time[below1_idx]
                end

                # 5) 写入结果 DataFrame（gap 列现在已经是 Float64，而不是解析字符串）
                push!(result_df, (
                    cut,
                    T,
                    num,
                    best_LB,
                    final_gap,
                    total_iter,
                    avg_iter_time,
                    gap_under_1_time,
                    gap_under_1_iter,
                ))
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
## -------------------------------------------------------------------------- comparison with partition rules ----------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
# 初始化DataFrame
algorithm = :SDDPL;
sparsity = true
partitionRule = :Bisection
result_df = DataFrame(
    cut=Symbol[], 
    T=Int[], 
    num=Int[], 
    best_LB=Float64[],         
    final_gap=Float64[], 
    total_iter=Int[], 
    avg_iter_time=String[],         
    # best_LB_time=Float64[], 
    # best_LB_iter=Int[],
    gap_under_1_time=Union{Missing, Float64}[],
    gap_under_1_iter=Union{Missing, Int}[]
);

for cut in [:LC, :LNC, :PLC, :SMC]
    for T in [10, 15]
        for num in [5, 10]
            try
                # 加载数据
                file_path = "/Users/aaron/SDDiP_with_EnhancedCut/src/GenerationExpansion/new_logger/case=GenerationExpansion/alg=$algorithm/T=$T/Real=$num/cut=$(cut)__partitionRule=$(partitionRule)__M=1__sparsity=true__discZ=true__run=20251213.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                LB = solHistory.LB
                UB = solHistory.UB
                n  = length(LB)

                # 1) 逐步更新“到目前为止最好的 UB”
                best_UB = similar(UB)
                best_UB[1] = UB[1]
                for i in 2:n
                    best_UB[i] = min(best_UB[i-1], UB[i])
                end

                # 2) 用 best_UB 来重算每一轮的 gap
                gap_best = (best_UB .- LB) ./ best_UB .* 100.0

                # 3) 你要的统计量：
                #    - best_LB 仍然用 LB 最大值
                best_LB, best_LB_idx = findmax(LB)

                #    - 最终 gap 用“best UB 定义”的 gap
                final_gap = gap_best[end]

                #    - 总迭代数
                total_iter = solHistory.iter[end]

                #    - 平均迭代时间（还是用 Time 列）
                iter_times = diff(solHistory.Time)
                avg_time   = mean(iter_times)
                std_time   = std(iter_times)
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time

                #    - 到 best LB 的时间 / 迭代数
                best_LB_time = solHistory.Time[best_LB_idx]
                best_LB_iter = solHistory.iter[best_LB_idx]

                # 4) 第一次 gap < 1% 的位置（用新 gap_best）
                below1_idx = findfirst(<(0.1), gap_best)

                gap_under_1_iter = missing
                gap_under_1_time = missing
                if below1_idx !== nothing
                    gap_under_1_iter = solHistory.iter[below1_idx]
                    gap_under_1_time = solHistory.Time[below1_idx]
                end

                # 5) 写入结果 DataFrame（gap 列现在已经是 Float64，而不是解析字符串）
                push!(result_df, (
                    cut,
                    T,
                    num,
                    best_LB,
                    final_gap,
                    total_iter,
                    avg_iter_time,
                    gap_under_1_time,
                    gap_under_1_iter,
                ))
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
## -------------------------------------------------------------------------- cut combinations with SBC ----------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
# 初始化DataFrame
algorithm = :SDDP;
sparsity = true;
cuts = [:SBCLC, :SBCLNC, :SBCPLC, :SBCSMC];
cuts = [:LC, :LNC, :PLC, :SMC];
result_df = DataFrame(
    cut=Symbol[], 
    T=Int[], 
    num=Int[], 
    best_LB=Float64[],         
    final_gap=Float64[], 
    total_iter=Int[], 
    avg_iter_time=String[],         
    best_LB_time=Float64[], 
    best_LB_iter=Int[],
    gap_under_1_time=Union{Missing, Float64}[],
    gap_under_1_iter=Union{Missing, Int}[]
);

for cut in cuts
    for T in [10, 15]
        for num in [5, 10]
            try
                # 加载数据
                file_path = "/Users/aaron/SDDiP_with_EnhancedCut/src/GenerationExpansion/logger/case=GenerationExpansion/alg=$algorithm/T=$T/Real=$num/cut=$(cut)__sparsity=$(sparsity)__discZ=true.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                LB = solHistory.LB
                UB = solHistory.UB
                n  = length(LB)

                # 1) 逐步更新“到目前为止最好的 UB”
                best_UB = similar(UB)
                best_UB[1] = UB[1]
                for i in 2:n
                    best_UB[i] = min(best_UB[i-1], UB[i])
                end

                # 2) 用 best_UB 来重算每一轮的 gap
                gap_best = (best_UB .- LB) ./ best_UB .* 100.0

                # 3) 你要的统计量：
                #    - best_LB 仍然用 LB 最大值
                best_LB, best_LB_idx = findmax(LB)

                #    - 最终 gap 用“best UB 定义”的 gap
                final_gap = gap_best[end]

                #    - 总迭代数
                total_iter = solHistory.iter[end]

                #    - 平均迭代时间（还是用 Time 列）
                iter_times = diff(solHistory.Time)
                avg_time   = mean(iter_times)
                std_time   = std(iter_times)
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time

                #    - 到 best LB 的时间 / 迭代数
                best_LB_time = solHistory.Time[best_LB_idx]
                best_LB_iter = solHistory.iter[best_LB_idx]

                # 4) 第一次 gap < 1% 的位置（用新 gap_best）
                below1_idx = findfirst(<(0.1), gap_best)

                gap_under_1_iter = missing
                gap_under_1_time = missing
                if below1_idx !== nothing
                    gap_under_1_iter = solHistory.iter[below1_idx]
                    gap_under_1_time = solHistory.Time[below1_idx]
                end

                # 5) 写入结果 DataFrame（gap 列现在已经是 Float64，而不是解析字符串）
                push!(result_df, (
                    cut,
                    T,
                    num,
                    best_LB,
                    final_gap,
                    total_iter,
                    avg_iter_time,
                    best_LB_time,
                    best_LB_iter,
                    gap_under_1_time,
                    gap_under_1_iter,
                ))
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
## -------------------------------------------------------------------------- sparsity ----------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
algorithm = :SDDPL

cuts = [:LC, :LNC, :PLC, :SMC]
Ts   = [10, 15]
nums = [5, 10]
sparsities = [true, false]

###############################################################
# 1. 每个 (cut, sparsity, T, num) 的结果表
###############################################################
result_df = DataFrame(
    cut             = Symbol[],
    sparsity        = Bool[],
    T               = Int[],
    num             = Int[],
    best_LB         = Float64[],
    final_gap       = Float64[],                 # 用 best_UB 计算的最终 gap
    total_iter      = Int[],
    avg_time_mean   = Float64[],                 # 平均每次迭代时间
    avg_time_std    = Float64[],                 # 迭代时间标准差
    gap_under_1_time = Union{Missing, Float64}[],
    gap_under_1_iter = Union{Missing, Int}[],
)

for sparsity in sparsities
    for cut in cuts
        for T in Ts
            for num in nums
                try
                    # ---------------------- 加载数据 ----------------------
                    file_path = "/Users/aaron/SDDiP_with_EnhancedCut/src/GenerationExpansion/logger/case=GenerationExpansion/alg=$algorithm/T=$T/Real=$num/cut=$(cut)__sparsity=$(sparsity)__discZ=true.jld2"
                    sddpResults = load(file_path)["sddpResults"]
                    solHistory  = sddpResults[:solHistory]

                    LB = solHistory.LB
                    UB = solHistory.UB
                    n  = length(LB)

                    # ---------------------- best_UB & gap ----------------------
                    best_UB = similar(UB)
                    best_UB[1] = UB[1]
                    for i in 2:n
                        best_UB[i] = min(best_UB[i-1], UB[i])
                    end

                    gap_best = (best_UB .- LB) ./ best_UB .* 100.0

                    # best LB 及 index
                    best_LB, best_LB_idx = findmax(LB)

                    # 最终 gap（用 best_UB）
                    final_gap  = gap_best[end]
                    total_iter = solHistory.iter[end]

                    # 迭代时间统计
                    iter_times = diff(solHistory.Time)
                    avg_time   = mean(iter_times)
                    std_time   = std(iter_times)

                    # 第一次 gap < 1% 的迭代/时间
                    below1_idx = findfirst(<(1.0), gap_best)

                    gap_under_1_iter = missing
                    gap_under_1_time = missing
                    if below1_idx !== nothing
                        gap_under_1_iter = solHistory.iter[below1_idx]
                        gap_under_1_time = solHistory.Time[below1_idx]
                    end

                    # 写入 result_df
                    push!(result_df, (
                        cut,
                        sparsity,
                        T,
                        num,
                        best_LB,
                        final_gap,
                        total_iter,
                        avg_time,
                        std_time,
                        gap_under_1_time,
                        gap_under_1_iter,
                    ))
                catch e
                    @warn "Error processing file: $file_path" exception=(e, catch_backtrace())
                end
            end
        end
    end
end

###############################################################
# 2. 按 (cut, sparsity) 聚合：把所有 (T,num) 合起来
###############################################################
cut_summary_df = combine(
    groupby(result_df, [:cut, :sparsity]),
    :avg_time_mean => mean => :mean_avg_iter_time,  # 平均迭代时间（秒）
    :total_iter    => mean => :mean_total_iter,     # 平均迭代次数
)

