using Pkg

const PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
@info "Project root detected as: $PROJECT_ROOT"

Pkg.activate(PROJECT_ROOT)
using Distributed
addprocs(5)

const GEP_SRC = abspath(joinpath(@__DIR__, ".."))
@info "GEP source dir: $GEP_SRC"
@everywhere const PROJECT_ROOT = $PROJECT_ROOT
@everywhere const GEP_SRC       = $GEP_SRC

@everywhere begin
    using JuMP, Gurobi, PowerModels
    using Statistics, StatsBase, Random, Dates, Distributions
    using Distributed, ParallelDataTransfer
    using CSV, DataFrames, Printf
    using JLD2, FileIO
    using Base.Filesystem: mkpath, dirname, isdir

    const GRB_ENV = Gurobi.Env()

    include(joinpath(GEP_SRC, "utilities", "structs.jl"))
    include(joinpath(GEP_SRC, "utilities", "utils.jl"))
    include(joinpath(GEP_SRC, "forwardPass.jl"))
    include(joinpath(GEP_SRC, "backwardPass.jl"))
    include(joinpath(GEP_SRC, "level_method.jl"))
    include(joinpath(GEP_SRC, "utilities", "setting.jl"))
    include(joinpath(GEP_SRC, "cut_variants.jl"))
    include(joinpath(GEP_SRC, "sddp.jl"))
end


"""
Run generation expansion experiments over multiple
algorithms, cut types, T, and num.

Returns:
    results :: Dict{Tuple{Symbol,Symbol,Int,Int}, Dict}
    keyed by (algorithm, cutType, T, num) => sddipResults
"""
function run_generation_expansion_experiments(;
    # algorithm list: SDDP / SDDPL / SDDiP
    algorithms::Vector{Symbol} = [:SDDP, :SDDPL, :SDDiP],

    # cut types: SMC / PLC / LC
    cutTypes::Vector{Symbol}   = [:SMC, :PLC, :LC],

    # time horizon and number of scenarios
    T_list::Vector{Int}        = [10, 15],
    num_list::Vector{Int}      = [5, 10],

    # SDDP / SDDiP parameters
    timeSDDP::Float64          = 3600.0,
    gapSDDP::Float64           = 1e-3,
    iterSDDP::Int              = 300,
    sample_size_SDDP::Int      = 500,
    solverGap::Float64         = 1e-6,
    solverTime::Float64        = 20.0,

    # model / level-set parameters
    ε::Float64                 = 1e-4,
    discreteZ::Bool            = true,
    cutSparsity::Bool          = true,
    partitionRule::Symbol      = :Incumbent,
    branchingStart::Int        = 3,
    M::Int                     = 1,
    verbose::Bool              = false,
    ℓ1::Float64                = 0.0,
    ℓ2::Float64                = 0.0,
    nxt_bound::Float64         = 1e8,
    logger_save::Bool          = true,
)

    # results[(algorithm, cutType, T, num)] = sddipResults
    results = Dict{Tuple{Symbol,Symbol,Int,Int}, Dict}()

    for algorithm in algorithms,
        cutType   in cutTypes,
        T         in T_list,
        num       in num_list

        @info "Running algorithm = $algorithm, cutType = $cutType, T = $T, num = $num"

        # ------------------ load data ------------------
        data_dir = joinpath(
            PROJECT_ROOT,
            "src", "GenerationExpansion", "numerical_data",
            "testData_stage($T)_real($num)",
        )

        stageDataList = load(joinpath(data_dir, "stageDataList.jld2"))["stageDataList"]
        Ω             = load(joinpath(data_dir, "Ω.jld2"))["Ω"]
        binaryInfo    = load(joinpath(data_dir, "binaryInfo.jld2"))["binaryInfo"]
        probList      = load(joinpath(data_dir, "probList.jld2"))["probList"]

        # ------------------ build parameter NamedTuple ------------------
        param = param_setup(
            timeSDDP         = timeSDDP,
            gapSDDP          = gapSDDP,
            iterSDDP         = iterSDDP,
            sample_size_SDDP = sample_size_SDDP,
            ε                = ε,
            discreteZ        = discreteZ,
            cutType          = cutType,
            cutSparsity      = cutSparsity,
            partitionRule    = partitionRule,
            branchingStart   = branchingStart,
            M                = M,
            T                = T,
            num              = num,
            verbose          = verbose,
            ℓ1               = ℓ1,
            ℓ2               = ℓ2,
            nxt_bound        = nxt_bound,
            logger_save      = logger_save,
            algorithm        = algorithm,
        )

        # ------------------ broadcast data to all workers ------------------
        @everywhere begin
            stageDataList = $stageDataList
            Ω             = $Ω
            binaryInfo    = $binaryInfo
            probList      = $probList
            param         = $param
        end

        # ------------------ run algorithm ------------------
        sddipResults = stochastic_dual_dynamic_programming_algorithm(
            Ω,
            probList,
            stageDataList;
            binaryInfo = binaryInfo,
            param      = param,
        )

        # 存结果（按 (algorithm, cutType, T, num) 作为 key）
        results[(algorithm, cutType, T, num)] = sddipResults

        # 清一下 GC
        @everywhere GC.gc()
    end

    return results
end