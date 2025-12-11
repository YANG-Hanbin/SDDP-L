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