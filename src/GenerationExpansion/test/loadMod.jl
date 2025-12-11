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