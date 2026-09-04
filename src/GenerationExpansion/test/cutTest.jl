const _GEP_PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const _JULIA11_BIN = joinpath(homedir(), ".juliaup", "bin", "julia")

"""
    ensure_julia_11!(project_root::AbstractString)

Relaunch the current script with Julia 1.11 when it is started from a
Julia 1.12+ process. The environment flag prevents recursive relaunching.
"""
function ensure_julia_11!(project_root::AbstractString)::Nothing
    if VERSION >= v"1.12" &&
       get(ENV, "SDDIP_SKIP_JULIA_REEXEC", "0") != "1" &&
       !isempty(PROGRAM_FILE) &&
       isfile(PROGRAM_FILE) &&
       isfile(_JULIA11_BIN)

        script_path = abspath(PROGRAM_FILE)
        cmd = `$_JULIA11_BIN +1.11 --project=$project_root $script_path $(ARGS...)`
        run(setenv(cmd, merge(copy(ENV), Dict("SDDIP_SKIP_JULIA_REEXEC" => "1"))))
        exit(0)
    end
    return
end

ensure_julia_11!(_GEP_PROJECT_ROOT)

include(joinpath(@__DIR__, "loadMod.jl"))


if abspath(PROGRAM_FILE) == @__FILE__
    algorithms = [:SDDP, :SDDPL, :SDDiP]
    cutTypes = [:LC, :SMC, :PLC, :LNC, :SBC, :SBCLC, :SBCSMC, :SBCPLC, :SBCLNC, :ReLUC, :NormalizedReLUC]
    T_list = [10, 15]
    num_list = [5, 10]
    partitionRule = :Incumbent
    cutSparsity = true

    results = run_generation_expansion_experiments(
        algorithms                  = algorithms,
        cutTypes                    = cutTypes,
        T_list                      = T_list,
        num_list                    = num_list,
        timeSDDP                    = 3600.0,
        gapSDDP                     = 1e-3,
        iterSDDP                    = 300,
        sample_size_SDDP            = 500,
        solverGap                   = 1e-6,
        solverTime                  = 20.0,
        ε                           = 1e-4,
        discreteZ                   = true,
        cutSparsity                 = cutSparsity,
        partitionRule               = partitionRule,
        branchingStart              = 3,
        M                           = 1,
        verbose                     = false,
        ℓ1                          = 0.0,
        ℓ2                          = 0.0,
        nxt_bound                   = 1e5,
        logger_save                 = true
    )
end
