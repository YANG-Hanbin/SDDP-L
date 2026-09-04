const _UC_PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
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

ensure_julia_11!(_UC_PROJECT_ROOT)

include(joinpath(@__DIR__, "loadMod.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    # Example usage:
    #   julia cutTest.jl 1 20
    # to run configurations 1 through 20 only.
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
            # algorithms   = [:SDDPL, :SDDiP, :SDDP],
            algorithms   = [:SDDP],
            # cuts         = [:PLC, :SMC, :LC, :LNC, :SBC, :SBCLC, :SBCSMC, :SBCPLC, :ReLUC, :NormalizedReLUC],
            cuts         = [:NormalizedReLUC],
            nums         = [10],
            Ts           = [6, 8, 12],
            numScenarios = 500,
            M            = 1,
            logger_save  = true,
            partitionRule= :Bisection,
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
    CSV.write(joinpath(PROJECT_ROOT, outname), summary)
end
