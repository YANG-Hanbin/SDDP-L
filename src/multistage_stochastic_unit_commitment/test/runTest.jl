cd(@__DIR__)
include(joinpath(@__DIR__, "loadMod.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    # Example usage:
    #   julia runTest.jl 1 20
    # to run configurations 1 through 20 only.
    # julia src/multistage_stochastic_unit_commitment/test/runTest.jl 1 50
    # julia src/multistage_stochastic_unit_commitment/test/runTest.jl 51 100
    # julia src/multistage_stochastic_unit_commitment/test/runTest.jl 101 144

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
            algorithms   = [:SDDPL],
            cuts         = [:LNC, :SMC],
            nums         = [5, 10],
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
