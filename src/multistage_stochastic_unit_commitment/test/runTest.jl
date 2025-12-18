# using Base.Filesystem  # 引入 Filesystem 模块
cd("/Users/aaron/SDDiP_with_EnhancedCut/src/multistage_stochastic_unit_commitment/test")  # 改变当前工作目录到脚本所在的目录
include(joinpath(@__DIR__, "loadMod.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    ## 用法示例：julia runTest.jl 1 20   # 跑第 1 到第 20 个组合
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
    CSV.write(joinpath(project_root, outname), summary)
end