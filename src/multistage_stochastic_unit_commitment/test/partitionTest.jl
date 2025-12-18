cd("/Users/aaron/SDDiP_with_EnhancedCut/src/multistage_stochastic_unit_commitment/test")  # 改变当前工作目录到脚本所在的目录
include(joinpath(@__DIR__, "loadMod.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    for partitionRule in [:Bisection, :Incumbent]
        summary = run_experiment_grid(
            case         = "case30",
            algorithms   = [:SDDPL],
            cuts         = [:LC, :NormalizedCut, :PLC, :SMC],
            nums         = [5, 10],
            Ts           = [6, 8, 12],
            numScenarios = 500,
            M            = 1,
            logger_save  = true,
            partitionRule= partitionRule,
            ε            = 1 / 2^8,
            ℓ            = 0.5,
            δ            = 1e-2,
            sparse_cut   = :sparse,
            tightness    = true,
            branch_variable   = :ALL,
            LiftIterThreshold = 2
        )
    end
end