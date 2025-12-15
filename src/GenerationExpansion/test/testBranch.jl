cd("/Users/aaron/SDDiP_with_EnhancedCut/src/GenerationExpansion/test")  # 改变当前工作目录到脚本所在的目录
include(joinpath(@__DIR__, "loadMod.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    algorithms = [:SDDPL]
    cutTypes = [:LC, :SMC]
    # cutTypes = [:PLC, :LNC]
    T_list = [10, 15]
    num_list = [5, 10]

    for partitionRule in [:Bisection]
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
            cutSparsity                 = true,
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
end