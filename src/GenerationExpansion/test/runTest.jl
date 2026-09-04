include(joinpath(@__DIR__, "loadMod.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    run_generation_expansion_experiments(
        algorithms = [:SDDP, :SDDPL, :SDDiP],
        cutTypes = [:LC, :LNC, :PLC, :SMC],
        T_list = [10, 15],
        num_list = [5, 10],
        timeSDDP = 3600.0,
        gapSDDP = 1e-3,
        iterSDDP = 300,
        levelMethodMaxIter = 200,
        sample_size_SDDP = 500,
        solverGap = 1e-6,
        solverTime = 20.0,
        ε = 1e-4,
        discreteZ = true,
        cutSparsity = true,
        partitionRule = :Bisection,
        branchingStart = 3,
        M = 1,
        verbose = false,
        ℓ1 = 0.0,
        ℓ2 = 0.0,
        nxt_bound = 1e5,
        logger_save = true,
    )
end
