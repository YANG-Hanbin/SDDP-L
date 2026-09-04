include(joinpath(@__DIR__, "loadMod.jl"))

const GEP_TIME_TEST = (
    algorithms = [:SDDP, :SDDiP, :SDDPL],
    cutTypes = [:LC, :LNC, :PLC, :SMC],
    T_list = [15],
    num_list = [10],
    kwargs = (
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
        branchingStart = 10,
        M = 1,
        verbose = false,
        ℓ1 = 0.0,
        ℓ2 = 0.0,
        nxt_bound = 1e5,
        logger_save = true,
        corePointStrategy = :Mid,
        corePointWeight = 0.5,
        corePointEpsilon = 1e-2,
        lncMinScale = 1e-3,
        lncCoreThetaMargin = 1e-4,
        cutDiagnostics = false,
    ),
)

function run_generation_expansion_time_test()
    cfg = GEP_TIME_TEST

    return run_generation_expansion_experiments(
        ;
        algorithms = cfg.algorithms,
        cutTypes = cfg.cutTypes,
        T_list = cfg.T_list,
        num_list = cfg.num_list,
        cfg.kwargs...,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_generation_expansion_time_test()
end
