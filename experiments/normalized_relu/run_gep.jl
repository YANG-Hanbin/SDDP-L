const SCRIPT_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))
const OUT_DIR = joinpath(REPOSITORY_ROOT, "src", "results", "normalized_relu")
mkpath(OUT_DIR)

include(joinpath(REPOSITORY_ROOT, "src", "GenerationExpansion", "test", "loadMod.jl"))
include(joinpath(SCRIPT_DIR, "table_helpers.jl"))

const GEP_INSTANCES = [(10, 5), (10, 10), (15, 5), (15, 10)]
const GEP_SUMMARY_PATH = joinpath(OUT_DIR, "gep_sddpl_b_normalized_relu_summary.csv")
const GEP_DETAIL_DIR = joinpath(OUT_DIR, "gep_runtime_detail")
const GEP_FULL_DIR = joinpath(OUT_DIR, "gep_full_results")
mkpath(GEP_DETAIL_DIR)
mkpath(GEP_FULL_DIR)

summary_rows = NamedTuple[]
failed_instances = Tuple{Int,Int}[]

for (T, num) in GEP_INSTANCES
    started = now()
    result = nothing
    status = "ok"
    error_message = ""
    @info "GEP SDDP-L-B NormalizedReLUC start" T num started

    try
        result_dict = run_generation_expansion_experiments(
            algorithms = [:SDDPL],
            cutTypes = [:NormalizedReLUC],
            T_list = [T],
            num_list = [num],
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
            logger_save = false,
            corePointStrategy = :Mid,
            corePointWeight = 0.5,
            corePointEpsilon = 1e-2,
            lncMinScale = 1e-3,
            lncCoreThetaMargin = 1e-4,
            cutDiagnostics = false,
        )
        result = result_dict[(:SDDPL, :NormalizedReLUC, T, num)]

        full_path = joinpath(GEP_FULL_DIR, "gep_T$(T)_R$(num)_sddpl_b_normalized_relu.jld2")
        JLD2.jldsave(
            full_path;
            result = result,
            config = (
                problem = "GEP",
                algorithm = :SDDPL,
                cut = :NormalizedReLUC,
                partition_rule = :Bisection,
                T = T,
                num = num,
            ),
        )

        save_runtime_detail!(
            joinpath(GEP_DETAIL_DIR, "gep_T$(T)_R$(num)_runtime.csv"),
            result[:runtimeHistory];
            problem = "GEP",
            algorithm = :SDDPL,
            cut = :NormalizedReLUC,
            partition_rule = :Bisection,
            T = T,
            num = num,
        )
    catch err
        status = "error"
        error_message = compact_error(err, catch_backtrace())
        push!(failed_instances, (T, num))
        @error "GEP SDDP-L-B NormalizedReLUC failed" T num error_message
    end

    elapsed = (now() - started).value / 1000
    push!(
        summary_rows,
        summarize_result(
            problem = "GEP",
            algorithm = :SDDPL,
            cut = :NormalizedReLUC,
            partition_rule = :Bisection,
            T = T,
            num = num,
            result = result,
            elapsed_wall_time = elapsed,
            status = status,
            error_message = error_message,
        ),
    )
    write_summary(GEP_SUMMARY_PATH, summary_rows)
    @info "GEP SDDP-L-B NormalizedReLUC finished" T num status elapsed
    @everywhere GC.gc()
end

if isempty(failed_instances)
    @info "GEP SDDP-L-B NormalizedReLUC batch complete" summary_path = GEP_SUMMARY_PATH
else
    @error "GEP SDDP-L-B NormalizedReLUC batch completed with failures" failed_instances summary_path = GEP_SUMMARY_PATH
    exit(1)
end
