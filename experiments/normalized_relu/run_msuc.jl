const SCRIPT_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))
const OUT_DIR = joinpath(REPOSITORY_ROOT, "src", "results", "normalized_relu")
mkpath(OUT_DIR)

include(joinpath(REPOSITORY_ROOT, "src", "multistage_stochastic_unit_commitment", "test", "loadMod.jl"))
include(joinpath(SCRIPT_DIR, "table_helpers.jl"))

const MSUC_INSTANCES = [(6, 5), (6, 10), (8, 5), (8, 10), (12, 5), (12, 10)]
const MSUC_SUMMARY_PATH = joinpath(OUT_DIR, "msuc_sddpl_b_normalized_relu_summary.csv")
const MSUC_DETAIL_DIR = joinpath(OUT_DIR, "msuc_runtime_detail")
const MSUC_FULL_DIR = joinpath(OUT_DIR, "msuc_full_results")
mkpath(MSUC_DETAIL_DIR)
mkpath(MSUC_FULL_DIR)

summary_rows = NamedTuple[]
failed_instances = Tuple{Int,Int}[]

for (T, num) in MSUC_INSTANCES
    started = now()
    result = nothing
    status = "ok"
    error_message = ""
    @info "MSUC SDDP-L-B NormalizedReLUC start" T num started

    try
        output = run_single_experiment(
            :SDDPL,
            :NormalizedReLUC,
            T,
            num;
            case = "case30",
            numScenarios = 500,
            M = 1,
            logger_save = false,
            terminate_time = 3600,
            TimeLimit = 10,
            terminate_threshold = 1e-3,
            MIPGap = 1e-4,
            MaxIter = 3000,
            partitionRule = :Bisection,
            ε = 1 / 2^8,
            δ = 1e-2,
            levelMethodMaxIter = 200,
            levelMethodVerbose = false,
            core_point_strategy = "Mid",
            core_point_weight = 0.5,
            core_point_epsilon = 1e-2,
            sparse_cut = :sparse,
            tightness = false,
            branch_variable = :ALL,
            LiftIterThreshold = 2,
            lncMinScale = 1e-3,
            lncCoreThetaMargin = 1e-4,
            cutDiagnostics = false,
        )
        result = output.sddpResults

        full_path = joinpath(MSUC_FULL_DIR, "msuc_case30_T$(T)_R$(num)_sddpl_b_normalized_relu.jld2")
        JLD2.jldsave(
            full_path;
            result = result,
            config = (
                problem = "MSUC",
                case = "case30",
                algorithm = :SDDPL,
                cut = :NormalizedReLUC,
                partition_rule = :Bisection,
                T = T,
                num = num,
            ),
        )

        save_runtime_detail!(
            joinpath(MSUC_DETAIL_DIR, "msuc_case30_T$(T)_R$(num)_runtime.csv"),
            result[:runtimeHistory];
            problem = "MSUC",
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
        @error "MSUC SDDP-L-B NormalizedReLUC failed" T num error_message
    end

    elapsed = (now() - started).value / 1000
    push!(
        summary_rows,
        summarize_result(
            problem = "MSUC",
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
    write_summary(MSUC_SUMMARY_PATH, summary_rows)
    @info "MSUC SDDP-L-B NormalizedReLUC finished" T num status elapsed
    @everywhere GC.gc()
end

if isempty(failed_instances)
    @info "MSUC SDDP-L-B NormalizedReLUC batch complete" summary_path = MSUC_SUMMARY_PATH
else
    @error "MSUC SDDP-L-B NormalizedReLUC batch completed with failures" failed_instances summary_path = MSUC_SUMMARY_PATH
    exit(1)
end
