# Bounded numerical validation for the supplied GEP inputs and public
# algorithm/cut entry points. Reports are written below the ignored
# src/results/validation/gep directory.
const SCRIPT_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))
const JULIA_11_BIN = joinpath(homedir(), ".juliaup", "bin", "julia")

function ensure_julia_11!()::Nothing
    if !(VERSION.major == 1 && VERSION.minor == 11) &&
       get(ENV, "SDDP_L_SKIP_JULIA_REEXEC", "0") != "1" &&
       isfile(JULIA_11_BIN) &&
       isfile(PROGRAM_FILE)

        script_path = abspath(PROGRAM_FILE)
        cmd = `$JULIA_11_BIN +1.11 --project=$REPOSITORY_ROOT $script_path $(ARGS...)`
        run(setenv(cmd, merge(copy(ENV), Dict("SDDP_L_SKIP_JULIA_REEXEC" => "1"))))
        exit(0)
    end

    VERSION.major == 1 && VERSION.minor == 11 || error(
        "GEP validation requires Julia 1.11; running $(VERSION).",
    )
    return nothing
end

ensure_julia_11!()

include(joinpath(REPOSITORY_ROOT, "src", "GenerationExpansion", "test", "loadMod.jl"))
using SHA
using LinearAlgebra

const OUTPUT_DIR = joinpath(REPOSITORY_ROOT, "src", "results", "validation", "gep")
const INPUT_SUMMARY_PATH = joinpath(OUTPUT_DIR, "input_validation.csv")
const SCHEDULE_SUMMARY_PATH = joinpath(OUTPUT_DIR, "schedule_validation.csv")
const COVERAGE_SUMMARY_PATH = joinpath(OUTPUT_DIR, "coverage.csv")
const INTERFACE_COVERAGE_PATH = joinpath(OUTPUT_DIR, "interface_coverage.csv")
const RUN_SUMMARY_PATH = joinpath(OUTPUT_DIR, "run_validation.csv")
const CROSS_CASE_SUMMARY_PATH = joinpath(OUTPUT_DIR, "cross_case_validation.csv")
const CONTRACT_SUMMARY_PATH = joinpath(OUTPUT_DIR, "contract_validation.csv")
const PARAMETER_CONTRACT_SUMMARY_PATH = joinpath(OUTPUT_DIR, "parameter_contract_validation.csv")
const WARM_ALIAS_DYNAMIC_SUMMARY_PATH = joinpath(OUTPUT_DIR, "warm_alias_dynamic_validation.csv")
const DEEP_SUMMARY_PATH = joinpath(OUTPUT_DIR, "deep_adaptive_validation.csv")
const SDDIP_SOLUTION_SUMMARY_PATH = joinpath(OUTPUT_DIR, "sddip_solution_validation.csv")
const METADATA_PATH = joinpath(OUTPUT_DIR, "metadata.csv")

const GEP_INSTANCES = [(10, 5), (10, 10), (15, 5), (15, 10)]
const GEP_ALGORITHMS = [:SDDP, :SDDPL, :SDDiP]
const GEP_BASE_CUTS = [
    :LC,
    :SMC,
    :PLC,
    :AdaptiveSMC,
    :AdaptivePLC,
    :LNC,
    :SBC,
    :ReLUC,
    :NormalizedReLUC,
]
const ALL_INPUT_CUTS = [:LC, :AdaptivePLC, :AdaptiveSMC, :NormalizedReLUC]
const WARM_CUT_TARGETS = [
    (:SBCLC, :LC),
    (:SBCSMC, :SMC),
    (:SBCPLC, :PLC),
    (:SBCLNC, :LNC),
    (:SBCReLUC, :ReLUC),
    (:SBCNormalizedReLUC, :NormalizedReLUC),
]
const REFERENCE_INSTANCE = (10, 5)
const EXPECTED_STATE_UPPER_BOUNDS = [4.0, 10.0, 10.0, 1.0, 45.0, 4.0]
const EXPECTED_INITIAL_STATE = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
const DEEP_ADAPTIVE_BASELINES = Dict(
    :AdaptivePLC => (
        LB = [
            4591.8207151999995,
            6721.450344829629,
            7796.805228971196,
            7796.805228971196,
        ],
        UB = [
            81937.0193568608,
            78401.80771430997,
            89217.79860122477,
            82561.54209906622,
        ],
    ),
    :AdaptiveSMC => (
        LB = [
            4591.8207151999995,
            6721.450344829629,
            7536.265159644444,
            9665.894789274073,
        ],
        UB = [
            81937.0193568608,
            70389.85110738366,
            86066.09875295933,
            83850.28782518965,
        ],
    ),
)
const DEEP_BASELINE_RTOL = 1e-5
const DEEP_BASELINE_ATOL = 1e-3

const EXPECTED_FILE_HASHES = Dict(
    (10, 5, "stageDataList.jld2") => "b255c10cdb6a9d3170d81ac671e9a8d4baf8eb804faaf8d550e8be1655f57b63",
    (10, 5, "Omega.jld2") => "325fdecf72fbc3fc7e50a80526d6ceb624ba5f660f4332ab383ce40e17d8fbad",
    (10, 5, "binaryInfo.jld2") => "238878349b2a9d70474204c4e84046f18ef91ca8209173cd47db1355570088e4",
    (10, 5, "probList.jld2") => "68cb1dd97c8f02d0e5c4f1a488730a5d3c417d8a22a1b307d6d71601a0126786",
    (10, 10, "stageDataList.jld2") => "b255c10cdb6a9d3170d81ac671e9a8d4baf8eb804faaf8d550e8be1655f57b63",
    (10, 10, "Omega.jld2") => "6aa00dda06ceb3d46956e9766e39d822f1c2bc7546e20c19eedd650abbc9a549",
    (10, 10, "binaryInfo.jld2") => "238878349b2a9d70474204c4e84046f18ef91ca8209173cd47db1355570088e4",
    (10, 10, "probList.jld2") => "0a97a1b7583e0ac1d63da452e82bee0be0040fd35a008354daeed309703f0042",
    (15, 5, "stageDataList.jld2") => "125166de3f77cf06fe7e57aecfb53d18509ab26a7f0ed126b5a7921df94ff69b",
    (15, 5, "Omega.jld2") => "a05e8e9eec1950acd976eb8d6e9d44ca6c3e0307f9bfc9809950c55519d2156b",
    (15, 5, "binaryInfo.jld2") => "238878349b2a9d70474204c4e84046f18ef91ca8209173cd47db1355570088e4",
    (15, 5, "probList.jld2") => "e5522623c38ba548f6349c8fdbd5b3cf38698f7a9bce0d13dff8a0c1f40babf8",
    (15, 10, "stageDataList.jld2") => "125166de3f77cf06fe7e57aecfb53d18509ab26a7f0ed126b5a7921df94ff69b",
    (15, 10, "Omega.jld2") => "978b13820daa3273796d92e371c8a52e48dfbb57910669e11e71ea33cb915ade",
    (15, 10, "binaryInfo.jld2") => "238878349b2a9d70474204c4e84046f18ef91ca8209173cd47db1355570088e4",
    (15, 10, "probList.jld2") => "aff6d13e868632241e9f933f82d0d614a6ebd743b7ee57e400ab781990e31edf",
)

const SOLUTION_HISTORY_COLUMNS = [
    :iter,
    :LB,
    :UB,
    :gap,
    :time,
    :LM_iter,
    :Time,
]
const RUNTIME_HISTORY_COLUMNS = [
    :Iter,
    :Algorithm,
    :Cut,
    :forward_time,
    :lifting_time,
    :backward_time,
    :other_time,
    :iteration_time,
    :total_time,
    :completed_backward,
    :min_lnc_scale,
    :max_lnc_tightness_gap,
    :num_lnc_fallback,
    :num_lnc_core_theta_fallback,
]

function compact_error(err, bt = nothing)::String
    rendered = bt === nothing ? sprint(showerror, err) : sprint(showerror, err, bt)
    return replace(strip(rendered), '\n' => ' ', '\r' => ' ')
end

function expect!(errors::Vector{String}, condition::Bool, message::AbstractString)::Nothing
    condition || push!(errors, String(message))
    return nothing
end

function write_rows(path::AbstractString, rows)::Nothing
    mkpath(dirname(path))
    CSV.write(path, DataFrame(rows))
    return nothing
end

sha256_hex(path::AbstractString)::String = bytes2hex(SHA.sha256(read(path)))

function fixture_directory(T::Int, num::Int)::String
    return joinpath(
        REPOSITORY_ROOT,
        "src",
        "GenerationExpansion",
        "numerical_data",
        "testData_stage($(T))_real($(num))",
    )
end

function load_single_key(path::AbstractString, expected_key::String, errors::Vector{String})
    payload = load(path)
    expect!(
        errors,
        Set(String.(keys(payload))) == Set([expected_key]),
        "$(basename(path)) must contain exactly the key $(expected_key)",
    )
    return get(payload, expected_key, nothing)
end

function validate_input(T::Int, num::Int)
    errors = String[]
    data_dir = fixture_directory(T, num)
    file_specs = [
        ("stageDataList.jld2", "stageDataList", "stageDataList.jld2"),
        ("Omega.jld2", "Ω", "Ω.jld2"),
        ("binaryInfo.jld2", "binaryInfo", "binaryInfo.jld2"),
        ("probList.jld2", "probList", "probList.jld2"),
    ]
    hashes = Dict{String, String}()
    loaded = Dict{String, Any}()

    try
        for (hash_name, key_name, disk_name) in file_specs
            path = joinpath(data_dir, disk_name)
            expect!(errors, isfile(path), "missing fixture file $(disk_name)")
            isfile(path) || continue

            actual_hash = sha256_hex(path)
            hashes[hash_name] = actual_hash
            expect!(
                errors,
                actual_hash == EXPECTED_FILE_HASHES[(T, num, hash_name)],
                "unexpected SHA-256 for $(disk_name)",
            )
            loaded[key_name] = load_single_key(path, key_name, errors)
        end

        stage_data = get(loaded, "stageDataList", nothing)
        omega = get(loaded, "Ω", nothing)
        binary_info = get(loaded, "binaryInfo", nothing)
        probabilities = get(loaded, "probList", nothing)

        expect!(errors, stage_data isa Dict{Int, StageData}, "stageDataList has the wrong concrete type")
        expect!(errors, omega isa Dict{Int, Dict{Int, RandomVariables}}, "Omega has the wrong concrete type")
        expect!(errors, binary_info isa BinaryInfo, "binaryInfo has the wrong concrete type")
        expect!(errors, probabilities isa Dict{Int, Vector{Float64}}, "probList has the wrong concrete type")

        if stage_data isa Dict{Int, StageData}
            expect!(errors, sort(collect(keys(stage_data))) == collect(1:T), "stageDataList stage keys are inconsistent")
            for t in sort(collect(keys(stage_data)))
                stage = stage_data[t]
                expect!(errors, length(stage.c1) == 6, "stage $(t) c1 dimension is not 6")
                expect!(errors, length(stage.c2) == 6, "stage $(t) c2 dimension is not 6")
                expect!(errors, length(stage.ū) == 6, "stage $(t) state-bound dimension is not 6")
                expect!(errors, size(stage.N) == (6, 6), "stage $(t) capacity matrix is not 6x6")
                expect!(errors, length(stage.s₀) == 6, "stage $(t) initial-state dimension is not 6")
                expect!(errors, all(isfinite, stage.c1) && all(>=(0.0), stage.c1), "stage $(t) c1 is not finite/nonnegative")
                expect!(errors, all(isfinite, stage.c2) && all(>=(0.0), stage.c2), "stage $(t) c2 is not finite/nonnegative")
                expect!(errors, all(isfinite, stage.ū) && all(>=(0.0), stage.ū), "stage $(t) state bounds are invalid")
                expect!(errors, all(isfinite, stage.N) && all(>=(0.0), stage.N), "stage $(t) capacity matrix is invalid")
                expect!(errors, all(isfinite, stage.s₀) && all(>=(0.0), stage.s₀), "stage $(t) initial state is invalid")
                expect!(errors, isfinite(stage.h) && stage.h > 0.0, "stage $(t) hours are invalid")
                expect!(errors, isfinite(stage.penalty) && stage.penalty > 0.0, "stage $(t) penalty is invalid")
                expect!(errors, stage.ū == EXPECTED_STATE_UPPER_BOUNDS, "stage $(t) has unexpected state bounds")
                expect!(errors, stage.s₀ == EXPECTED_INITIAL_STATE, "stage $(t) has an unexpected initial state")
                expect!(errors, all(stage.s₀ .<= stage.ū), "stage $(t) initial state exceeds its bounds")
                expect!(errors, all(iszero, stage.N - Diagonal(diag(stage.N))), "stage $(t) capacity matrix is not diagonal")
                expect!(errors, all(diag(stage.N) .> 0.0), "stage $(t) capacity diagonal is not positive")
            end
            if length(stage_data) >= 2
                first_stage = stage_data[1]
                for t in 2:T
                    stage = stage_data[t]
                    expect!(errors, stage.ū == first_stage.ū, "state bounds vary across stages")
                    expect!(errors, stage.N == first_stage.N, "capacity matrix varies across stages")
                    expect!(errors, stage.s₀ == first_stage.s₀, "initial state varies across stages")
                    expect!(errors, stage.h == first_stage.h, "hours vary across stages")
                    expect!(errors, stage.penalty == first_stage.penalty, "penalty varies across stages")
                end
                expect!(errors, all(g -> all(diff([stage_data[t].c1[g] for t in 1:T]) .< 0.0), 1:6), "investment costs are not strictly decreasing by stage")
                expect!(errors, all(g -> all(diff([stage_data[t].c2[g] for t in 1:T]) .> 0.0), 1:6), "operating costs are not strictly increasing by stage")
            end
        end

        if omega isa Dict{Int, Dict{Int, RandomVariables}}
            expect!(errors, sort(collect(keys(omega))) == collect(1:T), "Omega stage keys are inconsistent")
            for t in sort(collect(keys(omega)))
                expect!(errors, sort(collect(keys(omega[t]))) == collect(1:num), "Omega node keys are inconsistent at stage $(t)")
                for node in sort(collect(keys(omega[t])))
                    demand = omega[t][node].d
                    expect!(errors, length(demand) == 1, "demand dimension is not 1 at stage $(t), node $(node)")
                    expect!(errors, all(isfinite, demand) && all(>(0.0), demand), "demand is invalid at stage $(t), node $(node)")
                end
            end
            if haskey(omega, 1) && !isempty(omega[1])
                root_demand = first(values(omega[1])).d
                expect!(errors, all(value.d == root_demand for value in values(omega[1])), "root demand realizations are inconsistent")
                expect!(errors, root_demand == [5.68e6], "root demand has an unexpected value")
            end
            expect!(errors, all(node -> all(diff([omega[t][node].d[1] for t in 1:T]) .>= 0.0), 1:num), "demand is not nondecreasing along a node index")
        end

        if probabilities isa Dict{Int, Vector{Float64}}
            expect!(errors, sort(collect(keys(probabilities))) == collect(1:T), "probList stage keys are inconsistent")
            for t in sort(collect(keys(probabilities)))
                p = probabilities[t]
                expect!(errors, length(p) == num, "probability dimension is inconsistent at stage $(t)")
                expect!(errors, all(isfinite, p) && all(>(0.0), p), "probabilities are invalid at stage $(t)")
                expect!(errors, isapprox(sum(p), 1.0; atol = 1e-12, rtol = 0.0), "probabilities do not sum to one at stage $(t)")
                expect!(errors, all(isapprox.(p, 1.0 / num; atol = 1e-12, rtol = 0.0)), "probabilities are not uniform at stage $(t)")
            end
        end

        if omega isa Dict{Int, Dict{Int, RandomVariables}} &&
           probabilities isa Dict{Int, Vector{Float64}}
            Random.seed!(20260904)
            sampled_paths = SampleScenarios(omega, probabilities; M = 100)
            expect!(errors, sort(collect(keys(sampled_paths))) == collect(1:100), "sampled path keys are inconsistent")
            for path in values(sampled_paths)
                expect!(errors, length(path) == T, "a sampled path has the wrong horizon")
                length(path) == T || continue
                expect!(errors, path[1] == 1, "a sampled path does not start at the root node")
                expect!(errors, all(path[t] in keys(omega[t]) for t in 1:T), "a sampled path contains an unknown node")
            end
        end

        if binary_info isa BinaryInfo
            expect!(errors, binary_info.d == 6, "binaryInfo generator dimension is not 6")
            expect!(errors, binary_info.n == 21, "binaryInfo binary dimension is not 21")
            expect!(errors, size(binary_info.A) == (binary_info.d, binary_info.n), "binaryInfo matrix dimensions are inconsistent")
            expect!(errors, all(binary_info.A .>= 0), "binaryInfo contains negative weights")
            for column in eachcol(binary_info.A)
                expect!(errors, count(value -> !iszero(value), column) == 1, "a binaryInfo column maps to other than one state component")
            end
            for g in 1:binary_info.d
                weights = filter(value -> !iszero(value), collect(binary_info.A[g, :]))
                expect!(errors, weights == [2^(j - 1) for j in eachindex(weights)], "binaryInfo row $(g) is not a contiguous binary expansion")
                expect!(errors, sum(weights) >= EXPECTED_STATE_UPPER_BOUNDS[g], "binaryInfo row $(g) cannot represent its upper bound")
            end
        end
    catch err
        push!(errors, "input validation exception: $(compact_error(err, catch_backtrace()))")
    end

    return (
        application = "GEP",
        T = T,
        realizations = num,
        status = isempty(errors) ? "pass" : "fail",
        error_count = length(errors),
        errors = join(errors, " | "),
        stage_data_sha256 = get(hashes, "stageDataList.jld2", ""),
        omega_sha256 = get(hashes, "Omega.jld2", ""),
        binary_info_sha256 = get(hashes, "binaryInfo.jld2", ""),
        probability_sha256 = get(hashes, "probList.jld2", ""),
        expected_stages = T,
        expected_realizations_per_stage = num,
        state_dimension = 6,
        binary_dimension = 21,
    )
end

function validate_cut_schedules()
    rows = NamedTuple[]
    for (alias, target) in WARM_CUT_TARGETS
        observed = (get_cutType(alias, 1), get_cutType(alias, 3), get_cutType(alias, 4))
        expected = (:SBC, :SBC, target)
        errors = observed == expected ? "" : "expected $(expected), observed $(observed)"
        push!(
            rows,
            (
                alias = String(alias),
                target = String(target),
                iteration_1 = String(observed[1]),
                iteration_3 = String(observed[2]),
                iteration_4 = String(observed[3]),
                status = isempty(errors) ? "pass" : "fail",
                errors = errors,
            ),
        )
    end
    return rows
end

function validate_interface_coverage(run_rows, schedule_rows, warm_alias_rows)
    rows = NamedTuple[]

    for algorithm in GEP_ALGORITHMS, cut in GEP_BASE_CUTS
        is_supported_configuration(algorithm, cut) || continue
        matches = [
            row for row in run_rows if
            row.T == REFERENCE_INSTANCE[1] &&
            row.realizations == REFERENCE_INSTANCE[2] &&
            row.algorithm == String(algorithm) &&
            row.cut == String(cut)
        ]
        errors = String[]
        expect!(errors, length(matches) == 1, "expected one representative numerical run")
        expect!(errors, length(matches) == 1 && only(matches).status == "pass", "representative numerical run did not pass")
        push!(
            rows,
            (
                algorithm = String(algorithm),
                requested_cut = String(cut),
                category = "base",
                active_cut_from_iteration_4 = String(cut),
                validation_level = "two-iteration-numerical",
                status = isempty(errors) ? "pass" : "fail",
                errors = join(errors, " | "),
            ),
        )
    end

    for algorithm in GEP_ALGORITHMS, (alias, target) in WARM_CUT_TARGETS
        is_supported_configuration(algorithm, alias) || continue
        schedule_matches = [row for row in schedule_rows if row.alias == String(alias)]
        numerical_matches = [
            row for row in warm_alias_rows if
            row.algorithm == String(algorithm) &&
            row.requested_cut == String(alias)
        ]
        errors = String[]
        expect!(errors, length(schedule_matches) == 1, "expected one schedule-dispatch check")
        expect!(
            errors,
            length(schedule_matches) == 1 && only(schedule_matches).status == "pass",
            "schedule-dispatch check did not pass",
        )
        expect!(errors, length(numerical_matches) == 1, "expected one warm-alias numerical run")
        expect!(
            errors,
            length(numerical_matches) == 1 && only(numerical_matches).status == "pass",
            "warm-alias numerical run did not pass",
        )
        push!(
            rows,
            (
                algorithm = String(algorithm),
                requested_cut = String(alias),
                category = "warm-start-alias",
                active_cut_from_iteration_4 = String(target),
                validation_level = "four-iteration-numerical",
                status = isempty(errors) ? "pass" : "fail",
                errors = join(errors, " | "),
            ),
        )
    end

    length(rows) == 41 || error("Expected 41 supported GEP interface combinations, found $(length(rows)).")
    return rows
end

function validation_cases()
    cases = NamedTuple[]
    seen = Set{Tuple{Int, Int, Symbol, Symbol}}()

    function add_case!(T, num, algorithm, cut, coverage)
        key = (T, num, algorithm, cut)
        key in seen && return
        push!(seen, key)
        push!(cases, (; T, num, algorithm, cut, coverage))
    end

    for algorithm in GEP_ALGORITHMS, cut in GEP_BASE_CUTS
        is_supported_configuration(algorithm, cut) || continue
        add_case!(REFERENCE_INSTANCE..., algorithm, cut, "all-supported-combinations")
    end

    for (T, num) in GEP_INSTANCES, cut in ALL_INPUT_CUTS
        add_case!(T, num, :SDDPL, cut, "all-input-critical-paths")
    end

    return cases
end

function parse_gap_percent(value)::Float64
    return parse(Float64, replace(String(value), "%" => ""))
end

function validate_result(result, algorithm::Symbol, cut::Symbol, T::Int, num::Int)
    errors = String[]
    warnings = String[]
    metrics = Dict{Symbol, Any}(
        :history_rows => -1,
        :runtime_rows => -1,
        :solution_length => -1,
        :solution_values => "",
        :initial_lb => NaN,
        :final_lb => NaN,
        :initial_ub => NaN,
        :final_ub => NaN,
        :lb_monotone => false,
        :bound_order_observed => false,
        :gap_history_matches => false,
        :reported_lm_iter => -1,
        :completed_backward_rows => -1,
        :total_forward_time => NaN,
        :total_lifting_time => NaN,
        :total_backward_time => NaN,
        :reported_total_time => NaN,
    )

    if !(result isa AbstractDict)
        push!(errors, "result is not a dictionary")
        return (; errors, warnings, metrics)
    end

    expected_keys = Set([:solHistory, :solution, :gapHistory, :runtimeHistory])
    expect!(errors, Set(keys(result)) == expected_keys, "result dictionary keys do not match the public contract")
    all(haskey(result, key) for key in expected_keys) || return (; errors, warnings, metrics)

    history = result[:solHistory]
    runtime = result[:runtimeHistory]
    gap_history = result[:gapHistory]
    solution = result[:solution]

    expect!(errors, history isa DataFrame, "solHistory is not a DataFrame")
    expect!(errors, runtime isa DataFrame, "runtimeHistory is not a DataFrame")
    expect!(errors, gap_history isa AbstractVector, "gapHistory is not a vector")
    expect!(errors, solution isa AbstractVector, "solution is not a vector")

    if solution isa AbstractVector
        metrics[:solution_length] = length(solution)
        metrics[:solution_values] = join(Float64.(solution), ";")
        expect!(errors, length(solution) == 6, "solution is not in the six-dimensional original state space")
        expect!(errors, all(isfinite, solution), "solution contains a non-finite value")
        expect!(errors, all(solution .>= -1e-6), "solution violates a nonnegativity bound")
        expect!(errors, all(solution .<= EXPECTED_STATE_UPPER_BOUNDS .+ 1e-6), "solution exceeds a state upper bound")
        expect!(errors, all(abs.(solution .- round.(solution)) .<= 1e-6), "solution is not integral")
    end

    if history isa DataFrame
        metrics[:history_rows] = nrow(history)
        expect!(errors, Symbol.(names(history)) == SOLUTION_HISTORY_COLUMNS, "solHistory columns are inconsistent")
        expect!(errors, nrow(history) == 2, "solHistory does not contain exactly two requested iterations")

        if nrow(history) > 0 && all(name in Symbol.(names(history)) for name in SOLUTION_HISTORY_COLUMNS)
            lb = Float64.(history.LB)
            ub = Float64.(history.UB)
            metrics[:initial_lb] = first(lb)
            metrics[:final_lb] = last(lb)
            metrics[:initial_ub] = first(ub)
            metrics[:final_ub] = last(ub)
            expect!(errors, collect(history.iter) == collect(1:nrow(history)), "solHistory iteration indices are inconsistent")
            expect!(errors, all(isfinite, lb), "solHistory contains a non-finite lower bound")
            expect!(errors, all(isfinite, ub), "solHistory contains a non-finite statistical upper estimate")
            expect!(errors, all(isfinite, history.time) && all(history.time .>= 0.0), "solHistory iteration times are invalid")
            expect!(errors, all(isfinite, history.Time) && all(history.Time .>= 0.0), "solHistory cumulative times are invalid")
            expect!(errors, issorted(history.Time), "solHistory cumulative times are not monotone")
            expect!(errors, all(history.LM_iter .>= 0), "solHistory contains a negative level-method iteration count")

            lb_tolerance = 1e-5 * max(1.0, maximum(abs.(lb)))
            lb_monotone = all(diff(lb) .>= -lb_tolerance)
            metrics[:lb_monotone] = lb_monotone
            expect!(errors, lb_monotone, "lower bound decreases beyond numerical tolerance")

            bound_tolerance = 1e-5 .* max.(1.0, max.(abs.(lb), abs.(ub)))
            bound_order = all(lb .<= ub .+ bound_tolerance)
            metrics[:bound_order_observed] = bound_order
            bound_order || push!(warnings, "statistical upper estimate falls below the lower bound")

            if gap_history isa AbstractVector && length(gap_history) == nrow(history)
                expected_gap = [round((ub[i] - lb[i]) / ub[i] * 100; digits = 2) for i in eachindex(lb)]
                parsed_gap = try
                    parse_gap_percent.(history.gap)
                catch err
                    push!(errors, "could not parse solHistory gap strings: $(compact_error(err))")
                    fill(NaN, nrow(history))
                end
                gap_matches = all(isapprox.(Float64.(gap_history), expected_gap; atol = 1e-10, rtol = 0.0)) &&
                    all(isapprox.(parsed_gap, expected_gap; atol = 1e-10, rtol = 0.0))
                metrics[:gap_history_matches] = gap_matches
                expect!(errors, gap_matches, "gap fields do not match LB and UB")
            else
                push!(errors, "gapHistory length does not match solHistory")
            end

            if nrow(history) >= 2
                metrics[:reported_lm_iter] = history.LM_iter[1]
                expect!(errors, history.LM_iter[1] > 0, "completed backward pass reported no cut-generation iterations")
                expect!(errors, history.LM_iter[2] == 0, "terminal forward-only iteration reports cut-generation iterations")
            end
        end
    end

    if runtime isa DataFrame
        metrics[:runtime_rows] = nrow(runtime)
        expect!(errors, Symbol.(names(runtime)) == RUNTIME_HISTORY_COLUMNS, "runtimeHistory columns are inconsistent")
        expect!(errors, nrow(runtime) == 2, "runtimeHistory does not contain exactly two requested iterations")

        if nrow(runtime) > 0 && all(name in Symbol.(names(runtime)) for name in RUNTIME_HISTORY_COLUMNS)
            expect!(errors, collect(runtime.Iter) == collect(1:nrow(runtime)), "runtimeHistory iteration indices are inconsistent")
            expect!(errors, all(runtime.Algorithm .== algorithm), "runtimeHistory algorithm label is inconsistent")
            expect!(errors, all(runtime.Cut .== cut), "runtimeHistory cut label is inconsistent")

            timing_columns = [:forward_time, :lifting_time, :backward_time, :other_time, :iteration_time, :total_time]
            for column in timing_columns
                values = runtime[!, column]
                expect!(errors, all(isfinite, values) && all(values .>= 0.0), "runtimeHistory $(column) values are invalid")
            end
            expect!(errors, all(runtime.forward_time .> 0.0), "runtimeHistory contains a zero forward-pass time")
            reconstructed_other_time = max.(
                runtime.iteration_time .-
                runtime.forward_time .-
                runtime.lifting_time .-
                runtime.backward_time,
                0.0,
            )
            expect!(
                errors,
                all(isapprox.(runtime.other_time, reconstructed_other_time; atol = 1e-9, rtol = 1e-9)),
                "runtimeHistory component times do not reconstruct other_time",
            )
            expect!(errors, issorted(runtime.total_time), "runtimeHistory total_time is not monotone")

            if nrow(runtime) == 2
                expect!(errors, collect(runtime.completed_backward) == [true, false], "runtimeHistory backward-completion flags are inconsistent")
                expect!(errors, runtime.backward_time[1] > 0.0, "completed backward pass has zero runtime")
                expect!(errors, runtime.backward_time[2] == 0.0, "terminal forward-only iteration reports backward runtime")
                if history isa DataFrame && nrow(history) == 2
                    expect!(errors, all(isapprox.(history.time, runtime.iteration_time; atol = 1e-9, rtol = 1e-9)), "solHistory and runtimeHistory iteration times disagree")
                    expect!(errors, all(isapprox.(history.Time, runtime.total_time; atol = 1e-9, rtol = 1e-9)), "solHistory and runtimeHistory cumulative times disagree")
                end
                if algorithm == :SDDPL
                    expect!(errors, runtime.lifting_time[1] > 0.0, "SDDP-L completed iteration has zero lifting runtime")
                else
                    expect!(errors, all(runtime.lifting_time .== 0.0), "non-lifted algorithm reports lifting runtime")
                end

                if cut == :LNC
                    expect!(errors, isfinite(runtime.min_lnc_scale[1]), "completed LNC backward pass has no finite scale")
                    expect!(errors, runtime.min_lnc_scale[1] >= 1e-3, "completed LNC backward pass violates its minimum scale")
                    expect!(errors, isfinite(runtime.max_lnc_tightness_gap[1]) && runtime.max_lnc_tightness_gap[1] >= 0.0, "completed LNC backward pass has an invalid tightness gap")
                    max_subproblems = (T - 1) * num
                    expect!(errors, 0 <= runtime.num_lnc_fallback[1] <= max_subproblems, "LNC fallback count is out of range")
                    expect!(errors, 0 <= runtime.num_lnc_core_theta_fallback[1] <= max_subproblems, "LNC core-theta fallback count is out of range")
                    expect!(errors, isnan(runtime.min_lnc_scale[2]), "terminal forward-only LNC iteration reports a scale")
                else
                    expect!(errors, all(isnan, runtime.min_lnc_scale), "non-LNC run reports an LNC scale")
                    expect!(errors, all(isnan, runtime.max_lnc_tightness_gap), "non-LNC run reports an LNC tightness gap")
                    expect!(errors, all(runtime.num_lnc_fallback .== 0), "non-LNC run reports an LNC fallback")
                    expect!(errors, all(runtime.num_lnc_core_theta_fallback .== 0), "non-LNC run reports an LNC core-theta fallback")
                end
            end

            metrics[:completed_backward_rows] = count(runtime.completed_backward)
            metrics[:total_forward_time] = sum(runtime.forward_time)
            metrics[:total_lifting_time] = sum(runtime.lifting_time)
            metrics[:total_backward_time] = sum(runtime.backward_time)
            metrics[:reported_total_time] = last(runtime.total_time)
        end
    end

    return (; errors, warnings, metrics)
end

function validate_relative_gap_stopping()
    errors = String[]
    warnings = String[]
    observed_rows = -1
    observed_relative_gap = NaN
    stopped_before_backward = false
    solution_length = -1
    elapsed = @elapsed begin
        try
            result_dict = run_generation_expansion_experiments(
                algorithms = [:SDDP],
                cutTypes = [:SBC],
                T_list = [10],
                num_list = [5],
                timeSDDP = 30.0,
                gapSDDP = 1.0,
                iterSDDP = 5,
                levelMethodMaxIter = 2,
                sample_size_SDDP = 2,
                solverGap = 1e-6,
                solverTime = 10.0,
                logger_save = false,
                verbose = false,
            )
            result = result_dict[(:SDDP, :SBC, 10, 5)]
            history = result[:solHistory]
            runtime = result[:runtimeHistory]
            solution = result[:solution]
            observed_rows = nrow(history)
            solution_length = solution isa AbstractVector ? length(solution) : -1
            if nrow(history) > 0
                observed_relative_gap = (history.UB[1] - history.LB[1]) / history.UB[1]
            end
            stopped_before_backward = nrow(runtime) == 1 && runtime.completed_backward[1] == false
            expect!(errors, nrow(history) == 1, "relative gap tolerance 1.0 did not stop after the first bound observation")
            expect!(errors, isfinite(observed_relative_gap) && 0.0 <= observed_relative_gap <= 1.0, "observed relative gap is outside [0,1]")
            expect!(errors, stopped_before_backward, "gap-converged run executed a backward pass")
            expect!(errors, solution isa AbstractVector && length(solution) == 6, "gap-converged run returned an invalid solution")
        catch err
            push!(errors, "gap stopping validation exception: $(compact_error(err, catch_backtrace()))")
        end
    end

    return (
        application = "GEP",
        contract = "relative-gap-stopping",
        status = isempty(errors) ? "pass" : "fail",
        errors = join(errors, " | "),
        warnings = join(warnings, " | "),
        requested_relative_gap_tolerance = 1.0,
        observed_relative_gap = observed_relative_gap,
        observed_history_rows = observed_rows,
        stopped_before_backward = stopped_before_backward,
        solution_length = solution_length,
        elapsed_wall_time = elapsed,
    )
end

function validate_parameter_contracts()
    rows = NamedTuple[]

    function record_boolean(name::String, condition::Bool, details::String)
        push!(
            rows,
            (
                contract = name,
                status = condition ? "pass" : "fail",
                details = details,
            ),
        )
    end

    record_boolean(
        "reject-unknown-algorithm",
        !is_supported_configuration(:UnknownAlgorithm, :LC),
        "is_supported_configuration(:UnknownAlgorithm, :LC) must be false",
    )
    record_boolean(
        "reject-unknown-cut",
        !is_supported_configuration(:SDDP, :UnknownCut),
        "is_supported_configuration(:SDDP, :UnknownCut) must be false",
    )
    record_boolean(
        "reject-sddip-relu",
        !is_supported_configuration(:SDDiP, :ReLUC),
        "is_supported_configuration(:SDDiP, :ReLUC) must be false",
    )

    supported_count = count(
        is_supported_configuration(algorithm, cut)
        for algorithm in SUPPORTED_ALGORITHMS, cut in SUPPORTED_CUT_TYPES
    )
    record_boolean(
        "supported-combination-count",
        supported_count == 41,
        "expected 41 combinations; observed $(supported_count)",
    )

    data_dir = fixture_directory(10, 5)
    stage_data = load(joinpath(data_dir, "stageDataList.jld2"))["stageDataList"]
    omega = load(joinpath(data_dir, "Ω.jld2"))["Ω"]
    binary_info = load(joinpath(data_dir, "binaryInfo.jld2"))["binaryInfo"]
    probabilities = load(joinpath(data_dir, "probList.jld2"))["probList"]

    function record_argument_error(name::String, param)
        observed = "no exception"
        passed = false
        try
            stochastic_dual_dynamic_programming_algorithm(
                omega,
                probabilities,
                stage_data;
                binaryInfo = binary_info,
                param = param,
            )
        catch err
            observed = compact_error(err)
            passed = err isa ArgumentError
        end
        record_boolean(name, passed, observed)
    end

    common = (
        T = 10,
        num = 5,
        algorithm = :SDDP,
        cutType = :LC,
        iterSDDP = 2,
        levelMethodMaxIter = 2,
        sample_size_SDDP = 2,
        M = 1,
        gapSDDP = 1e-6,
        solverGap = 1e-6,
        solverTime = 10.0,
        logger_save = false,
    )
    record_argument_error(
        "reject-single-sample",
        param_setup(; merge(common, (sample_size_SDDP = 1,))...),
    )
    record_argument_error(
        "reject-zero-backward-samples",
        param_setup(; merge(common, (M = 0,))...),
    )
    record_argument_error(
        "reject-too-many-backward-samples",
        param_setup(; merge(common, (M = 3,))...),
    )

    return rows
end

function run_case(case)
    started = time()
    result = nothing
    execution_error = ""

    try
        result_dict = run_generation_expansion_experiments(
            algorithms = [case.algorithm],
            cutTypes = [case.cut],
            T_list = [case.T],
            num_list = [case.num],
            timeSDDP = 30.0,
            gapSDDP = 1e-6,
            iterSDDP = 2,
            levelMethodMaxIter = 5,
            sample_size_SDDP = 2,
            solverGap = 1e-6,
            solverTime = 10.0,
            ε = 1e-4,
            discreteZ = true,
            cutSparsity = true,
            partitionRule = :Bisection,
            branchingStart = 1,
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
        result = result_dict[(case.algorithm, case.cut, case.T, case.num)]
    catch err
        execution_error = compact_error(err, catch_backtrace())
    end

    elapsed = time() - started
    checked = result === nothing ? (
        errors = isempty(execution_error) ? ["result is missing"] : ["execution error: $(execution_error)"],
        warnings = String[],
        metrics = Dict{Symbol, Any}(),
    ) : validate_result(result, case.algorithm, case.cut, case.T, case.num)

    errors = checked.errors
    warnings = checked.warnings
    metrics = checked.metrics
    metric(name, fallback) = get(metrics, name, fallback)

    try
        @everywhere GC.gc()
    catch err
        push!(warnings, "worker garbage collection failed: $(compact_error(err))")
    end

    return (
        application = "GEP",
        T = case.T,
        realizations = case.num,
        algorithm = String(case.algorithm),
        cut = String(case.cut),
        coverage = case.coverage,
        status = isempty(errors) ? "pass" : "fail",
        error_count = length(errors),
        errors = join(errors, " | "),
        warnings = join(warnings, " | "),
        elapsed_wall_time = elapsed,
        history_rows = metric(:history_rows, -1),
        runtime_rows = metric(:runtime_rows, -1),
        solution_length = metric(:solution_length, -1),
        solution_values = metric(:solution_values, ""),
        initial_lb = metric(:initial_lb, NaN),
        final_lb = metric(:final_lb, NaN),
        initial_ub = metric(:initial_ub, NaN),
        final_ub = metric(:final_ub, NaN),
        lb_monotone = metric(:lb_monotone, false),
        bound_order_observed = metric(:bound_order_observed, false),
        gap_history_matches = metric(:gap_history_matches, false),
        reported_lm_iter = metric(:reported_lm_iter, -1),
        completed_backward_rows = metric(:completed_backward_rows, -1),
        total_forward_time = metric(:total_forward_time, NaN),
        total_lifting_time = metric(:total_lifting_time, NaN),
        total_backward_time = metric(:total_backward_time, NaN),
        reported_total_time = metric(:reported_total_time, NaN),
    )
end

function warm_alias_validation_cases()
    aliases = Set(alias for (alias, _) in WARM_CUT_TARGETS)
    missing_aliases = setdiff(aliases, Set(SUPPORTED_CUT_TYPES))
    isempty(missing_aliases) || error(
        "Warm-start aliases are missing from SUPPORTED_CUT_TYPES: " *
        join(sort!(String.(collect(missing_aliases))), ", "),
    )

    cases = [
        (; algorithm, alias, target)
        for algorithm in SUPPORTED_ALGORITHMS
        for (alias, target) in WARM_CUT_TARGETS
        if is_supported_configuration(algorithm, alias)
    ]
    length(cases) == 16 || error(
        "Expected 16 supported GEP warm-alias combinations, found $(length(cases)).",
    )
    return cases
end

function run_warm_alias_case(case)
    errors = String[]
    warnings = String[]
    result = nothing
    elapsed = @elapsed begin
        try
            result_dict = run_generation_expansion_experiments(
                algorithms = [case.algorithm],
                cutTypes = [case.alias],
                T_list = [REFERENCE_INSTANCE[1]],
                num_list = [REFERENCE_INSTANCE[2]],
                timeSDDP = 120.0,
                gapSDDP = 0.0,
                iterSDDP = 4,
                levelMethodMaxIter = 3,
                sample_size_SDDP = 2,
                solverGap = 1e-6,
                solverTime = 10.0,
                ε = 1e-4,
                discreteZ = true,
                cutSparsity = true,
                partitionRule = :Bisection,
                branchingStart = 1,
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
            result = get(
                result_dict,
                (case.algorithm, case.alias, REFERENCE_INSTANCE...),
                nothing,
            )
            result === nothing && push!(errors, "result dictionary does not contain the requested run")
        catch err
            push!(errors, "execution error: $(compact_error(err, catch_backtrace()))")
        end
    end

    expected_cut_sequence = [:SBC, :SBC, :SBC, case.target]
    expected_backward_sequence = [true, true, true, false]
    history_rows = -1
    runtime_rows = -1
    observed_cut_sequence = Symbol[]
    observed_backward_sequence = Bool[]
    lb_sequence = Float64[]
    ub_sequence = Float64[]
    gap_percent_sequence = Float64[]
    lm_iteration_sequence = Int[]
    solution_values = Float64[]
    total_forward_time = NaN
    total_lifting_time = NaN
    total_backward_time = NaN
    reported_total_time = NaN
    lb_monotone = false
    bound_order_observed = false
    same_row_timing = false

    if result !== nothing
        expected_keys = Set([:solHistory, :solution, :gapHistory, :runtimeHistory])
        expect!(errors, result isa AbstractDict, "result is not a dictionary")
        if result isa AbstractDict
            expect!(errors, Set(keys(result)) == expected_keys, "result keys do not match the public contract")
        end

        if result isa AbstractDict && all(haskey(result, key) for key in expected_keys)
            history = result[:solHistory]
            runtime = result[:runtimeHistory]
            gap_history = result[:gapHistory]
            solution = result[:solution]

            expect!(errors, history isa DataFrame, "solHistory is not a DataFrame")
            expect!(errors, runtime isa DataFrame, "runtimeHistory is not a DataFrame")
            expect!(errors, gap_history isa AbstractVector, "gapHistory is not a vector")
            expect!(errors, solution isa AbstractVector, "solution is not a vector")

            if solution isa AbstractVector
                solution_values = Float64.(solution)
                expect!(errors, length(solution_values) == 6, "solution is not six-dimensional")
                expect!(errors, all(isfinite, solution_values), "solution contains a non-finite value")
                if length(solution_values) == 6
                    expect!(errors, all(solution_values .>= -1e-6), "solution violates nonnegativity")
                    expect!(errors, all(solution_values .<= EXPECTED_STATE_UPPER_BOUNDS .+ 1e-6), "solution exceeds a state upper bound")
                    expect!(errors, all(abs.(solution_values .- round.(solution_values)) .<= 1e-6), "solution is nonintegral")
                end
            end

            if history isa DataFrame
                history_rows = nrow(history)
                expect!(errors, Symbol.(names(history)) == SOLUTION_HISTORY_COLUMNS, "solHistory columns are inconsistent")
                expect!(errors, history_rows == 4, "solHistory does not contain four iterations")
                if history_rows > 0 && all(name in Symbol.(names(history)) for name in SOLUTION_HISTORY_COLUMNS)
                    lb_sequence = Float64.(history.LB)
                    ub_sequence = Float64.(history.UB)
                    lm_iteration_sequence = Int.(history.LM_iter)
                    expect!(errors, collect(history.iter) == collect(1:history_rows), "solHistory iteration indices are inconsistent")
                    expect!(errors, all(isfinite, lb_sequence), "lower-bound sequence is non-finite")
                    expect!(errors, all(isfinite, ub_sequence), "upper-estimate sequence is non-finite")
                    expect!(errors, all(isfinite, history.time) && all(history.time .>= 0.0), "per-iteration times are invalid")
                    expect!(errors, all(isfinite, history.Time) && all(history.Time .>= 0.0), "cumulative times are invalid")
                    expect!(errors, issorted(history.Time), "cumulative times decrease")

                    if history_rows == 4
                        expect!(errors, all(lm_iteration_sequence[1:3] .> 0), "a completed backward iteration has no cut-generation work")
                        expect!(errors, lm_iteration_sequence[4] == 0, "terminal forward-only iteration reports cut-generation work")
                        lb_tolerance = 1e-5 * max(1.0, maximum(abs.(lb_sequence)))
                        lb_monotone = all(diff(lb_sequence) .>= -lb_tolerance)
                        expect!(errors, lb_monotone, "lower-bound sequence decreases beyond numerical tolerance")
                        bound_tolerance = 1e-5 .* max.(1.0, max.(abs.(lb_sequence), abs.(ub_sequence)))
                        bound_order_observed = all(lb_sequence .<= ub_sequence .+ bound_tolerance)
                        bound_order_observed || push!(warnings, "statistical upper estimate falls below the lower bound")
                    end

                    if gap_history isa AbstractVector && length(gap_history) == history_rows
                        gap_percent_sequence = Float64.(gap_history)
                        expected_gaps = [
                            round((ub_sequence[i] - lb_sequence[i]) / ub_sequence[i] * 100; digits = 2)
                            for i in eachindex(lb_sequence)
                        ]
                        parsed_gaps = try
                            parse_gap_percent.(history.gap)
                        catch err
                            push!(errors, "could not parse gap strings: $(compact_error(err))")
                            fill(NaN, history_rows)
                        end
                        expect!(errors, all(isfinite, gap_percent_sequence), "gapHistory contains a non-finite value")
                        expect!(
                            errors,
                            all(isapprox.(gap_percent_sequence, expected_gaps; atol = 1e-10, rtol = 0.0)) &&
                            all(isapprox.(parsed_gaps, expected_gaps; atol = 1e-10, rtol = 0.0)),
                            "gap outputs do not match LB and UB",
                        )
                    else
                        push!(errors, "gapHistory length does not match solHistory")
                    end
                end
            end

            if runtime isa DataFrame
                runtime_rows = nrow(runtime)
                expect!(errors, Symbol.(names(runtime)) == RUNTIME_HISTORY_COLUMNS, "runtimeHistory columns are inconsistent")
                expect!(errors, runtime_rows == 4, "runtimeHistory does not contain four iterations")
                if runtime_rows > 0 && all(name in Symbol.(names(runtime)) for name in RUNTIME_HISTORY_COLUMNS)
                    observed_cut_sequence = Symbol.(runtime.Cut)
                    observed_backward_sequence = Bool.(runtime.completed_backward)
                    expect!(errors, collect(runtime.Iter) == collect(1:runtime_rows), "runtimeHistory iteration indices are inconsistent")
                    expect!(errors, all(runtime.Algorithm .== case.algorithm), "runtimeHistory algorithm labels are inconsistent")
                    expect!(errors, observed_cut_sequence == expected_cut_sequence, "active cut sequence is inconsistent")
                    expect!(errors, observed_backward_sequence == expected_backward_sequence, "backward-completion sequence is inconsistent")

                    timing_columns = [:forward_time, :lifting_time, :backward_time, :other_time, :iteration_time, :total_time]
                    for column in timing_columns
                        values = runtime[!, column]
                        expect!(errors, all(isfinite, values) && all(values .>= 0.0), "runtimeHistory $(column) values are invalid")
                    end
                    expect!(errors, all(runtime.forward_time .> 0.0), "a forward pass has zero runtime")
                    if runtime_rows == 4
                        expect!(errors, all(runtime.backward_time[1:3] .> 0.0), "a completed backward pass has zero runtime")
                        expect!(errors, runtime.backward_time[4] == 0.0, "terminal forward-only iteration reports backward runtime")
                        if case.algorithm == :SDDPL
                            expect!(errors, all(runtime.lifting_time[1:3] .> 0.0), "a completed SDDP-L iteration has zero lifting runtime")
                            expect!(errors, runtime.lifting_time[4] == 0.0, "terminal SDDP-L iteration reports lifting runtime")
                        else
                            expect!(errors, all(runtime.lifting_time .== 0.0), "a non-lifted algorithm reports lifting runtime")
                        end
                    end

                    reconstructed_other_time = max.(
                        runtime.iteration_time .-
                        runtime.forward_time .-
                        runtime.lifting_time .-
                        runtime.backward_time,
                        0.0,
                    )
                    expect!(
                        errors,
                        all(isapprox.(runtime.other_time, reconstructed_other_time; atol = 1e-9, rtol = 1e-9)),
                        "runtime components do not reconstruct other_time",
                    )
                    expect!(errors, issorted(runtime.total_time), "runtimeHistory cumulative times decrease")

                    if history isa DataFrame && history_rows == runtime_rows
                        same_row_timing =
                            all(isapprox.(history.time, runtime.iteration_time; atol = 1e-9, rtol = 1e-9)) &&
                            all(isapprox.(history.Time, runtime.total_time; atol = 1e-9, rtol = 1e-9))
                        expect!(errors, same_row_timing, "solHistory and runtimeHistory times disagree within an iteration")
                    end

                    total_forward_time = sum(runtime.forward_time)
                    total_lifting_time = sum(runtime.lifting_time)
                    total_backward_time = sum(runtime.backward_time)
                    reported_total_time = last(runtime.total_time)
                end
            end
        end
    end

    try
        @everywhere GC.gc()
    catch err
        push!(warnings, "worker garbage collection failed: $(compact_error(err))")
    end

    return (
        application = "GEP",
        T = REFERENCE_INSTANCE[1],
        realizations = REFERENCE_INSTANCE[2],
        algorithm = String(case.algorithm),
        requested_cut = String(case.alias),
        target_cut = String(case.target),
        status = isempty(errors) ? "pass" : "fail",
        error_count = length(errors),
        errors = join(errors, " | "),
        warnings = join(warnings, " | "),
        elapsed_wall_time = elapsed,
        history_rows = history_rows,
        runtime_rows = runtime_rows,
        expected_cut_sequence = join(String.(expected_cut_sequence), ";"),
        observed_cut_sequence = join(String.(observed_cut_sequence), ";"),
        expected_backward_sequence = join(expected_backward_sequence, ";"),
        observed_backward_sequence = join(observed_backward_sequence, ";"),
        lb_sequence = join(lb_sequence, ";"),
        ub_sequence = join(ub_sequence, ";"),
        gap_percent_sequence = join(gap_percent_sequence, ";"),
        lm_iteration_sequence = join(lm_iteration_sequence, ";"),
        solution_values = join(solution_values, ";"),
        solution_length = length(solution_values),
        lb_monotone = lb_monotone,
        bound_order_observed = bound_order_observed,
        same_row_timing = same_row_timing,
        total_forward_time = total_forward_time,
        total_lifting_time = total_lifting_time,
        total_backward_time = total_backward_time,
        reported_total_time = reported_total_time,
    )
end

function run_warm_alias_dynamic_validation()
    cases = warm_alias_validation_cases()
    rows = NamedTuple[]
    for (index, case) in enumerate(cases)
        @info "GEP warm-alias validation start" index total = length(cases) algorithm = case.algorithm requested_cut = case.alias target_cut = case.target
        row = run_warm_alias_case(case)
        push!(rows, row)
        write_rows(WARM_ALIAS_DYNAMIC_SUMMARY_PATH, rows)
        @info "GEP warm-alias validation finish" index status = row.status elapsed = row.elapsed_wall_time errors = row.errors warnings = row.warnings
    end
    return rows
end

function run_deep_adaptive_validation()
    rows = NamedTuple[]

    for cut in (:AdaptivePLC, :AdaptiveSMC)
        errors = String[]
        warnings = String[]
        result = nothing
        baseline = DEEP_ADAPTIVE_BASELINES[cut]
        baseline_lb_max_abs_error = NaN
        baseline_ub_max_abs_error = NaN
        elapsed = @elapsed begin
            try
                result_dict = run_generation_expansion_experiments(
                    algorithms = [:SDDPL],
                    cutTypes = [cut],
                    T_list = [10],
                    num_list = [5],
                    timeSDDP = 120.0,
                    gapSDDP = 1e-8,
                    iterSDDP = 4,
                    levelMethodMaxIter = 8,
                    sample_size_SDDP = 5,
                    solverGap = 1e-6,
                    solverTime = 10.0,
                    ε = 1e-4,
                    discreteZ = true,
                    cutSparsity = true,
                    partitionRule = :Bisection,
                    branchingStart = 1,
                    M = 1,
                    verbose = false,
                    ℓ1 = 0.0,
                    ℓ2 = 0.0,
                    nxt_bound = 1e5,
                    logger_save = false,
                    corePointStrategy = :Mid,
                    corePointWeight = 0.5,
                    corePointEpsilon = 1e-2,
                    cutDiagnostics = false,
                )
                result = result_dict[(:SDDPL, cut, 10, 5)]
            catch err
                push!(errors, "execution error: $(compact_error(err, catch_backtrace()))")
            end
        end

        if result !== nothing
            history = result[:solHistory]
            runtime = result[:runtimeHistory]
            solution = result[:solution]
            expect!(errors, history isa DataFrame && nrow(history) == 4, "deep run does not have four solHistory rows")
            expect!(errors, runtime isa DataFrame && nrow(runtime) == 4, "deep run does not have four runtimeHistory rows")
            expect!(errors, solution isa AbstractVector && length(solution) == 6, "deep run returned an invalid solution dimension")
            if solution isa AbstractVector && length(solution) == 6
                expect!(errors, all(isfinite, solution), "deep-run solution is non-finite")
                expect!(errors, all(solution .>= -1e-6), "deep-run solution is negative")
                expect!(errors, all(solution .<= EXPECTED_STATE_UPPER_BOUNDS .+ 1e-6), "deep-run solution exceeds an upper bound")
                expect!(errors, all(abs.(solution .- round.(solution)) .<= 1e-6), "deep-run solution is nonintegral")
            end

            if history isa DataFrame && runtime isa DataFrame && nrow(history) == 4 && nrow(runtime) == 4
                expect!(errors, collect(history.iter) == collect(1:4), "deep solHistory iteration indices are inconsistent")
                expect!(errors, collect(runtime.Iter) == collect(1:4), "deep runtimeHistory iteration indices are inconsistent")
                expect!(errors, collect(runtime.completed_backward) == [true, true, true, false], "deep backward-completion flags are inconsistent")
                expect!(errors, all(history.LM_iter[1:3] .> 0), "a completed deep iteration has no level-method work")
                expect!(errors, history.LM_iter[4] == 0, "terminal deep iteration reports level-method work")
                expect!(errors, all(isfinite, history.LB), "deep lower-bound sequence is non-finite")
                expect!(errors, all(isfinite, history.UB), "deep upper-estimate sequence is non-finite")
                baseline_lb_max_abs_error = maximum(abs.(history.LB .- baseline.LB))
                baseline_ub_max_abs_error = maximum(abs.(history.UB .- baseline.UB))
                expect!(
                    errors,
                    all(
                        isapprox(history.LB[i], baseline.LB[i]; atol = DEEP_BASELINE_ATOL, rtol = DEEP_BASELINE_RTOL)
                        for i in eachindex(baseline.LB)
                    ),
                    "deep lower-bound sequence differs from the fixed-seed numerical baseline",
                )
                expect!(
                    errors,
                    all(
                        isapprox(history.UB[i], baseline.UB[i]; atol = DEEP_BASELINE_ATOL, rtol = DEEP_BASELINE_RTOL)
                        for i in eachindex(baseline.UB)
                    ),
                    "deep upper-estimate sequence differs from the fixed-seed numerical baseline",
                )
                lb_tolerance = 1e-5 * max(1.0, maximum(abs.(history.LB)))
                expect!(errors, all(diff(history.LB) .>= -lb_tolerance), "deep lower-bound sequence decreases")
                expect!(errors, all(isapprox.(history.time, runtime.iteration_time; atol = 1e-9, rtol = 1e-9)), "deep per-iteration times disagree")
                expect!(errors, all(isapprox.(history.Time, runtime.total_time; atol = 1e-9, rtol = 1e-9)), "deep cumulative times disagree")
                expect!(errors, all(runtime.Algorithm .== :SDDPL), "deep runtime algorithm label is inconsistent")
                expect!(errors, all(runtime.Cut .== cut), "deep runtime cut label is inconsistent")
                expect!(errors, all(isfinite, runtime.iteration_time) && all(runtime.iteration_time .>= 0.0), "deep iteration runtime is invalid")
                expect!(errors, issorted(runtime.total_time), "deep cumulative runtime decreases")
                bound_tolerance = 1e-5 .* max.(1.0, max.(abs.(history.LB), abs.(history.UB)))
                all(history.LB .<= history.UB .+ bound_tolerance) || push!(warnings, "statistical upper estimate falls below the lower bound")
            end
        end

        if result === nothing || !(result[:solHistory] isa DataFrame) || nrow(result[:solHistory]) == 0
            push!(
                rows,
                (
                    application = "GEP",
                    algorithm = "SDDPL",
                    cut = String(cut),
                    iteration = -1,
                    LB = NaN,
                    UB = NaN,
                    gap_percent = NaN,
                    LM_iter = -1,
                    iteration_time = NaN,
                    total_time = NaN,
                    completed_backward = false,
                    expected_LB = NaN,
                    expected_UB = NaN,
                    baseline_lb_max_abs_error = baseline_lb_max_abs_error,
                    baseline_ub_max_abs_error = baseline_ub_max_abs_error,
                    baseline_rtol = DEEP_BASELINE_RTOL,
                    baseline_atol = DEEP_BASELINE_ATOL,
                    status = "fail",
                    errors = join(errors, " | "),
                    warnings = join(warnings, " | "),
                    elapsed_wall_time = elapsed,
                ),
            )
        else
            history = result[:solHistory]
            runtime = result[:runtimeHistory]
            status = isempty(errors) ? "pass" : "fail"
            for i in 1:nrow(history)
                push!(
                    rows,
                    (
                        application = "GEP",
                        algorithm = "SDDPL",
                        cut = String(cut),
                        iteration = history.iter[i],
                        LB = history.LB[i],
                        UB = history.UB[i],
                        gap_percent = result[:gapHistory][i],
                        LM_iter = history.LM_iter[i],
                        iteration_time = history.time[i],
                        total_time = history.Time[i],
                        completed_backward = nrow(runtime) >= i ? runtime.completed_backward[i] : false,
                        expected_LB = baseline.LB[i],
                        expected_UB = baseline.UB[i],
                        baseline_lb_max_abs_error = baseline_lb_max_abs_error,
                        baseline_ub_max_abs_error = baseline_ub_max_abs_error,
                        baseline_rtol = DEEP_BASELINE_RTOL,
                        baseline_atol = DEEP_BASELINE_ATOL,
                        status = status,
                        errors = join(errors, " | "),
                        warnings = join(warnings, " | "),
                        elapsed_wall_time = elapsed,
                    ),
                )
            end
        end
    end

    return rows
end

function validate_sddip_solution_decoding()
    errors = String[]
    warnings = String[]
    returned_solution = Float64[]
    synthetic_bits = Float64[]
    synthetic_expected = Float64[]
    synthetic_observed = Float64[]
    elapsed = @elapsed begin
        try
            data_dir = fixture_directory(10, 5)
            binary_info = load(joinpath(data_dir, "binaryInfo.jld2"))["binaryInfo"]
            synthetic_bits = Float64.(isodd.(collect(1:binary_info.n)))
            synthetic_expected = Float64.(binary_info.A * synthetic_bits)
            synthetic_state = StageInfo(nothing, nothing, nothing, nothing, synthetic_bits)
            synthetic_observed = first_stage_integer_solution(synthetic_state, binary_info)
            expect!(errors, synthetic_observed == synthetic_expected, "SDDiP synthetic binary decoding is incorrect")

            result_dict = run_generation_expansion_experiments(
                algorithms = [:SDDiP],
                cutTypes = [:LC],
                T_list = [10],
                num_list = [5],
                timeSDDP = 60.0,
                gapSDDP = 1e-8,
                iterSDDP = 2,
                levelMethodMaxIter = 8,
                sample_size_SDDP = 5,
                solverGap = 1e-6,
                solverTime = 10.0,
                logger_save = false,
                verbose = false,
            )
            result = result_dict[(:SDDiP, :LC, 10, 5)]
            checked = validate_result(result, :SDDiP, :LC, 10, 5)
            append!(errors, checked.errors)
            append!(warnings, checked.warnings)
            if result[:solution] isa AbstractVector
                returned_solution = Float64.(result[:solution])
            end
        catch err
            push!(errors, "SDDiP decoding validation exception: $(compact_error(err, catch_backtrace()))")
        end
    end

    return [(
        application = "GEP",
        algorithm = "SDDiP",
        cut = "LC",
        contract = "original-state-solution-decoding",
        status = isempty(errors) ? "pass" : "fail",
        errors = join(errors, " | "),
        warnings = join(warnings, " | "),
        returned_solution = join(returned_solution, ";"),
        returned_solution_length = length(returned_solution),
        synthetic_binary_length = length(synthetic_bits),
        synthetic_expected_state = join(synthetic_expected, ";"),
        synthetic_observed_state = join(synthetic_observed, ";"),
        elapsed_wall_time = elapsed,
    )]
end

function validate_cross_case_consistency(rows)
    cross_rows = NamedTuple[]
    for (T, num) in GEP_INSTANCES
        group = [row for row in rows if row.T == T && row.realizations == num]
        errors = String[]
        expect!(errors, !isempty(group), "no numerical result was produced")
        expect!(errors, all(isfinite(row.initial_lb) for row in group), "an initial lower bound is missing or non-finite")
        expect!(errors, all(isfinite(row.initial_ub) for row in group), "an initial upper estimate is missing or non-finite")

        lb_spread = isempty(group) ? NaN : maximum(row.initial_lb for row in group) - minimum(row.initial_lb for row in group)
        ub_spread = isempty(group) ? NaN : maximum(row.initial_ub for row in group) - minimum(row.initial_ub for row in group)
        scale = isempty(group) ? 1.0 : max(
            1.0,
            maximum(abs(row.initial_lb) for row in group),
            maximum(abs(row.initial_ub) for row in group),
        )
        tolerance = 1e-8 * scale
        expect!(errors, isfinite(lb_spread) && lb_spread <= tolerance, "first-iteration lower bounds vary across algorithms/cuts")
        expect!(errors, isfinite(ub_spread) && ub_spread <= tolerance, "first-iteration upper estimates vary across algorithms/cuts")

        push!(
            cross_rows,
            (
                application = "GEP",
                T = T,
                realizations = num,
                compared_runs = length(group),
                initial_lb_spread = lb_spread,
                initial_ub_spread = ub_spread,
                tolerance = tolerance,
                status = isempty(errors) ? "pass" : "fail",
                errors = join(errors, " | "),
            ),
        )
    end
    return cross_rows
end

function repository_commit()::String
    try
        return readchomp(`git -C $REPOSITORY_ROOT rev-parse HEAD`)
    catch
        return "unknown"
    end
end

function main()::Nothing
    mkpath(OUTPUT_DIR)
    metadata = [(
        application = "GEP",
        generated_at = string(now()),
        repository_commit = repository_commit(),
        julia_version = string(VERSION),
        iteration_limit = 2,
        sample_size = 2,
        level_method_iteration_limit = 5,
        solver_gap = 1e-6,
        solver_time_limit = 10.0,
        outer_time_limit = 30.0,
    )]
    write_rows(METADATA_PATH, metadata)

    input_rows = [validate_input(T, num) for (T, num) in GEP_INSTANCES]
    write_rows(INPUT_SUMMARY_PATH, input_rows)
    for row in input_rows
        @info "GEP input validation" T = row.T realizations = row.realizations status = row.status errors = row.errors
    end

    schedule_rows = validate_cut_schedules()
    write_rows(SCHEDULE_SUMMARY_PATH, schedule_rows)

    contract_rows = [validate_relative_gap_stopping()]
    write_rows(CONTRACT_SUMMARY_PATH, contract_rows)

    parameter_contract_rows = validate_parameter_contracts()
    write_rows(PARAMETER_CONTRACT_SUMMARY_PATH, parameter_contract_rows)

    cases = validation_cases()
    expected_supported_combinations = count(
        is_supported_configuration(algorithm, cut)
        for algorithm in GEP_ALGORITHMS, cut in GEP_BASE_CUTS
    )
    expected_supported_combinations == 25 || error(
        "Expected 25 public supported algorithm/cut combinations, found $(expected_supported_combinations).",
    )
    length(cases) == 37 || error("Expected 37 numerical validation cases, found $(length(cases)).")
    coverage_rows = [
        (
            application = "GEP",
            T = case.T,
            realizations = case.num,
            algorithm = String(case.algorithm),
            cut = String(case.cut),
            coverage = case.coverage,
        ) for case in cases
    ]
    write_rows(COVERAGE_SUMMARY_PATH, coverage_rows)

    run_rows = NamedTuple[]
    for (index, case) in enumerate(cases)
        @info "GEP numerical validation start" index total = length(cases) T = case.T realizations = case.num algorithm = case.algorithm cut = case.cut
        row = run_case(case)
        push!(run_rows, row)
        write_rows(RUN_SUMMARY_PATH, run_rows)
        @info "GEP numerical validation finish" index status = row.status elapsed = row.elapsed_wall_time errors = row.errors warnings = row.warnings
    end

    cross_rows = validate_cross_case_consistency(run_rows)
    write_rows(CROSS_CASE_SUMMARY_PATH, cross_rows)

    warm_alias_rows = run_warm_alias_dynamic_validation()

    interface_rows = validate_interface_coverage(run_rows, schedule_rows, warm_alias_rows)
    write_rows(INTERFACE_COVERAGE_PATH, interface_rows)

    deep_rows = run_deep_adaptive_validation()
    write_rows(DEEP_SUMMARY_PATH, deep_rows)

    sddip_solution_rows = validate_sddip_solution_decoding()
    write_rows(SDDIP_SOLUTION_SUMMARY_PATH, sddip_solution_rows)

    input_failures = count(row.status != "pass" for row in input_rows)
    schedule_failures = count(row.status != "pass" for row in schedule_rows)
    run_failures = count(row.status != "pass" for row in run_rows)
    cross_failures = count(row.status != "pass" for row in cross_rows)
    contract_failures = count(row.status != "pass" for row in contract_rows)
    parameter_contract_failures = count(row.status != "pass" for row in parameter_contract_rows)
    warm_alias_failures = count(row.status != "pass" for row in warm_alias_rows)
    interface_failures = count(row.status != "pass" for row in interface_rows)
    deep_failures = count(row.status != "pass" for row in deep_rows)
    sddip_solution_failures = count(row.status != "pass" for row in sddip_solution_rows)
    @info "GEP validation complete" input_failures schedule_failures contract_failures parameter_contract_failures run_failures cross_failures warm_alias_failures interface_failures deep_failures sddip_solution_failures output_dir = OUTPUT_DIR

    if input_failures + schedule_failures + contract_failures +
       parameter_contract_failures + run_failures + cross_failures +
       warm_alias_failures + interface_failures + deep_failures +
       sddip_solution_failures > 0
        exit(1)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
