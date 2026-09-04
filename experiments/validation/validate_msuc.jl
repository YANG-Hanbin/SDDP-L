# Bounded numerical validation for the published case30 MSUC inputs and public
# algorithm/cut entry points. The default run checks all six serialized inputs
# and their hashes, parameter and cut-dispatch contracts, the complete supported
# configuration matrix on (T, R) = (6, 5), four core cuts on every other input,
# every public cut alias over four iterations, and fixed-seed adaptive baselines.
#
# Run from the repository root with:
#
#   julia +1.11 --project=. experiments/validation/validate_msuc.jl
#
# Set MSUC_VALIDATION_START and MSUC_VALIDATION_STOP to run a contiguous subset
# of the numerical task list. Reports are written below the ignored
# src/results/validation/msuc directory by default.
const MSUC_SCRIPT_DIR = @__DIR__
const MSUC_REPOSITORY_ROOT = normpath(joinpath(MSUC_SCRIPT_DIR, "..", ".."))
const MSUC_JULIA_11_BIN = joinpath(homedir(), ".juliaup", "bin", "julia")

function ensure_msuc_julia_11!()::Nothing
    if !(VERSION.major == 1 && VERSION.minor == 11) &&
       get(ENV, "SDDP_L_SKIP_JULIA_REEXEC", "0") != "1" &&
       isfile(MSUC_JULIA_11_BIN) &&
       isfile(PROGRAM_FILE)

        script_path = abspath(PROGRAM_FILE)
        cmd = `$MSUC_JULIA_11_BIN +1.11 --project=$MSUC_REPOSITORY_ROOT $script_path $(ARGS...)`
        run(setenv(cmd, merge(copy(ENV), Dict("SDDP_L_SKIP_JULIA_REEXEC" => "1"))))
        exit(0)
    end

    VERSION.major == 1 && VERSION.minor == 11 || error(
        "MSUC validation requires Julia 1.11; running $(VERSION).",
    )
    return nothing
end

ensure_msuc_julia_11!()

include(joinpath(
    MSUC_SCRIPT_DIR,
    "..", "..", "src", "multistage_stochastic_unit_commitment", "test", "loadMod.jl",
))

using CSV
using DataFrames
using Random
using SHA

const MSUC_INSTANCES = [(T, R) for T in (6, 8, 12) for R in (5, 10)]
const MSUC_ALGORITHMS = (:SDDPL, :SDDP, :SDDiP)
const MSUC_PUBLIC_CUTS = (
    :PLC,
    :AdaptivePLC,
    :SMC,
    :AdaptiveSMC,
    :LC,
    :LNC,
    :SBC,
    :SBCLC,
    :SBCSMC,
    :SBCPLC,
    :ReLUC,
    :NormalizedReLUC,
)
const MSUC_ALL_INPUT_CORE_CUTS = (:LC, :AdaptivePLC, :AdaptiveSMC, :NormalizedReLUC)
const MSUC_ALIAS_CUTS = (
    :SBCLC,
    :SBCSMC,
    :SBCPLC,
    :SBCLNC,
    :SBCNormalizedCut,
    :SBCReLUC,
    :SBCNormalizedReLUC,
    :NormalizedCut,
)
const MSUC_ALIAS_EXPECTED_SEQUENCES = Dict(
    :SBCLC => (:SBC, :SBC, :SBC, :LC),
    :SBCSMC => (:SBC, :SBC, :SBC, :SMC),
    :SBCPLC => (:SBC, :SBC, :SBC, :PLC),
    :SBCLNC => (:SBC, :SBC, :SBC, :LNC),
    :SBCNormalizedCut => (:SBC, :SBC, :SBC, :LNC),
    :SBCReLUC => (:SBC, :SBC, :SBC, :ReLUC),
    :SBCNormalizedReLUC => (:SBC, :SBC, :SBC, :NormalizedReLUC),
    :NormalizedCut => (:LNC, :LNC, :LNC, :LNC),
)

const MSUC_INITIAL_STATE_SHA256 =
    "5f0400c17655f85ac5b12fb78a5842ef9d4b3928922ec42373976612b075fe61"
const MSUC_INSTANCE_SHA256 = Dict(
    (6, 5, "indexSets.jld2") =>
        "4d83ad87dd6d273e7a4c89ac1c039a880b3b4da2ce47af6ea3568be0a8a2bddf",
    (6, 5, "paramDemand.jld2") =>
        "ac6df87b075c8c6e70d3bc62d574b0a2946bd0b64706752e6ece817d3f60ac34",
    (6, 5, "paramOPF.jld2") =>
        "1c1ed7a5029967d56830342f33cac1901828246f63bb32a3c0fe231cc47099ee",
    (6, 5, "scenarioTree.jld2") =>
        "728f86c44c50ebb39a0726c14ed887266bc2005cc3f1a9fa616c5bb6c7737553",
    (6, 10, "indexSets.jld2") =>
        "4d83ad87dd6d273e7a4c89ac1c039a880b3b4da2ce47af6ea3568be0a8a2bddf",
    (6, 10, "paramDemand.jld2") =>
        "0b31a8853e3a7ba7b57dd34a6d751a70b50f564938bc4c38f21c235233ba6f27",
    (6, 10, "paramOPF.jld2") =>
        "1c1ed7a5029967d56830342f33cac1901828246f63bb32a3c0fe231cc47099ee",
    (6, 10, "scenarioTree.jld2") =>
        "037bd8ab22256eba248206ac22958343517d82afcae9568e97f06e1ff4b30f48",
    (8, 5, "indexSets.jld2") =>
        "534a882664750fc8fc6c5055d71e503b4b03281c73e716718baaf5a8261a0a82",
    (8, 5, "paramDemand.jld2") =>
        "d35541681adc725d7722958d998caddddc31096b1f0dc431b9c7e7b5201fff0b",
    (8, 5, "paramOPF.jld2") =>
        "1c1ed7a5029967d56830342f33cac1901828246f63bb32a3c0fe231cc47099ee",
    (8, 5, "scenarioTree.jld2") =>
        "9cc29bb11835497332e51e97abcd0f636e9325a8ddf6a14cb03b03f12f01e39c",
    (8, 10, "indexSets.jld2") =>
        "534a882664750fc8fc6c5055d71e503b4b03281c73e716718baaf5a8261a0a82",
    (8, 10, "paramDemand.jld2") =>
        "b6e48b8a128856ceaae6ea484a108da24bce15de87f05892594f3176e2e75e6b",
    (8, 10, "paramOPF.jld2") =>
        "1c1ed7a5029967d56830342f33cac1901828246f63bb32a3c0fe231cc47099ee",
    (8, 10, "scenarioTree.jld2") =>
        "2c013407090e25e5deb6599b28080f1d6b892729aee4c7a7ce2b0ae0862b9e24",
    (12, 5, "indexSets.jld2") =>
        "6352ca9c21e5df7ed7e8a1c6439e01a3e6a69b9aeb404b0813f3cc98c7968ae8",
    (12, 5, "paramDemand.jld2") =>
        "6c727caad1088de7413ef466b5dfeb214183e37117378716aff663df59b6980c",
    (12, 5, "paramOPF.jld2") =>
        "1c1ed7a5029967d56830342f33cac1901828246f63bb32a3c0fe231cc47099ee",
    (12, 5, "scenarioTree.jld2") =>
        "f1dd33c77b2733d6cbd83453ae105e3d8512adaedbcdbb5a5d5ca9f7cb7e58cf",
    (12, 10, "indexSets.jld2") =>
        "6352ca9c21e5df7ed7e8a1c6439e01a3e6a69b9aeb404b0813f3cc98c7968ae8",
    (12, 10, "paramDemand.jld2") =>
        "581cb2fd92a67cd611afcbb9ae78a50f0675866aff57938a0098caba61df7249",
    (12, 10, "paramOPF.jld2") =>
        "1c1ed7a5029967d56830342f33cac1901828246f63bb32a3c0fe231cc47099ee",
    (12, 10, "scenarioTree.jld2") =>
        "a26f42d79c6ba52a61e83901d2a3e53ea26dc922998bff18acf6b7fe92f8873f",
)
const MSUC_ADAPTIVE_BASELINES = (
    (
        algorithm = :SDDPL, cut = :AdaptivePLC, T = 6, R = 5,
        seed = 41_321_404, iterations = 2, expected_LB = 3271.298145073255,
    ),
    (
        algorithm = :SDDPL, cut = :AdaptiveSMC, T = 6, R = 5,
        seed = 61_321_404, iterations = 2, expected_LB = 3271.767528006203,
    ),
)
const MSUC_BASELINE_ATOL = 0.5
const MSUC_BASELINE_RTOL = 5e-4

const MSUC_VALIDATION_ROOT = get(
    ENV,
    "MSUC_VALIDATION_ROOT",
    joinpath(PROJECT_ROOT, "src", "results", "validation", "msuc"),
)

finite_number(x)::Bool = x isa Real && isfinite(Float64(x))
same_keys(dict, expected)::Bool = Set(keys(dict)) == Set(expected)

function require_check(condition::Bool, message::AbstractString)::Nothing
    condition || error(message)
    return nothing
end

function load_one(path::AbstractString, expected_key::AbstractString, expected_type)
    require_check(isfile(path), "missing input file: $path")
    raw = load(path)
    require_check(
        Set(keys(raw)) == Set([expected_key]),
        "$(basename(path)) has keys $(sort!(collect(keys(raw)))); expected [$expected_key]",
    )
    value = raw[expected_key]
    require_check(
        value isa expected_type,
        "$(basename(path)) stores $(typeof(value)); expected $expected_type",
    )
    return value
end

function msuc_input_root()::String
    return joinpath(
        PROJECT_ROOT,
        "src", "multistage_stochastic_unit_commitment", "experiment_case30",
    )
end

function expected_input_hashes()::Dict{String, String}
    hashes = Dict("initialStateInfo.jld2" => MSUC_INITIAL_STATE_SHA256)
    for ((T, R, filename), digest) in MSUC_INSTANCE_SHA256
        hashes[joinpath("stage($T)real($R)", filename)] = digest
    end
    return hashes
end

function validate_input_hashes()::DataFrame
    root = msuc_input_root()
    expected = expected_input_hashes()
    actual_paths = String[]
    for (dir, _, filenames) in walkdir(root), filename in filenames
        endswith(filename, ".jld2") || continue
        push!(actual_paths, relpath(joinpath(dir, filename), root))
    end

    rows = DataFrame(
        path = String[], expected_sha256 = String[], actual_sha256 = String[],
        status = String[], error = String[],
    )
    for relative_path in sort!(collect(union(Set(keys(expected)), Set(actual_paths))))
        expected_digest = get(expected, relative_path, "")
        absolute_path = joinpath(root, relative_path)
        if isempty(expected_digest)
            push!(rows, (relative_path, "", bytes2hex(sha256(read(absolute_path))),
                         "fail", "unexpected JLD2 input"))
        elseif !isfile(absolute_path)
            push!(rows, (relative_path, expected_digest, "", "fail", "missing JLD2 input"))
        else
            actual_digest = bytes2hex(sha256(read(absolute_path)))
            status = actual_digest == expected_digest ? "pass" : "fail"
            error = status == "pass" ? "" : "SHA-256 mismatch"
            push!(rows, (relative_path, expected_digest, actual_digest, status, error))
        end
    end
    require_check(length(expected) == 25, "expected SHA-256 inventory must contain 25 inputs")
    return rows
end

function validate_index_sets(indexSets::IndexSets, T::Int)::Nothing
    require_check(indexSets.T == T, "IndexSets.T=$(indexSets.T), expected $T")
    for (name, values) in (
        (:D, indexSets.D), (:G, indexSets.G), (:L, indexSets.L), (:B, indexSets.B),
    )
        require_check(!isempty(values), "$name is empty")
        require_check(length(unique(values)) == length(values), "$name contains duplicates")
    end

    require_check(same_keys(indexSets.Dᵢ, indexSets.B), "Dᵢ keys do not equal B")
    require_check(same_keys(indexSets.Gᵢ, indexSets.B), "Gᵢ keys do not equal B")
    require_check(same_keys(indexSets.out_L, indexSets.B), "out_L keys do not equal B")
    require_check(same_keys(indexSets.in_L, indexSets.B), "in_L keys do not equal B")

    assigned_demands = reduce(vcat, values(indexSets.Dᵢ); init = Int[])
    assigned_generators = reduce(vcat, values(indexSets.Gᵢ); init = Int[])
    require_check(
        sort(assigned_demands) == sort(indexSets.D),
        "loads are not assigned exactly once to buses",
    )
    require_check(
        sort(assigned_generators) == sort(indexSets.G),
        "generators are not assigned exactly once to buses",
    )

    buses = Set(indexSets.B)
    require_check(
        all(l[1] in buses && l[2] in buses for l in indexSets.L),
        "a transmission line references a bus outside B",
    )
    outgoing = Set((from, to) for from in indexSets.B for to in indexSets.out_L[from])
    incoming = Set((from, to) for to in indexSets.B for from in indexSets.in_L[to])
    lines = Set(indexSets.L)
    require_check(outgoing == lines, "out_L is inconsistent with L")
    require_check(incoming == lines, "in_L is inconsistent with L")
    return nothing
end

function validate_parameters(
    indexSets::IndexSets,
    paramOPF::ParamOPF,
    paramDemand::ParamDemand,
)::Nothing
    generators = indexSets.G
    lines = indexSets.L
    buses = indexSets.B
    demands = indexSets.D

    for (name, dict, expected) in (
        (:b, paramOPF.b, lines),
        (:W, paramOPF.W, lines),
        (:smax, paramOPF.smax, generators),
        (:smin, paramOPF.smin, generators),
        (:M, paramOPF.M, generators),
        (:slope, paramOPF.slope, generators),
        (:intercept, paramOPF.intercept, generators),
        (:C_start, paramOPF.C_start, generators),
        (:C_down, paramOPF.C_down, generators),
        (:demand, paramDemand.demand, demands),
        (:w, paramDemand.w, demands),
        (:cb, paramDemand.cb, buses),
        (:cg, paramDemand.cg, generators),
        (:cl, paramDemand.cl, lines),
    )
        require_check(same_keys(dict, expected), "$name keys do not match their index set")
    end

    require_check(
        finite_number(paramOPF.θmin) && finite_number(paramOPF.θmax) &&
        paramOPF.θmin <= paramOPF.θmax,
        "invalid phase-angle bounds",
    )
    require_check(all(finite_number, values(paramOPF.b)), "b contains a non-finite value")
    require_check(
        all(finite_number(v) && v > 0.0 for v in values(paramOPF.W)),
        "W must be finite and positive",
    )
    for g in generators
        require_check(
            finite_number(paramOPF.smin[g]) && finite_number(paramOPF.smax[g]) &&
            0.0 <= paramOPF.smin[g] <= paramOPF.smax[g],
            "invalid production bounds for generator $g",
        )
        require_check(
            finite_number(paramOPF.M[g]) && paramOPF.M[g] >= 0.0,
            "invalid ramp bound for generator $g",
        )
        require_check(
            !isempty(paramOPF.slope[g]) &&
            same_keys(paramOPF.slope[g], keys(paramOPF.intercept[g])) &&
            all(finite_number, values(paramOPF.slope[g])) &&
            all(finite_number, values(paramOPF.intercept[g])),
            "invalid piecewise-linear cost for generator $g",
        )
    end
    for (name, dict) in (
        (:C_start, paramOPF.C_start), (:C_down, paramOPF.C_down),
        (:demand, paramDemand.demand), (:w, paramDemand.w),
        (:cb, paramDemand.cb), (:cg, paramDemand.cg), (:cl, paramDemand.cl),
    )
        require_check(all(finite_number, values(dict)), "$name contains a non-finite value")
    end
    require_check(
        finite_number(paramDemand.penalty) && paramDemand.penalty > 0.0,
        "penalty must be finite and positive",
    )
    return nothing
end

function validate_scenario_tree(
    scenarioTree::ScenarioTree,
    indexSets::IndexSets,
    T::Int,
    R::Int,
)::Nothing
    require_check(same_keys(scenarioTree.tree, 1:T), "scenario stages are not exactly 1:$T")
    demand_ids = Set(indexSets.D)
    expected_realizations = Set(1:R)

    for t in 1:T
        stage = scenarioTree.tree[t]
        expected_nodes = t == 1 ? Set([1]) : expected_realizations
        require_check(Set(keys(stage.nodes)) == expected_nodes, "stage $t has wrong node ids")
        require_check(
            Set(keys(stage.prob)) == expected_realizations,
            "stage $t has wrong probability ids",
        )
        probabilities = collect(values(stage.prob))
        require_check(
            all(finite_number(p) && p > 0.0 for p in probabilities),
            "stage $t has a non-positive or non-finite probability",
        )
        require_check(
            isapprox(sum(probabilities), 1.0; atol = 1e-12, rtol = 1e-12),
            "stage $t probabilities sum to $(sum(probabilities)), not one",
        )
        for (node_id, node) in stage.nodes
            require_check(
                Set(keys(node.deviation)) == demand_ids,
                "stage $t node $node_id has wrong demand ids",
            )
            require_check(
                all(finite_number(v) && v > 0.0 for v in values(node.deviation)),
                "stage $t node $node_id has an invalid demand multiplier",
            )
        end
    end
    return nothing
end

function validate_initial_state(
    initialStateInfo::StateInfo,
    indexSets::IndexSets,
    paramOPF::ParamOPF,
)::Nothing
    require_check(initialStateInfo.BinVar !== nothing, "initial BinVar is missing")
    require_check(initialStateInfo.ContVar !== nothing, "initial ContVar is missing")
    require_check(
        Set(keys(initialStateInfo.BinVar)) == Set((:y, :v, :w)),
        "initial binary-state names are not y/v/w",
    )
    require_check(
        Set(keys(initialStateInfo.ContVar)) == Set((:s,)),
        "initial continuous-state name is not s",
    )
    generators = Set(indexSets.G)
    for state_name in (:y, :v, :w)
        state = initialStateInfo.BinVar[state_name]
        require_check(Set(keys(state)) == generators, "initial $state_name has wrong generator ids")
        require_check(
            all(finite_number(v) && (isapprox(v, 0.0; atol = 1e-12) ||
                                     isapprox(v, 1.0; atol = 1e-12)) for v in values(state)),
            "initial $state_name is not binary",
        )
    end
    production = initialStateInfo.ContVar[:s]
    require_check(Set(keys(production)) == generators, "initial s has wrong generator ids")
    for g in indexSets.G
        require_check(
            finite_number(production[g]) &&
            paramOPF.smin[g] - 1e-12 <= production[g] <= paramOPF.smax[g] + 1e-12,
            "initial production for generator $g violates its bounds",
        )
        require_check(
            initialStateInfo.BinVar[:y][g] > 0.5 || isapprox(production[g], 0.0; atol = 1e-12),
            "offline generator $g has nonzero initial production",
        )
    end
    for field in (
        :IntVar, :IntVarLeaf, :ContVarLeaf, :StageValue, :StateValue,
        :IntAugState, :ContAugState, :IntStateBin, :ContStateBin,
    )
        require_check(getfield(initialStateInfo, field) === nothing, "initial $field should be nothing")
    end
    return nothing
end

function validate_inputs()::DataFrame
    rows = DataFrame(
        T = Int[], R = Int[], status = String[], loads = Int[], generators = Int[],
        lines = Int[], buses = Int[], stages = Int[], total_nodes = Int[],
        total_bytes = Int[], error = String[],
    )
    initial_path = joinpath(
        PROJECT_ROOT,
        "src", "multistage_stochastic_unit_commitment", "experiment_case30",
        "initialStateInfo.jld2",
    )

    for (T, R) in MSUC_INSTANCES
        try
            dir = experiment_dir("case30", T, R)
            expected_files = Set((
                "indexSets.jld2", "paramOPF.jld2", "paramDemand.jld2", "scenarioTree.jld2",
            ))
            actual_files = Set(filter(name -> endswith(name, ".jld2"), readdir(dir)))
            require_check(actual_files == expected_files, "unexpected JLD2 file set in $dir")

            indexSets = load_one(joinpath(dir, "indexSets.jld2"), "indexSets", IndexSets)
            paramOPF = load_one(joinpath(dir, "paramOPF.jld2"), "paramOPF", ParamOPF)
            paramDemand = load_one(joinpath(dir, "paramDemand.jld2"), "paramDemand", ParamDemand)
            scenarioTree = load_one(joinpath(dir, "scenarioTree.jld2"), "scenarioTree", ScenarioTree)
            initialStateInfo = load_one(initial_path, "initialStateInfo", StateInfo)

            validate_index_sets(indexSets, T)
            validate_parameters(indexSets, paramOPF, paramDemand)
            validate_scenario_tree(scenarioTree, indexSets, T, R)
            validate_initial_state(initialStateInfo, indexSets, paramOPF)

            input_paths = [joinpath(dir, name) for name in sort!(collect(expected_files))]
            push!(rows, (
                T, R, "pass", length(indexSets.D), length(indexSets.G), length(indexSets.L),
                length(indexSets.B), length(scenarioTree.tree),
                sum(length(stage.nodes) for stage in values(scenarioTree.tree)),
                sum(filesize, input_paths) + filesize(initial_path), "",
            ))
        catch err
            push!(rows, (T, R, "fail", 0, 0, 0, 0, 0, 0, 0, sprint(showerror, err)))
        end
    end
    return rows
end

function validate_cut_dispatch()::DataFrame
    cases = (
        (:LC, :LC, :LC, :LC),
        (:SBCLC, :SBC, :SBC, :LC),
        (:SBCSMC, :SBC, :SBC, :SMC),
        (:SBCPLC, :SBC, :SBC, :PLC),
        (:SBCLNC, :SBC, :SBC, :LNC),
        (:SBCNormalizedCut, :SBC, :SBC, :LNC),
        (:SBCReLUC, :SBC, :SBC, :ReLUC),
        (:SBCNormalizedReLUC, :SBC, :SBC, :NormalizedReLUC),
        (:NormalizedCut, :LNC, :LNC, :LNC),
    )
    rows = DataFrame(
        requested_cut = String[], iteration_1 = String[], iteration_3 = String[],
        iteration_4 = String[], status = String[],
    )
    for (requested, expected_1, expected_3, expected_4) in cases
        actual = (
            get_cut_selection(requested, 1),
            get_cut_selection(requested, 3),
            get_cut_selection(requested, 4),
        )
        expected = (expected_1, expected_3, expected_4)
        push!(rows, (
            string(requested), string(actual[1]), string(actual[2]), string(actual[3]),
            actual == expected ? "pass" : "fail",
        ))
    end
    require_check(
        is_supported_configuration(:SDDPL, :NormalizedReLUC),
        "production support matrix rejected SDDP-L with NormalizedReLUC",
    )
    require_check(
        is_supported_configuration(:SDDP, :AdaptivePLC),
        "production support matrix rejected SDDP with AdaptivePLC",
    )
    require_check(
        !is_supported_configuration(:SDDiP, :ReLUC) &&
        !is_supported_configuration(:SDDiP, :NormalizedReLUC),
        "production support matrix accepted an unsupported SDDiP/ReLU pair",
    )
    require_check(
        !is_supported_configuration(:Bogus, :Bogus) &&
        !is_supported_configuration(:Bogus, :LC) &&
        !is_supported_configuration(:SDDP, :Bogus),
        "production support matrix accepted an unknown algorithm or cut",
    )
    require_check(
        all(cut in SUPPORTED_CUT_TYPES for cut in MSUC_PUBLIC_CUTS),
        "the validation cut inventory disagrees with the production whitelist",
    )
    defaults = param_setup(; logger_save = false)
    require_check(defaults.partitionRule == :Bisection, "default partition rule is not implemented")
    require_check(defaults.OPT === nothing, "unknown default optimum is not represented as nothing")
    return rows
end

function numerical_tasks()
    coverage = Dict{Tuple{Symbol, Symbol, Int, Int}, Set{String}}()
    add_task!(algorithm, cut, T, R, label) =
        push!(get!(coverage, (algorithm, cut, T, R), Set{String}()), label)

    for algorithm in MSUC_ALGORITHMS, cut in MSUC_PUBLIC_CUTS
        is_supported_configuration(algorithm, cut) || continue
        add_task!(algorithm, cut, 6, 5, "representative_algorithm_cut_matrix")
    end
    for (T, R) in MSUC_INSTANCES, cut in MSUC_ALL_INPUT_CORE_CUTS
        add_task!(:SDDPL, cut, T, R, "all_published_inputs")
    end

    algorithm_rank = Dict(value => index for (index, value) in enumerate(MSUC_ALGORITHMS))
    cut_rank = Dict(value => index for (index, value) in enumerate(MSUC_PUBLIC_CUTS))
    tasks = [
        (
            algorithm = key[1], cut = key[2], T = key[3], R = key[4],
            coverage = join(sort!(collect(labels)), "+"),
        )
        for (key, labels) in coverage
    ]
    sort!(tasks; by = task -> (
        task.T, task.R, algorithm_rank[task.algorithm], cut_rank[task.cut],
    ))
    representative_count = count(
        task -> task.T == 6 && task.R == 5 &&
                occursin("representative_algorithm_cut_matrix", task.coverage),
        tasks,
    )
    require_check(
        representative_count == 34,
        "representative algorithm/cut matrix has $representative_count tasks; expected 34",
    )
    require_check(length(tasks) == 54, "numerical task list has $(length(tasks)) tasks; expected 54")
    return tasks
end

function validate_numerical_output(result, algorithm::Symbol, cut::Symbol, iterations::Int)
    errors = String[]
    check(condition, message) = condition || push!(errors, message)
    sddpResults = result.sddpResults
    check(
        Set(keys(sddpResults)) == Set((:solHistory, :gapHistory, :runtimeHistory)),
        "result dictionary has unexpected keys",
    )
    all(haskey(sddpResults, key) for key in (:solHistory, :gapHistory, :runtimeHistory)) ||
        return (errors = [errors; "result dictionary is missing a required table"],)

    sol = sddpResults[:solHistory]
    gaps = sddpResults[:gapHistory]
    runtime = sddpResults[:runtimeHistory]
    expected_sol_columns = (:Iter, :LB, :OPT, :UB, :gap, :time, :LM_iter, :Time, :Branch)
    expected_runtime_columns = (
        :Iter, :Algorithm, :Cut, :forward_time, :lifting_time, :backward_time,
        :other_time, :iteration_time, :total_time, :completed_backward,
        :min_lnc_scale, :max_lnc_tightness_gap, :num_lnc_fallback,
        :num_lnc_core_theta_fallback,
    )

    check(Tuple(Symbol.(names(sol))) == expected_sol_columns, "solHistory column contract changed")
    check(Tuple(Symbol.(names(runtime))) == expected_runtime_columns, "runtimeHistory column contract changed")
    check(nrow(sol) == iterations, "solHistory has $(nrow(sol)) rows, expected $iterations")
    check(nrow(runtime) == iterations, "runtimeHistory has $(nrow(runtime)) rows, expected $iterations")
    check(length(gaps) == iterations, "gapHistory has $(length(gaps)) values, expected $iterations")
    if nrow(sol) != iterations || nrow(runtime) != iterations || length(gaps) != iterations
        return (errors = errors,)
    end

    check(collect(sol.Iter) == collect(1:iterations), "solHistory iteration ids are not sequential")
    check(collect(runtime.Iter) == collect(1:iterations), "runtimeHistory iteration ids are not sequential")
    check(all(==(algorithm), runtime.Algorithm), "runtimeHistory algorithm label is incorrect")
    check(
        collect(runtime.Cut) == [get_cut_selection(cut, i) for i in 1:iterations],
        "runtimeHistory cut labels do not match dispatch",
    )
    check(all(ismissing(v) || v === nothing for v in sol.OPT), "unknown OPT must be nothing/missing")
    check(all(finite_number, sol.LB), "solHistory LB contains a non-finite value")
    check(all(finite_number, sol.UB), "solHistory UB contains a non-finite value")
    check(all(finite_number(v) && v >= 0.0 for v in sol.time), "invalid solHistory time")
    check(all(finite_number(v) && v >= 0.0 for v in sol.Time), "invalid solHistory cumulative time")
    check(all(v >= 0 for v in sol.LM_iter), "negative level-method count")
    check(all(v isa Bool for v in sol.Branch), "Branch contains a non-Boolean value")

    lb_tolerance = 1e-5 * max(1.0, maximum(abs, sol.LB))
    lb_monotone = all(diff(sol.LB) .>= -lb_tolerance)
    check(lb_monotone, "lower bound decreased beyond numerical tolerance")
    bound_tolerance = 1e-5 .* max.(1.0, abs.(sol.UB))
    bounds_ordered = all(sol.LB .<= sol.UB .+ bound_tolerance)

    gap_matches = true
    for row in eachrow(sol)
        expected_gap = round((row.UB - row.LB) / row.UB * 100; digits = 2)
        gap_matches &= row.gap == string(expected_gap, "%")
    end
    check(gap_matches, "gap strings do not match LB and UB")
    check(
        all(isapprox(Float64(gaps[i]), round((sol.UB[i] - sol.LB[i]) / sol.UB[i] * 100; digits = 2); atol = 1e-12)
            for i in 1:iterations),
        "gapHistory does not match solHistory",
    )

    runtime_metric_columns = (
        :forward_time, :lifting_time, :backward_time, :other_time,
        :iteration_time, :total_time,
    )
    for column in runtime_metric_columns
        check(
            all(finite_number(v) && v >= -1e-9 for v in runtime[!, column]),
            "$column contains an invalid value",
        )
    end
    check(all(diff(runtime.total_time) .>= -1e-9), "runtime total_time decreased")
    check(
        collect(runtime.completed_backward) == [fill(true, iterations - 1); false],
        "completed_backward flags are incorrect",
    )
    check(runtime.backward_time[1] > 0.0, "no completed backward work was timed")
    for row in eachrow(runtime)
        component_sum = row.forward_time + row.lifting_time + row.backward_time + row.other_time
        tolerance = max(0.05, 0.02 * row.iteration_time)
        check(
            isapprox(component_sum, row.iteration_time; atol = tolerance, rtol = 0.02),
            "runtime components do not sum to iteration_time at iteration $(row.Iter)",
        )
    end
    check(
        all(isapprox(sol.time[i], runtime.iteration_time[i]; atol = 0.1, rtol = 0.02)
            for i in 1:iterations),
        "solHistory time does not match the same iteration in runtimeHistory",
    )
    check(
        all(isapprox(sol.Time[i], runtime.total_time[i]; atol = 0.1, rtol = 0.02)
            for i in 1:iterations),
        "solHistory Time does not match the same iteration in runtimeHistory",
    )
    check(sol.LM_iter[end] == 0, "the forward-only final iteration reports level-method work")
    if cut in (:AdaptivePLC, :AdaptiveSMC)
        check(sol.LM_iter[1] > 0, "$cut did not execute adaptive level-method iterations")
    end
    if get_cut_selection(cut, 1) == :LNC
        check(finite_number(runtime.min_lnc_scale[1]), "LNC min scale is not finite")
        check(runtime.min_lnc_scale[1] > 0.0, "LNC min scale is not positive")
        check(
            finite_number(runtime.max_lnc_tightness_gap[1]) &&
            runtime.max_lnc_tightness_gap[1] >= 0.0,
            "LNC tightness gap is invalid",
        )
    end

    summary = result.summary
    check(isapprox(summary.LB, sol.LB[end]; atol = 1e-10), "summary LB does not match history")
    check(isapprox(summary.UB, sol.UB[end]; atol = 1e-10), "summary UB does not match history")
    check(summary.gap_str == sol.gap[end], "summary gap does not match history")
    check(isapprox(summary.time, sol.Time[end]; atol = 1e-10), "summary time does not match history")
    check(finite_number(summary.runtime) && summary.runtime >= 0.0, "summary runtime is invalid")

    return (
        errors = errors,
        iterations = nrow(sol),
        completed_backward = count(runtime.completed_backward),
        final_LB = sol.LB[end],
        final_UB = sol.UB[end],
        final_gap = sol.gap[end],
        max_LM_iter = maximum(sol.LM_iter),
        bounds_ordered = bounds_ordered,
        lb_monotone = lb_monotone,
    )
end

function empty_numerical_report()::DataFrame
    return DataFrame(
        task_id = Int[], coverage = String[], algorithm = String[], cut = String[],
        T = Int[], R = Int[], seed = Int[], status = String[], iterations = Int[],
        completed_backward = Int[], final_LB = Float64[], final_UB = Float64[],
        final_gap = String[], max_LM_iter = Int[], bounds_ordered = Bool[],
        lb_monotone = Bool[], elapsed_seconds = Float64[], error = String[],
    )
end

function run_validation_experiment(
    algorithm::Symbol,
    cut::Symbol,
    T::Int,
    R::Int;
    seed::Int,
    iterations::Int,
)
    Random.seed!(seed)
    return run_single_experiment(
        algorithm,
        cut,
        T,
        R;
        case = "case30",
        numScenarios = 2,
        M = 1,
        logger_save = false,
        terminate_time = 600,
        terminate_threshold = 0.0,
        TimeLimit = 20,
        MIPGap = 1e-4,
        MaxIter = iterations,
        partitionRule = :Bisection,
        ε = 1 / 2^8,
        δ = 1e-2,
        levelMethodMaxIter = 8,
        levelMethodVerbose = false,
        core_point_strategy = "Mid",
        core_point_weight = 0.75,
        core_point_epsilon = 1e-2,
        sparse_cut = :sparse,
        tightness = false,
        branch_variable = :ALL,
        LiftIterThreshold = 1,
        cutDiagnostics = false,
    )
end

function run_numerical_validation(tasks)::DataFrame
    rows = empty_numerical_report()
    total = length(tasks)
    start_id = parse(Int, get(ENV, "MSUC_VALIDATION_START", "1"))
    stop_id = parse(Int, get(ENV, "MSUC_VALIDATION_STOP", string(total)))
    require_check(1 <= start_id <= total, "MSUC_VALIDATION_START is outside 1:$total")
    require_check(start_id <= stop_id <= total, "MSUC_VALIDATION_STOP is outside $start_id:$total")
    output_path = joinpath(
        MSUC_VALIDATION_ROOT,
        "numerical_report_$(start_id)_$(stop_id)_of_$(total).csv",
    )

    for task_id in start_id:stop_id
        task = tasks[task_id]
        algorithm_index = findfirst(==(task.algorithm), MSUC_ALGORITHMS)
        cut_index = findfirst(==(task.cut), MSUC_PUBLIC_CUTS)
        seed = 20_260_904 + 10_000 * task.T + 100 * task.R +
               1_000_000 * algorithm_index + 10_000_000 * cut_index
        println(
            "[$task_id/$total] $(task.algorithm)/$(task.cut), " *
            "T=$(task.T), R=$(task.R), coverage=$(task.coverage)",
        )
        started = time_ns()
        try
            output = run_validation_experiment(
                task.algorithm, task.cut, task.T, task.R;
                seed = seed, iterations = 2,
            )
            checked = validate_numerical_output(output, task.algorithm, task.cut, 2)
            status = isempty(checked.errors) ? "pass" : "fail"
            push!(rows, (
                task_id, task.coverage, string(task.algorithm), string(task.cut), task.T, task.R,
                seed, status, checked.iterations, checked.completed_backward,
                checked.final_LB, checked.final_UB, checked.final_gap, checked.max_LM_iter,
                checked.bounds_ordered, checked.lb_monotone,
                (time_ns() - started) / 1e9, join(checked.errors, " | "),
            ))
        catch err
            message = sprint(showerror, err, catch_backtrace())
            if length(message) > 8_000
                message = first(message, 8_000)
            end
            push!(rows, (
                task_id, task.coverage, string(task.algorithm), string(task.cut), task.T, task.R,
                seed, "error", 0, 0, NaN, NaN, "", 0, false, false,
                (time_ns() - started) / 1e9, replace(message, '\n' => ' '),
            ))
        end
        mkpath(MSUC_VALIDATION_ROOT)
        CSV.write(output_path, rows)
        @everywhere GC.gc()
    end
    return rows
end

function alias_numerical_tasks()
    tasks = NamedTuple{(:algorithm, :cut), Tuple{Symbol, Symbol}}[]
    for cut in MSUC_ALIAS_CUTS, algorithm in MSUC_ALGORITHMS
        is_supported_configuration(algorithm, cut) || continue
        push!(tasks, (algorithm = algorithm, cut = cut))
    end
    require_check(length(tasks) == 22, "alias task list has $(length(tasks)) tasks; expected 22")
    for cut in MSUC_ALIAS_CUTS
        expected_count = cut in SDDIP_UNSUPPORTED_CUT_TYPES ? 2 : 3
        actual_count = count(task -> task.cut == cut, tasks)
        require_check(
            actual_count == expected_count,
            "$cut has $actual_count legal algorithm combinations; expected $expected_count",
        )
    end
    return tasks
end

function empty_alias_report()::DataFrame
    return DataFrame(
        task_id = Int[], algorithm = String[], cut = String[], T = Int[], R = Int[],
        seed = Int[], expected_sequence = String[], actual_sequence = String[],
        status = String[], iterations = Int[], completed_backward = Int[],
        final_LB = Float64[], final_UB = Float64[], final_gap = String[],
        max_LM_iter = Int[], bounds_ordered = Bool[], lb_monotone = Bool[],
        elapsed_seconds = Float64[], error = String[],
    )
end

function run_alias_numerical_validation()::DataFrame
    tasks = alias_numerical_tasks()
    rows = empty_alias_report()
    output_path = joinpath(MSUC_VALIDATION_ROOT, "alias_numerical_report.csv")
    mkpath(MSUC_VALIDATION_ROOT)

    for (task_id, task) in enumerate(tasks)
        algorithm_index = findfirst(==(task.algorithm), MSUC_ALGORITHMS)
        cut_index = findfirst(==(task.cut), MSUC_ALIAS_CUTS)
        seed = 300_000_000 + 10_000 * algorithm_index + 1_000_000 * cut_index
        expected = collect(MSUC_ALIAS_EXPECTED_SEQUENCES[task.cut])
        println(
            "[alias $task_id/$(length(tasks))] $(task.algorithm)/$(task.cut), " *
            "T=6, R=5, iterations=4",
        )
        started = time_ns()
        try
            output = run_validation_experiment(
                task.algorithm, task.cut, 6, 5; seed = seed, iterations = 4,
            )
            checked = validate_numerical_output(output, task.algorithm, task.cut, 4)
            runtime = output.sddpResults[:runtimeHistory]
            actual = collect(runtime.Cut)
            errors = copy(checked.errors)
            actual == expected || push!(
                errors,
                "runtime cut sequence $(Tuple(actual)) does not equal $(Tuple(expected))",
            )
            push!(rows, (
                task_id, string(task.algorithm), string(task.cut), 6, 5, seed,
                join(string.(expected), "->"), join(string.(actual), "->"),
                isempty(errors) ? "pass" : "fail", checked.iterations,
                checked.completed_backward, checked.final_LB, checked.final_UB,
                checked.final_gap, checked.max_LM_iter, checked.bounds_ordered,
                checked.lb_monotone, (time_ns() - started) / 1e9, join(errors, " | "),
            ))
        catch err
            message = sprint(showerror, err, catch_backtrace())
            length(message) > 8_000 && (message = first(message, 8_000))
            push!(rows, (
                task_id, string(task.algorithm), string(task.cut), 6, 5, seed,
                join(string.(expected), "->"), "", "error", 0, 0, NaN, NaN, "",
                0, false, false, (time_ns() - started) / 1e9,
                replace(message, '\n' => ' '),
            ))
        end
        CSV.write(output_path, rows)
        @everywhere GC.gc()
    end
    return rows
end

function validate_parameter_contracts()::DataFrame
    cases = (
        (
            name = "unknown_algorithm", algorithm = :Bogus, cut = :LC,
            case_name = "missing-contract-fixture", kwargs = NamedTuple(),
        ),
        (
            name = "unknown_cut", algorithm = :SDDPL, cut = :Bogus,
            case_name = "missing-contract-fixture", kwargs = NamedTuple(),
        ),
        (
            name = "unsupported_sddip_relu", algorithm = :SDDiP, cut = :ReLUC,
            case_name = "missing-contract-fixture", kwargs = NamedTuple(),
        ),
        (
            name = "negative_terminate_threshold", algorithm = :SDDPL, cut = :LC,
            case_name = "case30", kwargs = (terminate_threshold = -1.0,),
        ),
        (
            name = "invalid_mip_gap", algorithm = :SDDPL, cut = :LC,
            case_name = "case30", kwargs = (MIPGap = 1.1,),
        ),
        (
            name = "nonpositive_time_limit", algorithm = :SDDPL, cut = :LC,
            case_name = "case30", kwargs = (TimeLimit = 0.0,),
        ),
        (
            name = "nonpositive_epsilon", algorithm = :SDDPL, cut = :LC,
            case_name = "case30", kwargs = (ε = 0.0,),
        ),
    )
    rows = DataFrame(
        check = String[], status = String[], exception_type = String[], error = String[],
    )
    common = (
        numScenarios = 2,
        M = 1,
        logger_save = false,
        terminate_time = 1,
        terminate_threshold = 0.0,
        TimeLimit = 1.0,
        MIPGap = 1e-4,
        MaxIter = 1,
        partitionRule = :Bisection,
        ε = 1 / 2^8,
        levelMethodMaxIter = 1,
        sparse_cut = :sparse,
        branch_variable = :ALL,
        LiftIterThreshold = 1,
    )
    for test_case in cases
        caught = nothing
        try
            kwargs = merge(common, (case = test_case.case_name,), test_case.kwargs)
            run_single_experiment(
                test_case.algorithm, test_case.cut, 6, 5; kwargs...,
            )
        catch err
            caught = err
        end
        passed = caught isa ArgumentError
        push!(rows, (
            test_case.name,
            passed ? "pass" : "fail",
            caught === nothing ? "" : string(typeof(caught)),
            caught === nothing ? "no exception was thrown" : sprint(showerror, caught),
        ))
    end
    return rows
end

function run_adaptive_baseline_validation()::DataFrame
    rows = DataFrame(
        algorithm = String[], cut = String[], T = Int[], R = Int[], seed = Int[],
        iterations = Int[], expected_LB = Float64[], actual_LB = Float64[],
        atol = Float64[], rtol = Float64[], status = String[], error = String[],
    )
    for baseline in MSUC_ADAPTIVE_BASELINES
        try
            output = run_validation_experiment(
                baseline.algorithm, baseline.cut, baseline.T, baseline.R;
                seed = baseline.seed, iterations = baseline.iterations,
            )
            checked = validate_numerical_output(
                output, baseline.algorithm, baseline.cut, baseline.iterations,
            )
            actual_LB = Float64(output.summary.LB)
            errors = copy(checked.errors)
            isapprox(
                actual_LB, baseline.expected_LB;
                atol = MSUC_BASELINE_ATOL, rtol = MSUC_BASELINE_RTOL,
            ) || push!(
                errors,
                "final LB $actual_LB is outside the fixed-seed baseline tolerance",
            )
            push!(rows, (
                string(baseline.algorithm), string(baseline.cut), baseline.T, baseline.R,
                baseline.seed, baseline.iterations, baseline.expected_LB, actual_LB,
                MSUC_BASELINE_ATOL, MSUC_BASELINE_RTOL,
                isempty(errors) ? "pass" : "fail", join(errors, " | "),
            ))
        catch err
            push!(rows, (
                string(baseline.algorithm), string(baseline.cut), baseline.T, baseline.R,
                baseline.seed, baseline.iterations, baseline.expected_LB, NaN,
                MSUC_BASELINE_ATOL, MSUC_BASELINE_RTOL, "error",
                replace(sprint(showerror, err, catch_backtrace()), '\n' => ' '),
            ))
        end
        @everywhere GC.gc()
    end
    return rows
end

function run_msuc_additional_validation()
    mkpath(MSUC_VALIDATION_ROOT)
    numerical_tasks()
    hash_report = validate_input_hashes()
    parameter_report = validate_parameter_contracts()
    alias_report = run_alias_numerical_validation()
    baseline_report = run_adaptive_baseline_validation()
    CSV.write(joinpath(MSUC_VALIDATION_ROOT, "input_hash_report.csv"), hash_report)
    CSV.write(joinpath(MSUC_VALIDATION_ROOT, "parameter_contract_report.csv"), parameter_report)
    CSV.write(joinpath(MSUC_VALIDATION_ROOT, "adaptive_baseline_report.csv"), baseline_report)

    failures = sum(
        count(!=("pass"), report.status)
        for report in (hash_report, parameter_report, alias_report, baseline_report)
    )
    println(
        "Additional MSUC validation complete: " *
        "hashes=$(count(==("pass"), hash_report.status))/$(nrow(hash_report)), " *
        "parameters=$(count(==("pass"), parameter_report.status))/$(nrow(parameter_report)), " *
        "aliases=$(count(==("pass"), alias_report.status))/$(nrow(alias_report)), " *
        "baselines=$(count(==("pass"), baseline_report.status))/$(nrow(baseline_report)).",
    )
    return (
        hash_report = hash_report,
        parameter_report = parameter_report,
        alias_report = alias_report,
        baseline_report = baseline_report,
        passed = failures == 0,
    )
end

function run_msuc_validation()
    mkpath(MSUC_VALIDATION_ROOT)
    input_report = validate_inputs()
    dispatch_report = validate_cut_dispatch()
    hash_report = validate_input_hashes()
    parameter_report = validate_parameter_contracts()
    CSV.write(joinpath(MSUC_VALIDATION_ROOT, "input_report.csv"), input_report)
    CSV.write(joinpath(MSUC_VALIDATION_ROOT, "dispatch_report.csv"), dispatch_report)
    CSV.write(joinpath(MSUC_VALIDATION_ROOT, "input_hash_report.csv"), hash_report)
    CSV.write(joinpath(MSUC_VALIDATION_ROOT, "parameter_contract_report.csv"), parameter_report)

    tasks = numerical_tasks()
    numerical_report = run_numerical_validation(tasks)
    alias_report = run_alias_numerical_validation()
    baseline_report = run_adaptive_baseline_validation()
    CSV.write(joinpath(MSUC_VALIDATION_ROOT, "adaptive_baseline_report.csv"), baseline_report)
    input_failures = count(!=("pass"), input_report.status)
    dispatch_failures = count(!=("pass"), dispatch_report.status)
    hash_failures = count(!=("pass"), hash_report.status)
    parameter_failures = count(!=("pass"), parameter_report.status)
    numerical_failures = count(!=("pass"), numerical_report.status)
    alias_failures = count(!=("pass"), alias_report.status)
    baseline_failures = count(!=("pass"), baseline_report.status)
    println(
        "MSUC validation complete: inputs=$(nrow(input_report) - input_failures)/$(nrow(input_report)), " *
        "dispatch=$(nrow(dispatch_report) - dispatch_failures)/$(nrow(dispatch_report)), " *
        "hashes=$(nrow(hash_report) - hash_failures)/$(nrow(hash_report)), " *
        "parameters=$(nrow(parameter_report) - parameter_failures)/$(nrow(parameter_report)), " *
        "numerical=$(nrow(numerical_report) - numerical_failures)/$(nrow(numerical_report)), " *
        "aliases=$(nrow(alias_report) - alias_failures)/$(nrow(alias_report)), " *
        "baselines=$(nrow(baseline_report) - baseline_failures)/$(nrow(baseline_report)).",
    )
    println("Reports: $MSUC_VALIDATION_ROOT")
    return (
        input_report = input_report,
        dispatch_report = dispatch_report,
        hash_report = hash_report,
        parameter_report = parameter_report,
        numerical_report = numerical_report,
        alias_report = alias_report,
        baseline_report = baseline_report,
        passed = input_failures + dispatch_failures + hash_failures + parameter_failures +
                 numerical_failures + alias_failures + baseline_failures == 0,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    report = run_msuc_validation()
    report.passed || exit(1)
end
