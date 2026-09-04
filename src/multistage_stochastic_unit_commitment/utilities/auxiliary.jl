const CORE_POINT_STRATEGIES = ("Mid", "Eps", "Conv")
const CONT_AUG_STATE_TOLERANCE = 1e-6
const SUPPORTED_ALGORITHMS = (:SDDP, :SDDPL, :SDDiP)
const SUPPORTED_CUT_TYPES = (
    :LC,
    :PLC,
    :AdaptivePLC,
    :SMC,
    :AdaptiveSMC,
    :SBC,
    :LNC,
    :ReLUC,
    :NormalizedReLUC,
    :SBCLC,
    :SBCSMC,
    :SBCPLC,
    :SBCLNC,
    :SBCNormalizedCut,
    :SBCReLUC,
    :SBCNormalizedReLUC,
    :NormalizedCut,
)
const SDDIP_UNSUPPORTED_CUT_TYPES = (
    :ReLUC,
    :NormalizedReLUC,
    :SBCReLUC,
    :SBCNormalizedReLUC,
)

"""Return whether an algorithm/cut pair is implemented by the MSUC solver."""
function is_supported_configuration(algorithm::Symbol, cutSelection::Symbol)::Bool
    return algorithm in SUPPORTED_ALGORITHMS &&
           cutSelection in SUPPORTED_CUT_TYPES &&
           !(algorithm == :SDDiP && cutSelection in SDDIP_UNSUPPORTED_CUT_TYPES)
end

function _core_point_strategy(param_cut::NamedTuple)::String
    strategy = String(param_cut.core_point_strategy)
    if strategy ∉ CORE_POINT_STRATEGIES
        error(
            "Unsupported core point strategy: $(strategy). " *
            "Expected one of $(join(CORE_POINT_STRATEGIES, ", ")).",
        )
    end
    return strategy
end

function _core_point_weight(param_cut::NamedTuple)::Float64
    weight = haskey(param_cut, :core_point_weight) ? param_cut.core_point_weight : param_cut.ℓ
    0.0 <= weight <= 1.0 || error("core_point_weight must be in [0, 1]. Got $(weight).")
    return Float64(weight)
end

function _core_point_epsilon(param_cut::NamedTuple)::Float64
    epsilon = haskey(param_cut, :core_point_epsilon) ? param_cut.core_point_epsilon : 1e-2
    0.0 <= epsilon < 0.5 || error("core_point_epsilon must be in [0, 0.5). Got $(epsilon).")
    return Float64(epsilon)
end

function _core_value(
    incumbent::Real,
    lower_bound::Real,
    upper_bound::Real,
    param_cut::NamedTuple,
)::Float64
    lb = Float64(lower_bound)
    ub = Float64(upper_bound)
    x = Float64(incumbent)
    width = ub - lb
    width >= 0.0 || error("Invalid core point bounds: lower_bound=$(lb), upper_bound=$(ub).")

    strategy = _core_point_strategy(param_cut)
    if width == 0.0
        return lb
    elseif strategy == "Mid"
        return (lb + ub) / 2
    elseif strategy == "Eps"
        margin = _core_point_epsilon(param_cut) * width
        return clamp(x, lb + margin, ub - margin)
    else
        weight = _core_point_weight(param_cut)
        diagonal = lb + ub - x
        return weight * x + (1.0 - weight) * diagonal
    end
end

function _uses_dense_cont_aug_state(param::NamedTuple)::Bool
    return hasproperty(param, :sparse_cut) && param.sparse_cut == :dense
end

function _leaf_contains_value(
    leafInfo::Dict{Symbol, Any},
    value::Real;
    tol::Float64 = CONT_AUG_STATE_TOLERANCE,
)::Bool
    return Float64(leafInfo[:lb]) - tol <= Float64(value) <= Float64(leafInfo[:ub]) + tol
end

function _sorted_leaf_keys(keys_to_sort, leaves)
    return sort(
        collect(keys_to_sort);
        by = k -> (Float64(leaves[k][:ub]), Float64(leaves[k][:lb]), string(k)),
    )
end

function _active_cont_leaf_from_state(
    stateInfo::StateInfo,
    leaves::Dict,
    g,
)
    state_value = Float64(stateInfo.ContVar[:s][g])

    if stateInfo.ContAugState !== nothing &&
       haskey(stateInfo.ContAugState, :s) &&
       haskey(stateInfo.ContAugState[:s], g)
        active_from_aug = [
            k for (k, value) in stateInfo.ContAugState[:s][g]
            if abs(Float64(value) - 1.0) <= CONT_AUG_STATE_TOLERANCE &&
               haskey(leaves, k) &&
               _leaf_contains_value(leaves[k], state_value)
        ]
        if !isempty(active_from_aug)
            return first(_sorted_leaf_keys(active_from_aug, leaves))
        end
    end

    if stateInfo.ContVarLeaf !== nothing &&
       haskey(stateInfo.ContVarLeaf, :s) &&
       haskey(stateInfo.ContVarLeaf[:s], g)
        active_from_forward = [
            k for (k, value) in stateInfo.ContVarLeaf[:s][g]
            if get(value, :var, 0.0) > 0.5 &&
               haskey(leaves, k) &&
               _leaf_contains_value(leaves[k], state_value)
        ]
        if !isempty(active_from_forward)
            return first(_sorted_leaf_keys(active_from_forward, leaves))
        end
    end

    candidates = [
        k for (k, leafInfo) in leaves
        if _leaf_contains_value(leafInfo, state_value)
    ]
    if !isempty(candidates)
        return first(_sorted_leaf_keys(candidates, leaves))
    end

    error(
        "Unable to locate an active SDDP-L partition leaf for generator $g " *
        "at state s=$(state_value)."
    )
end

"""
Return the lifted leaf indicator implied by the current continuous state.

SDDP-L refines the partition tree during the algorithm, so an old forward
record can reference a leaf that has since been split. This helper rebuilds
the active augment state from `stateInfo.ContVar` and the current
`modelInfo.ContVarLeaf` before cut generation.
"""
function active_cont_aug_state(
    stateInfo::StateInfo,
    modelInfo::SDDPModel;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Any}}}}
    if modelInfo.ContVarLeaf === nothing || stateInfo.ContVar === nothing
        return nothing
    end

    dense_aug_state = _uses_dense_cont_aug_state(param)
    return Dict{Any, Dict{Any, Dict{Any, Any}}}(
        :s => Dict{Any, Dict{Any, Any}}(
            g => begin
                leaves = modelInfo.ContVarLeaf[:s][g]
                active_leaf = _active_cont_leaf_from_state(stateInfo, leaves, g)
                if dense_aug_state
                    Dict{Any, Any}(
                        k => (k == active_leaf ? 1.0 : 0.0)
                        for k in keys(leaves)
                    )
                else
                    Dict{Any, Any}(active_leaf => 1.0)
                end
            end for g in indexSets.G
        ),
    )
end

"""
Refresh `stateInfo.ContAugState` in place using the current SDDP-L partition.
"""
function sync_cont_aug_state!(
    stateInfo::StateInfo,
    modelInfo::SDDPModel;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::StateInfo
    stateInfo.ContAugState = active_cont_aug_state(
        stateInfo,
        modelInfo;
        indexSets = indexSets,
        param = param,
    )
    return stateInfo
end

function _setup_core_point(
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    paramOPF::ParamOPF = paramOPF,
    param::NamedTuple = param,
    param_cut::NamedTuple = param_cut,
)::StateInfo
    BinVar = Dict{Any, Dict{Any, Any}}(
        :y => Dict{Any, Any}(
            g => _core_value(stateInfo.BinVar[:y][g], 0.0, 1.0, param_cut)
            for g in indexSets.G
        ),
        :v => Dict{Any, Any}(
            g => 0.0 for g in indexSets.G
        ),
        :w => Dict{Any, Any}(
            g => 0.0 for g in indexSets.G
        ),
    )

    ContVar = Dict{Any, Dict{Any, Any}}(
        :s => Dict{Any, Any}(
            g => _core_value(stateInfo.ContVar[:s][g], paramOPF.smin[g], paramOPF.smax[g], param_cut)
            for g in indexSets.G
        ),
    )

    ContAugState = stateInfo.ContAugState === nothing ? nothing :
        Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    k => _core_value(stateInfo.ContAugState[:s][g][k], 0.0, 1.0, param_cut)
                    for k in keys(stateInfo.ContAugState[:s][g])
                ) for g in indexSets.G
            ),
        )

    ContStateBin = stateInfo.ContStateBin === nothing ? nothing :
        Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    i => _core_value(stateInfo.ContStateBin[:s][g][i], 0.0, 1.0, param_cut)
                    for i in 1:param.κ[g]
                ) for g in indexSets.G
            ),
        )

    return StateInfo(
        BinVar,
        nothing,
        ContVar,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        ContAugState,
        nothing,
        ContStateBin,
    )
end

function _state_minus_core(
    stateInfo::StateInfo,
    corePoint::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::StateInfo
    BinVar = Dict{Any, Dict{Any, Any}}(
        name => Dict{Any, Any}(
            g => stateInfo.BinVar[name][g] - corePoint.BinVar[name][g]
            for g in indexSets.G
        ) for name in (:y, :v, :w)
    )

    ContVar = Dict{Any, Dict{Any, Any}}(
        :s => Dict{Any, Any}(
            g => stateInfo.ContVar[:s][g] - corePoint.ContVar[:s][g]
            for g in indexSets.G
        ),
    )

    ContAugState = stateInfo.ContAugState === nothing ? nothing :
        Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    k => stateInfo.ContAugState[:s][g][k] - corePoint.ContAugState[:s][g][k]
                    for k in keys(stateInfo.ContAugState[:s][g])
                ) for g in indexSets.G
            ),
        )

    ContStateBin = stateInfo.ContStateBin === nothing ? nothing :
        Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    i => stateInfo.ContStateBin[:s][g][i] - corePoint.ContStateBin[:s][g][i]
                    for i in 1:param.κ[g]
                ) for g in indexSets.G
            ),
        )

    return StateInfo(
        BinVar,
        nothing,
        ContVar,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        ContAugState,
        nothing,
        ContStateBin,
    )
end

"""
    setup_PLC_core_point(stateInfo; ...)

Build the core point used by Pareto Lagrangian cuts. Supported strategies are:
`Mid`, `Eps`, and `Conv`.
"""
function setup_PLC_core_point(
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    paramOPF::ParamOPF = paramOPF,
    param::NamedTuple = param,
    param_cut::NamedTuple = param_cut,
)::StateInfo
    return _setup_core_point(
        stateInfo;
        indexSets = indexSets,
        paramOPF = paramOPF,
        param = param,
        param_cut = param_cut,
    )
end

"""
    setup_LNC_core_point(stateInfo; ...)

Build the normalization vector `(xhat - core)` used by the linear normalized
Lagrangian cut.
"""
function setup_LNC_core_point(
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    paramOPF::ParamOPF = paramOPF,
    param::NamedTuple = param,
    param_cut::NamedTuple = param_cut
)::StateInfo
    corePoint = setup_PLC_core_point(
        stateInfo;
        indexSets = indexSets,
        paramOPF = paramOPF,
        param = param,
        param_cut = param_cut,
    )
    return _state_minus_core(
        stateInfo,
        corePoint;
        indexSets = indexSets,
        param = param,
    )
end

"""
    binarize_continuous_variable(
        state::Float64,
        smax::Float64,
        param::NamedTuple,
    )::Vector

Encode a continuous state value on the `ε`-grid through its binary expansion.
"""
function binarize_continuous_variable(
    state::Float64,
    smax::Float64,
    param::NamedTuple
)::Vector
    if smax == 0
        return [0]
    end
    num_binary_vars = floor(Int, log2(smax / param.ε)) + 1
    max_integer = floor(Int, state / param.ε)
    binary_representation = digits(max_integer, base=2, pad=num_binary_vars)
    return binary_representation
end

"""
Return a zero lifted-dual block with the same leaf support as `stateInfo`.
"""
function zero_relu_aug_state(
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
)
    if stateInfo.ContAugState === nothing
        return nothing
    end

    return Dict{Any, Dict{Any, Dict{Any, Any}}}(
        :s => Dict{Any, Dict{Any, Any}}(
            g => Dict{Any, Any}(
                k => 0.0 for k in keys(stateInfo.ContAugState[:s][g])
            ) for g in indexSets.G
        ),
    )
end

"""
    setup_ReLU_initial_point(stateInfo::StateInfo)

Initialize the ReLU-dual iterate at the origin.
"""
function setup_ReLU_initial_point(
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
)::ReLUDualStateInfo
    return ReLUDualStateInfo(
        nothing,
        Dict(
            name => Dict(g => 0.0 for g in indexSets.G)
            for name in keys(stateInfo.BinVar)
        ),
        Dict(
            name => Dict(g => 0.0 for g in indexSets.G)
            for name in keys(stateInfo.BinVar)
        ),
        Dict(
            :s => Dict(g => 0.0 for g in indexSets.G)
        ),
        Dict(
            :s => Dict(g => 0.0 for g in indexSets.G)
        ),
        zero_relu_aug_state(stateInfo; indexSets = indexSets),
    )
end

const RELU_CORE_TOLERANCE = 1e-6
const RELU_CORE_EPS_BOUND = 1e-3
const RELU_DUAL_VAR_BOUND_SCALE = 3.0
const RELU_DUAL_N0_BOUND = 1.0
const RELU_MIN_NORMALIZED_PI0 = 1e-8
const NORMALIZED_RELU_OBJECTIVE_INTERIOR = 1e-6
const RELU_LOCAL_VALIDITY_TOLERANCE = 1e-6

function relu_aug_state_dot(
    dualInfo::ReLUDualStateInfo,
    stateInfo::StateInfo,
)
    if dualInfo.ContAugState === nothing || stateInfo.ContAugState === nothing
        return 0.0
    end

    return sum(
        sum(
            dualInfo.ContAugState[:s][g][k] * stateInfo.ContAugState[:s][g][k]
            for k in keys(dualInfo.ContAugState[:s][g]);
            init = 0.0,
        )
        for g in keys(dualInfo.ContAugState[:s]);
        init = 0.0,
    )
end

function relu_aug_state_dot(
    left::ReLUDualStateInfo,
    right::ReLUDualStateInfo,
)
    if left.ContAugState === nothing || right.ContAugState === nothing
        return 0.0
    end

    return sum(
        sum(
            left.ContAugState[:s][g][k] * right.ContAugState[:s][g][k]
            for k in keys(left.ContAugState[:s][g]);
            init = 0.0,
        )
        for g in keys(left.ContAugState[:s]);
        init = 0.0,
    )
end

function relu_cut_rhs(
    oracle_value::Real,
    dualInfo::ReLUDualStateInfo,
    stateInfo::StateInfo,
)::Float64
    return Float64(oracle_value) - relu_aug_state_dot(dualInfo, stateInfo)
end

function relu_aug_lagrangian_term(
    dualInfo::ReLUDualStateInfo,
    model::Model,
    stateInfo::StateInfo,
)
    if dualInfo.ContAugState === nothing || stateInfo.ContAugState === nothing
        return 0.0
    end

    return sum(
        sum(
            dualInfo.ContAugState[:s][g][k] *
            (stateInfo.ContAugState[:s][g][k] - model[:augmentVar_copy][g, k])
            for k in keys(dualInfo.ContAugState[:s][g]);
            init = 0.0,
        )
        for g in keys(dualInfo.ContAugState[:s]);
        init = 0.0,
    )
end

function relu_aug_value_gradient(
    model::Model,
    stateInfo::StateInfo,
)
    if stateInfo.ContAugState === nothing
        return nothing
    end

    return Dict{Any, Dict{Any, Dict{Any, Any}}}(
        :s => Dict{Any, Dict{Any, Any}}(
            g => Dict{Any, Any}(
                k => JuMP.value(model[:augmentVar_copy][g, k]) -
                     stateInfo.ContAugState[:s][g][k]
                for k in keys(stateInfo.ContAugState[:s][g])
            )
            for g in keys(stateInfo.ContAugState[:s])
        ),
    )
end

function zero_relu_aug_gradient(stateInfo::StateInfo)
    if stateInfo.ContAugState === nothing
        return nothing
    end

    return Dict{Any, Dict{Any, Dict{Any, Any}}}(
        :s => Dict{Any, Dict{Any, Any}}(
            g => Dict{Any, Any}(k => 0.0 for k in keys(stateInfo.ContAugState[:s][g]))
            for g in keys(stateInfo.ContAugState[:s])
        ),
    )
end

"""
    setup_NormalizedReLU_core_point(
        stateInfo::StateInfo;
        primal_bound::Float64,
        incumbent_theta::Float64,
    )

Construct normalization coefficients for the normalized ReLU dual following
the core-point logic used in the reference implementation. In particular, the
scalar normalization coefficient is set to
`primal_bound - incumbent_theta + 10^-6`.
"""
function setup_NormalizedReLU_core_point(
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    paramOPF::ParamOPF = paramOPF,
    param::NamedTuple = param,
    param_cut::NamedTuple = param_cut,
    primal_bound::Float64,
    incumbent_theta::Float64,
)::ReLUDualStateInfo
    BinVarPlus = Dict{Any, Dict{Any, Any}}()
    BinVarMinus = Dict{Any, Dict{Any, Any}}()
    for name in keys(stateInfo.BinVar)
        BinVarPlus[name] = Dict{Any, Any}()
        BinVarMinus[name] = Dict{Any, Any}()
        for g in indexSets.G
            x̂ = stateInfo.BinVar[name][g]
            if abs(x̂) ≤ RELU_CORE_TOLERANCE
                BinVarPlus[name][g] = RELU_CORE_EPS_BOUND
                BinVarMinus[name][g] = 0.0
            elseif abs(x̂ - 1.0) ≤ RELU_CORE_TOLERANCE
                BinVarPlus[name][g] = 0.0
                BinVarMinus[name][g] = RELU_CORE_EPS_BOUND
            else
                u = 0.5 * x̂ * (1.0 - x̂)
                BinVarPlus[name][g] = u
                BinVarMinus[name][g] = u
            end
        end
    end

    ContVarPlus = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}())
    ContVarMinus = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}())
    for g in indexSets.G
        x̂ = stateInfo.ContVar[:s][g]
        B = paramOPF.smax[g]
        if abs(x̂) ≤ RELU_CORE_TOLERANCE
            ContVarPlus[:s][g] = RELU_CORE_EPS_BOUND
            ContVarMinus[:s][g] = 0.0
        elseif abs(x̂ - B) ≤ RELU_CORE_TOLERANCE
            ContVarPlus[:s][g] = 0.0
            ContVarMinus[:s][g] = RELU_CORE_EPS_BOUND
        else
            u = 0.5 * x̂ * (1.0 - x̂ / B)
            ContVarPlus[:s][g] = u
            ContVarMinus[:s][g] = u
        end
    end

    ContAugState = nothing
    if stateInfo.ContAugState !== nothing
        corePoint = _setup_core_point(
            stateInfo;
            indexSets = indexSets,
            paramOPF = paramOPF,
            param = param,
            param_cut = param_cut,
        )
        linearNormalizer = _state_minus_core(
            stateInfo,
            corePoint;
            indexSets = indexSets,
            param = param,
        )
        ContAugState = linearNormalizer.ContAugState
    end

    return ReLUDualStateInfo(
        max(
            primal_bound - incumbent_theta + NORMALIZED_RELU_OBJECTIVE_INTERIOR,
            NORMALIZED_RELU_OBJECTIVE_INTERIOR,
        ),
        BinVarPlus,
        BinVarMinus,
        ContVarPlus,
        ContVarMinus,
        ContAugState,
    )
end

"""
    setup_NormalizedReLU_initial_point(
        stateInfo::StateInfo,
        scenario_objective::Float64,
        incumbent_theta::Float64,
        normalizationInfo::ReLUDualStateInfo;
        indexSets::IndexSets = indexSets,
    )::ReLUDualStateInfo

Construct the initial iterate for the normalized ReLU dual by following the
scaled interior-point rule used in the reference implementation. The iterate
is rescaled to satisfy the exact normalization constraint.
"""
function setup_NormalizedReLU_initial_point(
    stateInfo::StateInfo,
    scenario_objective::Float64,
    incumbent_theta::Float64,
    normalizationInfo::ReLUDualStateInfo;
    indexSets::IndexSets = indexSets,
)::ReLUDualStateInfo
    bound_value = RELU_DUAL_VAR_BOUND_SCALE * max(scenario_objective, 1.0)

    BinVarPlus = Dict(
        name => Dict(g => -bound_value for g in indexSets.G)
        for name in keys(stateInfo.BinVar)
    )
    BinVarMinus = Dict(
        name => Dict(g => -bound_value for g in indexSets.G)
        for name in keys(stateInfo.BinVar)
    )
    ContVarPlus = Dict(
        :s => Dict(g => -bound_value for g in indexSets.G)
    )
    ContVarMinus = Dict(
        :s => Dict(g => -bound_value for g in indexSets.G)
    )

    π₀ = clamp(scenario_objective - incumbent_theta, 0.0, RELU_DUAL_N0_BOUND)
    normalization_value =
        normalizationInfo.StateValue * π₀ +
        sum(
            normalizationInfo.ContVarPlus[:s][g] * ContVarPlus[:s][g] +
            normalizationInfo.ContVarMinus[:s][g] * ContVarMinus[:s][g] +
            normalizationInfo.BinVarPlus[:y][g] * BinVarPlus[:y][g] +
            normalizationInfo.BinVarMinus[:y][g] * BinVarMinus[:y][g] +
            normalizationInfo.BinVarPlus[:v][g] * BinVarPlus[:v][g] +
            normalizationInfo.BinVarMinus[:v][g] * BinVarMinus[:v][g] +
            normalizationInfo.BinVarPlus[:w][g] * BinVarPlus[:w][g] +
            normalizationInfo.BinVarMinus[:w][g] * BinVarMinus[:w][g]
            for g in indexSets.G
        )

    if normalization_value > 1.0
        for name in keys(BinVarPlus)
            for g in indexSets.G
                BinVarPlus[name][g] /= normalization_value
                BinVarMinus[name][g] /= normalization_value
            end
        end
        for g in indexSets.G
            ContVarPlus[:s][g] /= normalization_value
            ContVarMinus[:s][g] /= normalization_value
        end
        π₀ /= normalization_value
    end

    return ReLUDualStateInfo(
        clamp(π₀, 0.0, RELU_DUAL_N0_BOUND),
        BinVarPlus,
        BinVarMinus,
        ContVarPlus,
        ContVarMinus,
        zero_relu_aug_state(stateInfo; indexSets = indexSets),
    )
end

"""
    get_cut_selection(cutSelection::Symbol, i::Int)

    Base on cutSelection and iteration index i, determine the actual cut selection strategy to use.
"""
function get_cut_selection(cutSelection::Symbol, i::Int)
    if cutSelection == :SBCLC
        return i <= 3 ? :SBC : :LC
    elseif cutSelection == :SBCSMC
        return i <= 3 ? :SBC : :SMC
    elseif cutSelection == :SBCPLC
        return i <= 3 ? :SBC : :PLC
    elseif cutSelection ∈ (:SBCLNC, :SBCNormalizedCut)
        return i <= 3 ? :SBC : :LNC
    elseif cutSelection == :SBCReLUC
        return i <= 3 ? :SBC : :ReLUC
    elseif cutSelection == :SBCNormalizedReLUC
        return i <= 3 ? :SBC : :NormalizedReLUC
    elseif cutSelection == :NormalizedCut
        return :LNC
    end
    return cutSelection
end

is_pareto_lagrangian_cut(cutSelection::Symbol)::Bool =
    cutSelection in (:PLC, :AdaptivePLC)

is_square_minimization_cut(cutSelection::Symbol)::Bool =
    cutSelection in (:SMC, :AdaptiveSMC)

is_adaptive_level_cut(cutSelection::Symbol)::Bool =
    cutSelection in (:AdaptivePLC, :AdaptiveSMC)
