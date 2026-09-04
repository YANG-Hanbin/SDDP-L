"""
Modify the forward model for a given realization (demand, previous state).

- For all algorithms: update demand constraint.
- For SDDiP      : fix Lc to previous binary state `L̂`.
- For SDDP/SDDPL : fix Sc to previous integer state `Ŝ`.
"""

const RELU_CORE_SCALE_NONBOUND = 0.5
const RELU_CORE_EPS_BOUND = 1e-3
const RELU_CORE_TOLERANCE = 1e-6
const RELU_DUAL_VAR_BOUND_SCALE = 3.0
const RELU_DUAL_N0_BOUND = 1.0
const RELU_MIN_NORMALIZED_PI0 = 1e-8
const NORMALIZED_RELU_DUAL_OBJECTIVE_BOUND = 2.0
const NORMALIZED_RELU_OBJECTIVE_INTERIOR = 1e-6
const RELU_LOCAL_VALIDITY_TOLERANCE = 1e-6

"""
    setup_regular_relu_starting_point(
        stateInfo::StageInfo,
        scenario_objective::Float64,
    )::ReLUDualStageInfo

Construct the regular ReLU-dual starting iterate using the box bound of the
dual master problem.
"""
function setup_regular_relu_starting_point(
    stateInfo::StageInfo,
    scenario_objective::Float64,
)::ReLUDualStageInfo
    bound_value = RELU_DUAL_VAR_BOUND_SCALE * max(scenario_objective, 1.0)
    start_value = -bound_value

    return ReLUDualStageInfo(
        nothing,
        fill(start_value, length(stateInfo.IntVar)),
        fill(start_value, length(stateInfo.IntVar)),
        zero_relu_lifted_leaf_state(stateInfo),
    )
end

"""
    setup_normalized_relu_starting_point(
        stateInfo::StageInfo,
        scenario_objective::Float64,
        incumbent_theta::Float64,
        normalizationInfo::ReLUDualStageInfo,
    )::ReLUDualStageInfo

Construct the normalized ReLU-dual starting iterate from the incumbent
epigraph gap and the normalization vector.

The iterate is clipped to the explicit box used by the dual master:

- `π⁺, π⁻ ∈ [-M, M]`,
- `π₀ ∈ [0, 1]`.
"""
function setup_normalized_relu_starting_point(
    stateInfo::StageInfo,
    scenario_objective::Float64,
    incumbent_theta::Float64,
    normalizationInfo::ReLUDualStageInfo,
)::ReLUDualStageInfo
    bound_value = RELU_DUAL_VAR_BOUND_SCALE * max(scenario_objective, 1.0)
    start_value = -bound_value

    π_plus = fill(start_value, length(stateInfo.IntVar))
    π_minus = fill(start_value, length(stateInfo.IntVar))
    π0 = clamp(scenario_objective - incumbent_theta, 0.0, RELU_DUAL_N0_BOUND)

    normalization_value =
        normalizationInfo.StateValue * π0 +
        normalizationInfo.IntVarPlus' * π_plus +
        normalizationInfo.IntVarMinus' * π_minus

    if normalization_value > 1.0
        π_plus ./= normalization_value
        π_minus ./= normalization_value
        π0 /= normalization_value
    end

    return ReLUDualStageInfo(
        clamp(π0, 0.0, RELU_DUAL_N0_BOUND),
        π_plus,
        π_minus,
        zero_relu_lifted_leaf_state(stateInfo),
    )
end

function model_modification!(
    model::Model,
    demand::Vector{Float64};
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::Nothing

    # remove old constraints
    delete(model, model[:demandConstraint])
    unregister(model, :demandConstraint)

    # remove Non-Anticipativity constraint
    if :NonAnticipativity ∈ keys(model.obj_dict) 
        delete(model, model[:NonAnticipativity])
        unregister(model, :NonAnticipativity)
    end

    # new demand constraint: sum(y) + slack ≥ sum(demand)
    @constraint(
        model,
        demandConstraint,
        sum(model[:y]) + model[:slack] ≥ sum(demand),
    )
    return
end

"""
    compute_node_primal_bound(
        model::Model,
        stateInfo::StageInfo;
        param,
    )::Float64

Solve the current backward node primal problem with the previous-stage state
fixed at the incumbent state `x̂`. The returned value is the node-specific
cost-to-go `Q_t(x̂, ξ_t^j)` used by the ReLU-based cut generation routines.

For the SDDP benchmark, this amounts to fixing `Sc == x̂` and minimizing the
stage primal objective. When ReLU auxiliary variables already exist in the
reused model, they are refreshed so that the temporary primal solve is
consistent with the current incumbent state.
"""
function compute_node_primal_bound(
    model::Model,
    stateInfo::StageInfo;
    param::SDDPParam = param,
)::Float64
    if param.algorithm != :SDDP && param.algorithm != :SDDPL
        error("Node-specific ReLU primal bounds are only implemented for SDDP and SDDP-L.")
    end

    update_relu_dual_auxiliary_variables!(
        model,
        stateInfo;
        param = param,
    )

    local leaf_fix = nothing
    try
        @constraint(
            model,
            NodePrimalStateFix,
            model[:Sc] .== stateInfo.IntVar,
        )

        if param.algorithm == :SDDPL && stateInfo.IntVarLeaf !== nothing
            leaf_fix = Dict(
                g => Dict(
                    k => @constraint(
                        model,
                        model[:region_indicator_copy][g][k] == stateInfo.IntVarLeaf[g][k],
                    ) for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in keys(stateInfo.IntVarLeaf)
            )
        end

        @objective(model, Min, model[:primal_objective_expression])
        optimize!(model)

        termination = termination_status(model)
        if termination != MOI.OPTIMAL && termination != MOI.LOCALLY_SOLVED
            error("Failed to solve the node-specific primal problem (status = $termination).")
        end

        return objective_value(model)
    finally
        if :NodePrimalStateFix ∈ keys(model.obj_dict)
            delete(model, model[:NodePrimalStateFix])
            unregister(model, :NodePrimalStateFix)
        end
        if leaf_fix !== nothing
            for g in keys(leaf_fix), k in keys(leaf_fix[g])
                delete(model, leaf_fix[g][k])
            end
        end
    end
end

"""
    This function is to collect the necessary parameters for the level set method.
"""
const GEP_CORE_POINT_STRATEGIES = (:Mid, :Eps, :Conv)

function core_point_strategy(param::SDDPParam)::Symbol
    strategy = param.corePointStrategy
    if strategy ∉ GEP_CORE_POINT_STRATEGIES
        error(
            "Unsupported core point strategy: $(strategy). " *
            "Expected one of $(join(string.(GEP_CORE_POINT_STRATEGIES), ", ")).",
        )
    end
    return strategy
end

function core_value(
    incumbent::Real,
    lower_bound::Real,
    upper_bound::Real,
    param::SDDPParam,
)::Float64
    lb = Float64(lower_bound)
    ub = Float64(upper_bound)
    x = Float64(incumbent)
    width = ub - lb
    width >= 0.0 || error("Invalid core point bounds: lower_bound=$(lb), upper_bound=$(ub).")

    strategy = core_point_strategy(param)
    if width == 0.0
        return lb
    elseif strategy == :Mid
        return (lb + ub) / 2
    elseif strategy == :Eps
        0.0 <= param.corePointEpsilon < 0.5 ||
            error("corePointEpsilon must be in [0, 0.5). Got $(param.corePointEpsilon).")
        margin = param.corePointEpsilon * width
        return clamp(x, lb + margin, ub - margin)
    else
        0.0 <= param.corePointWeight <= 1.0 ||
            error("corePointWeight must be in [0, 1]. Got $(param.corePointWeight).")
        diagonal = lb + ub - x
        return param.corePointWeight * x + (1.0 - param.corePointWeight) * diagonal
    end
end

function setup_stage_core_point(
    stateInfo::StageInfo;
    stageData::StageData = stageData,
    param::SDDPParam = param,
)::StageInfo
    return StageInfo(
        -1.0,
        nothing,
        stateInfo.IntVar === nothing ?
            nothing :
            [
                core_value(stateInfo.IntVar[g], 0.0, stageData.ū[g], param)
                for g in eachindex(stateInfo.IntVar)
            ],
        stateInfo.IntVarLeaf === nothing ?
            nothing :
            Dict(
                g => Dict(
                    k => core_value(stateInfo.IntVarLeaf[g][k], 0.0, 1.0, param)
                    for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in keys(stateInfo.IntVarLeaf)
            ),
        stateInfo.IntVarBinaries === nothing ?
            nothing :
            [
                core_value(stateInfo.IntVarBinaries[i], 0.0, 1.0, param)
                for i in eachindex(stateInfo.IntVarBinaries)
            ],
    )
end

function setup_lnc_normalization_state(
    stateInfo::StageInfo,
    coreState::StageInfo,
    theta_anchor::Float64,
    core_theta::Float64,
)::StageInfo
    return StageInfo(
        theta_anchor - core_theta,
        nothing,
        stateInfo.IntVar === nothing ?
            nothing :
            stateInfo.IntVar .- coreState.IntVar,
        stateInfo.IntVarLeaf === nothing ?
            nothing :
            Dict(
                g => Dict(
                    k => stateInfo.IntVarLeaf[g][k] - coreState.IntVarLeaf[g][k]
                    for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in keys(stateInfo.IntVarLeaf)
            ),
        stateInfo.IntVarBinaries === nothing ?
            nothing :
            stateInfo.IntVarBinaries .- coreState.IntVarBinaries,
    )
end

function lnc_theta_margin(
    theta_anchor::Real,
    param::SDDPParam,
)::Float64
    return max(
        param.lncCoreThetaMargin * max(abs(float(theta_anchor)), 1.0),
        param.lncCoreThetaMargin,
    )
end

function fallback_lnc_core_theta(
    theta_anchor::Real,
    param::SDDPParam,
)::Float64
    return float(theta_anchor) + lnc_theta_margin(theta_anchor, param)
end

function solve_lnc_state_theta(
    model::Model,
    targetState::StageInfo;
    fix_leaf_state::Bool,
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::Union{Float64, Nothing}
    local state_fix
    local leaf_fix = nothing

    try
        if param.algorithm == :SDDiP
            state_fix = @constraint(
                model,
                LNCStateThetaFix,
                model[:Lc] .== targetState.IntVarBinaries,
            )
        else
            state_fix = @constraint(
                model,
                LNCStateThetaFix,
                model[:Sc] .== targetState.IntVar,
            )
        end

        if fix_leaf_state && param.algorithm == :SDDPL && targetState.IntVarLeaf !== nothing
            leaf_fix = Dict(
                g => Dict(
                    k => @constraint(
                        model,
                        model[:region_indicator_copy][g][k] == targetState.IntVarLeaf[g][k],
                    ) for k in keys(targetState.IntVarLeaf[g])
                ) for g in keys(targetState.IntVarLeaf)
            )
        end

        @objective(model, Min, model[:primal_objective_expression])
        optimize!(model)

        status = termination_status(model)
        if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
            value = objective_value(model)
            return isfinite(value) ? value : nothing
        end
        return nothing
    finally
        if @isdefined(state_fix)
            delete(model, state_fix)
            unregister(model, :LNCStateThetaFix)
        end
        if leaf_fix !== nothing
            for g in keys(leaf_fix), k in keys(leaf_fix[g])
                delete(model, leaf_fix[g][k])
            end
        end
    end
end

function lnc_core_theta_info(
    model::Model,
    coreState::StageInfo,
    theta_anchor::Float64;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::NamedTuple
    core_theta = solve_lnc_state_theta(
        model,
        coreState;
        fix_leaf_state = true,
        param = param,
        binaryInfo = binaryInfo,
    )

    margin = lnc_theta_margin(theta_anchor, param)
    used_fallback =
        core_theta === nothing ||
        !isfinite(core_theta) ||
        abs(theta_anchor - core_theta) < margin

    if used_fallback
        core_theta = fallback_lnc_core_theta(theta_anchor, param)
    end

    return (
        core_theta = Float64(core_theta),
        used_fallback = used_fallback,
    )
end

function lnc_state_cut_value(
    dualInfo::StageInfo,
    stateInfo::StageInfo;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::Float64
    value = 0.0
    if param.algorithm == :SDDiP
        value += dualInfo.IntVarBinaries' * stateInfo.IntVarBinaries
    else
        value += dualInfo.IntVar' * stateInfo.IntVar
    end
    if param.algorithm == :SDDPL
        value += sum(
            sum(
                dualInfo.IntVarLeaf[g][k] * stateInfo.IntVarLeaf[g][k]
                for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    end
    return value
end

function lnc_cut_value(
    cutInfo,
    stateInfo::StageInfo;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::Float64
    λ₀, λ₁ = cutInfo
    scale = -λ₁.StateValue
    if !isfinite(scale) || scale <= 0.0
        return NaN
    end
    return (λ₀ + lnc_state_cut_value(
        λ₁,
        stateInfo;
        param = param,
        binaryInfo = binaryInfo,
    )) / scale
end

function lnc_tightness_tolerance(reference_value::Float64)::Float64
    return max(1e-4, 1e-8 * max(abs(reference_value), 1.0))
end

function setupCutGenerationInfo(
    model::Model,
    stateInfo::StageInfo,
    primal_bound::Float64,
    incumbent_theta::Float64,
    cutType::Symbol;
    stageData::StageData = stageData,
    binaryInfo::BinaryInfo = binaryInfo,
    param::SDDPParam = param
)::NamedTuple

    coreStatePLC = setup_stage_core_point(
        stateInfo;
        stageData = stageData,
        param = param,
    )

    πₙ = StageInfo(
        - 1.0,
        nothing,
        stateInfo.IntVar === nothing ?
            nothing :
            stateInfo.IntVar .* 0.0,
        stateInfo.IntVarLeaf === nothing ?
            nothing :
            Dict(
                g => Dict(
                    k => 0.0 for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in keys(stateInfo.IntVarLeaf)
            ),
        stateInfo.IntVarBinaries === nothing ?
            nothing :
            stateInfo.IntVarBinaries .* 0.0,
    );

    πₙ_init =
        if cutType == :NormalizedReLUC
            ReLUDualStageInfo(
                0.0,
                stateInfo.IntVar .* 0.0,
                stateInfo.IntVar .* 0.0,
                stateInfo.IntVarLeaf === nothing ?
                    nothing :
                    Dict(
                        g => Dict(k => 0.0 for k in keys(stateInfo.IntVarLeaf[g]))
                        for g in keys(stateInfo.IntVarLeaf)
                    ),
            )
        elseif cutType == :ReLUC
            setup_regular_relu_starting_point(
                stateInfo,
                primal_bound,
            )
        else
            πₙ
        end

    if cutType == :LC
        cutGenerationProgramInfo = LagrangianCutGenerationProgram(0.0)
    elseif cutType == :ReLUC
        if param.algorithm != :SDDP && param.algorithm != :SDDPL
            error("ReLU Lagrangian cuts are implemented for SDDP and SDDP-L only.")
        end
        cutGenerationProgramInfo = ReLULagrangianCutGenerationProgram{Float64}(primal_bound)
    elseif cutType == :NormalizedReLUC
        if param.algorithm != :SDDP && param.algorithm != :SDDPL
            error("Normalized ReLU cuts are implemented for SDDP and SDDP-L only.")
        end

        # Build the normalization vector in the lifted ReLU space: use
        # one-sided boundary perturbations and scale each interior point by
        # `0.5 * x̂ * (1 - x̂ / B)`.
        u_plus  = similar(stateInfo.IntVar, Float64)
        u_minus = similar(stateInfo.IntVar, Float64)
        for g in 1:binaryInfo.d
            x̂ = stateInfo.IntVar[g]
            B = stageData.ū[g]
            if abs(x̂) ≤ RELU_CORE_TOLERANCE
                u_plus[g]  = RELU_CORE_EPS_BOUND
                u_minus[g] = 0.0
            elseif abs(x̂ - B) ≤ RELU_CORE_TOLERANCE
                u_plus[g]  = 0.0
                u_minus[g] = RELU_CORE_EPS_BOUND
            else
                u = RELU_CORE_SCALE_NONBOUND * x̂ * (1.0 - x̂ / B)
                u_plus[g]  = u
                u_minus[g] = u
            end
        end

        u0 = max(
            primal_bound - incumbent_theta + NORMALIZED_RELU_OBJECTIVE_INTERIOR,
            NORMALIZED_RELU_OBJECTIVE_INTERIOR,
        )
        u_leaf = nothing
        if stateInfo.IntVarLeaf !== nothing
            core_state = setup_stage_core_point(
                stateInfo;
                stageData = stageData,
                param = param,
            )
            u_leaf = Dict(
                g => Dict(
                    k => stateInfo.IntVarLeaf[g][k] - core_state.IntVarLeaf[g][k]
                    for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in keys(stateInfo.IntVarLeaf)
            )
        end

        # Keep θ̂_n and Q_n(x̂) separate. The normalized dual objective is
        # anchored at θ̂_n, while Q_n(x̂) is still needed for scaling and
        # validity checks.
        cutGenerationProgramInfo = NormalizedReLULagrangianCutGenerationProgram(
            ReLUDualStageInfo(u0, u_plus, u_minus, u_leaf),
            incumbent_theta,
            primal_bound,
        )

        πₙ_init = setup_normalized_relu_starting_point(
            stateInfo,
            primal_bound,
            incumbent_theta,
            cutGenerationProgramInfo.NormalizationInfo,
        )
    elseif is_pareto_lagrangian_cut(cutType)
        ## state / binary state constraint
        if param.algorithm == :SDDiP
            @constraint(
                model,
                NonAnticipativity,
                model[:Lc] .== stateInfo.IntVarBinaries,
            )
        else
            @constraint(
                model,
                NonAnticipativity,
                model[:Sc] .== stateInfo.IntVar,
            )
        end
        @objective(model, Min, model[:primal_objective_expression])
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            cutGenerationProgramInfo = ParetoLagrangianCutGenerationProgram(
                coreStatePLC,
                objective_value(model),
                param.ε
            )
        else 
            cutGenerationProgramInfo = ParetoLagrangianCutGenerationProgram(
                coreStatePLC,
                primal_bound,
                param.ε
            )
        end 
        delete(model, model[:NonAnticipativity])
        unregister(model, :NonAnticipativity)
    elseif is_square_minimization_cut(cutType)
        ## state / binary state constraint
        if param.algorithm == :SDDiP
            @constraint(
                model,
                NonAnticipativity,
                model[:Lc] .== stateInfo.IntVarBinaries,
            )
        else
            @constraint(
                model,
                NonAnticipativity,
                model[:Sc] .== stateInfo.IntVar,
            )
        end
        @objective(model, Min, model[:primal_objective_expression])
        optimize!(model)
        cutGenerationProgramInfo = SquareMinimizationCutGenerationProgram(
            param.ε,
            objective_value(model)
        )
        delete(model, model[:NonAnticipativity])
        unregister(model, :NonAnticipativity)

        # cutGenerationProgramInfo = SquareMinimizationCutGenerationProgram(
        #     param.ε,
        #     primal_bound
        # )
    elseif cutType == :SBC
        @objective(model, Min, model[:primal_objective_expression])
        if param.algorithm == :SDDP 
            @constraint(
                model, 
                StateNonAnticipativity, 
                model[:Sc] .== stateInfo.IntVar
            );  
            lp_model = relax_integrality(model);
            optimize!(model);
            # obtain the Benders cut coefficients
            coefficientBendersCut = StageInfo(
                -1., 
                nothing, 
                dual.(model[:StateNonAnticipativity]), 
                nothing,
                nothing
            );
            delete(model, model[:StateNonAnticipativity])
            unregister(model, :StateNonAnticipativity)
            lp_model();
        elseif param.algorithm == :SDDPL 
            @constraint(
                model, 
                StateNonAnticipativity, 
                model[:Sc] .== stateInfo.IntVar
            );  
            @constraint(
                model, 
                NonAnticipativity[g in 1:binaryInfo.d, i in keys(stateInfo.IntVarLeaf[g])], 
                model[:region_indicator_copy][g][i] .== stateInfo.IntVarLeaf[g][i]
            );  
            lp_model = relax_integrality(model);
            optimize!(model);
            # obtain the Benders cut coefficients
            coefficientBendersCut = StageInfo(
                -1., 
                nothing, 
                dual.(model[:StateNonAnticipativity]), 
                Dict(
                    g => Dict(
                        k => dual.(model[:NonAnticipativity][g, k]) for k in keys(stateInfo.IntVarLeaf[g])
                        # k => 0.0 for k in keys(stateInfo.IntVarLeaf[g])
                    ) for g in keys(stateInfo.IntVarLeaf)
                ),
                nothing
            );
            delete(model, model[:StateNonAnticipativity])
            for g in 1:binaryInfo.d, i in keys(stateInfo.IntVarLeaf[g])
                delete(model, model[:NonAnticipativity][g, i])
            end
            unregister(model, :StateNonAnticipativity)
            unregister(model, :NonAnticipativity)
            lp_model();

        elseif param.algorithm == :SDDiP 
            @constraint(
                model, 
                StateNonAnticipativity, 
                model[:Lc] .== stateInfo.IntVarBinaries
            );   
            lp_model = relax_integrality(model);
            optimize!(model);
            # obtain the Benders cut coefficients
            coefficientBendersCut = StageInfo(
                -1., 
                nothing, 
                nothing, 
                nothing,
                dual.(model[:StateNonAnticipativity])
            );
            delete(model, model[:StateNonAnticipativity])
            unregister(model, :StateNonAnticipativity)
            lp_model();
        end
        cutGenerationProgramInfo = StrengthenedBendersCutGenerationProgram{Float64}(
            coefficientBendersCut
        )
    elseif cutType == :LNC 
        theta_anchor_result = solve_lnc_state_theta(
            model,
            stateInfo;
            fix_leaf_state = true,
            param = param,
            binaryInfo = binaryInfo,
        )
        theta_anchor = theta_anchor_result === nothing ? primal_bound : theta_anchor_result
        core_theta_info = lnc_core_theta_info(
            model,
            coreStatePLC,
            theta_anchor;
            param = param,
            binaryInfo = binaryInfo,
        )
        uₙ = setup_lnc_normalization_state(
            stateInfo,
            coreStatePLC,
            theta_anchor,
            core_theta_info.core_theta,
        )
        cutGenerationProgramInfo = LinearNormalizationLagrangianCutGenerationProgram(
            uₙ,
            theta_anchor,
            theta_anchor,
            param.lncMinScale,
            core_theta_info.used_fallback,
        )
    end

    level_method_tolerance =
        cutType == :ReLUC || cutType == :NormalizedReLUC ?
        min(param.gapSDDP, 1e-4) :
        max(param.gapSDDP, 1e-4)

    cutGenerationParamInfo = CutGenerationParamInfo(
        0.95,
        0.5,
        level_method_tolerance,
        param.levelMethodMaxIter,
        is_adaptive_level_cut(cutType) ? param.nxt_bound : 1e8,
        param.verbose,
        stateInfo,
        cutType,
        πₙ_init,
    )

    return (
        cutGenerationParamInfo = cutGenerationParamInfo,
        cutGenerationProgramInfo = cutGenerationProgramInfo
    )
end

"""
    backwardPass(backwardNodeInfo)

    function for backward pass in parallel computing
"""
function backwardPass(
    backwardNodeInfo::Tuple, 
    solCollection::Dict{Any, Any}; 
    forwardInfoList::Dict{Int64, StageModel} = forwardInfoList,
    stageDataList::Dict{Int64, StageData} = stageDataList,
    Ω::Dict{Int64, Dict{Int64, RandomVariables}} = Ω,
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::NamedTuple

    (i, t, j, k) = backwardNodeInfo; 
    model_modification!(
        forwardInfoList[t].model,
        Ω[t][j].d;
        param = param,
        binaryInfo = binaryInfo,
    );

    cutType = get_cutType(
        param.cutType,
        i
    )

    node_primal_bound =
        (cutType == :ReLUC || cutType == :NormalizedReLUC) ?
        compute_node_primal_bound(
            forwardInfoList[t].model,
            solCollection[t-1, k];
            param = param,
        ) :
        solCollection[t, k].StateValue

    (cutGenerationParamInfo, cutTypeInfo) = setupCutGenerationInfo(
        forwardInfoList[t].model,
        solCollection[t-1,k], 
        node_primal_bound,
        solCollection[t-1,k].StateValue - solCollection[t-1,k].StageValue,
        cutType;
        stageData = stageDataList[t],
        binaryInfo = binaryInfo,
        param = param
    );

    levelSetResult = LevelSetMethod_optimization!(
        forwardInfoList[t],
        cutGenerationParamInfo,
        cutTypeInfo;
        stageData = stageDataList[t],
        binaryInfo = binaryInfo,
        param = param,
    )
    cutInfo = levelSetResult.cutInfo
    LMiter = levelSetResult.iter
    lnc_fallback = false
    lnc_scale = NaN
    lnc_tightness_gap = NaN
    lnc_core_theta_fallback =
        cutType == :LNC ? cutTypeInfo.core_theta_fallback : false

    if cutType == :LNC
        reference_cut_info_cache = nothing

        function obtain_lc_reference_cut!()
            if reference_cut_info_cache !== nothing
                return reference_cut_info_cache
            end

            regularParamInfo = CutGenerationParamInfo(
                cutGenerationParamInfo.μ,
                cutGenerationParamInfo.λ,
                cutGenerationParamInfo.gapBM,
                cutGenerationParamInfo.iterBM,
                cutGenerationParamInfo.nxt_bound,
                cutGenerationParamInfo.verbose,
                cutGenerationParamInfo.stateInfo,
                :LC,
                cutGenerationParamInfo.πₙ,
            )

            regularLevelSetResult = LevelSetMethod_optimization!(
                forwardInfoList[t],
                regularParamInfo,
                LagrangianCutGenerationProgram{Float64}(0.0);
                stageData = stageDataList[t],
                binaryInfo = binaryInfo,
                param = param,
            )
            LMiter += regularLevelSetResult.iter
            reference_cut_info_cache = regularLevelSetResult.cutInfo
            return reference_cut_info_cache
        end

        lnc_scale = -cutInfo[2].StateValue
        if !isfinite(lnc_scale) || lnc_scale < param.lncMinScale
            cutInfo = obtain_lc_reference_cut!()
            lnc_fallback = true
            lnc_scale = -cutInfo[2].StateValue
        end

        reference_cut_info = obtain_lc_reference_cut!()
        reference_cut_value = lnc_cut_value(
            reference_cut_info,
            cutGenerationParamInfo.stateInfo;
            param = param,
            binaryInfo = binaryInfo,
        )
        candidate_cut_value = lnc_cut_value(
            cutInfo,
            cutGenerationParamInfo.stateInfo,
            param = param,
            binaryInfo = binaryInfo,
        )

        lnc_tightness_gap = abs(candidate_cut_value - reference_cut_value)
        lnc_tightness_tol = lnc_tightness_tolerance(reference_cut_value)
        if !isfinite(lnc_tightness_gap) || lnc_tightness_gap > lnc_tightness_tol
            cutInfo = reference_cut_info
            lnc_fallback = true
            lnc_scale = -cutInfo[2].StateValue
            lnc_tightness_gap = 0.0
            candidate_cut_value = reference_cut_value
        end

        if param.cutDiagnostics
            lnc_epigraph_gap = abs(cutTypeInfo.theta_anchor - candidate_cut_value)
            @info "LNC cut diagnostics" scale=lnc_scale tightness_gap=lnc_tightness_gap tightness_tol=lnc_tightness_tol epigraph_gap=lnc_epigraph_gap fallback=lnc_fallback core_theta_fallback=lnc_core_theta_fallback
        end
    end

    if cutType == :NormalizedReLUC
        scenario_objective = node_primal_bound
        regularCutInfo = nothing

        function obtain_regular_fallback_cut!()::ReLUCutInfo
            if regularCutInfo === nothing
                regularParamInfo = CutGenerationParamInfo(
                    cutGenerationParamInfo.μ,
                    cutGenerationParamInfo.λ,
                    cutGenerationParamInfo.gapBM,
                    cutGenerationParamInfo.iterBM,
                    cutGenerationParamInfo.nxt_bound,
                    cutGenerationParamInfo.verbose,
                    cutGenerationParamInfo.stateInfo,
                    :ReLUC,
                    setup_regular_relu_starting_point(
                        cutGenerationParamInfo.stateInfo,
                        scenario_objective,
                    ),
                )

                regularLevelSetResult = LevelSetMethod_optimization!(
                    forwardInfoList[t],
                    regularParamInfo,
                    ReLULagrangianCutGenerationProgram{Float64}(scenario_objective);
                    stageData = stageDataList[t],
                    binaryInfo = binaryInfo,
                    param = param,
                )
                regularCutInfo = regularLevelSetResult.cutInfo
                LMiter += regularLevelSetResult.iter
            end

            return regularCutInfo
        end

        dualInfo = cutInfo.dualInfo
        has_degenerate_pi0 =
            dualInfo.StateValue === nothing ||
            !isfinite(dualInfo.StateValue) ||
            dualInfo.StateValue ≤ RELU_MIN_NORMALIZED_PI0
        has_zero_dual =
            maximum(abs.(dualInfo.IntVarPlus)) ≤ 1e-8 &&
            maximum(abs.(dualInfo.IntVarMinus)) ≤ 1e-8 &&
            (
                dualInfo.IntVarLeaf === nothing ||
                maximum(
                    maximum(
                        abs(dualInfo.IntVarLeaf[g][k])
                        for k in keys(dualInfo.IntVarLeaf[g])
                    )
                    for g in keys(dualInfo.IntVarLeaf)
                ) ≤ 1e-8
            )

        if has_degenerate_pi0 || has_zero_dual
            regularCutInfo = obtain_regular_fallback_cut!()
            cutInfo = ReLUCutInfo(
                regularCutInfo.rhs,
                ReLUDualStageInfo(
                    1.0,
                    regularCutInfo.dualInfo.IntVarPlus,
                    regularCutInfo.dualInfo.IntVarMinus,
                    regularCutInfo.dualInfo.IntVarLeaf,
                ),
            )
        end

        local_validity_tolerance = max(
            RELU_LOCAL_VALIDITY_TOLERANCE,
            1e-8 * max(abs(node_primal_bound), 1.0),
        )
        local_tightness_gap = abs(
            cutInfo.rhs +
            relu_lifted_leaf_dot(cutInfo.dualInfo, cutGenerationParamInfo.stateInfo) -
            cutInfo.dualInfo.StateValue * node_primal_bound
        )
        if local_tightness_gap > local_validity_tolerance
            regularCutInfo = obtain_regular_fallback_cut!()

            cutInfo = ReLUCutInfo(
                regularCutInfo.rhs,
                ReLUDualStageInfo(
                    1.0,
                    regularCutInfo.dualInfo.IntVarPlus,
                    regularCutInfo.dualInfo.IntVarMinus,
                    regularCutInfo.dualInfo.IntVarLeaf,
                ),
            )
        end
    end

    if cutType == :ReLUC || cutType == :NormalizedReLUC
        return (
            rhs = cutInfo.rhs,
            dualInfo = cutInfo.dualInfo,
            iter = LMiter,
            lnc_scale = NaN,
            lnc_tightness_gap = NaN,
            lnc_fallback = false,
            lnc_core_theta_fallback = false,
        )
    else
        (λ₀, λ₁) = cutInfo
        return (
            rhs = λ₀,
            dualInfo = λ₁,
            iter = LMiter,
            lnc_scale = lnc_scale,
            lnc_tightness_gap = lnc_tightness_gap,
            lnc_fallback = lnc_fallback,
            lnc_core_theta_fallback = lnc_core_theta_fallback,
        )
    end
end
