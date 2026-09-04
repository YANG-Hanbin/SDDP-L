export backwardPass
"""
    RemoveContVarNonAnticipative!(
        model::Model = model;
        indexSets::IndexSets = indexSets,
        param::NamedTuple = param,
    )::Nothing

Remove the parent-state non-anticipativity constraints from a backward node
model so that the node problem can be reused as a cut-generation oracle.
"""
function RemoveContVarNonAnticipative!(
    model::Model = model;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param
)::Nothing
    if :λ_copy ∉ keys(model.obj_dict)
        # For SDDP-L or SDDP algorithms
        for g in indexSets.G
            delete(model, model[:ContVarNonAnticipative][g]);
            delete(model, model[:BinVarNonAnticipative_y][g]);
            delete(model, model[:BinVarNonAnticipative_v][g]);
            delete(model, model[:BinVarNonAnticipative_w][g]);
        end
        unregister(model, :ContVarNonAnticipative);
        unregister(model, :BinVarNonAnticipative_y);
        unregister(model, :BinVarNonAnticipative_v);
        unregister(model, :BinVarNonAnticipative_w);
    else
        # For SDDiP algorithm
        for g in indexSets.G
            delete(model, model[:BinVarNonAnticipative_y][g]);
            delete(model, model[:BinVarNonAnticipative_v][g]);
            delete(model, model[:BinVarNonAnticipative_w][g]);
            for i in 1:param.κ[g]
                delete(model, model[:BinarizationNonAnticipative][g, i]);
            end
        end
        unregister(model, :BinVarNonAnticipative_y);
        unregister(model, :BinVarNonAnticipative_v);
        unregister(model, :BinVarNonAnticipative_w);
        unregister(model, :BinarizationNonAnticipative);
    end
    
    return
end

"""
    setup_initial_point(
        stateInfo::StateInfo;
        indexSets::IndexSets = indexSets,
        param::NamedTuple = param,
    )::StateInfo

Construct the zero vector used to initialize the classical level-set dual for
LC/PLC/SMC/LNC-style cut generation.
"""
function setup_initial_point(
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param
)::StateInfo
    BinVar = Dict{Any, Dict{Any, Any}}(
        :y => Dict{Any, Any}(
            g => 0.0 for g in indexSets.G
        ),
        :v => Dict{Any, Any}(
            g => 0.0 for g in indexSets.G
        ),
        :w => Dict{Any, Any}(
            g => 0.0 for g in indexSets.G
        ),
    );
    ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
        g => 0.0 for g in indexSets.G)
    );
    if stateInfo.ContAugState == nothing 
        ContAugState = nothing
    else
        ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    k => 0.0 for k in keys(stateInfo.ContAugState[:s][g])
                ) for g in indexSets.G
            )
        );
    end

    if stateInfo.ContStateBin == nothing 
        ContStateBin = nothing
    else
        ContStateBin = Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    i => 0.0 for i in 1:param.κ[g]
                ) for g in indexSets.G
            )
        );
    end

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
        ContStateBin
    );
end

"""
    setup_Benders_warm_start(
        model::Model,
        stateInfo::StateInfo;
        indexSets::IndexSets = indexSets,
        param::NamedTuple = param,
    )::StateInfo

Extract LP dual multipliers from the node model and use them as a warm start
for the strengthened Benders cut generation path.
"""
function setup_Benders_warm_start(
    model::Model,
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param
)::StateInfo
    lp_model = relax_integrality(model);
    optimize!(model);

    BinVar = Dict{Any, Dict{Any, Any}}(
        :y => Dict{Any, Any}(
            g => dual(model[:BinVarNonAnticipative_y][g]) for g in indexSets.G
        ),
        :v => Dict{Any, Any}(
            g => dual(model[:BinVarNonAnticipative_v][g]) for g in indexSets.G
        ),
        :w => Dict{Any, Any}(
            g => dual(model[:BinVarNonAnticipative_w][g]) for g in indexSets.G
        ),
    );
    if :ContVarNonAnticipative ∈ keys(model.obj_dict)
        ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
            g => dual(model[:ContVarNonAnticipative][g]) for g in indexSets.G)
        );
    else
        ContVar = nothing
    end
    if stateInfo.ContAugState == nothing 
        ContAugState = nothing
    else
        ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    k => 0.0 for k in keys(stateInfo.ContAugState[:s][g])
                ) for g in indexSets.G
            )
        );
    end

    if stateInfo.ContStateBin == nothing 
        ContStateBin = nothing
    else
        ContStateBin = Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    i => dual(model[:BinarizationNonAnticipative][g, i]) for i in 1:param.κ[g]
                ) for g in indexSets.G
            )
        );
    end

    lp_model()
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
        ContStateBin
    );
end

"""
    compute_node_primal_bound(model::Model)::Float64

Solve the current backward node primal problem with the parent-state
non-anticipativity constraints still enforced. The returned objective value is
the node-specific cost-to-go `Q_n(x̂)` required by the ReLU-based cut
generation routines.
"""
function compute_node_primal_bound(
    model::Model,
    stateInfo::Union{Nothing, StateInfo} = nothing;
    indexSets::IndexSets = indexSets,
)::Float64
    local aug_fix = nothing

    try
        if stateInfo !== nothing && stateInfo.ContAugState !== nothing
            aug_fix = Dict(
                g => Dict(
                    k => @constraint(
                        model,
                        model[:augmentVar_copy][g, k] == stateInfo.ContAugState[:s][g][k],
                    )
                    for k in keys(stateInfo.ContAugState[:s][g])
                )
                for g in indexSets.G
            )
        end

        @objective(
            model,
            Min,
            model[:primal_objective_expression]
        );
        optimize!(model);

        status = termination_status(model);
        if status != MOI.OPTIMAL && status != MOI.LOCALLY_SOLVED
            error("Failed to solve the node-specific primal problem (status = $status).")
        end

        return JuMP.objective_value(model)
    finally
        if aug_fix !== nothing
            for g in keys(aug_fix), k in keys(aug_fix[g])
                delete(model, aug_fix[g][k])
            end
        end
    end
end

"""
Return the minimum positive master-cut scale accepted for LNC.
"""
function lnc_min_scale(param::NamedTuple)::Float64
    scale = Float64(get(param, :lncMinScale, 1e-3))
    scale > 0.0 || error("lncMinScale must be positive. Got $(scale).")
    return scale
end

"""
Return the epigraph margin used to place the LNC core point strictly inside
the relaxed node epigraph.
"""
function lnc_theta_margin(theta_anchor::Real, param::NamedTuple)::Float64
    margin = Float64(get(param, :lncCoreThetaMargin, 1e-4))
    margin > 0.0 || error("lncCoreThetaMargin must be positive. Got $(margin).")
    return max(margin * max(abs(float(theta_anchor)), 1.0), margin)
end

function has_lnc_state_fix(model::Model)::Bool
    return :BinVarNonAnticipative_y ∈ keys(model.obj_dict) ||
           :BinarizationNonAnticipative ∈ keys(model.obj_dict)
end

"""
Solve the node value at a fixed parent state for LNC normalization.

The incumbent anchor is evaluated in the original node MIP. Fractional core
states are evaluated with integrality relaxed, because MSUC transition
constraints can make fractional parent copy states infeasible when current
commitment/startup/shutdown variables remain binary.
"""
function solve_lnc_state_theta(
    model::Model,
    randomVariables::RandomVariables,
    paramDemand::ParamDemand,
    targetState::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
    relax_integrality_for_state::Bool = false,
)::Union{Float64, Nothing}
    undo_integrality_relaxation = nothing
    try
        ModelModification!(
            model,
            randomVariables,
            paramDemand,
            targetState;
            indexSets = indexSets,
            param = param,
        )
        if relax_integrality_for_state
            # LNC core points are usually fractional in the parent copy-state
            # space. Evaluating them in the original MSUC node MIP can be
            # infeasible because transition equations couple fractional
            # copy states with binary current-stage decisions. The relaxed
            # node gives the convexified epigraph value used for normalization.
            undo_integrality_relaxation = relax_integrality(model)
        end
        @objective(model, Min, model[:primal_objective_expression])
        optimize!(model)

        status = termination_status(model)
        if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
            value = JuMP.objective_value(model)
            return isfinite(value) ? value : nothing
        end
        return nothing
    finally
        if undo_integrality_relaxation !== nothing
            undo_integrality_relaxation()
        end
        if has_lnc_state_fix(model)
            RemoveContVarNonAnticipative!(
                model;
                indexSets = indexSets,
                param = param,
            )
        end
    end
end

"""
Compute the LNC core epigraph coordinate from the relaxed core-state node.

If the relaxed solve fails or produces a point too close to the incumbent
anchor, use a strictly interior anchor-based fallback and expose that decision
through diagnostics.
"""
function lnc_core_theta_info(
    model::Model,
    randomVariables::RandomVariables,
    paramDemand::ParamDemand,
    coreState::StateInfo,
    theta_anchor::Float64;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::NamedTuple
    core_value = solve_lnc_state_theta(
        model,
        randomVariables,
        paramDemand,
        coreState;
        indexSets = indexSets,
        param = param,
        relax_integrality_for_state = true,
    )

    margin = lnc_theta_margin(theta_anchor, param)
    core_theta = core_value === nothing || !isfinite(core_value) ?
        theta_anchor + margin :
        Float64(core_value) + margin

    used_fallback =
        core_value === nothing ||
        !isfinite(core_value) ||
        abs(theta_anchor - core_theta) < margin

    if used_fallback
        core_theta = theta_anchor + margin
    end

    return (
        core_theta = core_theta,
        used_fallback = used_fallback,
    )
end

function lnc_cut_rhs_at_state(
    λ₀::Real,
    λ₁::StateInfo,
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::Float64
    rhs = float(λ₀)
    for g in indexSets.G
        rhs += param.algorithm == :SDDiP ?
            sum(
                λ₁.ContStateBin[:s][g][i] * stateInfo.ContStateBin[:s][g][i]
                for i in 1:param.κ[g];
                init = 0.0,
            ) :
            λ₁.ContVar[:s][g] * stateInfo.ContVar[:s][g]
        rhs += λ₁.BinVar[:y][g] * stateInfo.BinVar[:y][g]
        rhs += λ₁.BinVar[:v][g] * stateInfo.BinVar[:v][g]
        rhs += λ₁.BinVar[:w][g] * stateInfo.BinVar[:w][g]
        if param.algorithm == :SDDPL
            rhs += sum(
                λ₁.ContAugState[:s][g][k] * stateInfo.ContAugState[:s][g][k]
                for k in keys(stateInfo.ContAugState[:s][g]);
                init = 0.0,
            )
        end
    end
    return rhs
end

"""
Evaluate a normalized LNC/LC cut at a concrete parent state.
"""
function lnc_cut_value_at_state(
    cutInfo,
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::Float64
    λ₀, λ₁, πₙ₀ = cutInfo
    if !isfinite(πₙ₀) || πₙ₀ <= 0.0
        return Inf
    end
    return lnc_cut_rhs_at_state(
        λ₀,
        λ₁,
        stateInfo;
        indexSets = indexSets,
        param = param,
    ) / πₙ₀
end

"""
    extract_incumbent_theta(
        thetaValueCollection::Dict{Any, Any},
        scenarioTree::ScenarioTree,
        i::Int,
        t::Int,
        n::Int,
        ω::Int,
    )::Float64

Return the incumbent value of the child epigraph variable corresponding to
node `n`, scaled back from the stored weighted variable `θ[n]` to the
unweighted quantity required by the normalized ReLU theory.
"""
function extract_incumbent_theta(
    thetaValueCollection::Dict{Any, Any},
    scenarioTree::ScenarioTree,
    i::Int,
    t::Int,
    n::Int,
    ω::Int,
)::Float64
    node_probability = scenarioTree.tree[t-1].prob[n];
    if node_probability ≤ 0.0
        error("Encountered a non-positive child-node probability for node $n at stage $(t-1).")
    end

    return thetaValueCollection[i, t-1, ω][n] / node_probability
end

"""
    backwardPass(
        backwardNodeInfo::Tuple;
        ModelList::Dict{Int64, SDDPModel} = ModelList,
        indexSets::IndexSets = indexSets,
        paramDemand::ParamDemand = paramDemand,
        paramOPF::ParamOPF = paramOPF,
        scenarioTree::ScenarioTree = scenarioTree,
        stateInfoCollection::Dict{Any, Any} = stateInfoCollection,
        thetaValueCollection::Dict{Any, Any} = thetaValueCollection,
        param::NamedTuple = param,
        param_cut::NamedTuple = param_cut,
        param_levelsetmethod::NamedTuple = param_levelsetmethod,
    )

Solve one backward node, generate the requested cut, and return the cut
coefficients together with the number of level-method iterations used.
"""
function backwardPass(
    backwardNodeInfo::Tuple; 
    ModelList::Dict{Int64, SDDPModel} = ModelList,
    indexSets::IndexSets = indexSets, 
    paramDemand::ParamDemand = paramDemand, 
    paramOPF::ParamOPF = paramOPF,
    scenarioTree::ScenarioTree = scenarioTree, 
    stateInfoCollection::Dict{Any, Any} = stateInfoCollection,
    thetaValueCollection::Dict{Any, Any} = thetaValueCollection,
    param::NamedTuple = param, param_cut::NamedTuple = param_cut, param_levelsetmethod::NamedTuple = param_levelsetmethod
)

    (i, t, n, ω, cutSelection) = backwardNodeInfo;
    parentStateInfo = stateInfoCollection[i, t-1, ω]
    if param.algorithm == :SDDPL &&
       (cutSelection == :ReLUC || cutSelection == :NormalizedReLUC)
        sync_cont_aug_state!(
            parentStateInfo,
            ModelList[t-1];
            indexSets = indexSets,
            param = param,
        )
    end

    ModelModification!( 
        ModelList[t].model, 
        scenarioTree.tree[t].nodes[n],
        paramDemand,
        parentStateInfo;
        indexSets = indexSets,
        param = param
    );

    node_primal_bound = nothing;
    incumbent_theta = nothing;
    normalizationInfo = nothing;

    if cutSelection == :ReLUC || cutSelection == :NormalizedReLUC
        # ReLU-based cut generation is scaled by the current node-specific
        # primal value Q_n(x̂), not by the sampled forward-path value stored in
        # `stateInfoCollection[i, t, ω].StateValue`. Solving the node primal
        # problem here also provides a stable incumbent for the subsequent
        # lifted ReLU oracle.
        node_primal_bound = compute_node_primal_bound(
            ModelList[t].model,
            parentStateInfo;
            indexSets = indexSets,
        );
    end

    if cutSelection == :NormalizedReLUC
        # The normalized ReLU dual additionally requires the incumbent child
        # epigraph value θ̂_n. It is intentionally kept separate from
        # Q_n(x̂) because the two quantities enter different parts of the
        # normalized dual theory.
        incumbent_theta = extract_incumbent_theta(
            thetaValueCollection,
            scenarioTree,
            i,
            t,
            n,
            ω,
        );
        normalizationInfo = setup_NormalizedReLU_core_point(
            parentStateInfo;
            indexSets = indexSets,
            paramOPF = paramOPF,
            primal_bound = node_primal_bound,
            incumbent_theta = incumbent_theta,
        );
    end

    if cutSelection == :SBC
        initial_point = setup_Benders_warm_start(
            ModelList[t].model,
            stateInfoCollection[i, t-1, ω];
            indexSets = indexSets,
            param = param
        );
        levelsetmethodOracleParam = SetupLevelSetMethodOracleParam(
            initial_point;
            indexSets = indexSets,
            param = param,
            param_levelsetmethod = param_levelsetmethod
        );
    elseif cutSelection == :ReLUC
        levelsetmethodOracleParam = SetupLevelSetMethodOracleParam(
            setup_ReLU_initial_point(
                parentStateInfo;
                indexSets = indexSets,
            );
            indexSets = indexSets,
            param = param,
            param_levelsetmethod = param_levelsetmethod
        );
    elseif cutSelection == :NormalizedReLUC
        levelsetmethodOracleParam = SetupLevelSetMethodOracleParam(
            setup_NormalizedReLU_initial_point(
                parentStateInfo,
                node_primal_bound,
                incumbent_theta,
                normalizationInfo;
                indexSets = indexSets,
            );
            indexSets = indexSets,
            param = param,
            param_levelsetmethod = param_levelsetmethod
        );
    else 
        levelsetmethodOracleParam = SetupLevelSetMethodOracleParam(
            stateInfoCollection[i, t-1, ω];
            indexSets = indexSets,
            param = param,
            param_levelsetmethod = param_levelsetmethod
        );
    end

    RemoveContVarNonAnticipative!(
        ModelList[t].model;
        indexSets = indexSets,
        param = param
    );

    lnc_anchor_value = NaN

    if cutSelection == :PLC
        CutGenerationInfo = ParetoLagrangianCutGeneration{Float64}(
            param_cut.core_point_strategy,
            setup_PLC_core_point(
                stateInfoCollection[i, t-1, ω];
                indexSets = indexSets,
                paramOPF = paramOPF, 
                param_cut = param_cut,
                param = param
            ), 
            param_cut.δ, 
            stateInfoCollection[i, t, ω].StateValue
        );
    elseif cutSelection == :LC
        CutGenerationInfo = LagrangianCutGeneration{Float64}(
            stateInfoCollection[i, t, ω].StateValue
        );
    elseif cutSelection == :ReLUC
        if param.algorithm != :SDDP && param.algorithm != :SDDPL
            error("ReLU Lagrangian cuts are implemented for SDDP and SDDP-L only.")
        end
        CutGenerationInfo = ReLULagrangianCutGeneration{Float64}(
            node_primal_bound
        );
    elseif cutSelection == :NormalizedReLUC
        if param.algorithm != :SDDP && param.algorithm != :SDDPL
            error("Normalized ReLU cuts are implemented for SDDP and SDDP-L only.")
        end
        CutGenerationInfo = NormalizedReLULagrangianCutGeneration{Float64}(
            normalizationInfo,
            incumbent_theta,
            node_primal_bound,
        );
    elseif cutSelection == :SMC
        CutGenerationInfo = SquareMinimizationCutGeneration{Float64}(
            param_cut.δ, 
            stateInfoCollection[i, t, ω].StateValue
        );
    elseif cutSelection == :SBC
        CutGenerationInfo = StrengthenedBendersCutGeneration{Float64}();
    elseif cutSelection == :LNC
        stateInfo = stateInfoCollection[i, t-1, ω]
        randomVariables = scenarioTree.tree[t].nodes[n]
        theta_anchor = solve_lnc_state_theta(
            ModelList[t].model,
            randomVariables,
            paramDemand,
            stateInfo;
            indexSets = indexSets,
            param = param,
        )
        theta_anchor === nothing && error(
            "Failed to solve the LNC incumbent epigraph value at " *
            "iteration $i, stage $t, node $n, sample $ω.",
        )
        lnc_anchor_value = Float64(theta_anchor)

        coreState = setup_PLC_core_point(
            stateInfo;
            indexSets = indexSets,
            paramOPF = paramOPF,
            param_cut = param_cut,
            param = param,
        )
        core_theta_info = lnc_core_theta_info(
            ModelList[t].model,
            randomVariables,
            paramDemand,
            coreState,
            lnc_anchor_value;
            indexSets = indexSets,
            param = param,
        )
        normalizationState = _state_minus_core(
            stateInfo,
            coreState;
            indexSets = indexSets,
            param = param,
        )

        CutGenerationInfo = LinearNormalizationLagrangianCutGeneration{Float64}(
            normalizationState,
            lnc_anchor_value - core_theta_info.core_theta,
            lnc_anchor_value,
            lnc_anchor_value,
            core_theta_info.core_theta,
            lnc_min_scale(param),
            core_theta_info.used_fallback,
        )
    end

    function solve_strengthened_benders_fallback!()
        fallback_initial_point = setup_Benders_warm_start(
            ModelList[t].model,
            stateInfoCollection[i, t-1, ω];
            indexSets = indexSets,
            param = param,
        )
        fallback_param = SetupLevelSetMethodOracleParam(
            fallback_initial_point;
            indexSets = indexSets,
            param = param,
            param_levelsetmethod = param_levelsetmethod,
        )
        return LevelSetMethod_optimization!(
            ModelList[t].model,
            fallback_param,
            stateInfoCollection[i, t-1, ω],
            StrengthenedBendersCutGeneration{Float64}();
            indexSets = indexSets,
            paramDemand = paramDemand,
            paramOPF = paramOPF,
            param = param,
            param_levelsetmethod = param_levelsetmethod,
        )
    end

    local cut_generation_result
    try
        cut_generation_result = LevelSetMethod_optimization!(
            ModelList[t].model,
            levelsetmethodOracleParam,
            stateInfoCollection[i, t-1, ω],
            CutGenerationInfo;
            indexSets = indexSets,
            paramDemand = paramDemand,
            paramOPF = paramOPF,
            param = param,
            param_levelsetmethod = param_levelsetmethod,
        )
    catch err
        if err isa ReLUInnerOracleSolveError &&
           (cutSelection == :ReLUC || cutSelection == :NormalizedReLUC)
            cut_generation_result = solve_strengthened_benders_fallback!()
        else
            rethrow(err)
        end
    end

    if cutSelection == :ReLUC || cutSelection == :NormalizedReLUC
        if cut_generation_result isa NamedTuple
            (λ₀, λ₁, πₙ₀) = cut_generation_result.cutInfo
            LMiter = cut_generation_result.iter
        else
            ((λ₀, λ₁, πₙ₀), LMiter) = cut_generation_result
        end
    else
        (λ₀, λ₁, πₙ₀) = cut_generation_result.cutInfo
        LMiter = cut_generation_result.iter
    end
    lnc_diagnostics = (
        scale = NaN,
        tightness_gap = NaN,
        fallback = false,
        core_theta_fallback = false,
    )

    if cutSelection == :LNC
        scale = πₙ₀
        if !isfinite(scale) || scale < CutGenerationInfo.min_scale
            error(
                "LNC failed to produce a valid positive scale at iteration $i, " *
                "stage $t, node $n, sample $ω. " *
                "scale = $(scale), min_scale = $(CutGenerationInfo.min_scale)."
            )
        end

        selected_cut_value = lnc_cut_value_at_state(
            (λ₀, λ₁, πₙ₀),
            stateInfoCollection[i, t-1, ω];
            indexSets = indexSets,
            param = param,
        )
        tightness_gap = isfinite(selected_cut_value) ?
            abs(CutGenerationInfo.theta_anchor - selected_cut_value) :
            Inf
        lnc_diagnostics = (
            scale = scale,
            tightness_gap = tightness_gap,
            fallback = false,
            core_theta_fallback = CutGenerationInfo.core_theta_fallback,
        )

        if get(param, :cutDiagnostics, false)
            @info "LNC cut diagnostics" stage=t node=n sample=ω scale=scale tightness_gap=tightness_gap fallback=false core_theta_fallback=CutGenerationInfo.core_theta_fallback theta_anchor=CutGenerationInfo.theta_anchor core_theta=CutGenerationInfo.core_theta
        end
    end

    if cutSelection == :NormalizedReLUC
        regularCutInfo = nothing;

        function obtain_regular_fallback_cut!()
            if regularCutInfo === nothing
                regularOracleParam = SetupLevelSetMethodOracleParam(
                    setup_ReLU_initial_point(
                        stateInfoCollection[i, t-1, ω];
                        indexSets = indexSets,
                    );
                    indexSets = indexSets,
                    param = param,
                    param_levelsetmethod = param_levelsetmethod,
                );

                regular_result = LevelSetMethod_optimization!(
                    ModelList[t].model,
                    regularOracleParam,
                    stateInfoCollection[i, t-1, ω],
                    ReLULagrangianCutGeneration{Float64}(node_primal_bound);
                    indexSets = indexSets,
                    paramDemand = paramDemand,
                    paramOPF = paramOPF,
                    param = param,
                    param_levelsetmethod = param_levelsetmethod,
                );
                if regular_result isa NamedTuple
                    (rhs_regular, dual_regular, π0_regular) = regular_result.cutInfo
                    LMiter_regular = regular_result.iter
                else
                    ((rhs_regular, dual_regular, π0_regular), LMiter_regular) = regular_result
                end

                regularCutInfo = (rhs_regular, dual_regular, π0_regular, LMiter_regular);
            end

            return regularCutInfo
        end

        dualInfo = λ₁;
        dual_magnitude = maximum([
            maximum(abs, values(dualInfo.ContVarPlus[:s])),
            maximum(abs, values(dualInfo.ContVarMinus[:s])),
            maximum(abs, values(dualInfo.BinVarPlus[:y])),
            maximum(abs, values(dualInfo.BinVarMinus[:y])),
            maximum(abs, values(dualInfo.BinVarPlus[:v])),
            maximum(abs, values(dualInfo.BinVarMinus[:v])),
            maximum(abs, values(dualInfo.BinVarPlus[:w])),
            maximum(abs, values(dualInfo.BinVarMinus[:w])),
            dualInfo.ContAugState === nothing ?
                0.0 :
                maximum(
                    maximum(
                        abs(dualInfo.ContAugState[:s][g][k])
                        for k in keys(dualInfo.ContAugState[:s][g])
                    )
                    for g in keys(dualInfo.ContAugState[:s])
                ),
        ]);

        has_degenerate_pi0 =
            dualInfo.StateValue === nothing ||
            !isfinite(dualInfo.StateValue) ||
            dualInfo.StateValue ≤ RELU_MIN_NORMALIZED_PI0;
        has_zero_dual = dual_magnitude ≤ 1e-8;

        local_validity_tolerance = max(
            RELU_LOCAL_VALIDITY_TOLERANCE,
            1e-8 * max(abs(node_primal_bound), 1.0),
        );
        local_tightness_gap = has_degenerate_pi0 ?
            Inf :
            abs(
                λ₀ +
                relu_aug_state_dot(dualInfo, stateInfoCollection[i, t-1, ω]) -
                dualInfo.StateValue * node_primal_bound
            );

        # If the normalized solution degenerates or is not locally tight at the
        # incumbent, fall back to the regular ReLU cut. The fallback preserves
        # validity while avoiding a weak normalized cut in the SDDP master.
        if has_degenerate_pi0 || has_zero_dual || local_tightness_gap > local_validity_tolerance
            (λ₀, λ₁, πₙ₀, LMiter_regular) = obtain_regular_fallback_cut!();
            LMiter += LMiter_regular;
        end
    end

    return ((λ₀, λ₁, πₙ₀), LMiter, lnc_diagnostics)
end
