export backwardPass
"""
RemoveContVarNonAnticipative!(model::Model)

# Arguments

    1. `model::Model` : a nodal problem
  
# Modification
    1. Remove the Non-anticipativity constraints
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
    setup_initial_point(stateInfo::StateInfo)
# Arguments
    stateInfo::StateInfo : the parent's node decisions
    function for setting up the initial dual variables
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
    setup_Benders_warm_start(stateInfo::StateInfo)
# Arguments
    stateInfo::StateInfo : the parent's node decisions
    function for setting up the initial dual variables
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
    backwardPass(backwardNodeInfo)

    function for backward pass in parallel computing
"""
function backwardPass(
    backwardNodeInfo::Tuple; 
    ModelList::Dict{Int64, SDDPModel} = ModelList,
    indexSets::IndexSets = indexSets, 
    paramDemand::ParamDemand = paramDemand, 
    paramOPF::ParamOPF = paramOPF,
    scenarioTree::ScenarioTree = scenarioTree, 
    stateInfoCollection::Dict{Any, Any} = stateInfoCollection,
    param::NamedTuple = param, param_cut::NamedTuple = param_cut, param_levelsetmethod::NamedTuple = param_levelsetmethod
)

    (i, t, n, ω, cutSelection, core_point_strategy) = backwardNodeInfo; 
    ModelModification!( 
        ModelList[t].model, 
        scenarioTree.tree[t].nodes[n],
        paramDemand,
        stateInfoCollection[i, t-1, ω];
        indexSets = indexSets
    );

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
            param_levelsetmethod = param_levelsetmethod
        );
    else 
        levelsetmethodOracleParam = SetupLevelSetMethodOracleParam(
            stateInfoCollection[i, t-1, ω];
            indexSets = indexSets,
            param_levelsetmethod = param_levelsetmethod
        );
    end

    RemoveContVarNonAnticipative!(
        ModelList[t].model;
        indexSets = indexSets,
        param = param
    );

    if cutSelection == :PLC
        CutGenerationInfo = ParetoLagrangianCutGeneration{Float64}(
            core_point_strategy, 
            setup_core_point(
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
    elseif cutSelection == :SMC
        CutGenerationInfo = SquareMinimizationCutGeneration{Float64}(
            param_cut.δ, 
            stateInfoCollection[i, t, ω].StateValue
        );
    elseif cutSelection == :SBC
        CutGenerationInfo = StrengthenedBendersCutGeneration{Float64}();
    elseif cutSelection == :LNC
        ModelModification!( 
            ModelList[t].model, 
            scenarioTree.tree[t].nodes[n],
            paramDemand,
            stateInfoCollection[i, t-1, ω];
            indexSets = indexSets,
            param = param
        );
        optimize!(ModelList[t].model);

        CutGenerationInfo = LinearNormalizationLagrangianCutGeneration{Float64}(
            setup_core_point(
                stateInfoCollection[i, t-1, ω];
                indexSets = indexSets,
                paramOPF = paramOPF, 
                param_cut = param_cut,
                param = param
            ),
            1.0,
            JuMP.objective_value(ModelList[t].model)
        );
        RemoveContVarNonAnticipative!(
            ModelList[t].model;
            indexSets = indexSets,
            param = param
        );

        # CutGenerationInfo = LinearNormalizationLagrangianCutGeneration{Float64}(
        #     setup_core_point(
        #         stateInfoCollection[i, t-1, ω];
        #         indexSets = indexSets,
        #         paramOPF = paramOPF, 
        #         param_cut = param_cut,
        #         param = param
        #     ),
        #     1.0,
        #     stateInfoCollection[i, t, ω].StateValue
        # );
    end

    ((λ₀, λ₁, πₙ₀), LMiter) = LevelSetMethod_optimization!(
        ModelList[t].model, 
        levelsetmethodOracleParam, 
        stateInfoCollection[i, t-1, ω],
        CutGenerationInfo;
        indexSets = indexSets, 
        paramDemand = paramDemand, 
        paramOPF = paramOPF, 
        param = param, param_levelsetmethod = param_levelsetmethod
    );
                                                    
    return ((λ₀, λ₁, πₙ₀), LMiter)  
end
