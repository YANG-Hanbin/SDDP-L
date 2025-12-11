export forwardModel!, forwardPass
"""
    function forwardModel!( 
        paramDemand::ParamDemand, 
        paramOPF::ParamOPF, 
        stageRealization::StageRealization;
        indexSets::IndexSets = indexSets, 
        para::NamedTuple = para
    )
# Arguments

    1. `indexSets::IndexSets` : index sets for the power network
    2. `paramDemand::ParamDemand` : demand parameters
    3. `paramOPF::ParamOPF` : OPF parameters
    4. `stageRealization::StageRealization` : realization of the stage
  
# Returns
    1. `SDDPModel`
"""
function forwardModel!( 
    paramDemand::ParamDemand, 
    paramOPF::ParamOPF, 
    stageRealization::StageRealization;
    indexSets::IndexSets = indexSets, 
    param::NamedTuple = param
)::SDDPModel
    ## build the forward model
    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV))); 
    MOI.set(model, MOI.Silent(), !param.verbose);
    set_optimizer_attribute(model, "MIPGap", param.MIPGap);
    set_optimizer_attribute(model, "Threads", 1);
    set_optimizer_attribute(model, "TimeLimit", param.TimeLimit);
                
    ## define variables
    @variable(model, θ_angle[indexSets.B]);                                                ## phase angle of the bus i
    @variable(model, P[indexSets.L]);                                                      ## real power flow on line l; elements in L is Tuple (i, j)
    @variable(model, 0 ≤ s[g in indexSets.G] ≤ paramOPF.smax[g]);                          ## real power generation at generator g
    @variable(model, 0 ≤ x[indexSets.D] ≤ 1);                                              ## load shedding

    @variable(model, y[indexSets.G], Bin);                                                 ## binary variable for generator commitment status
    @variable(model, v[indexSets.G], Bin);                                                 ## binary variable for generator startup decision
    @variable(model, w[indexSets.G], Bin);                                                 ## binary variable for generator shutdown decision

    @variable(model, h[indexSets.G] ≥ 0);                                                  ## production cost at generator g

    @variable(model, θ[keys(stageRealization.prob)] ≥ param.θ̲);                            ## auxiliary variable for approximation of the value function

    if param.algorithm == :SDDPL
        ## augmented variables
        # define the augmented variables for cont. variables
        augmentVar = Dict(
            (g, k) => @variable(
                model, 
                base_name = "augmentVar[$g, $k]", 
                binary = true
            ) for g in indexSets.G for k in 1:1
        );
        model[:augmentVar] = augmentVar;
        
        ## define copy variables
        @variable(model, 0 ≤ s_copy[g in indexSets.G] ≤ paramOPF.smax[g]);
        if param.tightness
            @variable(model, y_copy[indexSets.G], Bin);        
            @variable(model, v_copy[indexSets.G], Bin);        
            @variable(model, w_copy[indexSets.G], Bin);        
            augmentVar_copy = Dict(
                (g, k) => @variable(
                    model, 
                    base_name = "augmentVar_copy[$g, $k]", 
                    binary = true
                ) for g in indexSets.G for k in 1:1
            );
            model[:augmentVar_copy] = augmentVar_copy;     
        else
            @variable(model, 0 ≤ y_copy[indexSets.G] ≤ 1); 
            @variable(model, 0 ≤ v_copy[indexSets.G] ≤ 1);  
            @variable(model, 0 ≤ w_copy[indexSets.G] ≤ 1);  
            augmentVar_copy = Dict(
                (g, k) => @variable(
                    model, 
                    base_name = "augmentVar_copy[$g, $k]", 
                    lower_bound = 0, 
                    upper_bound = 1
                ) for g in indexSets.G for k in 1:1
            );
            model[:augmentVar_copy] = augmentVar_copy;     
        end

        # constraints for augmented variables: Choosing one leaf node
        @constraint(model, [g in indexSets.G, k in [1]], augmentVar[g, k] == 1);
        @constraint(model, [g in indexSets.G, k in [1]], augmentVar_copy[g, k] == 1);
        ContVarLeaf = Dict(
            :s => Dict{Any, Dict{Any, Dict{Symbol, Any}}}(
                g => Dict(
                    k => Dict(
                        :lb => 0.0, 
                        :ub => paramOPF.smax[g], 
                        :parent => nothing, 
                        :sibling => nothing, 
                        :var => augmentVar[g,1]
                    ) for k in 1:1
                ) for g in indexSets.G
            )
        );
        ContVarBinaries = nothing;

        @constraint(
            model,
            partition_lower_bound[g in indexSets.G],
            s[g] ≥ sum(
                ContVarLeaf[:s][g][k][:lb] *
                augmentVar[g, k]
                for k in keys(ContVarLeaf[:s][g])
            )
        )
        @constraint(
            model,
            partition_upper_bound[g in indexSets.G],
            s[g] ≤ sum(
                ContVarLeaf[:s][g][k][:ub] *
                augmentVar[g, k]
                for k in keys(ContVarLeaf[:s][g])
            )
        )
        @constraint(
            model,
            partition_lower_bound_copy[g in indexSets.G],
            s_copy[g] ≥ sum(
                ContVarLeaf[:s][g][k][:lb] *
                augmentVar_copy[g, k]
                for k in keys(ContVarLeaf[:s][g])
            )
        )
        @constraint(
            model,
            partition_upper_bound_copy[g in indexSets.G],
            s_copy[g] ≤ sum(
                ContVarLeaf[:s][g][k][:ub] *
                augmentVar_copy[g, k]
                for k in keys(ContVarLeaf[:s][g])
            )
        )

    elseif param.algorithm == :SDDP 
        ## define copy variables
        @variable(model, 0 ≤ s_copy[g in indexSets.G] ≤ paramOPF.smax[g]);
        if param.tightness
            @variable(model, y_copy[indexSets.G], Bin);
            @variable(model, v_copy[indexSets.G], Bin);
            @variable(model, w_copy[indexSets.G], Bin);
        else
            @variable(model, 0 ≤ y_copy[indexSets.G] ≤ 1);
            @variable(model, 0 ≤ v_copy[indexSets.G] ≤ 1);
            @variable(model, 0 ≤ w_copy[indexSets.G] ≤ 1);
        end
        ContVarLeaf = nothing;
        ContVarBinaries = nothing;
    elseif param.algorithm == :SDDiP
        ## approximate the continuous state s[g], s[g] = ∑_{i=0}^{κ-1} 2ⁱ * λ[g, i] * ε, κ = log2(paramOPF.smax[g] / ε) + 1
        @variable(model, λ[g in indexSets.G, i in 1:param.κ[g]], Bin);  
        @constraint(model, ContiApprox[g in indexSets.G], param.ε * sum(2^(i-1) * λ[g, i] for i in 1:param.κ[g]) == s[g]);
        ContVarLeaf = nothing;
        ContVarBinaries = Dict(
            :s => Dict{Any, Dict{Any, VariableRef}}(
                g => Dict(
                    i => λ[g, i] for i in 1:param.κ[g]
                ) for g in indexSets.G
            )
        );
        @variable(model, 0 ≤ s_copy[g in indexSets.G] ≤ paramOPF.smax[g]);
        if param.tightness
            @variable(model, λ_copy[g in indexSets.G, i in 1:param.κ[g]], Bin);
            @variable(model, y_copy[indexSets.G], Bin);        
            @variable(model, v_copy[indexSets.G], Bin);        
            @variable(model, w_copy[indexSets.G], Bin);        
        else
            @variable(model, 0 ≤ λ_copy[g in indexSets.G, i in 1:param.κ[g]] ≤ 1);      
            @variable(model, 0 ≤ y_copy[indexSets.G] ≤ 1);    
            @variable(model, 0 ≤ v_copy[indexSets.G] ≤ 1);    
            @variable(model, 0 ≤ w_copy[indexSets.G] ≤ 1);       
        end
        @constraint(model, [g in indexSets.G], param.ε * sum(2^(i-1) * λ_copy[g, i] for i in 1:param.κ[g]) == s_copy[g]);
    end

    ## problem constraints:
    # power flow constraints
    for l in indexSets.L
        i = l[1];
        j = l[2];
        @constraint(model, P[l] ≤ - paramOPF.b[l] * (θ_angle[i] - θ_angle[j]));
        @constraint(model, P[l] ≥ - paramOPF.b[l] * (θ_angle[i] - θ_angle[j]));
    end
    
    # power flow limitation
    @constraint(model, [l in indexSets.L], P[l] ≥ - paramOPF.W[l]);
    @constraint(model, [l in indexSets.L], P[l] ≤   paramOPF.W[l]);
    # generator limitation
    @constraint(model, [g in indexSets.G], s[g] ≥ paramOPF.smin[g] * y[g]);
    @constraint(model, [g in indexSets.G], s[g] ≤ paramOPF.smax[g] * y[g]);

    # power balance constraints
    @constraint(
        model, 
        PowerBalance[i in indexSets.B], 
        sum(s[g] for g in indexSets.Gᵢ[i]) -
        sum(P[(i, j)] for j in indexSets.out_L[i]) + 
        sum(P[(j, i)] for j in indexSets.in_L[i]) 
        .== sum(paramDemand.demand[d] * x[d] for d in indexSets.Dᵢ[i]) 
    );
    
    # on/off status with startup and shutdown decision
    @constraint(
        model, 
        ShutUpDown[g in indexSets.G], 
        v[g] - w[g] == y[g] - y_copy[g]
    );
    @constraint(
        model, 
        Ramping1[g in indexSets.G], 
        # s[g] - s_copy[g] ≤ paramOPF.M[g] * y_copy[g] + paramOPF.smin[g] * v[g]
        s[g] - s_copy[g] ≤ paramOPF.M[g] * (y_copy[g] + v[g])
    );
    @constraint(
        model, 
        Ramping2[g in indexSets.G], 
        # s[g] - s_copy[g] ≥ - paramOPF.M[g] * y[g] - paramOPF.smin[g] * w[g]
        s[g] - s_copy[g] ≥ - paramOPF.M[g] * (y[g] + w[g])
    );

    # minimum up and down constraints
    @constraint(
        model, 
        MinimumUp[g in indexSets.G], 
        v_copy[g] + v[g] ≤ y[g]
    );
    @constraint(
        model, 
        MinimumDown[g in indexSets.G], 
        w_copy[g] + w[g] ≤ 1 - y[g]
    );

    # production cost
    @constraint(
        model, 
        production[g in indexSets.G, o in keys(paramOPF.slope[g])], 
        h[g] ≥ paramOPF.slope[g][o] * s[g] + paramOPF.intercept[g][o] * y[g]
    );

    # objective function
    @expression(
        model, 
        primal_objective_expression, 
        sum(h[g] + paramOPF.C_start[g] * v[g] + 
        paramOPF.C_down[g] * w[g] for g in indexSets.G) + 
        sum(paramDemand.w[d] * (1 - x[d]) for d in indexSets.D) + 
        sum(θ)
    );

    @objective(
        model, 
        Min, 
        primal_objective_expression
    );
    
    return SDDPModel(
        model, 
        Dict{Any, Dict{Any, VariableRef}}(
            :y => Dict{Any, VariableRef}(g => y[g] for g in indexSets.G),
            :v => Dict{Any, VariableRef}(g => v[g] for g in indexSets.G),
            :w => Dict{Any, VariableRef}(g => w[g] for g in indexSets.G)
        ), 
        nothing, 
        Dict{Any, Dict{Any, VariableRef}}(:s => Dict{Any, VariableRef}(g => s[g] for g in indexSets.G)), 
        nothing, 
        ContVarLeaf,
        nothing,
        ContVarBinaries
    )
end

"""
ModelModification!(; model::Model = model)

# Arguments

    1. `model::Model` : a forward pass model of stage t
    2. `randomVariables::RandomVariables` : random variables
    3. `paramDemand::ParamDemand` : demand parameters
    4. `stateInfo::StateInfo` : the last stage decisions
  
# Modification
    1. Remove the other scenario's demand balance constraints
    2. Add the current scenario's demand balance constraints
    3. Update its last stage decision with
"""
function ModelModification!( 
    model::Model, 
    randomVariables::RandomVariables,
    paramDemand::ParamDemand,
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param
)::Nothing
    if param.algorithm !== :SDDiP
        if :ContVarNonAnticipative ∉ keys(model.obj_dict) 
            @constraint(
                model, 
                ContVarNonAnticipative[g in indexSets.G], 
                model[:s_copy][g] == stateInfo.ContVar[:s][g]
            );
        end
        if :BinVarNonAnticipative_y ∉ keys(model.obj_dict) 
            @constraint(
                model, 
                BinVarNonAnticipative_y[g in indexSets.G], 
                model[:y_copy][g] == stateInfo.BinVar[:y][g]
            );
            @constraint(
                model, 
                BinVarNonAnticipative_v[g in indexSets.G], 
                model[:v_copy][g] == stateInfo.BinVar[:v][g]
            );
            @constraint(
                model, 
                BinVarNonAnticipative_w[g in indexSets.G], 
                model[:w_copy][g] == stateInfo.BinVar[:w][g]
            );
        end
    elseif param.algorithm == :SDDiP
        if :BinarizationNonAnticipative ∉ keys(model.obj_dict) 
            @constraint(
                model, 
                BinarizationNonAnticipative[g in indexSets.G, i in 1:param.κ[g]], 
                model[:λ_copy][g, i] == stateInfo.ContStateBin[:s][g][i]
            );
        end
        if :BinVarNonAnticipative_y ∉ keys(model.obj_dict) 
            @constraint(
                model, 
                BinVarNonAnticipative_y[g in indexSets.G], 
                model[:y_copy][g] == stateInfo.BinVar[:y][g]
            );
            @constraint(
                model, 
                BinVarNonAnticipative_v[g in indexSets.G], 
                model[:v_copy][g] == stateInfo.BinVar[:v][g]
            );
            @constraint(
                model, 
                BinVarNonAnticipative_w[g in indexSets.G], 
                model[:w_copy][g] == stateInfo.BinVar[:w][g]
            );
        end
    end

    # power balance constraints
    for i in indexSets.B
        delete(model, model[:PowerBalance][i])
    end
    unregister(model, :PowerBalance)
    @constraint(model, PowerBalance[i in indexSets.B], 
                            sum(model[:s][g]      for g in indexSets.Gᵢ[i]) -
                            sum(model[:P][(i, j)] for j in indexSets.out_L[i]) + 
                            sum(model[:P][(j, i)] for j in indexSets.in_L[i]) .==
                            sum(paramDemand.demand[d] * randomVariables.deviation[d] * model[:x][d] for d in indexSets.Dᵢ[i])
    )

    @objective(
        model, 
        Min, 
        model[:primal_objective_expression]
    );
    return
end

"""
forwardPass(ξ): function for forward pass in parallel computing

# Arguments

  1. `ξ`: A sampled scenario path

# Returns
  1. `stateInfoList`: forward pass solution collection

"""
function forwardPass(
    ξ::Dict{Int64, RandomVariables};
    ModelList::Dict{Int, SDDPModel} = ModelList,
    paramDemand::ParamDemand = paramDemand, 
    paramOPF::ParamOPF = paramOPF, 
    indexSets::IndexSets = indexSets, 
    initialStateInfo::StateInfo = initialStateInfo, 
    param::NamedTuple = param
)
    stateInfoList = Dict();
    stateInfoList[0] = deepcopy(initialStateInfo);
    for t in 1:indexSets.T
        ModelModification!( 
            ModelList[t].model, 
            ξ[t],
            paramDemand,
            stateInfoList[t-1];
            indexSets = indexSets,
            param = param
        )
        optimize!(ModelList[t].model);

        # record the solution
        BinVar = Dict{Any, Dict{Any, Any}}(
            :y => Dict{Any, Any}(
                g => round(JuMP.value(ModelList[t].BinVar[:y][g]), digits = 2) for g in indexSets.G
            ),
            :v => Dict{Any, Any}(
                g => round(JuMP.value(ModelList[t].BinVar[:v][g]), digits = 2) for g in indexSets.G
            ),
            :w => Dict{Any, Any}(
                g => round(JuMP.value(ModelList[t].BinVar[:w][g]), digits = 2) for g in indexSets.G
            ),
        );
        ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
            g => JuMP.value(ModelList[t].ContVar[:s][g]) for g in indexSets.G)
        );
        if param.algorithm == :SDDPL
            ContVarLeaf = Dict{Any, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}(
                :s => Dict{Any, Dict{Any, Dict{Symbol, Any}}}(
                    g => Dict(
                        k => Dict(
                            :var => round(JuMP.value(ModelList[t].ContVarLeaf[:s][g][k][:var]), digits = 2)
                        ) for k in keys(ModelList[t].ContVarLeaf[:s][g]) if JuMP.value(ModelList[t].ContVarLeaf[:s][g][k][:var]) > .5
                    ) for g in indexSets.G
                )
            );
            ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(:s => Dict{Any, Dict{Any, Any}}(g => Dict{Any, Any}() for g in indexSets.G));
            ContStateBin = nothing;
        elseif param.algorithm == :SDDiP
            ContVarLeaf  = nothing;
            ContAugState = nothing;
            ContStateBin = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                :s => Dict{Any, Dict{Any, Any}}(
                    g => Dict{Any, Any}(
                        i => round(JuMP.value(ModelList[t].ContVarBinaries[:s][g][i]), digits = 2) for i in 1:param.κ[g]
                    ) for g in indexSets.G
                )
            );
        elseif param.algorithm == :SDDP
            ContVarLeaf  = nothing;
            ContAugState = nothing;
            ContStateBin = nothing;
        end
        
        stageValue = JuMP.objective_value(ModelList[t].model) - sum(JuMP.value.(ModelList[t].model[:θ])); 
        stateValue = JuMP.objective_value(ModelList[t].model);
        stateInfoList[t] = StateInfo(
            BinVar, 
            nothing, 
            ContVar, 
            nothing, 
            ContVarLeaf, 
            stageValue, 
            stateValue, 
            nothing, 
            ContAugState,
            nothing,
            ContStateBin
        );
    end  
    return stateInfoList  
end
