"""
Modify the forward model for a given realization (demand, previous state).

- For all algorithms: update demand constraint.
- For SDDiP      : fix Lc to previous binary state `L̂`.
- For SDDP/SDDPL : fix Sc to previous integer state `Ŝ`.
"""
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
    This function is to collect the necessary parameters for the level set method.
"""
function setupCutGenerationInfo(
    model::Model,
    stateInfo::StageInfo, 
    primal_bound::Float64, 
    cutType::Symbol;    
    stageData::StageData = stageData,
    binaryInfo::BinaryInfo = binaryInfo,
    param::SDDPParam = param
)::NamedTuple

    coreState = StageInfo(
        1.0,
        nothing,
        stateInfo.IntVar === nothing ?
            nothing :
            stageData.ū ./ 2,
            # stateInfo.IntVar .* 0 .+ 0.5,
        stateInfo.IntVarLeaf === nothing ?
            nothing :
            Dict(
                g => Dict(
                    k => 0.5 for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in keys(stateInfo.IntVarLeaf)
            ),
        stateInfo.IntVarBinaries === nothing ?
            nothing :
            stateInfo.IntVarBinaries .* 0 .+ 0.5,
    );

    πₙ = StageInfo(
        - 0.0,
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

    cutGenerationParamInfo = CutGenerationParamInfo(
        0.95,
        0.5,
        param.gapSDDP,
        param.iterSDDP,
        1e8,
        param.verbose,
        stateInfo,
        param.cutType,
        πₙ
    )

    if cutType == :LC
        cutGenerationProgramInfo = LagrangianCutGenerationProgram(0.0)
    elseif cutType == :PLC
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
                coreState,
                objective_value(model),
                param.ε
            )
        else 
            cutGenerationProgramInfo = ParetoLagrangianCutGenerationProgram(
                coreState,
                primal_bound,
                param.ε
            )
        end 
        delete(model, model[:NonAnticipativity])
        unregister(model, :NonAnticipativity)
    elseif cutType == :SMC
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
        # state / binary state constraint
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
            cutGenerationProgramInfo = LinearNormalizationLagrangianCutGenerationProgram(
                coreState,
                objective_value(model)
            )
        else 
            cutGenerationProgramInfo = LinearNormalizationLagrangianCutGenerationProgram(
                coreState,
                primal_bound
            )
        end 
        delete(model, model[:NonAnticipativity])
        unregister(model, :NonAnticipativity)
    end

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
)::Vector

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

    (cutGenerationParamInfo, cutTypeInfo) = setupCutGenerationInfo(
        forwardInfoList[t].model,
        solCollection[t-1,k], 
        solCollection[t,k].StateValue,
        cutType;
        stageData = stageDataList[t],
        binaryInfo = binaryInfo,
        param = param
    );

    (λ₀, λ₁) = LevelSetMethod_optimization!(
        forwardInfoList[t],
        cutGenerationParamInfo,
        cutTypeInfo;
        stageData = stageDataList[t],
        binaryInfo = binaryInfo,
        param = param,
    );                 
    return [λ₀, λ₁]
end