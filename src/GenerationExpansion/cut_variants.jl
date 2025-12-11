"""
    function solve_inner_minimization_problem(
        cutTypeInfo::ParetoLagrangianCutGenerationProgram,
        model::Model, 
        πₙ::StageInfo, 
        stateInfo::StageInfo
    )

# Arguments

    1. `cutTypeInfo::ParetoLagrangianCutGenerationProgram` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `πₙ::StageInfo` : the dual information
    4. `stateInfo::StageInfo` : the last stage decision
  
# Returns
    1. `currentInfo::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    cutTypeInfo::ParetoLagrangianCutGenerationProgram,
    model::Model, 
    πₙ::StageInfo, 
    stateInfo::StageInfo;
    param::SDDPParam = param
)
    base_obj = model[:primal_objective_expression]

    state_penalty = if param.algorithm == :SDDiP
        πₙ.IntVarBinaries' * (stateInfo.IntVarBinaries .- model[:Lc])
    else
        πₙ.IntVar' * (stateInfo.IntVar .- model[:Sc])
    end

    leaf_penalty = if param.algorithm == :SDDPL
        sum(
            sum(
                πₙ.IntVarLeaf[g][k] *
                (stateInfo.IntVarLeaf[g][k] - model[:region_indicator_copy][g][k])
                for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    else
        0.0
    end

    @objective(
        model,
        Min,
        base_obj + state_penalty + leaf_penalty,
    )
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);
    
    state_term = if param.algorithm == :SDDiP
        πₙ.IntVarBinaries' * (
            cutTypeInfo.CoreState.IntVarBinaries .- stateInfo.IntVarBinaries
        )
    else
        πₙ.IntVar' * (
            cutTypeInfo.CoreState.IntVar .- stateInfo.IntVar
        )
    end

    leaf_term = if param.algorithm == :SDDPL
        sum(
            sum(
                πₙ.IntVarLeaf[g][k] * (
                    cutTypeInfo.CoreState.IntVarLeaf[g][k] - stateInfo.IntVarLeaf[g][k]
                )
                for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    else
        0.0
    end

    expr = - F - state_term - leaf_term

    d_obj = if param.algorithm == :SDDPL
        Dict{Symbol, Any}(
            :St => value.(model[:Sc]) .- cutTypeInfo.CoreState.IntVar,
            :region_indicator => Dict(
                g => Dict(
                    k => JuMP.value(model[:region_indicator_copy][g][k]) - cutTypeInfo.CoreState.IntVarLeaf[g][k]
                    for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in 1:binaryInfo.d
            ),
        )
    elseif param.algorithm == :SDDiP
        # SDDiP: binary 表示的状态差值
        Dict{Symbol, Any}(
            :Lt => JuMP.value.(model[:Lc]) .- cutTypeInfo.CoreState.IntVarBinaries
        )
    else
        # SDDP（或者其他默认情况）：只用 St
        Dict{Symbol, Any}(
            :St => value.(model[:Sc]) .- cutTypeInfo.CoreState.IntVar
        )
    end

    d_con = if param.algorithm == :SDDPL
        Dict{Symbol, Any}(
            :St => value.(model[:Sc]) .- stateInfo.IntVar,
            :region_indicator => Dict(
                g => Dict(
                    k => JuMP.value(model[:region_indicator_copy][g][k]) - stateInfo.IntVarLeaf[g][k]
                    for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in 1:binaryInfo.d
            ),
        )
    elseif param.algorithm == :SDDiP
        Dict{Symbol, Any}(
            :Lt => JuMP.value.(model[:Lc]) - stateInfo.IntVarBinaries
        )
    else
        Dict{Symbol, Any}(
            :St => value.(model[:Sc]) .- stateInfo.IntVar
        )
    end

    currentInfo = CurrentInfo(  
        πₙ, 
        expr,
        Dict(1 => cutTypeInfo.primal_bound - F - cutTypeInfo.δ),
        d_obj,
        Dict(1 => d_con)
    );
    return (currentInfo = currentInfo, currentInfo_f = F)
end    

"""
    function solve_inner_minimization_problem(
        cutTypeInfo::SquareMinimizationCutGenerationProgram,
        model::Model, 
        πₙ::StageInfo, 
        stateInfo::StageInfo
    )

# Arguments

    1. `cutTypeInfo::SquareMinimizationCutGenerationProgram` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `πₙ::StageInfo` : the dual information
    4. `stateInfo::StageInfo` : the last stage decision
  
# Returns
    1. `currentInfo::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    cutTypeInfo::SquareMinimizationCutGenerationProgram,
    model::Model, 
    πₙ::StageInfo, 
    stateInfo::StageInfo;
    param::SDDPParam = param
)
    base_obj = model[:primal_objective_expression]

    state_penalty = if param.algorithm == :SDDiP
        πₙ.IntVarBinaries' * (stateInfo.IntVarBinaries .- model[:Lc])
    else
        πₙ.IntVar' * (stateInfo.IntVar .- model[:Sc])
    end

    leaf_penalty = if param.algorithm == :SDDPL
        sum(
            sum(
                πₙ.IntVarLeaf[g][k] *
                (stateInfo.IntVarLeaf[g][k] - model[:region_indicator_copy][g][k])
                for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    else
        0.0
    end

    @objective(
        model,
        Min,
        base_obj + state_penalty + leaf_penalty,
    )
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);

    currentInfo = CurrentInfo(  
        πₙ, 
        1/2 * (param.algorithm == :SDDiP ?
                πₙ.IntVarBinaries' * πₙ.IntVarBinaries : 
                πₙ.IntVar' * πₙ.IntVar
        ) +
        (param.algorithm == :SDDPL ? 
        1/2 * sum(
                sum(
                    πₙ.IntVarLeaf[g][k] * πₙ.IntVarLeaf[g][k] for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
                ) for g in 1:binaryInfo.d
            ) : 0.0
        ),                                                                                                                                                              ## obj function value
        Dict(
            1 => cutTypeInfo.primal_bound - F - cutTypeInfo.δ
        ),                                                                                                                                                              ## constraint value
        param.algorithm == :SDDPL ? 
        Dict{Symbol, Any}(
            :St => πₙ.IntVar,
            :region_indicator => Dict(
                g => Dict(
                    k => πₙ.IntVarLeaf[g][k] for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in 1:binaryInfo.d
            )
        ) : param.algorithm == :SDDiP ? 
        Dict{Symbol, Any}(
            :Lt => πₙ.IntVarBinaries
        ) : 
        Dict{Symbol, Any}(
            :St => πₙ.IntVar
        ),                                                                                                                                                              ## obj gradient
        Dict(1 => 
            param.algorithm == :SDDPL ? 
            Dict{Symbol, Any}(
                :St => value.(model[:Sc]) .- stateInfo.IntVar,
                :region_indicator => Dict(
                    g => Dict(
                        k => JuMP.value(model[:region_indicator_copy][g][k]) - stateInfo.IntVarLeaf[g][k]
                                for k in keys(stateInfo.IntVarLeaf[g])
                    ) for g in 1:binaryInfo.d
                )
            ) : param.algorithm == :SDDiP ? 
            Dict{Symbol, Any}(
                :Lt => JuMP.value.(model[:Lc]) .- stateInfo.IntVarBinaries
            ) : Dict{Symbol, Any}(
                :St => value.(model[:Sc]) .- stateInfo.IntVar
            ) 
        )                                                                                                                             ## constraint gradient
    );
    return (currentInfo = currentInfo, currentInfo_f = F)
end    

"""
    function solve_inner_minimization_problem(
        cutTypeInfo::LagrangianCutGenerationProgram,
        model::Model, 
        πₙ::StageInfo, 
        stateInfo::StageInfo
    )

# Arguments

    1. `cutTypeInfo::LagrangianCutGenerationProgram` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `πₙ::StageInfo` : the dual information
    4. `stateInfo::StageInfo` : the last stage decision
  
# Returns
    1. `currentInfo::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    cutTypeInfo::LagrangianCutGenerationProgram,
    model::Model, 
    πₙ::StageInfo, 
    stateInfo::StageInfo;
    param::SDDPParam = param
)
    base_obj = model[:primal_objective_expression]

    state_penalty = if param.algorithm == :SDDiP
        πₙ.IntVarBinaries' * (stateInfo.IntVarBinaries .- model[:Lc])
    else
        πₙ.IntVar' * (stateInfo.IntVar .- model[:Sc])
    end

    leaf_penalty = if param.algorithm == :SDDPL
        sum(
            sum(
                πₙ.IntVarLeaf[g][k] *
                (stateInfo.IntVarLeaf[g][k] - model[:region_indicator_copy][g][k])
                for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    else
        0.0
    end

    @objective(
        model,
        Min,
        base_obj + state_penalty + leaf_penalty,
    )
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);

    d_obj = if param.algorithm == :SDDPL
        Dict{Symbol, Any}(
            :St => value.(model[:Sc]) .- stateInfo.IntVar,
            :region_indicator => Dict(
                g => Dict(
                    k => JuMP.value(model[:region_indicator_copy][g][k]) -
                        stateInfo.IntVarLeaf[g][k]
                    for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in 1:binaryInfo.d
            ),
        )
    elseif param.algorithm == :SDDiP
        # SDDiP: binary 表示的状态差值
        Dict{Symbol, Any}(
            :Lt => JuMP.value.(model[:Lc]) .- stateInfo.IntVarBinaries,
        )
    else
        # SDDP（或者其他默认情况）：只用 St
        Dict{Symbol, Any}(
            :St => value.(model[:Sc]) .- stateInfo.IntVar,
        )
    end

    d_con = if param.algorithm == :SDDPL
        Dict{Symbol, Any}(
            :St => zeros(binaryInfo.d),
            :region_indicator => Dict(
                g => Dict(
                    k => 0.0
                    for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in 1:binaryInfo.d
            ),
        )
    elseif param.algorithm == :SDDiP
        Dict{Symbol, Any}(
            :Lt => zeros(binaryInfo.n),
        )
    else
        Dict{Symbol, Any}(
            :St => zeros(binaryInfo.d),
        )
    end

    currentInfo = CurrentInfo(  
        πₙ, 
        - F,
        Dict(1 => 0.0),
        d_obj,
        Dict(1 => d_con)
    );
    return (currentInfo = currentInfo, currentInfo_f = F)
end    

"""
    function solve_inner_minimization_problem(
        cutTypeInfo::StrengthenedBendersCutGenerationProgram,
        model::Model, 
        πₙ::StageInfo, 
        stateInfo::StageInfo
    )

# Arguments

    1. `cutTypeInfo::StrengthenedBendersCutGenerationProgram` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `πₙ::StageInfo` : the dual information
    4. `stateInfo::StageInfo` : the last stage decision
  
# Returns
    1. `currentInfo::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    cutTypeInfo::StrengthenedBendersCutGenerationProgram,
    model::Model, 
    πₙ::StageInfo, 
    stateInfo::StageInfo;
    param::SDDPParam = param
)
    base_obj = model[:primal_objective_expression]

    state_penalty = if param.algorithm == :SDDiP
        πₙ.IntVarBinaries' * (stateInfo.IntVarBinaries .- model[:Lc])
    else
        πₙ.IntVar' * (stateInfo.IntVar .- model[:Sc])
    end

    leaf_penalty = if param.algorithm == :SDDPL
        sum(
            sum(
                πₙ.IntVarLeaf[g][k] *
                (stateInfo.IntVarLeaf[g][k] - model[:region_indicator_copy][g][k])
                for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    else
        0.0
    end

    @objective(
        model,
        Min,
        base_obj + state_penalty + leaf_penalty,
    )
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);

    currentInfo = CurrentInfo(  
        πₙ, 
        - F,
        Dict(
            1 => 0.
        ),
        param.algorithm == :SDDPL ? 
        Dict{Symbol, Any}(
            :St => value.(model[:Sc]) .- stateInfo.IntVar,
            :region_indicator => Dict(
                g => Dict(
                    k => JuMP.value(model[:region_indicator_copy][g][k]) - stateInfo.IntVarLeaf[g][k]
                            for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in 1:binaryInfo.d
            )
        ) : param.algorithm == :SDDiP ? 
        Dict{Symbol, Any}(
            :Lt => JuMP.value.(model[:Lc]) - stateInfo.IntVarBinaries
        ) : Dict{Symbol, Any}(
            :St => value.(model[:Sc]) .- stateInfo.IntVar
        ),
        Dict(1 => 
            param.algorithm == :SDDPL ? 
            Dict{Symbol, Any}(
                :St => stateInfo.IntVar .* 0.0,
                :region_indicator => Dict(
                    g => Dict(
                        k => 0.0 for k in keys(stateInfo.IntVarLeaf[g])
                    ) for g in 1:binaryInfo.d
                )
            ) : param.algorithm == :SDDiP ? 
            Dict{Symbol, Any}(
                :Lt => 0.0
            ) : Dict{Symbol, Any}(
                :St => 0.0
            ) 
        )
    );
    return (currentInfo = currentInfo, currentInfo_f = F)
end    

"""
    function solve_inner_minimization_problem(
        cutTypeInfo::LinearNormalizationLagrangianCutGenerationProgram,
        model::Model, 
        πₙ::StageInfo, 
        stateInfo::StageInfo
    )

# Arguments

    1. `cutTypeInfo::LinearNormalizationLagrangianCutGenerationProgram` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `πₙ::StageInfo` : the dual information
    4. `stateInfo::StageInfo` : the last stage decision
  
# Returns
    1. `currentInfo::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    cutTypeInfo::LinearNormalizationLagrangianCutGenerationProgram,
    model::Model, 
    πₙ::StageInfo, 
    stateInfo::StageInfo;
    param::SDDPParam = param
)
    base_obj = πₙ.StateValue * (cutTypeInfo.primal_bound - model[:primal_objective_expression])

    state_penalty = if param.algorithm == :SDDiP
        πₙ.IntVarBinaries' * (stateInfo.IntVarBinaries .- model[:Lc])
    else
        πₙ.IntVar' * (stateInfo.IntVar .- model[:Sc])
    end

    leaf_penalty = if param.algorithm == :SDDPL
        sum(
            sum(
                πₙ.IntVarLeaf[g][k] * (stateInfo.IntVarLeaf[g][k] - model[:region_indicator_copy][g][k]) for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    else
        0.0
    end
    
    @objective(
        model,
        Min,
        base_obj + state_penalty + leaf_penalty,
    )
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);

    normalization_function = cutTypeInfo.CoreState.StateValue * πₙ.StateValue + (
    stateInfo.IntVar === nothing ? 0.0 : cutTypeInfo.CoreState.IntVar' * πₙ.IntVar) + (
        stateInfo.IntVarLeaf === nothing ? 0.0 : sum(
            sum(
                cutTypeInfo.CoreState.IntVarLeaf[g][k] * πₙ.IntVarLeaf[g][k] for k in keys(πₙ.IntVarLeaf[g])
        ) for g in 1:binaryInfo.d) 
    ) + (
        stateInfo.IntVarBinaries === nothing ? 0.0 : cutTypeInfo.CoreState.IntVarBinaries' * πₙ.IntVarBinaries
    );
    
    
    d_obj = if param.algorithm == :SDDPL
        Dict{Symbol, Any}(
            :obj => value(model[:primal_objective_expression]) - cutTypeInfo.primal_bound,
            :St => value.(model[:Sc]) .- stateInfo.IntVar,
            :region_indicator => Dict(
                g => Dict(
                    k => JuMP.value(model[:region_indicator_copy][g][k]) - stateInfo.IntVarLeaf[g][k]
                    for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in 1:binaryInfo.d
            ),
        )
    elseif param.algorithm == :SDDiP
        # SDDiP: binary 表示的状态差值
        Dict{Symbol, Any}(
            :obj => value(model[:primal_objective_expression]) - cutTypeInfo.primal_bound,
            :Lt => JuMP.value.(model[:Lc]) .- stateInfo.IntVarBinaries
        )
    elseif param.algorithm == :SDDP
        # SDDP（或者其他默认情况）：只用 St
        Dict{Symbol, Any}(
            :obj => value(model[:primal_objective_expression]) - cutTypeInfo.primal_bound,
            :St => value.(model[:Sc]) .- stateInfo.IntVar
        )
    end

    d_con = if param.algorithm == :SDDPL
        Dict{Symbol, Any}(
            :obj => cutTypeInfo.CoreState.StateValue,
            :St => cutTypeInfo.CoreState.IntVar,
            :region_indicator => Dict(
                g => Dict(
                    k => cutTypeInfo.CoreState.IntVarLeaf[g][k]
                    for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in 1:binaryInfo.d
            ),
        )
    elseif param.algorithm == :SDDiP
        Dict{Symbol, Any}(
            :obj => cutTypeInfo.CoreState.StateValue,
            :Lt => cutTypeInfo.CoreState.IntVarBinaries
        )
    else
        Dict{Symbol, Any}(
            :obj => cutTypeInfo.CoreState.StateValue,
            :St => cutTypeInfo.CoreState.IntVar
        )
    end

    currentInfo = CurrentInfo(  
        πₙ, 
        - F,
        Dict(
            1 => normalization_function - 1.0, 
            2 => πₙ.StateValue),
        d_obj,
        Dict(
            1 => d_con, 
            2 => 1
        )
    );
    return (currentInfo = currentInfo, currentInfo_f = F)
end   