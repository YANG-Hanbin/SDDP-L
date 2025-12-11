"""
    function solve_inner_minimization_problem(
        CutGenerationInfo::ParetoLagrangianCutGeneration,
        model::Model, 
        πₙ::StateInfo, 
        stateInfo::StateInfo
    )

# Arguments

    1. `CutGenerationInfo::ParetoLagrangianCutGeneration` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `πₙ::StateInfo` : the dual information
    4. `stateInfo::StateInfo` : the last stage decision
  
# Returns
    1. `currentInfo::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    CutGenerationInfo::ParetoLagrangianCutGeneration,
    model::Model, 
    πₙ::StateInfo, 
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param
)
    @objective(
        model, 
        Min,  
        model[:primal_objective_expression] +
        sum(
            πₙ.BinVar[:y][g] * (stateInfo.BinVar[:y][g] - model[:y_copy][g]) + 
            πₙ.BinVar[:v][g] * (stateInfo.BinVar[:v][g] - model[:v_copy][g]) + 
            πₙ.BinVar[:w][g] * (stateInfo.BinVar[:w][g] - model[:w_copy][g]) + 
            (param.algorithm == :SDDiP ?
                sum(
                    πₙ.ContStateBin[:s][g][i] * (stateInfo.ContStateBin[:s][g][i] - model[:λ_copy][g, i]) for i in 1:param.κ[g]
                ) : πₙ.ContVar[:s][g] * (stateInfo.ContVar[:s][g] - model[:s_copy][g])
            ) +
            (param.algorithm == :SDDPL ?
                sum(
                    πₙ.ContAugState[:s][g][k] * (stateInfo.ContAugState[:s][g][k] - model[:augmentVar_copy][g, k]) 
                    for k in keys(stateInfo.ContAugState[:s][g]); init = 0.0
                ) : 0.0
            ) 
            for g in indexSets.G
        )
    );
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);
    
    negative_∇F = StateInfo(
        Dict{Any, Dict{Any, Any}}(
            :y => Dict{Any, Any}(
                g => JuMP.value(model[:y_copy][g]) - stateInfo.BinVar[:y][g] for g in indexSets.G
            ),
            :v => Dict{Any, Any}(
                g => JuMP.value(model[:v_copy][g]) - stateInfo.BinVar[:v][g] for g in indexSets.G
            ),
            :w => Dict{Any, Any}(
                g => JuMP.value(model[:w_copy][g]) - stateInfo.BinVar[:w][g] for g in indexSets.G
            )
        ), 
        nothing, 
        Dict{Any, Dict{Any, Any}}(
            :s => Dict{Any, Any}(
                g => JuMP.value(model[:s_copy][g]) - stateInfo.ContVar[:s][g] for g in indexSets.G
            )
        ), 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        param.algorithm == :SDDPL ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        k => JuMP.value(model[:augmentVar_copy][g, k]) - stateInfo.ContAugState[:s][g][k]
                        for k in keys(stateInfo.ContAugState[:s][g])
                    ) 
                for g in indexSets.G
            )
        ) : nothing,
        nothing,
        param.algorithm == :SDDiP ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        i => JuMP.value(model[:λ_copy][g, i]) - stateInfo.ContStateBin[:s][g][i]
                        for i in 1:param.κ[g]
                    ) 
                for g in indexSets.G
            )
        ) : nothing
    );
    currentInfo = CurrentInfo(  
        πₙ, 
        - F - sum(
            (param.algorithm == :SDDiP ?
                sum(
                    πₙ.ContStateBin[:s][g][i] * (CutGenerationInfo.core_point.ContStateBin[:s][g][i] - stateInfo.ContStateBin[:s][g][i]) for i in 1:param.κ[g]
                ) : πₙ.ContVar[:s][g] * (CutGenerationInfo.core_point.ContVar[:s][g] - stateInfo.ContVar[:s][g])
            ) +
            πₙ.BinVar[:y][g] * (CutGenerationInfo.core_point.BinVar[:y][g] - stateInfo.BinVar[:y][g]) +
            πₙ.BinVar[:v][g] * (CutGenerationInfo.core_point.BinVar[:v][g] - stateInfo.BinVar[:v][g]) +
            πₙ.BinVar[:w][g] * (CutGenerationInfo.core_point.BinVar[:w][g] - stateInfo.BinVar[:w][g]) +
            (param.algorithm == :SDDPL ?
                sum(
                    πₙ.ContAugState[:s][g][k] * (CutGenerationInfo.core_point.ContAugState[:s][g][k] - stateInfo.ContAugState[:s][g][k])
                    for k in keys(stateInfo.ContAugState[:s][g]); init = 0.0
                ) : 0.0
            )
            for g in indexSets.G
        ),                                                                                                                                                              ## obj function value
        Dict(
            1 => CutGenerationInfo.primal_bound - F - CutGenerationInfo.δ
        ),                                                                                                                                                              ## constraint value
        param.algorithm == :SDDPL ? 
        Dict{Symbol, Dict{Int64, Any}}(
            :s => Dict(g => JuMP.value(model[:s_copy][g]) - CutGenerationInfo.core_point.ContVar[:s][g] for g in indexSets.G),
            :y => Dict(g => JuMP.value(model[:y_copy][g]) - CutGenerationInfo.core_point.BinVar[:y][g] for g in indexSets.G),
            :v => Dict(g => JuMP.value(model[:v_copy][g]) - CutGenerationInfo.core_point.BinVar[:v][g] for g in indexSets.G),
            :w => Dict(g => JuMP.value(model[:w_copy][g]) - CutGenerationInfo.core_point.BinVar[:w][g] for g in indexSets.G),
            :sur => Dict(
                g => Dict(
                    k => JuMP.value(model[:augmentVar_copy][g, k]) - CutGenerationInfo.core_point.ContAugState[:s][g][k] 
                            for k in keys(stateInfo.ContAugState[:s][g])
                ) for g in indexSets.G
            ),
            :λ => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(
                        i => JuMP.value(model[:λ_copy][g, i]) - CutGenerationInfo.core_point.ContStateBin[:s][g][i] for i in 1:param.κ[g]
                    ) : nothing
                ) for g in indexSets.G
            )
        ) :
        Dict{Symbol, Dict{Int64, Any}}(
            :s => Dict(g => JuMP.value(model[:s_copy][g]) - CutGenerationInfo.core_point.ContVar[:s][g] for g in indexSets.G),
            :y => Dict(g => JuMP.value(model[:y_copy][g]) - CutGenerationInfo.core_point.BinVar[:y][g] for g in indexSets.G),
            :v => Dict(g => JuMP.value(model[:v_copy][g]) - CutGenerationInfo.core_point.BinVar[:v][g] for g in indexSets.G),
            :w => Dict(g => JuMP.value(model[:w_copy][g]) - CutGenerationInfo.core_point.BinVar[:w][g] for g in indexSets.G),
            :λ => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(
                        i => JuMP.value(model[:λ_copy][g, i]) - CutGenerationInfo.core_point.ContStateBin[:s][g][i] for i in 1:param.κ[g]
                    ) : nothing
                ) for g in indexSets.G
            )
        ),                                                                                                                                                              ## obj gradient
        Dict(1 => negative_∇F )                                                                                                                                         ## constraint gradient
    );
    return (currentInfo = currentInfo, currentInfo_f = F)
end    

"""
    function solve_inner_minimization_problem(
        CutGenerationInfo::SquareMinimizationCutGeneration,
        model::Model, 
        πₙ::StateInfo, 
        stateInfo::StateInfo
    )

# Arguments

    1. `CutGenerationInfo::SquareMinimizationCutGeneration` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `πₙ::StateInfo` : the dual information
    4. `stateInfo::StateInfo` : the last stage decision
  
# Returns
    1. `currentInfo::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    CutGenerationInfo::SquareMinimizationCutGeneration,
    model::Model, 
    πₙ::StateInfo, 
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets
)
    @objective(
        model, 
        Min,  
        model[:primal_objective_expression] +
        sum(
            πₙ.BinVar[:y][g] * (stateInfo.BinVar[:y][g] - model[:y_copy][g]) + 
            πₙ.BinVar[:v][g] * (stateInfo.BinVar[:v][g] - model[:v_copy][g]) + 
            πₙ.BinVar[:w][g] * (stateInfo.BinVar[:w][g] - model[:w_copy][g]) + 
            (param.algorithm == :SDDiP ?
                sum(
                    πₙ.ContStateBin[:s][g][i] * (stateInfo.ContStateBin[:s][g][i] - model[:λ_copy][g, i]) for i in 1:param.κ[g]
                ) : πₙ.ContVar[:s][g] * (stateInfo.ContVar[:s][g] - model[:s_copy][g])
            ) +
            (param.algorithm == :SDDPL ?
                sum(
                    πₙ.ContAugState[:s][g][k] * (stateInfo.ContAugState[:s][g][k] - model[:augmentVar_copy][g, k]) 
                    for k in keys(stateInfo.ContAugState[:s][g]); init = 0.0
                ) : 0.0
            ) 
            for g in indexSets.G
        )
    );
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);
    
    negative_∇F = StateInfo(
        Dict{Any, Dict{Any, Any}}(
            :y => Dict{Any, Any}(
                g => JuMP.value(model[:y_copy][g]) - stateInfo.BinVar[:y][g] for g in indexSets.G
            ),
            :v => Dict{Any, Any}(
                g => JuMP.value(model[:v_copy][g]) - stateInfo.BinVar[:v][g] for g in indexSets.G
            ),
            :w => Dict{Any, Any}(
                g => JuMP.value(model[:w_copy][g]) - stateInfo.BinVar[:w][g] for g in indexSets.G
            )
        ), 
        nothing, 
        Dict{Any, Dict{Any, Any}}(
            :s => Dict{Any, Any}(
                g => JuMP.value(model[:s_copy][g]) - stateInfo.ContVar[:s][g] for g in indexSets.G
            )
        ), 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        param.algorithm == :SDDPL ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        k => JuMP.value(model[:augmentVar_copy][g, k]) - stateInfo.ContAugState[:s][g][k]
                        for k in keys(stateInfo.ContAugState[:s][g])
                    ) 
                for g in indexSets.G
            )
        ) : nothing,
        nothing,
        param.algorithm == :SDDiP ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        i => JuMP.value(model[:λ_copy][g, i]) - stateInfo.ContStateBin[:s][g][i]
                        for i in 1:param.κ[g]
                    ) 
                for g in indexSets.G
            )
        ) : nothing
    );
    currentInfo = CurrentInfo(  
        πₙ, 
        1/2 * (param.algorithm == :SDDiP ?
                sum(
                    sum(πₙ.ContStateBin[:s][g][i] * πₙ.ContStateBin[:s][g][i] for i in 1:param.κ[g]) 
                    for g in indexSets.G
                ) : 
                sum(πₙ.ContVar[:s][g] * πₙ.ContVar[:s][g] for g in indexSets.G)
        ) +
        1/2 * sum(πₙ.BinVar[:y][g] * πₙ.BinVar[:y][g] + πₙ.BinVar[:v][g] * πₙ.BinVar[:v][g] + πₙ.BinVar[:w][g] * πₙ.BinVar[:w][g] for g in indexSets.G) + 
        (param.algorithm == :SDDPL ? 
        1/2 * sum(
                sum(
                    πₙ.ContAugState[:s][g][k] * πₙ.ContAugState[:s][g][k] for k in keys(stateInfo.ContAugState[:s][g]); init = 0.0
                ) for g in indexSets.G
            ) : 0.0
        ),                                                                                                                                                              ## obj function value
        Dict(
            1 => CutGenerationInfo.primal_bound - F - CutGenerationInfo.δ
        ),                                                                                                                                                              ## constraint value
        param.algorithm == :SDDPL ?
        Dict{Symbol, Dict{Int64, Any}}(
            :s => Dict(g => πₙ.ContVar[:s][g] for g in indexSets.G),
            :y => Dict(g => πₙ.BinVar[:y][g] for g in indexSets.G), 
            :v => Dict(g => πₙ.BinVar[:v][g] for g in indexSets.G), 
            :w => Dict(g => πₙ.BinVar[:w][g] for g in indexSets.G), 
            :sur => Dict(g => Dict(k => πₙ.ContAugState[:s][g][k] for k in keys(stateInfo.ContAugState[:s][g])) for g in indexSets.G),
            :λ => Dict(
                g => (
                    param.algorithm == :SDDiP ?
                        Dict(i => πₙ.ContStateBin[:s][g][i] for i in 1:param.κ[g]) : nothing
                ) for g in indexSets.G
            )
        ) : 
        Dict{Symbol, Dict{Int64, Any}}(
            :s => Dict(g => πₙ.ContVar[:s][g] for g in indexSets.G),
            :y => Dict(g => πₙ.BinVar[:y][g] for g in indexSets.G),
            :v => Dict(g => πₙ.BinVar[:v][g] for g in indexSets.G),
            :w => Dict(g => πₙ.BinVar[:w][g] for g in indexSets.G),
            :λ => Dict(
                g => (
                    param.algorithm == :SDDiP ?
                        Dict(i => πₙ.ContStateBin[:s][g][i] for i in 1:param.κ[g]) : nothing
                ) for g in indexSets.G
            )
        ),                                                                                                                                                              ## obj gradient
        Dict(1 => negative_∇F )                                                                                                                                         ## constraint gradient
    );
    return (currentInfo = currentInfo, currentInfo_f = F)
end    

"""
    function solve_inner_minimization_problem(
        CutGenerationInfo::LagrangianCutGeneration,
        model::Model, 
        πₙ::StateInfo, 
        stateInfo::StateInfo
    )

# Arguments

    1. `CutGenerationInfo::LagrangianCutGeneration` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `πₙ::StateInfo` : the dual information
    4. `stateInfo::StateInfo` : the last stage decision
  
# Returns
    1. `currentInfo::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    CutGenerationInfo::LagrangianCutGeneration,
    model::Model, 
    πₙ::StateInfo, 
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets
)
    @objective(
        model, 
        Min,  
        model[:primal_objective_expression] +
        sum(
            πₙ.BinVar[:y][g] * (stateInfo.BinVar[:y][g] - model[:y_copy][g]) + 
            πₙ.BinVar[:v][g] * (stateInfo.BinVar[:v][g] - model[:v_copy][g]) + 
            πₙ.BinVar[:w][g] * (stateInfo.BinVar[:w][g] - model[:w_copy][g]) + 
            (param.algorithm == :SDDiP ?
                sum(
                    πₙ.ContStateBin[:s][g][i] * (stateInfo.ContStateBin[:s][g][i] - model[:λ_copy][g, i]) for i in 1:param.κ[g]
                ) : πₙ.ContVar[:s][g] * (stateInfo.ContVar[:s][g] - model[:s_copy][g])
            ) +
            (param.algorithm == :SDDPL ?
                sum(
                    πₙ.ContAugState[:s][g][k] * (stateInfo.ContAugState[:s][g][k] - model[:augmentVar_copy][g, k]) 
                    for k in keys(stateInfo.ContAugState[:s][g]); init = 0.0
                ) : 0.0
            ) 
            for g in indexSets.G
        )
    );
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);
    negative_∇F = StateInfo(
        Dict{Any, Dict{Any, Any}}(
            :y => Dict{Any, Any}(
                g => 0. for g in indexSets.G
            ),
            :v => Dict{Any, Any}(
                g => 0. for g in indexSets.G
            ),
            :w => Dict{Any, Any}(
                g => 0. for g in indexSets.G
            )
        ), 
        nothing, 
        Dict{Any, Dict{Any, Any}}(
            :s => Dict{Any, Any}(
                g => 0. for g in indexSets.G
            )
        ), 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        param.algorithm == :SDDPL ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        k => 0.
                        for k in keys(stateInfo.ContAugState[:s][g])
                    ) 
                for g in indexSets.G
            )
        ) : nothing,
        nothing,
        param.algorithm == :SDDiP ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        i => 0.
                        for i in 1:param.κ[g]
                    ) 
                for g in indexSets.G
            )
        ) : nothing
    );
    currentInfo = CurrentInfo(  
        πₙ, 
        - F,                                                                                                                                                            ## obj function value
        Dict(
            1 => 0.
        ),                                                                                                                                                              ## constraint value
        param.algorithm == :SDDPL ?                                                                                                                            
        Dict{Symbol, Dict{Int64, Any}}(
            :s   => Dict(g => JuMP.value(model[:s_copy][g]) - stateInfo.ContVar[:s][g] for g in indexSets.G),
            :y   => Dict(g => JuMP.value(model[:y_copy][g]) - stateInfo.BinVar[:y][g]  for g in indexSets.G), 
            :v   => Dict(g => JuMP.value(model[:v_copy][g]) - stateInfo.BinVar[:v][g]  for g in indexSets.G), 
            :w   => Dict(g => JuMP.value(model[:w_copy][g]) - stateInfo.BinVar[:w][g]  for g in indexSets.G), 
            :sur => Dict(g => Dict(
                k => JuMP.value(model[:augmentVar_copy][g, k]) - stateInfo.ContAugState[:s][g][k] for k in keys(stateInfo.ContAugState[:s][g])
                ) for g in indexSets.G),
            :λ   => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(i => JuMP.value(model[:λ_copy][g, i]) - stateInfo.ContStateBin[:s][g][i] for i in 1:param.κ[g]) : nothing
                ) for g in indexSets.G
            )
        ) : 
        Dict{Symbol, Dict{Int64, Any}}(
            :s   => Dict(g => JuMP.value(model[:s_copy][g]) - stateInfo.ContVar[:s][g] for g in indexSets.G),
            :y   => Dict(g => JuMP.value(model[:y_copy][g]) - stateInfo.BinVar[:y][g]  for g in indexSets.G),
            :v   => Dict(g => JuMP.value(model[:v_copy][g]) - stateInfo.BinVar[:v][g]  for g in indexSets.G),
            :w   => Dict(g => JuMP.value(model[:w_copy][g]) - stateInfo.BinVar[:w][g]  for g in indexSets.G),
            :λ   => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(i => JuMP.value(model[:λ_copy][g, i]) - stateInfo.ContStateBin[:s][g][i] for i in 1:param.κ[g]) : nothing
                ) for g in indexSets.G
            )
        ),                                                                                                                                                              ## obj gradient
        Dict(1 => negative_∇F )                                                                                                                                                               ## constraint gradient
    );
    return (currentInfo = currentInfo, currentInfo_f = F)
end    

"""
    function solve_inner_minimization_problem(
        CutGenerationInfo::StrengthenedBendersCutGeneration,
        model::Model, 
        πₙ::StateInfo, 
        stateInfo::StateInfo
    )

# Arguments

    1. `CutGenerationInfo::StrengthenedBendersCutGeneration` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `πₙ::StateInfo` : the dual information
    4. `stateInfo::StateInfo` : the last stage decision
  
# Returns
    1. `currentInfo::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    CutGenerationInfo::StrengthenedBendersCutGeneration,
    model::Model, 
    πₙ::StateInfo, 
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets
)
    @objective(
        model, 
        Min,  
        model[:primal_objective_expression] +
        sum(
            πₙ.BinVar[:y][g] * (stateInfo.BinVar[:y][g] - model[:y_copy][g]) + 
            πₙ.BinVar[:v][g] * (stateInfo.BinVar[:v][g] - model[:v_copy][g]) + 
            πₙ.BinVar[:w][g] * (stateInfo.BinVar[:w][g] - model[:w_copy][g]) + 
            (param.algorithm == :SDDiP ?
                sum(
                    πₙ.ContStateBin[:s][g][i] * (stateInfo.ContStateBin[:s][g][i] - model[:λ_copy][g, i]) for i in 1:param.κ[g]
                ) : πₙ.ContVar[:s][g] * (stateInfo.ContVar[:s][g] - model[:s_copy][g])
            ) +
            (param.algorithm == :SDDPL ?
                sum(
                    πₙ.ContAugState[:s][g][k] * (stateInfo.ContAugState[:s][g][k] - model[:augmentVar_copy][g, k]) 
                    for k in keys(stateInfo.ContAugState[:s][g]); init = 0.0
                ) : 0.0
            ) 
            for g in indexSets.G
        )
    );
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);
    negative_∇F = StateInfo(
        Dict{Any, Dict{Any, Any}}(
            :y => Dict{Any, Any}(
                g => 0. for g in indexSets.G
            ),
            :v => Dict{Any, Any}(
                g => 0. for g in indexSets.G
            ),
            :w => Dict{Any, Any}(
                g => 0. for g in indexSets.G
            )
        ), 
        nothing, 
        Dict{Any, Dict{Any, Any}}(
            :s => Dict{Any, Any}(
                g => 0. for g in indexSets.G
            )
        ), 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        param.algorithm == :SDDPL ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        k => 0.
                        for k in keys(stateInfo.ContAugState[:s][g])
                    ) 
                for g in indexSets.G
            )
        ) : nothing,
        nothing,
        param.algorithm == :SDDiP ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        i => 0.
                        for i in 1:param.κ[g]
                    ) 
                for g in indexSets.G
            )
        ) : nothing
    );
    currentInfo = CurrentInfo(  
        πₙ, 
        - F,                                                                                                                                                            ## obj function value
        Dict(
            1 => 0.
        ),                                                                                                                                                              ## constraint value
        param.algorithm == :SDDPL ?                                                                                                                            
        Dict{Symbol, Dict{Int64, Any}}(
            :s   => Dict(g => JuMP.value(model[:s_copy][g]) - stateInfo.ContVar[:s][g] for g in indexSets.G),
            :y   => Dict(g => JuMP.value(model[:y_copy][g]) - stateInfo.BinVar[:y][g]  for g in indexSets.G), 
            :v   => Dict(g => JuMP.value(model[:v_copy][g]) - stateInfo.BinVar[:v][g]  for g in indexSets.G), 
            :w   => Dict(g => JuMP.value(model[:w_copy][g]) - stateInfo.BinVar[:w][g]  for g in indexSets.G), 
            :sur => Dict(g => Dict(
                k => JuMP.value(model[:augmentVar_copy][g, k]) - stateInfo.ContAugState[:s][g][k] for k in keys(stateInfo.ContAugState[:s][g])
                ) for g in indexSets.G),
            :λ   => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(i => JuMP.value(model[:λ_copy][g, i]) - stateInfo.ContStateBin[:s][g][i] for i in 1:param.κ[g]) : nothing
                ) for g in indexSets.G
            )
        ) : 
        Dict{Symbol, Dict{Int64, Any}}(
            :s   => Dict(g => JuMP.value(model[:s_copy][g]) - stateInfo.ContVar[:s][g] for g in indexSets.G),
            :y   => Dict(g => JuMP.value(model[:y_copy][g]) - stateInfo.BinVar[:y][g]  for g in indexSets.G),
            :v   => Dict(g => JuMP.value(model[:v_copy][g]) - stateInfo.BinVar[:v][g]  for g in indexSets.G),
            :w   => Dict(g => JuMP.value(model[:w_copy][g]) - stateInfo.BinVar[:w][g]  for g in indexSets.G),
            :λ   => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(i => JuMP.value(model[:λ_copy][g, i]) - stateInfo.ContStateBin[:s][g][i] for i in 1:param.κ[g]) : nothing
                ) for g in indexSets.G
            )
        ),                                                                                                                                                              ## obj gradient
        Dict(1 => negative_∇F )                                                                                                                                                               ## constraint gradient
    );
    return (currentInfo = currentInfo, currentInfo_f = F)
end     