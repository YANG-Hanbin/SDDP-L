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
function update_relu_dual_auxiliary_variables!(
    model::Model,
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    paramOPF::ParamOPF = paramOPF,
    param::NamedTuple = param,
)::Nothing
    if param.algorithm != :SDDP && param.algorithm != :SDDPL
        error("ReLU cut generation is only implemented for SDDP and SDDP-L.")
    end

    if :ReLUDualWPlus_s ∉ keys(model.obj_dict)
        @variable(model, ReLUDualWPlus_s[g in indexSets.G] ≥ 0);
        @variable(model, ReLUDualWMinus_s[g in indexSets.G] ≥ 0);
        @variable(model, ReLUDualR_s[g in indexSets.G], Bin);
        @variable(model, ReLUDualWPlus_y[g in indexSets.G] ≥ 0);
        @variable(model, ReLUDualWMinus_y[g in indexSets.G] ≥ 0);
        @variable(model, ReLUDualR_y[g in indexSets.G], Bin);

        @variable(model, ReLUDualWPlus_v[g in indexSets.G] ≥ 0);
        @variable(model, ReLUDualWMinus_v[g in indexSets.G] ≥ 0);
        @variable(model, ReLUDualR_v[g in indexSets.G], Bin);

        @variable(model, ReLUDualWPlus_w[g in indexSets.G] ≥ 0);
        @variable(model, ReLUDualWMinus_w[g in indexSets.G] ≥ 0);
        @variable(model, ReLUDualR_w[g in indexSets.G], Bin);
    end

    for key in (
        :ReLUDualDeviation_s,
        :ReLUDualPlusBound_s,
        :ReLUDualMinusBound_s,
        :ReLUDualDeviation_y,
        :ReLUDualPlusBound_y,
        :ReLUDualMinusBound_y,
        :ReLUDualDeviation_v,
        :ReLUDualPlusBound_v,
        :ReLUDualMinusBound_v,
        :ReLUDualDeviation_w,
        :ReLUDualPlusBound_w,
        :ReLUDualMinusBound_w,
    )
        if key ∈ keys(model.obj_dict)
            delete(model, [model[key][g] for g in indexSets.G]);
            unregister(model, key);
        end
    end

    @constraint(model, ReLUDualDeviation_s[g in indexSets.G],
        model[:ReLUDualWPlus_s][g] - model[:ReLUDualWMinus_s][g] == model[:s_copy][g] - stateInfo.ContVar[:s][g]
    );
    @constraint(model, ReLUDualPlusBound_s[g in indexSets.G],
        model[:ReLUDualWPlus_s][g] <= (paramOPF.smax[g] - stateInfo.ContVar[:s][g]) * model[:ReLUDualR_s][g]
    );
    @constraint(model, ReLUDualMinusBound_s[g in indexSets.G],
        model[:ReLUDualWMinus_s][g] <= stateInfo.ContVar[:s][g] * (1 - model[:ReLUDualR_s][g])
    );

    @constraint(model, ReLUDualDeviation_y[g in indexSets.G],
        model[:ReLUDualWPlus_y][g] - model[:ReLUDualWMinus_y][g] == model[:y_copy][g] - stateInfo.BinVar[:y][g]
    );
    @constraint(model, ReLUDualPlusBound_y[g in indexSets.G],
        model[:ReLUDualWPlus_y][g] <= (1.0 - stateInfo.BinVar[:y][g]) * model[:ReLUDualR_y][g]
    );
    @constraint(model, ReLUDualMinusBound_y[g in indexSets.G],
        model[:ReLUDualWMinus_y][g] <= stateInfo.BinVar[:y][g] * (1 - model[:ReLUDualR_y][g])
    );

    @constraint(model, ReLUDualDeviation_v[g in indexSets.G],
        model[:ReLUDualWPlus_v][g] - model[:ReLUDualWMinus_v][g] == model[:v_copy][g] - stateInfo.BinVar[:v][g]
    );
    @constraint(model, ReLUDualPlusBound_v[g in indexSets.G],
        model[:ReLUDualWPlus_v][g] <= (1.0 - stateInfo.BinVar[:v][g]) * model[:ReLUDualR_v][g]
    );
    @constraint(model, ReLUDualMinusBound_v[g in indexSets.G],
        model[:ReLUDualWMinus_v][g] <= stateInfo.BinVar[:v][g] * (1 - model[:ReLUDualR_v][g])
    );

    @constraint(model, ReLUDualDeviation_w[g in indexSets.G],
        model[:ReLUDualWPlus_w][g] - model[:ReLUDualWMinus_w][g] == model[:w_copy][g] - stateInfo.BinVar[:w][g]
    );
    @constraint(model, ReLUDualPlusBound_w[g in indexSets.G],
        model[:ReLUDualWPlus_w][g] <= (1.0 - stateInfo.BinVar[:w][g]) * model[:ReLUDualR_w][g]
    );
    @constraint(model, ReLUDualMinusBound_w[g in indexSets.G],
        model[:ReLUDualWMinus_w][g] <= stateInfo.BinVar[:w][g] * (1 - model[:ReLUDualR_w][g])
    );

    return
end

"""
    ReLUInnerOracleSolveError

Structured exception raised when a lifted ReLU inner minimization problem does
not return an exact oracle point. The level method may safely terminate and
reuse the best previously verified cut, but it must not continue with an
inexact subgradient.
"""
struct ReLUInnerOracleSolveError <: Exception
    context::String
    termination_status::Any
    primal_status::Any
    dual_status::Any
    result_count::Int
end

function Base.showerror(io::IO, err::ReLUInnerOracleSolveError)
    print(
        io,
        "The lifted ReLU inner minimization problem did not return an optimal ",
        "solution in ", err.context, ". ",
        "termination_status = ", err.termination_status, ", ",
        "primal_status = ", err.primal_status, ", ",
        "dual_status = ", err.dual_status, ", ",
        "result_count = ", err.result_count, ".",
    )
end

"""
    require_optimal_relu_inner_solution!(
        model::Model;
        context::AbstractString,
    )::Float64

Return the objective value of a lifted ReLU inner minimization problem only
when the solver has produced an optimal oracle point. The level method relies
on the exact oracle value together with the corresponding primal minimizer in
order to construct a valid subgradient cut.
"""
function require_optimal_relu_inner_solution!(
    model::Model;
    context::AbstractString,
)::Float64
    status = termination_status(model)
    has_solution = result_count(model) > 0
    if has_solution &&
       (status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.ALMOST_OPTIMAL)
        return JuMP.objective_value(model)
    end

    throw(ReLUInnerOracleSolveError(
        context,
        status,
        primal_status(model),
        dual_status(model),
        result_count(model),
    ))
end

"""
    return_best_relu_cut_on_oracle_failure(err, cutInfo, iter; normalized = false)

Terminate the current lifted ReLU level-method call when a later oracle query
cannot be solved to optimality within the allotted time. The returned cut is
the best cut assembled from previously verified oracle evaluations, so
validity is preserved even though the bundle method stops early.
"""
function return_best_relu_cut_on_oracle_failure(
    err::ReLUInnerOracleSolveError,
    cutInfo::ReLUCutInfoUC,
    iter::Int;
    normalized::Bool = false,
)
    if normalized
        return ((cutInfo.rhs, cutInfo.dualInfo, cutInfo.dualInfo.StateValue), iter)
    end

    return ((cutInfo.rhs, cutInfo.dualInfo, 1.0), iter)
end

function return_best_relu_cut_on_oracle_failure(
    err,
    cutInfo::ReLUCutInfoUC,
    iter::Int;
    normalized::Bool = false,
)
    rethrow(err)
end

"""
    try_update_relu_oracle_point(f, cutInfo, iter; normalized = false)

Evaluate the closure `f()` that computes the next exact ReLU oracle point. If
the underlying inner MIP only returns a heuristic incumbent, terminate the
bundle method cleanly and keep the best previously verified cut.
"""
function try_update_relu_oracle_point(
    f::Function,
    cutInfo::ReLUCutInfoUC,
    iter::Int;
    normalized::Bool = false,
)
    try
        return (success = true, value = f())
    catch err
        return (
            success = false,
            value = return_best_relu_cut_on_oracle_failure(
                err,
                cutInfo,
                iter;
                normalized = normalized,
            ),
        )
    end
end

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

"""
    solve_inner_minimization_problem(
        CutGenerationInfo::ReLULagrangianCutGeneration,
        model::Model,
        πₙ::ReLUDualStateInfo,
        stateInfo::StateInfo
    )

Solve the ReLU-dual inner minimization problem for the SDDP benchmark.
"""
function solve_inner_minimization_problem(
    CutGenerationInfo::ReLULagrangianCutGeneration,
    model::Model,
    πₙ::ReLUDualStateInfo,
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    paramOPF::ParamOPF = paramOPF,
    param::NamedTuple = param,
)
    update_relu_dual_auxiliary_variables!(
        model,
        stateInfo;
        indexSets = indexSets,
        paramOPF = paramOPF,
        param = param,
    )

    @objective(
        model,
        Min,
        model[:primal_objective_expression] +
        sum(
            πₙ.ContVarPlus[:s][g] * model[:ReLUDualWPlus_s][g] +
            πₙ.ContVarMinus[:s][g] * model[:ReLUDualWMinus_s][g] +
            πₙ.BinVarPlus[:y][g] * model[:ReLUDualWPlus_y][g] +
            πₙ.BinVarMinus[:y][g] * model[:ReLUDualWMinus_y][g] +
            πₙ.BinVarPlus[:v][g] * model[:ReLUDualWPlus_v][g] +
            πₙ.BinVarMinus[:v][g] * model[:ReLUDualWMinus_v][g] +
            πₙ.BinVarPlus[:w][g] * model[:ReLUDualWPlus_w][g] +
            πₙ.BinVarMinus[:w][g] * model[:ReLUDualWMinus_w][g]
            for g in indexSets.G
        ) +
        relu_aug_lagrangian_term(πₙ, model, stateInfo)
    );
    optimize!(model);
    F = require_optimal_relu_inner_solution!(
        model;
        context = "the regular ReLU oracle",
    );

    currentInfo = ReLUDualCurrentInfo(
        πₙ,
        -F,
        Dict(1 => 0.0),
        Dict{Symbol, Any}(
            :s_plus => Dict(g => -JuMP.value(model[:ReLUDualWPlus_s][g]) for g in indexSets.G),
            :s_minus => Dict(g => -JuMP.value(model[:ReLUDualWMinus_s][g]) for g in indexSets.G),
            :y_plus => Dict(g => -JuMP.value(model[:ReLUDualWPlus_y][g]) for g in indexSets.G),
            :y_minus => Dict(g => -JuMP.value(model[:ReLUDualWMinus_y][g]) for g in indexSets.G),
            :v_plus => Dict(g => -JuMP.value(model[:ReLUDualWPlus_v][g]) for g in indexSets.G),
            :v_minus => Dict(g => -JuMP.value(model[:ReLUDualWMinus_v][g]) for g in indexSets.G),
            :w_plus => Dict(g => -JuMP.value(model[:ReLUDualWPlus_w][g]) for g in indexSets.G),
            :w_minus => Dict(g => -JuMP.value(model[:ReLUDualWMinus_w][g]) for g in indexSets.G),
            :sur => relu_aug_value_gradient(model, stateInfo),
        ),
        Dict(
            1 => Dict{Symbol, Any}(
                :s_plus => Dict(g => 0.0 for g in indexSets.G),
                :s_minus => Dict(g => 0.0 for g in indexSets.G),
                :y_plus => Dict(g => 0.0 for g in indexSets.G),
                :y_minus => Dict(g => 0.0 for g in indexSets.G),
                :v_plus => Dict(g => 0.0 for g in indexSets.G),
                :v_minus => Dict(g => 0.0 for g in indexSets.G),
                :w_plus => Dict(g => 0.0 for g in indexSets.G),
                :w_minus => Dict(g => 0.0 for g in indexSets.G),
                :sur => zero_relu_aug_gradient(stateInfo),
            )
        )
    );
    return (currentInfo = currentInfo, currentInfo_f = F)
end
