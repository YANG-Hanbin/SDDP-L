"""
    add_constraint(
        currentInfo::NormalizationCurrentInfo,
        modelInfo::ModelInfo;
        indexSets::IndexSets = indexSets,
        param::NamedTuple = param,
    )::Nothing

Add the bundle linearizations used by the linear-normalization dual program to
either the oracle model or the next-iterate model.
"""
function add_constraint(
    currentInfo::NormalizationCurrentInfo,
    modelInfo::ModelInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param
)::Nothing
    # add constraints     
    @constraint(
        modelInfo.model, 
        modelInfo.z .≥ currentInfo.f + 
        currentInfo.df[:obj] * (modelInfo.model[:xθ] - currentInfo.x[2]) + 
        (param.algorithm == :SDDiP ?  
            sum(
                sum(currentInfo.df[:λ][g][i] * (modelInfo.sur[g, i] - currentInfo.x[1].ContStateBin[:s][g][i]) for i in 1:param.κ[g]; init = 0.0) for g in indexSets.G
            ) : sum(currentInfo.df[:s][g] * (modelInfo.xs[g] - currentInfo.x[1].ContVar[:s][g]) for g in indexSets.G)
        ) + 
        sum(currentInfo.df[:y][g] * (modelInfo.xy[g] - currentInfo.x[1].BinVar[:y][g]) for g in indexSets.G) + 
        sum(currentInfo.df[:v][g] * (modelInfo.xv[g] - currentInfo.x[1].BinVar[:v][g]) for g in indexSets.G) + 
        sum(currentInfo.df[:w][g] * (modelInfo.xw[g] - currentInfo.x[1].BinVar[:w][g]) for g in indexSets.G) + 
        (param.algorithm == :SDDPL ?  
            sum(
                sum(currentInfo.df[:sur][g][k] * (modelInfo.sur[g, k] - currentInfo.x[1].ContAugState[:s][g][k]) for k in keys(currentInfo.df[:sur][g]); init = 0.0) for g in indexSets.G
            ) : 0.0
        )
    );

    @constraint(
        modelInfo.model, 
        [k = 1:1], 
        modelInfo.y .≥ currentInfo.G[k] + 
        (param.algorithm == :SDDiP ?  
            sum(
                sum(currentInfo.dG[k][1].ContStateBin[:s][g][j] * (modelInfo.sur[g, j] .- currentInfo.x[1].ContStateBin[:s][g][j]) for j in 1:param.κ[g]; init = 0.0) for g in indexSets.G
            ) : sum(currentInfo.dG[k][1].ContVar[:s][g] * (modelInfo.xs[g] .- currentInfo.x[1].ContVar[:s][g]) for g in keys(currentInfo.df[:s]))
        ) + 
        sum(currentInfo.dG[k][1].BinVar[:y][g] * (modelInfo.xy[g] .- currentInfo.x[1].BinVar[:y][g]) for g in indexSets.G) + 
        sum(currentInfo.dG[k][1].BinVar[:v][g] * (modelInfo.xv[g] .- currentInfo.x[1].BinVar[:v][g]) for g in indexSets.G) + 
        sum(currentInfo.dG[k][1].BinVar[:w][g] * (modelInfo.xw[g] .- currentInfo.x[1].BinVar[:w][g]) for g in indexSets.G) + 
        (param.algorithm == :SDDPL ?  
            sum(
                sum(currentInfo.dG[k][1].ContAugState[:s][g][j] * (modelInfo.sur[g, j] .- currentInfo.x[1].ContAugState[:s][g][j]) for j in keys(currentInfo.df[:sur][g]); init = 0.0) for g in indexSets.G
            ) : 0.0
        ) + 
        currentInfo.dG[k][2] * (modelInfo.model[:xθ] - currentInfo.x[2])
    );

    @constraint(
        modelInfo.model, 
        [k = 2:2], 
        modelInfo.y .≥ currentInfo.G[k] + currentInfo.dG[k][2] * (modelInfo.model[:xθ] - currentInfo.x[2])
    );

    return                                                                              
end

"""
    relu_dual_bound_value(cutGenerationInfo::NormalizedReLULagrangianCutGeneration)::Float64

Return the explicit box bound used for the ReLU-dual variables in the bundle
models. This follows the same scaling rule as the reference implementation
and therefore uses the incumbent node primal value `Q_n(x̂)` as the magnitude
reference, not the incumbent master epigraph value.
"""
function relu_dual_bound_value(
    cutGenerationInfo::NormalizedReLULagrangianCutGeneration,
)::Float64
    return RELU_DUAL_VAR_BOUND_SCALE * max(cutGenerationInfo.primal_bound, 1.0)
end

"""
    add_normalized_relu_feasibility_region!(
        model::Model,
        cutGenerationInfo::NormalizedReLULagrangianCutGeneration,
        s_plus,
        s_minus,
        y_plus,
        y_minus,
        v_plus,
        v_minus,
        w_plus,
        w_minus,
        x0;
        indexSets::IndexSets = indexSets,
    )::Nothing

Add the exact normalization constraint used by the normalized ReLU dual.
"""
function add_normalized_relu_feasibility_region!(
    model::Model,
    cutGenerationInfo::NormalizedReLULagrangianCutGeneration,
    s_plus,
    s_minus,
    y_plus,
    y_minus,
    v_plus,
    v_minus,
    w_plus,
    w_minus,
    x0;
    indexSets::IndexSets = indexSets,
    sur = nothing,
)::Nothing
    aug_term =
        sur === nothing ||
        cutGenerationInfo.uₙ.ContAugState === nothing ?
        0.0 :
        sum(
            sum(
                cutGenerationInfo.uₙ.ContAugState[:s][g][k] * sur[g, k]
                for k in keys(cutGenerationInfo.uₙ.ContAugState[:s][g]);
                init = 0.0,
            )
            for g in keys(cutGenerationInfo.uₙ.ContAugState[:s]);
            init = 0.0,
        )

    @constraint(
        model,
        relu_normalization_constraint,
        cutGenerationInfo.uₙ.StateValue * x0 +
        sum(
            cutGenerationInfo.uₙ.ContVarPlus[:s][g] * s_plus[g] +
            cutGenerationInfo.uₙ.ContVarMinus[:s][g] * s_minus[g] +
            cutGenerationInfo.uₙ.BinVarPlus[:y][g] * y_plus[g] +
            cutGenerationInfo.uₙ.BinVarMinus[:y][g] * y_minus[g] +
            cutGenerationInfo.uₙ.BinVarPlus[:v][g] * v_plus[g] +
            cutGenerationInfo.uₙ.BinVarMinus[:v][g] * v_minus[g] +
            cutGenerationInfo.uₙ.BinVarPlus[:w][g] * w_plus[g] +
            cutGenerationInfo.uₙ.BinVarMinus[:w][g] * w_minus[g]
            for g in indexSets.G
        ) +
        aug_term ≤ 1.0,
    );

    return
end

"""
    function solve_inner_minimization_problem(
        CutGenerationInfo::LinearNormalizationLagrangianCutGeneration,
        model::Model, 
        πₙ::StateInfo, 
        stateInfo::StateInfo
    )

# Arguments

    1. `CutGenerationInfo::LinearNormalizationLagrangianCutGeneration` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `πₙ::StateInfo` : the dual information
    4. `stateInfo::StateInfo` : the last stage decision
  
# Returns
    1. `currentInfo::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    CutGenerationInfo::LinearNormalizationLagrangianCutGeneration,
    model::Model, 
    πₙ₀::Float64,
    πₙ::StateInfo, 
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param
)
    @objective(
        model, 
        Min,  
        πₙ₀ * (CutGenerationInfo.primal_bound - model[:primal_objective_expression]) +
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
    F = JuMP.objective_value(model);
    # The cut constant needs a certified value of the dual function. The
    # incumbent objective is still used for the bundle gradient below.
    F_cut = JuMP.objective_bound(model);
    if !isfinite(F_cut)
        error(
            "LNC inner minimization did not return a finite objective bound " *
            "for cut generation. Status = $(termination_status(model)).",
        )
    end
    F_cut = min(F_cut, F)
    
    normalization_function = CutGenerationInfo.uₙ₀ * πₙ₀ + sum(
        πₙ.BinVar[:w][g] * CutGenerationInfo.uₙ.BinVar[:w][g] +
        πₙ.BinVar[:y][g] * CutGenerationInfo.uₙ.BinVar[:y][g] +
        πₙ.BinVar[:v][g] * CutGenerationInfo.uₙ.BinVar[:v][g] +
        (
            param.algorithm == :SDDiP ?
                sum(
                    πₙ.ContStateBin[:s][g][i] *
                    CutGenerationInfo.uₙ.ContStateBin[:s][g][i]
                    for i in 1:param.κ[g];
                    init = 0.0,
                ) :
                πₙ.ContVar[:s][g] * CutGenerationInfo.uₙ.ContVar[:s][g]
        ) +
        (
            param.algorithm == :SDDPL ?
                sum(
                    πₙ.ContAugState[:s][g][k] *
                    CutGenerationInfo.uₙ.ContAugState[:s][g][k]
                    for k in keys(CutGenerationInfo.uₙ.ContAugState[:s][g]);
                    init = 0.0,
                ) :
                0.0
        )
        for g in indexSets.G
    );

    currentInfo = NormalizationCurrentInfo(  
        (πₙ, πₙ₀),  
        - F,                                                                                                                                                            ## obj function value
        Dict(
            1 => normalization_function - 1, 
            2 => πₙ₀ + CutGenerationInfo.min_scale
        ),                                                                                                                                                             ## constraint value
        param.algorithm == :SDDPL ?                                                                                                                            
        Dict{Symbol, Any}(
            :obj => value(model[:primal_objective_expression]) - CutGenerationInfo.primal_bound,
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
        Dict{Symbol, Any}(
            :obj => value(model[:primal_objective_expression]) - CutGenerationInfo.primal_bound,
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
        Dict(1 => (CutGenerationInfo.uₙ, CutGenerationInfo.uₙ₀), 
             2 => (nothing, 1.0)
        )                                                                                                                                                               ## constraint gradient
    );

    return (currentInfo = currentInfo, currentInfo_f = F_cut)
end  

function solve_inner_minimization_problem(
    CutGenerationInfo::NormalizedReLULagrangianCutGeneration,
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
        πₙ.StateValue * model[:primal_objective_expression] +
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
        context = "the normalized ReLU oracle",
    );

    normalization_function =
        CutGenerationInfo.uₙ.StateValue * πₙ.StateValue +
        sum(
            πₙ.ContVarPlus[:s][g] * CutGenerationInfo.uₙ.ContVarPlus[:s][g] +
            πₙ.ContVarMinus[:s][g] * CutGenerationInfo.uₙ.ContVarMinus[:s][g] +
            πₙ.BinVarPlus[:y][g] * CutGenerationInfo.uₙ.BinVarPlus[:y][g] +
            πₙ.BinVarMinus[:y][g] * CutGenerationInfo.uₙ.BinVarMinus[:y][g] +
            πₙ.BinVarPlus[:v][g] * CutGenerationInfo.uₙ.BinVarPlus[:v][g] +
            πₙ.BinVarMinus[:v][g] * CutGenerationInfo.uₙ.BinVarMinus[:v][g] +
            πₙ.BinVarPlus[:w][g] * CutGenerationInfo.uₙ.BinVarPlus[:w][g] +
            πₙ.BinVarMinus[:w][g] * CutGenerationInfo.uₙ.BinVarMinus[:w][g]
            for g in indexSets.G
        ) +
        relu_aug_state_dot(πₙ, CutGenerationInfo.uₙ)

    currentInfo = ReLUDualCurrentInfo(
        πₙ,
        # The normalized dual objective is rhs - π₀ * θ̂_n.
        -F + πₙ.StateValue * CutGenerationInfo.incumbent_theta,
        Dict(1 => normalization_function - 1.0),
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
            :pi0 => CutGenerationInfo.incumbent_theta - JuMP.value(model[:primal_objective_expression]),
        ),
        Dict(
            1 => Dict{Symbol, Any}(
                :s_plus => CutGenerationInfo.uₙ.ContVarPlus[:s],
                :s_minus => CutGenerationInfo.uₙ.ContVarMinus[:s],
                :y_plus => CutGenerationInfo.uₙ.BinVarPlus[:y],
                :y_minus => CutGenerationInfo.uₙ.BinVarMinus[:y],
                :v_plus => CutGenerationInfo.uₙ.BinVarPlus[:v],
                :v_minus => CutGenerationInfo.uₙ.BinVarMinus[:v],
                :w_plus => CutGenerationInfo.uₙ.BinVarPlus[:w],
                :w_minus => CutGenerationInfo.uₙ.BinVarMinus[:w],
                :sur => CutGenerationInfo.uₙ.ContAugState,
                :pi0 => CutGenerationInfo.uₙ.StateValue,
            ),
        ),
    )

    return (currentInfo = currentInfo, currentInfo_f = F)
end

function LevelSetMethod_optimization!(
    model::Model,
    levelsetmethodOracleParam::LevelSetMethodOracleParam,
    stateInfo::StateInfo,
    CutGenerationInfo::NormalizedReLULagrangianCutGeneration;
    indexSets::IndexSets = indexSets,
    paramDemand::ParamDemand = paramDemand,
    paramOPF::ParamOPF = paramOPF,
    param::NamedTuple = param,
    param_levelsetmethod::NamedTuple = param_levelsetmethod
)
    G = indexSets.G
    iter = 1
    α = 1 / 2
    Δ = Inf

    currentInfo, currentInfo_f = solve_inner_minimization_problem(
        CutGenerationInfo,
        model,
        levelsetmethodOracleParam.x₀,
        stateInfo;
        indexSets = indexSets,
        paramOPF = paramOPF,
        param = param,
    )

    functionHistory = FunctionHistory(
        Dict(1 => currentInfo.f),
        Dict(1 => maximum(currentInfo.G[k] for k in keys(currentInfo.G))),
    )

    cutInfo = ReLUCutInfoUC(
        relu_cut_rhs(currentInfo_f, currentInfo.x, stateInfo),
        currentInfo.x,
    )
    best_dual_lower_bound = -currentInfo.f

    dual_bound_value = relu_dual_bound_value(CutGenerationInfo)

    oracleModel = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV))
    )
    MOI.set(oracleModel, MOI.Silent(), true)
    set_optimizer_attribute(oracleModel, "Threads", 1)
    set_optimizer_attribute(oracleModel, "MIPGap", param.MIPGap)
    set_optimizer_attribute(oracleModel, "TimeLimit", param.TimeLimit)
    @variable(oracleModel, z ≥ -1e8)
    @variable(oracleModel, 0.0 ≤ x0 ≤ RELU_DUAL_N0_BOUND)
    @variable(oracleModel, -dual_bound_value ≤ s_plus_oracle[G] ≤ dual_bound_value)
    @variable(oracleModel, -dual_bound_value ≤ s_minus_oracle[G] ≤ dual_bound_value)
    @variable(oracleModel, -dual_bound_value ≤ y_plus_oracle[G] ≤ dual_bound_value)
    @variable(oracleModel, -dual_bound_value ≤ y_minus_oracle[G] ≤ dual_bound_value)
    @variable(oracleModel, -dual_bound_value ≤ v_plus_oracle[G] ≤ dual_bound_value)
    @variable(oracleModel, -dual_bound_value ≤ v_minus_oracle[G] ≤ dual_bound_value)
    @variable(oracleModel, -dual_bound_value ≤ w_plus_oracle[G] ≤ dual_bound_value)
    @variable(oracleModel, -dual_bound_value ≤ w_minus_oracle[G] ≤ dual_bound_value)
    if stateInfo.ContAugState === nothing
        sur_oracle = nothing
    else
        @variable(
            oracleModel,
            -dual_bound_value ≤
            sur_oracle[g in G, k in keys(stateInfo.ContAugState[:s][g])] ≤
            dual_bound_value,
        )
    end
    @variable(oracleModel, y ≤ 0)
    @objective(oracleModel, Min, z)
    add_normalized_relu_feasibility_region!(
        oracleModel,
        CutGenerationInfo,
        s_plus_oracle,
        s_minus_oracle,
        y_plus_oracle,
        y_minus_oracle,
        v_plus_oracle,
        v_minus_oracle,
        w_plus_oracle,
        w_minus_oracle,
        x0;
        indexSets = indexSets,
        sur = sur_oracle,
    )
    oracleInfo = ReLUModelInfo(
        oracleModel,
        s_plus_oracle,
        s_minus_oracle,
        y_plus_oracle,
        y_minus_oracle,
        v_plus_oracle,
        v_minus_oracle,
        w_plus_oracle,
        w_minus_oracle,
        sur_oracle,
        x0,
        y,
        z,
    )

    nxtModel = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV))
    )
    MOI.set(nxtModel, MOI.Silent(), true)
    set_optimizer_attribute(nxtModel, "MIPGap", param.MIPGap)
    set_optimizer_attribute(nxtModel, "Threads", 1)
    set_optimizer_attribute(nxtModel, "TimeLimit", param.TimeLimit)
    @variable(nxtModel, 0.0 ≤ x0 ≤ RELU_DUAL_N0_BOUND)
    @variable(nxtModel, -dual_bound_value ≤ s_plus[G] ≤ dual_bound_value)
    @variable(nxtModel, -dual_bound_value ≤ s_minus[G] ≤ dual_bound_value)
    @variable(nxtModel, -dual_bound_value ≤ y_plus[G] ≤ dual_bound_value)
    @variable(nxtModel, -dual_bound_value ≤ y_minus[G] ≤ dual_bound_value)
    @variable(nxtModel, -dual_bound_value ≤ v_plus[G] ≤ dual_bound_value)
    @variable(nxtModel, -dual_bound_value ≤ v_minus[G] ≤ dual_bound_value)
    @variable(nxtModel, -dual_bound_value ≤ w_plus[G] ≤ dual_bound_value)
    @variable(nxtModel, -dual_bound_value ≤ w_minus[G] ≤ dual_bound_value)
    if stateInfo.ContAugState === nothing
        sur = nothing
    else
        @variable(
            nxtModel,
            -dual_bound_value ≤
            sur[g in G, k in keys(stateInfo.ContAugState[:s][g])] ≤
            dual_bound_value,
        )
    end
    @variable(nxtModel, z1)
    @variable(nxtModel, y1 ≤ 0)
    add_normalized_relu_feasibility_region!(
        nxtModel,
        CutGenerationInfo,
        s_plus,
        s_minus,
        y_plus,
        y_minus,
        v_plus,
        v_minus,
        w_plus,
        w_minus,
        x0;
        indexSets = indexSets,
        sur = sur,
    )
    nxtInfo = ReLUModelInfo(
        nxtModel,
        s_plus,
        s_minus,
        y_plus,
        y_minus,
        v_plus,
        v_minus,
        w_plus,
        w_minus,
        sur,
        x0,
        y1,
        z1,
    )

    while true
        add_constraint(currentInfo, oracleInfo; indexSets = indexSets)
        optimize!(oracleModel)
        st = termination_status(oracleModel)
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
            f_star = objective_value(oracleModel)
        else
            return ((cutInfo.rhs, cutInfo.dualInfo, cutInfo.dualInfo.StateValue), iter)
        end

        result = Δ_model_formulation(functionHistory, f_star, iter; param = param)
        Δ, a_min, a_max = result[1], result[2], result[3]

        if param_levelsetmethod.verbose
            if iter == 1
                println("------------------------------------ Iteration Info --------------------------------------")
                println("Iter |   Gap                              Objective                             Constraint")
            end
            @printf("%3d  |   %5.3g                         %5.3g                              %5.3g\n", iter, Δ, -currentInfo.f, currentInfo.G[1])
        end

        x₀ = currentInfo.x
        current_dual_lower_bound = -currentInfo.f
        if current_dual_lower_bound ≥ best_dual_lower_bound - 1e-10
            best_dual_lower_bound = current_dual_lower_bound
            cutInfo = ReLUCutInfoUC(
                relu_cut_rhs(currentInfo_f, currentInfo.x, stateInfo),
                currentInfo.x,
            )
        end

        if !(param_levelsetmethod.μ / 2 ≤ (α - a_min) / (a_max - a_min + 1e-12) ≤ 1 - param_levelsetmethod.μ / 2)
            α = round.((a_min + a_max) / 2, digits = 6)
        end

        w = α * f_star
        W = minimum(α * functionHistory.f_his[j] + (1 - α) * functionHistory.G_max_his[j] for j in 1:iter)
        level = w + param_levelsetmethod.λ * (W - w)

        if iter == 1
            @constraint(nxtModel, levelConstraint, α * z1 + (1 - α) * y1 ≤ level)
        else
            delete(nxtModel, nxtModel[:levelConstraint])
            unregister(nxtModel, :levelConstraint)
            @constraint(nxtModel, levelConstraint, α * z1 + (1 - α) * y1 ≤ level)
        end

        add_constraint(currentInfo, nxtInfo; indexSets = indexSets)
        obj_expr = sum(
            (s_plus[g] - x₀.ContVarPlus[:s][g])^2 +
            (s_minus[g] - x₀.ContVarMinus[:s][g])^2 +
            (y_plus[g] - x₀.BinVarPlus[:y][g])^2 +
            (y_minus[g] - x₀.BinVarMinus[:y][g])^2 +
            (v_plus[g] - x₀.BinVarPlus[:v][g])^2 +
            (v_minus[g] - x₀.BinVarMinus[:v][g])^2 +
            (w_plus[g] - x₀.BinVarPlus[:w][g])^2 +
            (w_minus[g] - x₀.BinVarMinus[:w][g])^2
            for g in G
        ) + (x0 - x₀.StateValue)^2
        if x₀.ContAugState !== nothing
            obj_expr += sum(
                sum(
                    (sur[g, k] - x₀.ContAugState[:s][g][k])^2
                    for k in keys(x₀.ContAugState[:s][g]);
                    init = 0.0,
                )
                for g in keys(x₀.ContAugState[:s]);
                init = 0.0,
            )
        end
        @objective(nxtModel, Min, obj_expr)
        optimize!(nxtModel)
        st = termination_status(nxtModel)
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
            ContAugState = x₀.ContAugState === nothing ?
                nothing :
                Dict{Any, Dict{Any, Dict{Any, Any}}}(
                    :s => Dict{Any, Dict{Any, Any}}(
                        g => Dict{Any, Any}(
                            k => JuMP.value(sur[g, k])
                            for k in keys(x₀.ContAugState[:s][g])
                        ) for g in keys(x₀.ContAugState[:s])
                    ),
                )
            x_nxt = ReLUDualStateInfo(
                JuMP.value(x0),
                Dict(
                    :y => Dict(g => JuMP.value(y_plus[g]) for g in G),
                    :v => Dict(g => JuMP.value(v_plus[g]) for g in G),
                    :w => Dict(g => JuMP.value(w_plus[g]) for g in G),
                ),
                Dict(
                    :y => Dict(g => JuMP.value(y_minus[g]) for g in G),
                    :v => Dict(g => JuMP.value(v_minus[g]) for g in G),
                    :w => Dict(g => JuMP.value(w_minus[g]) for g in G),
                ),
                Dict(:s => Dict(g => JuMP.value(s_plus[g]) for g in G)),
                Dict(:s => Dict(g => JuMP.value(s_minus[g]) for g in G)),
                ContAugState,
            )
        else
            return ((cutInfo.rhs, cutInfo.dualInfo, cutInfo.dualInfo.StateValue), iter)
        end

        dual_gap_scale = max(abs(best_dual_lower_bound), 1.0)
        if Δ ≤ param_levelsetmethod.threshold * dual_gap_scale || iter >= param_levelsetmethod.MaxIter
            return ((cutInfo.rhs, cutInfo.dualInfo, cutInfo.dualInfo.StateValue), iter)
        end

        oracle_update = try_update_relu_oracle_point(
            () -> solve_inner_minimization_problem(
                CutGenerationInfo,
                model,
                x_nxt,
                stateInfo;
                indexSets = indexSets,
                paramOPF = paramOPF,
                param = param,
            ),
            cutInfo,
            iter;
            normalized = true,
        )
        if !oracle_update.success
            return oracle_update.value
        end
        currentInfo, currentInfo_f = oracle_update.value
        iter += 1
        functionHistory.f_his[iter] = currentInfo.f
        functionHistory.G_max_his[iter] = maximum(currentInfo.G[k] for k in keys(currentInfo.G))
    end
end

"""
    function LevelSetMethod_optimization!(
        model::Model, 
        levelsetmethodOracleParam::LevelSetMethodOracleParam, 
        stateInfo::StateInfo,
        CutGenerationInfo::CutGeneration;
        indexSets::IndexSets = indexSets, 
        paramDemand::ParamDemand = paramDemand, 
        paramOPF::ParamOPF = paramOPF, 
        param::NamedTuple = param, param_levelsetmethod::NamedTuple = param_levelsetmethod
    )  
# Arguments

    1. `stageDecision::Dict{Symbol, Dict{Int64, Float64}}` : the decision of the last stage
    2. `f_star_value::Float64` : the optimal value of the current approximate value function
    3. `x_interior::Union{Dict{Symbol, Dict{Int64, Float64}}, Nothing}` : an interior point
    4. `x₀::Dict{Symbol, Dict{Int64, Float64}}` : the initial point of the lagrangian dual variables
    5. `model::Model` : backward model
  
# Returns
    1. `cutInfo::Array{Any,1}` : the cut information
  
"""
function LevelSetMethod_optimization!(
    model::Model, 
    levelsetmethodOracleParam::LevelSetMethodOracleParam, 
    stateInfo::StateInfo,
    CutGenerationInfo::LinearNormalizationLagrangianCutGeneration;
    indexSets::IndexSets = indexSets, 
    paramDemand::ParamDemand = paramDemand, 
    paramOPF::ParamOPF = paramOPF, 
    param::NamedTuple = param, param_levelsetmethod::NamedTuple = param_levelsetmethod
)    
    ## ==================================================== Level-set Method ============================================== ##    
    (D, G, L, B) = (indexSets.D, indexSets.G, indexSets.L, indexSets.B);
    iter = 1;
    α = 1/2;
    Δ = Inf; 

    # trajectory
    currentInfo, currentInfo_f = solve_inner_minimization_problem(
        CutGenerationInfo,
        model, 
        -max(1.0, CutGenerationInfo.min_scale),
        levelsetmethodOracleParam.x₀, 
        stateInfo;
        indexSets = indexSets,
        param = param
    );

    functionHistory = FunctionHistory(  
        Dict(1 => currentInfo.f), 
        Dict(1 => maximum(currentInfo.G[k] for k in keys(currentInfo.G)) )
    );

    function build_trivial_lnc_cut_info()
        return [
            0.0,
            deepcopy(levelsetmethodOracleParam.x₀),
            1.0,
        ]
    end

    function build_lnc_cut_info(currentInfo, currentInfo_f)
        return [
            currentInfo_f -
            currentInfo.x[2] * CutGenerationInfo.primal_bound -
            sum(
                (
                    param.algorithm == :SDDiP ?
                    sum(
                        currentInfo.x[1].ContStateBin[:s][g][i] *
                        stateInfo.ContStateBin[:s][g][i] for i in 1:param.κ[g]
                    ) : currentInfo.x[1].ContVar[:s][g] * stateInfo.ContVar[:s][g]
                ) +
                currentInfo.x[1].BinVar[:y][g] * stateInfo.BinVar[:y][g] +
                currentInfo.x[1].BinVar[:v][g] * stateInfo.BinVar[:v][g] +
                currentInfo.x[1].BinVar[:w][g] * stateInfo.BinVar[:w][g]
                for g in G
            ) - (
                param.algorithm == :SDDPL ?
                sum(
                    sum(
                        currentInfo.x[1].ContAugState[:s][g][k] *
                        stateInfo.ContAugState[:s][g][k]
                        for k in keys(stateInfo.ContAugState[:s][g]); init = 0.0
                    ) for g in G
                ) : 0.0
            ),
            currentInfo.x[1],
            -currentInfo.x[2],
        ]
    end

    function incumbent_cut_value(currentInfo, currentInfo_f)
        pi0 = -currentInfo.x[2]
        return pi0 > 0.0 ?
            (currentInfo_f - currentInfo.x[2] * CutGenerationInfo.primal_bound) / pi0 :
            -Inf
    end

    # At the incumbent state, exact LNC has F <= 0. If the solver cannot
    # certify that inequality, returning a nontrivial cut can overestimate Q.
    safe_lnc_value(currentInfo, currentInfo_f) =
        -currentInfo.x[2] >= CutGenerationInfo.min_scale &&
        currentInfo.G[1] ≤ 1e-7 &&
        currentInfo.G[2] ≤ 1e-7 &&
        currentInfo_f ≤ 1e-7

    cutInfo = safe_lnc_value(currentInfo, currentInfo_f) ?
        build_lnc_cut_info(currentInfo, currentInfo_f) :
        build_trivial_lnc_cut_info();
    best_incumbent_cut_value = safe_lnc_value(currentInfo, currentInfo_f) ?
        incumbent_cut_value(currentInfo, currentInfo_f) :
        -Inf;

    # model for oracle
    oracleModel = Model(optimizer_with_attributes(
        ()->Gurobi.Optimizer(GRB_ENV))
    ); 
    MOI.set(oracleModel, MOI.Silent(), true);
    set_optimizer_attribute(oracleModel, "Threads", 1);
    set_optimizer_attribute(oracleModel, "MIPGap", param.MIPGap);
    set_optimizer_attribute(oracleModel, "TimeLimit", param.TimeLimit);

    @variable(oracleModel, z ≥ - 1e8);
    @variable(oracleModel, xθ <= -CutGenerationInfo.min_scale);
    @variable(oracleModel, xs_oracle[G]);
    @variable(oracleModel, xy_oracle[G]);
    @variable(oracleModel, xv_oracle[G]);
    @variable(oracleModel, xw_oracle[G]);
    if param.algorithm == :SDDPL
        @variable(oracleModel, sur_oracle[g in G, k in keys(stateInfo.ContAugState[:s][g])]);
    elseif param.algorithm == :SDDP
        sur_oracle = nothing;
    elseif param.algorithm == :SDDiP
        @variable(oracleModel, sur_oracle[g in G, i in 1:param.κ[g]]);
    end
    @variable(oracleModel, y ≤ 0);

    @objective(oracleModel, Min, z);
    oracleInfo = ModelInfo(oracleModel, xs_oracle, xy_oracle, xv_oracle, xw_oracle, sur_oracle, y, z);


    nxtModel = Model(optimizer_with_attributes(
        ()->Gurobi.Optimizer(GRB_ENV))
    ); 
    MOI.set(nxtModel, MOI.Silent(), true);
    set_optimizer_attribute(nxtModel, "MIPGap", param.MIPGap);
    set_optimizer_attribute(nxtModel, "Threads", 1);
    set_optimizer_attribute(nxtModel, "TimeLimit", param.TimeLimit);
    @variable(nxtModel, xθ <= -CutGenerationInfo.min_scale);
    @variable(nxtModel, xs[G]);
    @variable(nxtModel, xy[G]);
    @variable(nxtModel, xv[G]);
    @variable(nxtModel, xw[G]);
    if param.algorithm == :SDDPL
        @variable(nxtModel, sur[g in G, k in keys(stateInfo.ContAugState[:s][g])]);
    elseif param.algorithm == :SDDP
        sur = nothing;
    elseif param.algorithm == :SDDiP
        @variable(nxtModel, sur[g in G, i in 1:param.κ[g]])
    end
    @variable(nxtModel, z1);
    @variable(nxtModel, y1 ≤ 0);
    nxtInfo = ModelInfo(nxtModel, xs, xy, xv, xw, sur, y1, z1);

    while true
        add_constraint(currentInfo, oracleInfo; indexSets = indexSets, param = param);
        optimize!(oracleModel);
        st = termination_status(oracleModel);
        if termination_status(oracleModel) == MOI.OPTIMAL
            f_star = JuMP.objective_value(oracleModel);
        else 
            # @info "Oracle Model is $(st)!"
            return (cutInfo = cutInfo, iter = iter)
        end

        # formulate alpha model
        result = Δ_model_formulation(functionHistory, f_star, iter; param = param);
        previousΔ = Δ;
        Δ, a_min, a_max = result[1], result[2], result[3];

        if param_levelsetmethod.verbose # && (iter % 30 == 0)
            if iter == 1
                println("------------------------------------ Iteration Info --------------------------------------")
                println("Iter |   Gap                              Objective                             Constraint")
            end
            @printf("%3d  |   %5.3g                         %5.3g                              %5.3g\n", iter, Δ, - currentInfo.f, currentInfo.G[1])
        end

        x₀ = currentInfo.x[1];
        candidate_incumbent_cut_value = incumbent_cut_value(currentInfo, currentInfo_f)
        if safe_lnc_value(currentInfo, currentInfo_f) &&
           isfinite(candidate_incumbent_cut_value) &&
           candidate_incumbent_cut_value > best_incumbent_cut_value + 1e-8
            cutInfo = build_lnc_cut_info(currentInfo, currentInfo_f);
            best_incumbent_cut_value = candidate_incumbent_cut_value;
        end

        # update α
        if !(param_levelsetmethod.μ/2 ≤ (α-a_min)/(a_max-a_min) .≤ 1 - param_levelsetmethod.μ/2)
            α = round.((a_min+a_max)/2, digits = 6);
        end

        # update level
        w = α * f_star;
        W = minimum( α * functionHistory.f_his[j] + (1-α) * functionHistory.G_max_his[j] for j in 1:iter);
        
        level = w + param_levelsetmethod.λ * (W - w);
        
        ## ==================================================== next iteration point ============================================== ##
        # obtain the next iteration point
        if iter == 1
            @constraint(nxtModel, levelConstraint, α * z1 + (1 - α) * y1 ≤ level);
        else 
            delete(nxtModel, nxtModel[:levelConstraint]);
            unregister(nxtModel, :levelConstraint);
            @constraint(nxtModel, levelConstraint, α * z1 + (1 - α) * y1 ≤ level);
        end
        add_constraint(currentInfo, nxtInfo; indexSets = indexSets);
        @objective(
            nxtModel,
            Min,
            sum(
                (param.algorithm == :SDDiP ?  
                    sum((sur[g, i] - x₀.ContStateBin[:s][g][i]) * (sur[g, i] - x₀.ContStateBin[:s][g][i]) for i in 1:param.κ[g]; init = 0.0
                    ) : (xs[g] - x₀.ContVar[:s][g]) * (xs[g] - x₀.ContVar[:s][g])
                ) + 
                (xy[g] - x₀.BinVar[:y][g]) * (xy[g] - x₀.BinVar[:y][g]) + 
                (xv[g] - x₀.BinVar[:v][g]) * (xv[g] - x₀.BinVar[:v][g]) + 
                (xw[g] - x₀.BinVar[:w][g]) * (xw[g] - x₀.BinVar[:w][g]) + 
                (param.algorithm == :SDDPL ?  
                    sum((sur[g, k] - x₀.ContAugState[:s][g][k]) * (sur[g, k] - x₀.ContAugState[:s][g][k]) for k in keys(x₀.ContAugState[:s][g]); init = 0.0
                    ) : 0.0 
                )
                for g in G
            ) +
            (xθ - currentInfo.x[2]) * (xθ - currentInfo.x[2])
        );
        optimize!(nxtModel);
        st = termination_status(nxtModel);
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
            BinVar = Dict{Any, Dict{Any, Any}}(
                :y => Dict{Any, Any}(
                    g => JuMP.value(xy[g]) for g in indexSets.G
                ),
                :v => Dict{Any, Any}(
                    g => JuMP.value(xv[g]) for g in indexSets.G
                ),
                :w => Dict{Any, Any}(
                    g => JuMP.value(xw[g]) for g in indexSets.G
                ),
            );
            ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
                g => JuMP.value(xs[g]) for g in indexSets.G)
            );
            if param.algorithm == :SDDP
                ContAugState = nothing; 
                ContStateBin = nothing;
            elseif param.algorithm == :SDDPL
                ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                    :s => Dict{Any, Dict{Any, Any}}(
                        g => Dict{Any, Any}(
                            k => JuMP.value(sur[g, k]) for k in keys(stateInfo.ContAugState[:s][g])
                        ) for g in indexSets.G
                    )
                );
                ContStateBin = nothing;
            elseif param.algorithm == :SDDiP
                ContAugState = nothing;
                ContStateBin = Dict{Any, Dict{Any, Any}}(
                    :s => Dict{Any, Dict{Any, Any}}(
                        g => Dict{Any, Any}(
                            i => JuMP.value(sur[g, i]) for i in 1:param.κ[g]
                        ) for g in indexSets.G
                    )
                );
            end

            x_nxt = StateInfo(
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
            x0_nxt = min(JuMP.value(xθ), -CutGenerationInfo.min_scale);
        elseif st == MOI.NUMERICAL_ERROR ||
               st == MOI.INFEASIBLE_OR_UNBOUNDED ||
               st == MOI.INFEASIBLE
            # @info "Re-compute Next Iteration Point -- change to a safe level!"
            set_normalized_rhs( levelConstraint, w + .99 * (W - w))
            optimize!(nxtModel)
            st = termination_status(nxtModel);
            if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
                BinVar = Dict{Any, Dict{Any, Any}}(
                    :y => Dict{Any, Any}(g => JuMP.value(xy[g]) for g in indexSets.G),
                    :v => Dict{Any, Any}(g => JuMP.value(xv[g]) for g in indexSets.G),
                    :w => Dict{Any, Any}(g => JuMP.value(xw[g]) for g in indexSets.G),
                );
                ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
                    g => JuMP.value(xs[g]) for g in indexSets.G)
                );
                if param.algorithm == :SDDP
                    ContAugState = nothing; 
                    ContStateBin = nothing;
                elseif param.algorithm == :SDDPL
                    ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                        :s => Dict{Any, Dict{Any, Any}}(
                            g => Dict{Any, Any}(
                                k => JuMP.value(sur[g, k]) for k in keys(stateInfo.ContAugState[:s][g])
                            ) for g in indexSets.G
                        )
                    );
                    ContStateBin = nothing;
                elseif param.algorithm == :SDDiP
                    ContAugState = nothing;
                    ContStateBin = Dict{Any, Dict{Any, Any}}(
                        :s => Dict{Any, Dict{Any, Any}}(
                            g => Dict{Any, Any}(
                                i => JuMP.value(sur[g, i]) for i in 1:param.κ[g]
                            ) for g in indexSets.G
                        )
                    );
                end

                x_nxt = StateInfo(
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
                x0_nxt = min(JuMP.value(xθ), -CutGenerationInfo.min_scale);
            else
                return (cutInfo = cutInfo, iter = iter)
            end
        else 
            return (cutInfo = cutInfo, iter = iter)
        end

        ## stop rules
        if Δ ≤ param_levelsetmethod.threshold * CutGenerationInfo.primal_bound || iter > param_levelsetmethod.MaxIter
            return (cutInfo = cutInfo, iter = iter)
        end
        ## ==================================================== end ============================================== ##
        ## save the trajectory
        currentInfo, currentInfo_f = solve_inner_minimization_problem(
            CutGenerationInfo,
            model, 
            x0_nxt, 
            x_nxt,
            stateInfo;
            indexSets = indexSets,
            param = param
        );
        iter = iter + 1;
        functionHistory.f_his[iter] = currentInfo.f;
        functionHistory.G_max_his[iter] = maximum(currentInfo.G[k] for k in keys(currentInfo.G));
    end

end
