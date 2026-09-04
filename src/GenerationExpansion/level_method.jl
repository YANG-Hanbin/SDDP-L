#############################################################################################
###########################  auxiliary functions for level set method #######################
#############################################################################################

"""
    Δ_model_formulation(functionHistory, f_star, iter)

Build and solve the auxiliary model used to compute

- Δ      : gap value
- α_min  : minimal feasible alpha
- α_max  : maximal feasible alpha

given the history of objective values and max constraint violation.

`functionHistory` is assumed to have fields:
- `obj_history[j]`     : objective value at iteration j
- `max_con_history[j]` : max constraint value at iteration j
"""
function Δ_model_formulation(
    functionHistory::FunctionHistory,
    f_star::Float64,
    iter::Int,
)::Dict{Int, Float64}

    alphaModel = Model(
        optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV)
        ),
    )
    MOI.set(alphaModel, MOI.Silent(), true)
    set_optimizer_attribute(alphaModel, "MIPGap", 1e-4)
    set_optimizer_attribute(alphaModel, "Threads", 1)
    set_optimizer_attribute(alphaModel, "TimeLimit", 10.0)
    set_optimizer_attribute(alphaModel, "FeasibilityTol", 1e-8);

    @variable(alphaModel, z)
    @variable(alphaModel, 0 ≤ α ≤ 1)

    @constraint(alphaModel, con[j = 1:iter],
        z ≤ α * (functionHistory.obj_history[j] - f_star) +
             (1 - α) * functionHistory.max_con_history[j],
    )

    ######################
    # 1. solve for Δ
    ######################
    @objective(alphaModel, Max, z)
    optimize!(alphaModel)

    st_Δ = termination_status(alphaModel)
    ps_Δ = primal_status(alphaModel)

    if st_Δ != MOI.OPTIMAL && st_Δ != MOI.LOCALLY_SOLVED
        @warn "Δ_model_formulation: gap model not optimal (status = $st_Δ, ps = $ps_Δ). Using Δ = Inf."
        Δ = Inf
    else
        Δ = objective_value(alphaModel)
    end

    ######################
    # 2. α_min
    ######################
    @constraint(alphaModel, z_nonneg, z ≥ 0)
    @objective(alphaModel, Min, α)
    optimize!(alphaModel)

    st_min = termination_status(alphaModel)
    ps_min = primal_status(alphaModel)

    if st_min != MOI.OPTIMAL && st_min != MOI.LOCALLY_SOLVED
        # @warn "Δ_model_formulation: α_min model did not solve to optimality (status = $st_min, ps = $ps_min). Using α_min = 0."
        α_min = 0.0
    else
        α_min = value(α)
    end

    ######################
    # 3. α_max
    ######################
    @objective(alphaModel, Max, α)
    optimize!(alphaModel)

    st_max = termination_status(alphaModel)
    ps_max = primal_status(alphaModel)

    if st_max != MOI.OPTIMAL && st_max != MOI.LOCALLY_SOLVED
        # @warn "Δ_model_formulation: α_max model did not solve to optimality (status = $st_max, ps = $ps_max). Using α_max = 1."
        α_max = 1.0
    else
        α_max = value(α)
    end

    return Dict(
        1 => Δ,
        2 => α_min,
        3 => α_max,
    )
end


"""
Add linearization constraints for f and g_k at the current point.

- `currentInfo` :
    * `currentInfo.obj`                 – f(xⱼ)
    * `currentInfo.con[k]`             – g_k(xⱼ)
    * `currentInfo.d_obj[:St]`         – ∂f/∂St at xⱼ
    * `currentInfo.d_obj[:region_indicator][g][k]`
    * `currentInfo.d_con[k][:St]`
    * `currentInfo.d_con[k][:region_indicator][g][km]`
    * `currentInfo.var`                – struct with fields:
        - `IntVar`       – state vector xⱼ (for SDDP/SDDPL)
        - `IntVarLeaf`   – nested dict for surrogate vars
        - `IntVarBinaries` – binary state (used only by the SDDiP path)

- `model` :
    * variables `:x`, `:x_sur`, `:z`, and `:y` are already attached to `model`
"""
function add_constraint(
    currentInfo::CurrentInfo,
    model::Model;
    binaryInfo = binaryInfo
)::Nothing
    m  = length(currentInfo.con)
    xⱼ = currentInfo.var

    # shorthand
    x_var   = model[:x]
    x_sur   = haskey(model, :x_sur) ? model[:x_sur] : nothing

    # ---------- f(x) linearization ----------
    # z ≥ f(xⱼ) + ∇f_St(xⱼ)' (x - xⱼ) + sum_g,k ∂f/∂sur[g,k] * (x_sur[g,k] - xⱼ_sur[g,k])
    rhs = currentInfo.obj
    if :St ∈ keys(currentInfo.d_obj)
        rhs += currentInfo.d_obj[:St]' * (x_var .- xⱼ.IntVar)
    end
    if :region_indicator ∈ keys(currentInfo.d_obj)
        for g in 1:binaryInfo.d
            for k in keys(currentInfo.d_obj[:region_indicator][g])
                rhs += currentInfo.d_obj[:region_indicator][g][k] * (x_sur[g, k] - xⱼ.IntVarLeaf[g][k])
            end
        end
    end
    if :Lt ∈ keys(currentInfo.d_obj)
        rhs += currentInfo.d_obj[:Lt]' * (x_var .- xⱼ.IntVarBinaries)
    end
    
    @constraint(
        model,
        model[:z] ≥ rhs
    )

    # ---------- g_k(x) linearization ----------
    # y ≥ g_k(xⱼ) + ∂g_k/∂St (x - xⱼ) + surrogate contribution
    @constraint(
        model,
        [k = 1:m],
        model[:y] ≥
        currentInfo.con[k] + ( param.algorithm == :SDDiP ? 
        currentInfo.d_con[k][:Lt]' * (x_var .- xⱼ.IntVarBinaries) : 
        currentInfo.d_con[k][:St]' * (x_var .- xⱼ.IntVar) 
        ) + ( param.algorithm == :SDDPL ?
        sum(
            currentInfo.d_con[k][:region_indicator][g][km] *
            (x_sur[g, km] - xⱼ.IntVarLeaf[g][km])
            for g in 1:binaryInfo.d, km in keys(currentInfo.d_obj[:region_indicator][g])
        ) : 0.0)
    )

    return
end

"""
Add linearization constraints for the ReLU-dual bundle model.
"""
function add_constraint(
    currentInfo::ReLUCurrentInfo,
    model::Model,
)::Nothing
    x_plus = model[:x_plus]
    x_minus = model[:x_minus]
    πⱼ = currentInfo.var
    leaf_obj_term = relu_leaf_bundle_term(
        get(currentInfo.d_obj, :leaf, nothing),
        model,
        πⱼ.IntVarLeaf,
    )

    rhs = currentInfo.obj +
          currentInfo.d_obj[:plus]' * (x_plus .- πⱼ.IntVarPlus) +
          currentInfo.d_obj[:minus]' * (x_minus .- πⱼ.IntVarMinus) +
          leaf_obj_term

    if :x0 ∈ keys(model.obj_dict)
        rhs += currentInfo.d_obj[:pi0] * (model[:x0] - πⱼ.StateValue)
    end

    @constraint(
        model,
        model[:z] ≥ rhs,
    )

    @constraint(
        model,
        [k in 1:length(currentInfo.con)],
        model[:y] ≥ currentInfo.con[k] +
        currentInfo.d_con[k][:plus]' * (x_plus .- πⱼ.IntVarPlus) +
        currentInfo.d_con[k][:minus]' * (x_minus .- πⱼ.IntVarMinus) +
        relu_leaf_bundle_term(
            get(currentInfo.d_con[k], :leaf, nothing),
            model,
            πⱼ.IntVarLeaf,
        ) +
        (:x0 ∈ keys(model.obj_dict) ?
            currentInfo.d_con[k][:pi0] * (model[:x0] - πⱼ.StateValue) :
            0.0),
    )

    return
end

function relu_leaf_bundle_term(
    gradient,
    model::Model,
    center,
)
    if gradient === nothing || center === nothing || :x_leaf ∉ keys(model.obj_dict)
        return 0.0
    end

    return sum(
        sum(
            gradient[g][k] * (model[:x_leaf][g, k] - center[g][k])
            for k in keys(gradient[g]);
            init = 0.0,
        )
        for g in keys(gradient);
        init = 0.0,
    )
end

"""
    relu_dual_bound_value(cutTypeInfo)

Return the symmetric box bound used for the ReLU-dual variables in the
bundle models. It uses `dual_var_bound_scale * (scenario_objective -
lower_bound)`, with lower bound `0.0` for the generation-expansion
subproblems.
"""
function relu_dual_bound_value(
    cutTypeInfo::ReLULagrangianCutGenerationProgram,
)::Float64
    return RELU_DUAL_VAR_BOUND_SCALE * max(cutTypeInfo.primal_bound, 1.0)
end

function relu_dual_bound_value(
    cutTypeInfo::NormalizedReLULagrangianCutGenerationProgram,
)::Float64
    return RELU_DUAL_VAR_BOUND_SCALE * max(cutTypeInfo.primal_bound, 1.0)
end

"""
    add_normalized_relu_feasibility_region!(model, cutTypeInfo)

Add the exact normalization constraint used by the normalized ReLU dual. The
constraint is linear, so it can be enforced explicitly and exactly.
"""
function add_normalized_relu_feasibility_region!(
    model::Model,
    cutTypeInfo::NormalizedReLULagrangianCutGenerationProgram,
)::Nothing
    leaf_term =
        cutTypeInfo.NormalizationInfo.IntVarLeaf === nothing ||
        :x_leaf ∉ keys(model.obj_dict) ?
        0.0 :
        sum(
            sum(
                cutTypeInfo.NormalizationInfo.IntVarLeaf[g][k] *
                model[:x_leaf][g, k]
                for k in keys(cutTypeInfo.NormalizationInfo.IntVarLeaf[g]);
                init = 0.0,
            )
            for g in keys(cutTypeInfo.NormalizationInfo.IntVarLeaf);
            init = 0.0,
        )

    @constraint(
        model,
        relu_normalization_constraint,
        cutTypeInfo.NormalizationInfo.IntVarPlus' * model[:x_plus] +
        cutTypeInfo.NormalizationInfo.IntVarMinus' * model[:x_minus] +
        cutTypeInfo.NormalizationInfo.StateValue * model[:x0] +
        leaf_term ≤ 1.0,
    )
    return
end

"""
    relu_gradient_statistics(gradientInfo::Dict{Symbol, Any})::NamedTuple

Return compact infinity-norm summaries of the ReLU objective/constraint
subgradients for verbose tracing.
"""
function relu_gradient_statistics(
    gradientInfo::Dict{Symbol, Any},
)::NamedTuple
    plus_inf = maximum(abs.(gradientInfo[:plus]))
    minus_inf = maximum(abs.(gradientInfo[:minus]))
    pi0 = :pi0 ∈ keys(gradientInfo) ? gradientInfo[:pi0] : 0.0
    return (plus_inf = plus_inf, minus_inf = minus_inf, pi0 = pi0)
end

"""
    relu_dual_lower_bound(cutInfo, cutTypeInfo)

Return the dual lower-bound value associated with the incumbent ReLU cut.

- Regular ReLU: the cut right-hand side equals the dual objective value.
- Normalized ReLU: the dual objective is `rhs - π₀ * θ̂`.
"""
function relu_dual_lower_bound(
    cutInfo::ReLUCutInfo,
    cutTypeInfo::ReLULagrangianCutGenerationProgram,
)::Float64
    return cutInfo.rhs
end

function relu_dual_lower_bound(
    cutInfo::ReLUCutInfo,
    cutTypeInfo::NormalizedReLULagrangianCutGenerationProgram,
)::Float64
    return cutInfo.rhs - cutInfo.dualInfo.StateValue * cutTypeInfo.incumbent_theta
end

####################################################################################################
## -------------------------------------- Level-set main routine -------------------------------- ##
####################################################################################################

function LevelSetMethod_optimization!(
    stageModel::StageModel,
    cutGenerationParamInfo::CutGenerationParamInfo,
    cutTypeInfo::CutGenerationProgram;
    stageData::StageData = stageData,
    binaryInfo::BinaryInfo = binaryInfo,
    param::SDDPParam = param,
)::Any

    if is_adaptive_level_cut(cutGenerationParamInfo.cutSelection)
        return gep_adaptive_level_method!(
            stageModel,
            cutGenerationParamInfo,
            cutTypeInfo;
            stageData = stageData,
            binaryInfo = binaryInfo,
            param = param,
        )
    end

    # unpack parameters (μ > 0 larger means more aggressive α adjustment)
    μ          = cutGenerationParamInfo.μ
    λ          = cutGenerationParamInfo.λ
    threshold  = cutGenerationParamInfo.gapBM        # not directly used below
    nxt_bound  = cutGenerationParamInfo.nxt_bound
    iterBM     = cutGenerationParamInfo.iterBM       # not directly used below

    stateInfo  = cutGenerationParamInfo.stateInfo

    A, n, d    = binaryInfo.A, binaryInfo.n, binaryInfo.d

    # initial cut center (πₙ)
    πₙ = cutGenerationParamInfo.πₙ

    ############################################
    ## initial inner minimization (current point)
    ############################################
    iter = 1
    α    = 0.5

    currentInfo, currentInfo_f = solve_inner_minimization_problem(
        cutTypeInfo,
        stageModel.model,
        πₙ,
        stateInfo;
        param = param,
    )

    # initial cut info: [constant term, "point" info]
    cutInfo = [
        currentInfo_f -
        (param.algorithm == :SDDiP ?
            currentInfo.var.IntVarBinaries' * stateInfo.IntVarBinaries : currentInfo.var.IntVar' * stateInfo.IntVar
        ) -
        (param.algorithm == :SDDPL ?
            sum(
                sum(
                    currentInfo.var.IntVarLeaf[g][k] * stateInfo.IntVarLeaf[g][k]
                    for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
                ) for g in 1:binaryInfo.d
            ) : 0.0),
        currentInfo.var,
    ]

    # function history (f, max g_k) for level-set Δ model
    functionHistory = FunctionHistory(
        Dict(1 => currentInfo.obj),
        Dict(1 => maximum(currentInfo.con[k] for k in keys(currentInfo.con))),
    )

    ############################################
    ## oracle model (computes f_star)
    ############################################
    oracleModel = Model(
        optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV)
        ),
    )
    MOI.set(oracleModel, MOI.Silent(), !param.verbose)
    set_optimizer_attribute(oracleModel, "MIPGap", param.gapSDDP)
    set_optimizer_attribute(oracleModel, "Threads", 1)
    set_optimizer_attribute(oracleModel, "TimeLimit", param.timeSDDP)

    @variable(oracleModel, z ≥ -param.nxt_bound)
    if param.algorithm == :SDDPL
        @variable(oracleModel, x[i = 1:d])
        @variable(oracleModel, x_sur[g in 1:d, k in keys(stateInfo.IntVarLeaf[g])])
    elseif param.algorithm == :SDDP
        @variable(oracleModel, x[i = 1:d])
    elseif param.algorithm == :SDDiP
        @variable(oracleModel, x[i = 1:n])
    end
    @variable(oracleModel, y ≤ 0)
    @objective(oracleModel, Min, z)

    ############################################
    ## next-iterate model (nxtModel)
    ############################################
    nxtModel = Model(
        optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV)
        ),
    )
    MOI.set(nxtModel, MOI.Silent(), !param.verbose)
    set_optimizer_attribute(nxtModel, "MIPGap", param.gapSDDP)
    set_optimizer_attribute(nxtModel, "Threads", 1)
    set_optimizer_attribute(nxtModel, "TimeLimit", param.timeSDDP)

    @variable(nxtModel, z)
    if param.algorithm == :SDDPL
        @variable(nxtModel, x[i = 1:d])
        @variable(nxtModel, x_sur[g in 1:d, k in keys(stateInfo.IntVarLeaf[g])])
    elseif param.algorithm == :SDDP
        @variable(nxtModel, x[i = 1:d])
    elseif param.algorithm == :SDDiP
        @variable(nxtModel, x[i = 1:n])
    end
    @variable(nxtModel, y ≤ 0)
    @objective(nxtModel, Min, z)

    ############################################
    ## Level-set loop
    ############################################
    Δ   = Inf
    τₖ  = 1.0
    τₘ  = 0.5
    μₖ  = 1.0

    while true
        # --- Oracle step: compute f_star ---
        add_constraint(currentInfo, oracleModel)
        optimize!(oracleModel)
        st = termination_status(oracleModel)

        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
            f_star = objective_value(oracleModel)
        else
            return (cutInfo = cutInfo, iter = iter)
        end

        # --- Level-set Δ model: compute Δ, α_min, α_max ---
        result    = Δ_model_formulation(functionHistory, f_star, iter)
        previousΔ = Δ
        Δ         = result[1]
        a_min     = result[2]
        a_max     = result[3]

        # print progress (optional)
        if param.verbose
            if iter == 1
                println("-------------- Level-set Iteration Info --------------")
                println("Iter |   Gap               Objective                Constraint")
            end
            @printf(
                "%3d  |   %5.3g           %5.3g                 %5.3g\n",
                iter,
                Δ,
                -currentInfo.obj,
                currentInfo.con[1],
            )
        end

        # --- update x₀ and cutInfo if Δ improved ---
        x₀ = currentInfo.var
        if round(previousΔ; digits = 8) > round(Δ; digits = 8)
            τₖ  = μₖ * τₖ
            cutInfo = [
                currentInfo_f -
                (param.algorithm == :SDDiP ?
                    currentInfo.var.IntVarBinaries' * stateInfo.IntVarBinaries :
                    currentInfo.var.IntVar'          * stateInfo.IntVar
                ) -
                (param.algorithm == :SDDPL ?
                    sum(
                        sum(
                            currentInfo.var.IntVarLeaf[g][k] * stateInfo.IntVarLeaf[g][k]
                            for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
                        ) for g in 1:binaryInfo.d
                    ) : 0.0),
                currentInfo.var,
            ]
        else
            τₖ = (τₖ + τₘ) / 2
        end

        # --- update α ---
        denom = a_max - a_min
        if denom ≈ 0.0
            # avoid numerical issues
            α = (a_min + a_max) / 2
        else
            ratio = (α - a_min) / denom
            if !(μ / 2 ≤ ratio ≤ 1 - μ / 2)
                α = (a_min + a_max) / 2
            end
        end

        # --- update level ---
        w = α * f_star
        W = minimum(
            α * functionHistory.obj_history[j] +
            (1 - α) * functionHistory.max_con_history[j] for j in 1:iter
        )

        λ = iter ≤ 10 ? 0.05 : 0.1
        λ = iter ≥ 20 ? 0.2 : λ
        λ = iter ≥ 40 ? 0.3 : λ
        λ = iter ≥ 50 ? 0.4 : λ
        λ = iter ≥ 60 ? 0.5 : λ
        λ = iter ≥ 70 ? 0.6 : λ
        λ = iter ≥ 80 ? 0.7 : λ
        λ = iter ≥ 90 ? 0.8 : λ

        level = w + λ * (W - w)

        ############################################
        ## Next iteration point (nxtModel)
        ############################################
        # level constraint: α z + (1 - α) y ≤ level
        if iter == 1
            @constraint(
                nxtModel,
                level_constraint,
                α * nxtModel[:z] + (1 - α) * nxtModel[:y] ≤ level,
            )
        else
            delete(nxtModel, nxtModel[:level_constraint])
            unregister(nxtModel, :level_constraint)
            @constraint(
                nxtModel,
                level_constraint,
                α * nxtModel[:z] + (1 - α) * nxtModel[:y] ≤ level,
            )
        end

        add_constraint(currentInfo, nxtModel)

        # --- proximity objective: stay close to x₀ ---
        # Base term: distance in the original state space whenever it exists.
        obj_expr = x₀.IntVar === nothing ?
            zero(A[1, 1]) :                       # some 0.0 scalar
            sum((nxtModel[:x] .- x₀.IntVar).^2)

        # if binary state exists, add that distance term
        if x₀.IntVarBinaries !== nothing
            obj_expr += sum((nxtModel[:x] .- x₀.IntVarBinaries).^2)
        end

        # if leaf representation exists, add surrogate part
        if x₀.IntVarLeaf !== nothing
            obj_expr += sum(
                sum(
                    (nxtModel[:x_sur][g, k] - x₀.IntVarLeaf[g][k])^2
                    for k in keys(x₀.IntVarLeaf[g])
                ) for g in 1:d
            )
        end

        @objective(nxtModel, Min, obj_expr)
        optimize!(nxtModel)
        st = termination_status(nxtModel)

        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
            πₙ = StageInfo(
                -1.,
                nothing,
                x₀.IntVar === nothing ?
                    nothing :
                    value.(nxtModel[:x]),
                x₀.IntVarLeaf === nothing ?
                    nothing :
                    Dict(
                        g => Dict(
                            k => value(nxtModel[:x_sur][g, k])
                            for k in keys(x₀.IntVarLeaf[g])
                        ) for g in keys(x₀.IntVarLeaf)
                    ),
                x₀.IntVarBinaries === nothing ?
                    nothing :
                    value.(nxtModel[:x]),
            )
            λₖ = abs(dual(level_constraint))
            μₖ = λₖ + 1
        else
            return (cutInfo = cutInfo, iter = iter)
        end

        # --- stopping rule ---
        if Δ ≤ abs(cutGenerationParamInfo.stateInfo.StateValue) * threshold || iter > iterBM
            return (cutInfo = cutInfo, iter = iter)
        end

        ############################################
        ## prepare next iteration
        ############################################
        currentInfo, currentInfo_f = solve_inner_minimization_problem(
            cutTypeInfo,
            stageModel.model,
            πₙ,
            stateInfo;
            param = param,
        )

        iter += 1
        functionHistory.obj_history[iter] = currentInfo.obj
        functionHistory.max_con_history[iter] = 
        maximum(currentInfo.con[k] for k in keys(currentInfo.con))
    end

end

function LevelSetMethod_optimization!(
    stageModel::StageModel,
    cutGenerationParamInfo::CutGenerationParamInfo,
    cutTypeInfo::Union{
        ReLULagrangianCutGenerationProgram,
        NormalizedReLULagrangianCutGenerationProgram,
    };
    stageData::StageData = stageData,
    binaryInfo::BinaryInfo = binaryInfo,
    param::SDDPParam = param,
)::Any

    μ          = cutGenerationParamInfo.μ
    threshold  = cutGenerationParamInfo.gapBM
    iterBM     = cutGenerationParamInfo.iterBM
    stateInfo  = cutGenerationParamInfo.stateInfo

    normalized = cutTypeInfo isa NormalizedReLULagrangianCutGenerationProgram
    dual_bound_value = relu_dual_bound_value(cutTypeInfo)
    z_lower_bound = normalized ? NORMALIZED_RELU_DUAL_OBJECTIVE_BOUND : param.nxt_bound

    iter = 1
    α    = 0.5
    Δ    = Inf
    τₖ   = 1.0
    τₘ   = 0.5
    μₖ   = 1.0

    πₙ = cutGenerationParamInfo.πₙ
    currentInfo, currentInfo_f = solve_inner_minimization_problem(
        cutTypeInfo,
        stageModel.model,
        πₙ,
        stateInfo;
        param = param,
    )

    cutInfo = ReLUCutInfo(
        relu_cut_rhs(currentInfo_f, currentInfo.var, stateInfo),
        currentInfo.var,
    )
    best_dual_lower_bound = -currentInfo.obj

    functionHistory = FunctionHistory(
        Dict(1 => currentInfo.obj),
        Dict(1 => maximum(currentInfo.con[k] for k in keys(currentInfo.con))),
    )

    oracleModel = Model(
        optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV)
        ),
    )
    MOI.set(oracleModel, MOI.Silent(), !param.verbose)
    set_optimizer_attribute(oracleModel, "MIPGap", param.gapSDDP)
    set_optimizer_attribute(oracleModel, "Threads", 1)
    set_optimizer_attribute(oracleModel, "TimeLimit", param.timeSDDP)

    @variable(oracleModel, z ≥ -z_lower_bound)
    @variable(oracleModel, -dual_bound_value ≤ x_plus[i = 1:length(stateInfo.IntVar)] ≤ dual_bound_value)
    @variable(oracleModel, -dual_bound_value ≤ x_minus[i = 1:length(stateInfo.IntVar)] ≤ dual_bound_value)
    if stateInfo.IntVarLeaf !== nothing
        @variable(
            oracleModel,
            -dual_bound_value ≤
            x_leaf[g in keys(stateInfo.IntVarLeaf), k in keys(stateInfo.IntVarLeaf[g])] ≤
            dual_bound_value,
        )
    end
    if normalized
        @variable(oracleModel, 0.0 ≤ x0 ≤ RELU_DUAL_N0_BOUND)
    end
    @variable(oracleModel, y ≤ 0)
    @objective(oracleModel, Min, z)
    if normalized
        add_normalized_relu_feasibility_region!(oracleModel, cutTypeInfo)
    end

    nxtModel = Model(
        optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV)
        ),
    )
    MOI.set(nxtModel, MOI.Silent(), !param.verbose)
    set_optimizer_attribute(nxtModel, "MIPGap", param.gapSDDP)
    set_optimizer_attribute(nxtModel, "Threads", 1)
    set_optimizer_attribute(nxtModel, "TimeLimit", param.timeSDDP)

    @variable(nxtModel, z ≥ -z_lower_bound)
    @variable(nxtModel, -dual_bound_value ≤ x_plus[i = 1:length(stateInfo.IntVar)] ≤ dual_bound_value)
    @variable(nxtModel, -dual_bound_value ≤ x_minus[i = 1:length(stateInfo.IntVar)] ≤ dual_bound_value)
    if stateInfo.IntVarLeaf !== nothing
        @variable(
            nxtModel,
            -dual_bound_value ≤
            x_leaf[g in keys(stateInfo.IntVarLeaf), k in keys(stateInfo.IntVarLeaf[g])] ≤
            dual_bound_value,
        )
    end
    if normalized
        @variable(nxtModel, 0.0 ≤ x0 ≤ RELU_DUAL_N0_BOUND)
    end
    @variable(nxtModel, y ≤ 0)
    @objective(nxtModel, Min, z)
    if normalized
        add_normalized_relu_feasibility_region!(nxtModel, cutTypeInfo)
    end

    while true
        add_constraint(currentInfo, oracleModel)
        optimize!(oracleModel)
        st = termination_status(oracleModel)

        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
            f_star = objective_value(oracleModel)
        else
            return (cutInfo = cutInfo, iter = iter)
        end

        result    = Δ_model_formulation(functionHistory, f_star, iter)
        previousΔ = Δ
        Δ         = result[1]
        a_min     = result[2]
        a_max     = result[3]

        if param.verbose
            if iter == 1
                println("-------------- Level-set Iteration Info --------------")
                println("Iter |   Gap               Objective                Constraint")
            end
            @printf(
                "%3d  |   %5.3g           %5.3g                 %5.3g\n",
                iter,
                Δ,
                -currentInfo.obj,
                maximum(values(currentInfo.con)),
            )

            obj_grad_info = relu_gradient_statistics(currentInfo.d_obj)
            con_grad_info = relu_gradient_statistics(currentInfo.d_con[1])
            @printf(
                "      grad-f: |plus|∞ = %9.3g, |minus|∞ = %9.3g, pi0 = %9.3g\n",
                obj_grad_info.plus_inf,
                obj_grad_info.minus_inf,
                obj_grad_info.pi0,
            )
            @printf(
                "      grad-g: |plus|∞ = %9.3g, |minus|∞ = %9.3g, pi0 = %9.3g\n",
                con_grad_info.plus_inf,
                con_grad_info.minus_inf,
                con_grad_info.pi0,
            )
        end

        x₀ = currentInfo.var
        if round(previousΔ; digits = 8) > round(Δ; digits = 8)
            τₖ = μₖ * τₖ
        else
            τₖ = (τₖ + τₘ) / 2
        end

        current_dual_lower_bound = -currentInfo.obj
        if current_dual_lower_bound ≥ best_dual_lower_bound - 1e-10
            best_dual_lower_bound = current_dual_lower_bound
            cutInfo = ReLUCutInfo(
                relu_cut_rhs(currentInfo_f, currentInfo.var, stateInfo),
                currentInfo.var,
            )
        end

        denom = a_max - a_min
        if denom ≈ 0.0
            α = (a_min + a_max) / 2
        else
            ratio = (α - a_min) / denom
            if !(μ / 2 ≤ ratio ≤ 1 - μ / 2)
                α = (a_min + a_max) / 2
            end
        end

        w = α * f_star
        W = minimum(
            α * functionHistory.obj_history[j] +
            (1 - α) * functionHistory.max_con_history[j] for j in 1:iter
        )

        λ = iter ≤ 10 ? 0.05 : 0.1
        λ = iter ≥ 20 ? 0.2 : λ
        λ = iter ≥ 40 ? 0.3 : λ
        λ = iter ≥ 50 ? 0.4 : λ
        λ = iter ≥ 60 ? 0.5 : λ
        λ = iter ≥ 70 ? 0.6 : λ
        λ = iter ≥ 80 ? 0.7 : λ
        λ = iter ≥ 90 ? 0.8 : λ

        level = w + λ * (W - w)

        if iter == 1
            @constraint(
                nxtModel,
                level_constraint,
                α * nxtModel[:z] + (1 - α) * nxtModel[:y] ≤ level,
            )
        else
            delete(nxtModel, nxtModel[:level_constraint])
            unregister(nxtModel, :level_constraint)
            @constraint(
                nxtModel,
                level_constraint,
                α * nxtModel[:z] + (1 - α) * nxtModel[:y] ≤ level,
            )
        end

        add_constraint(currentInfo, nxtModel)

        obj_expr =
            sum((nxtModel[:x_plus] .- x₀.IntVarPlus).^2) +
            sum((nxtModel[:x_minus] .- x₀.IntVarMinus).^2)
        if x₀.IntVarLeaf !== nothing
            obj_expr += sum(
                sum(
                    (nxtModel[:x_leaf][g, k] - x₀.IntVarLeaf[g][k])^2
                    for k in keys(x₀.IntVarLeaf[g]);
                    init = 0.0,
                )
                for g in keys(x₀.IntVarLeaf);
                init = 0.0,
            )
        end
        if normalized
            obj_expr += (nxtModel[:x0] - x₀.StateValue)^2
        end

        @objective(nxtModel, Min, obj_expr)
        optimize!(nxtModel)
        st = termination_status(nxtModel)

        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
            πₙ = ReLUDualStageInfo(
                normalized ? JuMP.value(nxtModel[:x0]) : nothing,
                JuMP.value.(nxtModel[:x_plus]),
                JuMP.value.(nxtModel[:x_minus]),
                x₀.IntVarLeaf === nothing ?
                    nothing :
                    Dict(
                        g => Dict(
                            k => JuMP.value(nxtModel[:x_leaf][g, k])
                            for k in keys(x₀.IntVarLeaf[g])
                        ) for g in keys(x₀.IntVarLeaf)
                    ),
            )
            λₖ = abs(dual(level_constraint))
            μₖ = λₖ + 1
        else
            return (cutInfo = cutInfo, iter = iter)
        end

        dual_gap_scale = max(abs(best_dual_lower_bound), 1.0)
        if Δ ≤ dual_gap_scale * threshold || iter >= iterBM
            return (cutInfo = cutInfo, iter = iter)
        end

        currentInfo, currentInfo_f = solve_inner_minimization_problem(
            cutTypeInfo,
            stageModel.model,
            πₙ,
            stateInfo;
            param = param,
        )

        iter += 1
        functionHistory.obj_history[iter] = currentInfo.obj
        functionHistory.max_con_history[iter] =
            maximum(currentInfo.con[k] for k in keys(currentInfo.con))
    end
end

function LevelSetMethod_optimization!(
    stageModel::StageModel,
    cutGenerationParamInfo::CutGenerationParamInfo,
    cutTypeInfo::StrengthenedBendersCutGenerationProgram;
    stageData::StageData = stageData,
    binaryInfo::BinaryInfo = binaryInfo,
    param::SDDPParam = param,
)::Any
    stateInfo  = cutGenerationParamInfo.stateInfo
    # initial cut center (πₙ)
    πₙ = cutTypeInfo.πₙ

    currentInfo, currentInfo_f = solve_inner_minimization_problem(
        cutTypeInfo,
        stageModel.model,
        πₙ,
        stateInfo;
        param = param,
    )

    return (
        cutInfo = [
        currentInfo_f -
        (param.algorithm == :SDDiP ?
            currentInfo.var.IntVarBinaries' * stateInfo.IntVarBinaries : currentInfo.var.IntVar' * stateInfo.IntVar
        ) -
        (param.algorithm == :SDDPL ?
            sum(
                sum(
                    currentInfo.var.IntVarLeaf[g][k] * stateInfo.IntVarLeaf[g][k]
                    for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
                ) for g in 1:binaryInfo.d
            ) : 0.0),
        currentInfo.var,
        ],
        iter = 1,
    )
end


"""
Add linearization constraints for f and g_k at the current point.

- `currentInfo` :
    * `currentInfo.obj`                 – f(xⱼ)
    * `currentInfo.con[k]`             – g_k(xⱼ)
    * `currentInfo.d_obj[:obj]`         – ∂f/∂St at xⱼ
    * `currentInfo.d_obj[:St]`         – ∂f/∂St at xⱼ
    * `currentInfo.d_obj[:Lt]`         – ∂f/∂St at xⱼ
    * `currentInfo.d_obj[:region_indicator][g][k]`
    * `currentInfo.d_con[k][:obj]`
    * `currentInfo.d_con[k][:St]`
    * `currentInfo.d_con[k][:Lt]`
    * `currentInfo.d_con[k][:region_indicator][g][km]`
    * `currentInfo.var`                – struct with fields:
        - `StateValue`   – coeff vector θ  
        - `IntVar`       – coeff vector xⱼ (for SDDP/SDDPL)
        - `IntVarLeaf`   – nested dict for surrogate vars
        - `IntVarBinaries` – binary coefficient vector used by the SDDiP path

- `model` :
    * variables `:x`, `:x_sur`, `:z`, and `:y` are already attached to `model`
"""
function add_constraint!(
    currentInfo::CurrentInfo,
    model::Model;
    binaryInfo = binaryInfo
)::Nothing
    xⱼ = currentInfo.var

    # shorthand
    x_var   = model[:x]
    x_sur   = haskey(model, :x_sur) ? model[:x_sur] : nothing

    # ---------- f(x) linearization ----------
    # z ≥ f(xⱼ) + ∇f_St(xⱼ)' (x - xⱼ) + sum_g,k ∂f/∂sur[g,k] * (x_sur[g,k] - xⱼ_sur[g,k])
    rhs = currentInfo.obj + currentInfo.d_obj[:obj] * (model[:xθ] - xⱼ.StateValue) 
    if :St ∈ keys(currentInfo.d_obj)
        rhs += currentInfo.d_obj[:St]' * (x_var .- xⱼ.IntVar)
    end
    if :region_indicator ∈ keys(currentInfo.d_obj)
        for g in 1:binaryInfo.d
            for k in keys(currentInfo.d_obj[:region_indicator][g])
                rhs += currentInfo.d_obj[:region_indicator][g][k] * (x_sur[g, k] - xⱼ.IntVarLeaf[g][k])
            end
        end
    end
    if :Lt ∈ keys(currentInfo.d_obj)
        rhs += currentInfo.d_obj[:Lt]' * (x_var .- xⱼ.IntVarBinaries)
    end
    
    @constraint(
        model,
        model[:z] ≥ rhs   
    )

    # ---------- g_k(x) linearization ----------
    # y ≥ g_k(xⱼ) + ∂g_k/∂St (x - xⱼ) + surrogate contribution
    @constraint(
        model,
        model[:y] ≥
        currentInfo.con[1] + 
        (param.algorithm == :SDDiP ? 
        currentInfo.d_con[1][:Lt]' * (x_var .- xⱼ.IntVarBinaries) : 
        currentInfo.d_con[1][:St]' * (x_var .- xⱼ.IntVar) 
        ) + 
        ( param.algorithm == :SDDPL ?
        sum(
            sum(
                currentInfo.d_con[1][:region_indicator][g][km] * (x_sur[g, km] - xⱼ.IntVarLeaf[g][km]) for km in keys(currentInfo.d_obj[:region_indicator][g])
            ) for g in 1:binaryInfo.d
        ) : 0.0 ) + 
        currentInfo.d_con[1][:obj] * (model[:xθ] - xⱼ.StateValue) 
    )

    @constraint(
        model, 
        model[:y] .≥ currentInfo.con[2] +  
        currentInfo.d_con[2] * (model[:xθ] - xⱼ.StateValue)                                                                     
    ); 

    return
end


function LevelSetMethod_optimization!(
    stageModel::StageModel,
    cutGenerationParamInfo::CutGenerationParamInfo,
    cutTypeInfo::LinearNormalizationLagrangianCutGenerationProgram;
    stageData::StageData = stageData,
    binaryInfo::BinaryInfo = binaryInfo,
    param::SDDPParam = param,
)::Any

    # unpack parameters (μ > 0 larger means more aggressive α adjustment)
    μ          = cutGenerationParamInfo.μ
    λ          = cutGenerationParamInfo.λ
    threshold  = cutGenerationParamInfo.gapBM        # not directly used below
    nxt_bound  = cutGenerationParamInfo.nxt_bound
    iterBM     = cutGenerationParamInfo.iterBM       # not directly used below

    stateInfo  = cutGenerationParamInfo.stateInfo

    A, n, d    = binaryInfo.A, binaryInfo.n, binaryInfo.d

    # initial cut center (πₙ)
    πₙ = cutGenerationParamInfo.πₙ

    ############################################
    ## initial inner minimization (current point)
    ############################################
    iter = 1
    α    = 0.5

    currentInfo, currentInfo_f = solve_inner_minimization_problem(
        cutTypeInfo,
        stageModel.model,
        πₙ,
        stateInfo;
        param = param,
    )

    function build_lnc_cut_info(currentInfo, currentInfo_f)
        return [
            currentInfo_f - currentInfo.var.StateValue * cutTypeInfo.primal_bound -
            (param.algorithm == :SDDiP ?
                currentInfo.var.IntVarBinaries' * stateInfo.IntVarBinaries :
                currentInfo.var.IntVar' * stateInfo.IntVar
            ) -
            (param.algorithm == :SDDPL ?
                sum(
                    sum(
                        currentInfo.var.IntVarLeaf[g][k] * stateInfo.IntVarLeaf[g][k]
                        for k in keys(stateInfo.IntVarLeaf[g])
                    ) for g in 1:binaryInfo.d
                ) : 0.0),
            currentInfo.var,
        ]
    end

    # initial cut info: [constant term, "point" info]
    cutInfo = build_lnc_cut_info(currentInfo, currentInfo_f)

    # function history (f, max g_k) for level-set Δ model
    functionHistory = FunctionHistory(
        Dict(1 => currentInfo.obj),
        Dict(1 => maximum(currentInfo.con[k] for k in keys(currentInfo.con))),
    )

    ############################################
    ## oracle model (computes f_star)
    ############################################
    oracleModel = Model(
        optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV)
        ),
    )
    MOI.set(oracleModel, MOI.Silent(), !param.verbose)
    set_optimizer_attribute(oracleModel, "MIPGap", param.gapSDDP)
    set_optimizer_attribute(oracleModel, "Threads",1)
    set_optimizer_attribute(oracleModel, "TimeLimit", param.timeSDDP)

    @variable(oracleModel, z ≥ -param.nxt_bound)
    @variable(oracleModel, xθ <= -cutTypeInfo.min_scale)
    if param.algorithm == :SDDPL
        @variable(oracleModel, x[i = 1:d])
        @variable(oracleModel, x_sur[g in 1:d, k in keys(stateInfo.IntVarLeaf[g])])
    elseif param.algorithm == :SDDP
        @variable(oracleModel, x[i = 1:d])
    elseif param.algorithm == :SDDiP
        @variable(oracleModel, x[i = 1:n])
    end
    @variable(oracleModel, y ≤ 0)
    @objective(oracleModel, Min, z)

    ############################################
    ## next-iterate model (nxtModel)
    ############################################
    nxtModel = Model(
        optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV)
        ),
    )
    MOI.set(nxtModel, MOI.Silent(), !param.verbose)
    set_optimizer_attribute(nxtModel, "MIPGap", param.gapSDDP)
    set_optimizer_attribute(nxtModel, "Threads", 1)
    set_optimizer_attribute(nxtModel, "TimeLimit", param.timeSDDP)

    @variable(nxtModel, z)
    @variable(nxtModel, xθ <= -cutTypeInfo.min_scale)
    if param.algorithm == :SDDPL
        @variable(nxtModel, x[i = 1:d])
        @variable(nxtModel, x_sur[g in 1:d, k in keys(stateInfo.IntVarLeaf[g])])
    elseif param.algorithm == :SDDP
        @variable(nxtModel, x[i = 1:d])
    elseif param.algorithm == :SDDiP
        @variable(nxtModel, x[i = 1:n])
    end
    @variable(nxtModel, y ≤ 0)
    @objective(nxtModel, Min, z)

    ############################################
    ## Level-set loop
    ############################################
    Δ   = Inf
    τₖ  = 1.0
    τₘ  = 0.5
    μₖ  = 1.0

    while true
        # --- Oracle step: compute f_star ---
        add_constraint!(currentInfo, oracleModel)
        optimize!(oracleModel)
        st = termination_status(oracleModel)

        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
            f_star = objective_value(oracleModel)
        else
            return (cutInfo = cutInfo, iter = iter)
        end

        # --- Level-set Δ model: compute Δ, α_min, α_max ---
        result    = Δ_model_formulation(functionHistory, f_star, iter)
        previousΔ = Δ
        Δ         = result[1]
        a_min     = result[2]
        a_max     = result[3]

        # print progress (optional)
        if param.verbose
            if iter == 1
                println("-------------- Level-set Iteration Info --------------")
                println("Iter |   Gap               Objective                Constraint")
            end
            @printf(
                "%3d  |   %5.3g           %5.3g                 %5.3g\n",
                iter,
                Δ,
                -currentInfo.obj,
                maximum(values(currentInfo.con)),
            )
        end

        # --- update x₀ and cutInfo if Δ improved ---
        x₀ = currentInfo.var
        candidate_pi0 = -currentInfo.var.StateValue
        if candidate_pi0 ≥ cutTypeInfo.min_scale &&
           round(previousΔ; digits = 8) > round(Δ; digits = 8)
            τₖ  = μₖ * τₖ
            cutInfo = build_lnc_cut_info(currentInfo, currentInfo_f)
        else
            τₖ = (τₖ + τₘ) / 2
        end

        # --- update α ---
        denom = a_max - a_min
        if denom ≈ 0.0
            α = (a_min + a_max) / 2
        else
            ratio = (α - a_min) / denom
            if !(μ / 2 ≤ ratio ≤ 1 - μ / 2)
                α = (a_min + a_max) / 2
            end
        end

        # --- update level ---
        w = α * f_star
        W = minimum(
            α * functionHistory.obj_history[j] +
            (1 - α) * functionHistory.max_con_history[j] for j in 1:iter
        )

        λ = iter ≤ 10 ? 0.05 : 0.1
        λ = iter ≥ 20 ? 0.2 : λ
        λ = iter ≥ 40 ? 0.3 : λ
        λ = iter ≥ 50 ? 0.4 : λ
        λ = iter ≥ 60 ? 0.5 : λ
        λ = iter ≥ 70 ? 0.6 : λ
        λ = iter ≥ 80 ? 0.7 : λ
        λ = iter ≥ 90 ? 0.8 : λ

        level = w + λ * (W - w)

        ############################################
        ## Next iteration point (nxtModel)
        ############################################
        # level constraint: α z + (1 - α) y ≤ level
        if iter == 1
            @constraint(
                nxtModel,
                level_constraint,
                α * nxtModel[:z] + (1 - α) * nxtModel[:y] ≤ level,
            )
        else
            delete(nxtModel, nxtModel[:level_constraint])
            unregister(nxtModel, :level_constraint)
            @constraint(
                nxtModel,
                level_constraint,
                α * nxtModel[:z] + (1 - α) * nxtModel[:y] ≤ level,
            )
        end

        add_constraint!(currentInfo, nxtModel)

        # --- proximity objective: stay close to x₀ ---
        # Base term: distance in the original state space whenever it exists.
        obj_expr = x₀.IntVar === nothing ?
            zero(A[1, 1]) :                       # some 0.0 scalar
            sum((nxtModel[:x] .- x₀.IntVar).^2)

        # if binary state exists, add that distance term
        if x₀.IntVarBinaries !== nothing
            obj_expr += sum((nxtModel[:x] .- x₀.IntVarBinaries).^2)
        end

        # if leaf representation exists, add surrogate part
        if x₀.IntVarLeaf !== nothing
            obj_expr += sum(
                sum(
                    (nxtModel[:x_sur][g, k] - x₀.IntVarLeaf[g][k])^2
                    for k in keys(x₀.IntVarLeaf[g])
                ) for g in 1:d
            )
        end   
        @objective(nxtModel, Min, obj_expr + (nxtModel[:xθ] - x₀.StateValue)^2)
        optimize!(nxtModel)
        st = termination_status(nxtModel)

        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
            πₙ = StageInfo(
                JuMP.value(nxtModel[:xθ]),
                nothing,
                x₀.IntVar === nothing ?
                    nothing :
                    value.(nxtModel[:x]),
                x₀.IntVarLeaf === nothing ?
                    nothing :
                    Dict(
                        g => Dict(
                            k => value(nxtModel[:x_sur][g, k])
                            for k in keys(x₀.IntVarLeaf[g])
                        ) for g in keys(x₀.IntVarLeaf)
                    ),
                x₀.IntVarBinaries === nothing ?
                    nothing :
                    value.(nxtModel[:x]),
            )
            λₖ = abs(dual(level_constraint))
            μₖ = λₖ + 1
        else
            return (cutInfo = cutInfo, iter = iter)
        end

        # --- stopping rule ---
        if Δ ≤ abs(cutGenerationParamInfo.stateInfo.StateValue) * threshold || iter > iterBM
            return (cutInfo = cutInfo, iter = iter)
        end

        ############################################
        ## prepare next iteration
        ############################################
        currentInfo, currentInfo_f = solve_inner_minimization_problem(
            cutTypeInfo,
            stageModel.model,
            πₙ,
            stateInfo;
            param = param,
        )

        iter += 1
        functionHistory.obj_history[iter] = currentInfo.obj
        functionHistory.max_con_history[iter] = maximum(currentInfo.con[k] for k in keys(currentInfo.con))
    end

end
