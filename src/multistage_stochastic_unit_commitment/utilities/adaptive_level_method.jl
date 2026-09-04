"""
Adaptive-level bundle routines for Pareto Lagrangian and square-minimization
cut generation. The routines maintain bounds on the Lagrangian dual value and
select a cut from oracle points that are within `δ` of the current dual bound.
"""

function _uc_adaptive_level_bound(bound::Real, fallback::Real)::Float64
    candidate = Float64(bound)
    if !isfinite(candidate) || candidate <= 0.0
        candidate = Float64(fallback)
    end
    return (!isfinite(candidate) || candidate <= 0.0) ? 1e8 : candidate
end

function _uc_adaptive_status_ok(model::Model)::Bool
    status = termination_status(model)
    return status == MOI.OPTIMAL ||
           status == MOI.LOCALLY_SOLVED ||
           status == MOI.ALMOST_OPTIMAL
end

function _uc_adaptive_min_oracle_bounds(
    incumbent_value::Real,
    primal_bound::Real,
)
    upper_value = Float64(incumbent_value)
    lower_value = upper_value
    primal_cap = Float64(primal_bound)
    if isfinite(primal_cap)
        lower_value = min(lower_value, primal_cap)
    end

    return (
        lower = lower_value,
        upper = max(upper_value, lower_value),
    )
end

function _uc_dual_dot(
    π::StateInfo,
    state::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::Float64
    value = sum(
        (param.algorithm == :SDDiP ?
            sum(
                π.ContStateBin[:s][g][i] * state.ContStateBin[:s][g][i]
                for i in 1:param.κ[g]; init = 0.0
            ) :
            π.ContVar[:s][g] * state.ContVar[:s][g]
        ) +
        π.BinVar[:y][g] * state.BinVar[:y][g] +
        π.BinVar[:v][g] * state.BinVar[:v][g] +
        π.BinVar[:w][g] * state.BinVar[:w][g]
        for g in indexSets.G
    )

    if param.algorithm == :SDDPL
        value += sum(
            sum(
                π.ContAugState[:s][g][k] * state.ContAugState[:s][g][k]
                for k in keys(state.ContAugState[:s][g]); init = 0.0
            ) for g in indexSets.G
        )
    end

    return Float64(value)
end

function _uc_dual_dot_difference(
    π::StateInfo,
    left::StateInfo,
    right::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::Float64
    value = sum(
        (param.algorithm == :SDDiP ?
            sum(
                π.ContStateBin[:s][g][i] *
                (left.ContStateBin[:s][g][i] - right.ContStateBin[:s][g][i])
                for i in 1:param.κ[g]; init = 0.0
            ) :
            π.ContVar[:s][g] *
            (left.ContVar[:s][g] - right.ContVar[:s][g])
        ) +
        π.BinVar[:y][g] * (left.BinVar[:y][g] - right.BinVar[:y][g]) +
        π.BinVar[:v][g] * (left.BinVar[:v][g] - right.BinVar[:v][g]) +
        π.BinVar[:w][g] * (left.BinVar[:w][g] - right.BinVar[:w][g])
        for g in indexSets.G
    )

    if param.algorithm == :SDDPL
        value += sum(
            sum(
                π.ContAugState[:s][g][k] *
                (left.ContAugState[:s][g][k] - right.ContAugState[:s][g][k])
                for k in keys(right.ContAugState[:s][g]); init = 0.0
            ) for g in indexSets.G
        )
    end

    return Float64(value)
end

function _uc_dual_norm2(
    π::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::Float64
    value = sum(
        (param.algorithm == :SDDiP ?
            sum(π.ContStateBin[:s][g][i]^2 for i in 1:param.κ[g]; init = 0.0) :
            π.ContVar[:s][g]^2
        ) +
        π.BinVar[:y][g]^2 +
        π.BinVar[:v][g]^2 +
        π.BinVar[:w][g]^2
        for g in indexSets.G
    )

    if param.algorithm == :SDDPL
        value += sum(
            sum(
                π.ContAugState[:s][g][k]^2
                for k in keys(π.ContAugState[:s][g]); init = 0.0
            ) for g in indexSets.G
        )
    end

    return Float64(value)
end

function _uc_master_dot_difference(
    modelInfo::ModelInfo,
    left::StateInfo,
    right::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)
    expression = sum(
        (param.algorithm == :SDDiP ?
            sum(
                modelInfo.sur[g, i] *
                (left.ContStateBin[:s][g][i] - right.ContStateBin[:s][g][i])
                for i in 1:param.κ[g]; init = 0.0
            ) :
            modelInfo.xs[g] * (left.ContVar[:s][g] - right.ContVar[:s][g])
        ) +
        modelInfo.xy[g] * (left.BinVar[:y][g] - right.BinVar[:y][g]) +
        modelInfo.xv[g] * (left.BinVar[:v][g] - right.BinVar[:v][g]) +
        modelInfo.xw[g] * (left.BinVar[:w][g] - right.BinVar[:w][g])
        for g in indexSets.G
    )

    if param.algorithm == :SDDPL
        expression += sum(
            sum(
                modelInfo.sur[g, k] *
                (left.ContAugState[:s][g][k] - right.ContAugState[:s][g][k])
                for k in keys(right.ContAugState[:s][g]); init = 0.0
            ) for g in indexSets.G
        )
    end

    return expression
end

function _uc_master_norm2(
    modelInfo::ModelInfo,
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)
    expression = sum(
        (param.algorithm == :SDDiP ?
            sum(modelInfo.sur[g, i]^2 for i in 1:param.κ[g]; init = 0.0) :
            modelInfo.xs[g]^2
        ) +
        modelInfo.xy[g]^2 +
        modelInfo.xv[g]^2 +
        modelInfo.xw[g]^2
        for g in indexSets.G
    )

    if param.algorithm == :SDDPL
        expression += sum(
            sum(
                modelInfo.sur[g, k]^2
                for k in keys(stateInfo.ContAugState[:s][g]); init = 0.0
            ) for g in indexSets.G
        )
    end

    return expression
end

function _uc_bundle_cut_expr(
    modelInfo::ModelInfo,
    point;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)
    gradient = point.gradient
    center = point.π
    expression = sum(
        (param.algorithm == :SDDiP ?
            sum(
                gradient.ContStateBin[:s][g][i] *
                (modelInfo.sur[g, i] - center.ContStateBin[:s][g][i])
                for i in 1:param.κ[g]; init = 0.0
            ) :
            gradient.ContVar[:s][g] *
            (modelInfo.xs[g] - center.ContVar[:s][g])
        ) +
        gradient.BinVar[:y][g] * (modelInfo.xy[g] - center.BinVar[:y][g]) +
        gradient.BinVar[:v][g] * (modelInfo.xv[g] - center.BinVar[:v][g]) +
        gradient.BinVar[:w][g] * (modelInfo.xw[g] - center.BinVar[:w][g])
        for g in indexSets.G
    )

    if param.algorithm == :SDDPL
        expression += sum(
            sum(
                gradient.ContAugState[:s][g][k] *
                (modelInfo.sur[g, k] - center.ContAugState[:s][g][k])
                for k in keys(center.ContAugState[:s][g]); init = 0.0
            ) for g in indexSets.G
        )
    end

    return point.upper_value - expression
end

function _uc_adaptive_master_model(
    stateInfo::StateInfo,
    dual_bound::Float64;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::ModelInfo
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV)))
    MOI.set(model, MOI.Silent(), true)
    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "MIPGap", param.MIPGap)
    set_optimizer_attribute(model, "TimeLimit", param.TimeLimit)

    @variable(model, η)
    @variable(model, -dual_bound <= xs[indexSets.G] <= dual_bound)
    @variable(model, -dual_bound <= xy[indexSets.G] <= dual_bound)
    @variable(model, -dual_bound <= xv[indexSets.G] <= dual_bound)
    @variable(model, -dual_bound <= xw[indexSets.G] <= dual_bound)
    if param.algorithm == :SDDPL
        @variable(
            model,
            -dual_bound <=
                sur[g in indexSets.G, k in keys(stateInfo.ContAugState[:s][g])] <=
                dual_bound,
        )
    elseif param.algorithm == :SDDP
        sur = nothing
    elseif param.algorithm == :SDDiP
        @variable(
            model,
            -dual_bound <= sur[g in indexSets.G, i in 1:param.κ[g]] <= dual_bound,
        )
    else
        error("Unsupported algorithm $(param.algorithm) for adaptive cut generation.")
    end

    return ModelInfo(model, xs, xy, xv, xw, sur, η, η)
end

function _uc_master_point(
    modelInfo::ModelInfo,
    stateInfo::StateInfo;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::StateInfo
    bin_vars = Dict{Any, Dict{Any, Any}}(
        :y => Dict{Any, Any}(g => JuMP.value(modelInfo.xy[g]) for g in indexSets.G),
        :v => Dict{Any, Any}(g => JuMP.value(modelInfo.xv[g]) for g in indexSets.G),
        :w => Dict{Any, Any}(g => JuMP.value(modelInfo.xw[g]) for g in indexSets.G),
    )
    cont_vars = Dict{Any, Dict{Any, Any}}(
        :s => Dict{Any, Any}(g => JuMP.value(modelInfo.xs[g]) for g in indexSets.G),
    )

    cont_aug_state = param.algorithm == :SDDPL ?
        Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    k => JuMP.value(modelInfo.sur[g, k])
                    for k in keys(stateInfo.ContAugState[:s][g])
                ) for g in indexSets.G
            )
        ) :
        nothing

    cont_state_bin = param.algorithm == :SDDiP ?
        Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    i => JuMP.value(modelInfo.sur[g, i]) for i in 1:param.κ[g]
                ) for g in indexSets.G
            )
        ) :
        nothing

    return StateInfo(
        bin_vars,
        nothing,
        cont_vars,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        cont_aug_state,
        nothing,
        cont_state_bin,
    )
end

function _uc_add_bundle_cuts!(
    modelInfo::ModelInfo,
    bundle;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)::Nothing
    for point in bundle
        @constraint(
            modelInfo.model,
            modelInfo.z <= _uc_bundle_cut_expr(
                modelInfo,
                point;
                indexSets = indexSets,
                param = param,
            ),
        )
    end
    return
end

function _uc_solve_dual_upper_master(
    bundle,
    stateInfo::StateInfo,
    dual_bound::Float64;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)
    modelInfo = _uc_adaptive_master_model(
        stateInfo,
        dual_bound;
        indexSets = indexSets,
        param = param,
    )
    _uc_add_bundle_cuts!(modelInfo, bundle; indexSets = indexSets, param = param)
    @objective(modelInfo.model, Max, modelInfo.z)
    optimize!(modelInfo.model)

    if !_uc_adaptive_status_ok(modelInfo.model)
        return (success = false, value = Inf, π = nothing)
    end

    return (
        success = true,
        value = Float64(objective_value(modelInfo.model)),
        π = _uc_master_point(modelInfo, stateInfo; indexSets = indexSets, param = param),
    )
end

function _uc_solve_adaptive_selection_master(
    bundle,
    stateInfo::StateInfo,
    CutGenerationInfo::CutGeneration,
    level::Float64,
    dual_bound::Float64,
    use_square_minimization::Bool;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)
    modelInfo = _uc_adaptive_master_model(
        stateInfo,
        dual_bound;
        indexSets = indexSets,
        param = param,
    )
    _uc_add_bundle_cuts!(modelInfo, bundle; indexSets = indexSets, param = param)
    @constraint(modelInfo.model, modelInfo.z >= level)

    if use_square_minimization
        @objective(
            modelInfo.model,
            Min,
            _uc_master_norm2(
                modelInfo,
                stateInfo;
                indexSets = indexSets,
                param = param,
            ),
        )
    else
        @objective(
            modelInfo.model,
            Max,
            modelInfo.z +
            _uc_master_dot_difference(
                modelInfo,
                CutGenerationInfo.core_point,
                stateInfo;
                indexSets = indexSets,
                param = param,
            ),
        )
    end

    optimize!(modelInfo.model)
    if !_uc_adaptive_status_ok(modelInfo.model)
        return (success = false, value = NaN, π = nothing)
    end

    return (
        success = true,
        value = Float64(objective_value(modelInfo.model)),
        π = _uc_master_point(modelInfo, stateInfo; indexSets = indexSets, param = param),
    )
end

function _uc_adaptive_bundle_point(
    currentInfo::CurrentInfo,
    currentInfo_f::Float64,
    currentInfo_upper_value::Float64,
    CutGenerationInfo::CutGeneration,
    stateInfo::StateInfo,
    use_square_minimization::Bool;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)
    π = currentInfo.x
    score = use_square_minimization ? NaN :
        currentInfo_f + _uc_dual_dot_difference(
            π,
            CutGenerationInfo.core_point,
            stateInfo;
            indexSets = indexSets,
            param = param,
        )
    norm2 = _uc_dual_norm2(π; indexSets = indexSets, param = param)

    return (
        π = π,
        f = Float64(currentInfo_f),
        upper_value = Float64(currentInfo_upper_value),
        gradient = currentInfo.dG[1],
        score = Float64(score),
        norm2 = Float64(norm2),
        intercept = Float64(
            currentInfo_f -
            _uc_dual_dot(π, stateInfo; indexSets = indexSets, param = param)
        ),
    )
end

function _uc_evaluate_adaptive_oracle!(
    bundle::Vector{Any},
    model::Model,
    CutGenerationInfo::CutGeneration,
    π::StateInfo,
    stateInfo::StateInfo,
    use_square_minimization::Bool;
    indexSets::IndexSets = indexSets,
    param::NamedTuple = param,
)
    currentInfo, currentInfo_f = solve_inner_minimization_problem(
        CutGenerationInfo,
        model,
        π,
        stateInfo;
        indexSets = indexSets,
        param = param,
    )
    oracle_values = _uc_adaptive_min_oracle_bounds(
        currentInfo_f,
        CutGenerationInfo.primal_bound,
    )
    point = _uc_adaptive_bundle_point(
        currentInfo,
        oracle_values.lower,
        oracle_values.upper,
        CutGenerationInfo,
        stateInfo,
        use_square_minimization;
        indexSets = indexSets,
        param = param,
    )
    push!(bundle, point)
    return point
end

function _uc_best_adaptive_point(
    bundle,
    dual_lower::Float64,
    delta::Float64,
    tolerance::Float64,
    use_square_minimization::Bool,
)
    feasible = [
        point for point in bundle
        if point.f >= dual_lower - delta - tolerance
    ]
    candidates = isempty(feasible) ? bundle : feasible

    index = 1
    if use_square_minimization
        best_value = candidates[index].norm2
        for k in 2:length(candidates)
            if candidates[k].norm2 < best_value
                index = k
                best_value = candidates[k].norm2
            end
        end
    else
        best_value = candidates[index].score
        for k in 2:length(candidates)
            if candidates[k].score > best_value
                index = k
                best_value = candidates[k].score
            end
        end
    end
    return candidates[index]
end

function _uc_adaptive_level_method!(
    model::Model,
    levelsetmethodOracleParam::LevelSetMethodOracleParam,
    stateInfo::StateInfo,
    CutGenerationInfo::CutGeneration;
    indexSets::IndexSets = indexSets,
    paramDemand::ParamDemand = paramDemand,
    paramOPF::ParamOPF = paramOPF,
    param::NamedTuple = param,
    param_levelsetmethod::NamedTuple = param_levelsetmethod,
)
    use_square_minimization = param.cutSelection == :AdaptiveSMC
    delta = CutGenerationInfo.δ
    threshold = levelsetmethodOracleParam.threshold === nothing ?
        1e-4 :
        levelsetmethodOracleParam.threshold
    iter_limit = levelsetmethodOracleParam.MaxIter
    λ = clamp(
        levelsetmethodOracleParam.λ === nothing ? 0.5 : levelsetmethodOracleParam.λ,
        0.0,
        1.0,
    )
    dual_bound = _uc_adaptive_level_bound(
        levelsetmethodOracleParam.nxt_bound,
        get(param_levelsetmethod, :nxt_bound, 1e10),
    )

    if levelsetmethodOracleParam.verbose
        @info "Starting adaptive level method" cut=param.cutSelection dual_bound=dual_bound max_iterations=iter_limit
    end

    bundle = Any[]
    iter = 1
    primal_upper = Float64(CutGenerationInfo.primal_bound)
    first_point = _uc_evaluate_adaptive_oracle!(
        bundle,
        model,
        CutGenerationInfo,
        levelsetmethodOracleParam.x₀,
        stateInfo,
        use_square_minimization;
        indexSets = indexSets,
        param = param,
    )

    dual_lower = first_point.f
    dual_upper = isfinite(primal_upper) ? max(dual_lower, primal_upper) : Inf
    trial_level = isfinite(dual_upper) ? dual_upper : dual_lower
    best_point = first_point
    primal_gap = Inf

    while true
        upper_result = _uc_solve_dual_upper_master(
            bundle,
            stateInfo,
            dual_bound;
            indexSets = indexSets,
            param = param,
        )
        if upper_result.success && isfinite(upper_result.value)
            dual_upper = min(dual_upper, upper_result.value)
            if isfinite(primal_upper)
                dual_upper = min(dual_upper, primal_upper)
            end
            dual_upper = max(dual_upper, dual_lower)

            if iter < iter_limit
                upper_point = _uc_evaluate_adaptive_oracle!(
                    bundle,
                    model,
                    CutGenerationInfo,
                    upper_result.π,
                    stateInfo,
                    use_square_minimization;
                    indexSets = indexSets,
                    param = param,
                )
                iter += 1
                dual_lower = max(dual_lower, upper_point.f)
                if isfinite(primal_upper)
                    dual_upper = min(dual_upper, primal_upper)
                end
                dual_upper = max(dual_upper, dual_lower)
            end
        end

        if isfinite(dual_upper)
            trial_level = clamp(trial_level, dual_lower, dual_upper)
        else
            trial_level = max(trial_level, dual_lower)
        end

        trial_result = _uc_solve_adaptive_selection_master(
            bundle,
            stateInfo,
            CutGenerationInfo,
            trial_level - delta,
            dual_bound,
            use_square_minimization;
            indexSets = indexSets,
            param = param,
        )

        while !trial_result.success &&
              isfinite(dual_upper) &&
              dual_upper - dual_lower > threshold * max(abs(dual_lower), 1.0)
            dual_upper = min(dual_upper, trial_level - delta)
            dual_upper = max(dual_upper, dual_lower)
            trial_level = dual_lower + λ * (dual_upper - dual_lower)
            trial_result = _uc_solve_adaptive_selection_master(
                bundle,
                stateInfo,
                CutGenerationInfo,
                trial_level - delta,
                dual_bound,
                use_square_minimization;
                indexSets = indexSets,
                param = param,
            )
        end

        if trial_result.success && iter < iter_limit
            trial_point = _uc_evaluate_adaptive_oracle!(
                bundle,
                model,
                CutGenerationInfo,
                trial_result.π,
                stateInfo,
                use_square_minimization;
                indexSets = indexSets,
                param = param,
            )
            iter += 1
            dual_lower = max(dual_lower, trial_point.f)
        end

        serious_result = _uc_solve_adaptive_selection_master(
            bundle,
            stateInfo,
            CutGenerationInfo,
            dual_lower - delta,
            dual_bound,
            use_square_minimization;
            indexSets = indexSets,
            param = param,
        )

        if serious_result.success && iter < iter_limit
            serious_point = _uc_evaluate_adaptive_oracle!(
                bundle,
                model,
                CutGenerationInfo,
                serious_result.π,
                stateInfo,
                use_square_minimization;
                indexSets = indexSets,
                param = param,
            )
            iter += 1
            dual_lower = max(dual_lower, serious_point.f)
        end

        best_point = _uc_best_adaptive_point(
            bundle,
            dual_lower,
            delta,
            1e-8 * max(abs(dual_lower), 1.0),
            use_square_minimization,
        )

        if serious_result.success
            primal_gap = use_square_minimization ?
                max(0.0, best_point.norm2 - serious_result.value) :
                max(0.0, serious_result.value - best_point.score)
        end

        dual_gap = isfinite(dual_upper) ? max(0.0, dual_upper - dual_lower) : Inf
        dual_tolerance = threshold * max(abs(dual_lower), 1.0)
        primal_reference = use_square_minimization ? best_point.norm2 : best_point.score
        primal_tolerance = threshold * max(abs(primal_reference), 1.0)

        if get(param, :cutDiagnostics, false)
            @info "Adaptive enhanced cut diagnostics" cut=param.cutSelection iter=iter D_lower=dual_lower D_upper=dual_upper D_gap=dual_gap p_gap=primal_gap selected_f=best_point.f selected_intercept=best_point.intercept
        end

        diagnostics = (
            D_lower = dual_lower,
            D_upper = dual_upper,
            d_gap = dual_gap,
            p_gap = primal_gap,
            selected_f = best_point.f,
            selected_upper_value = best_point.upper_value,
            selected_score = best_point.score,
            selected_norm2 = best_point.norm2,
            bundle_size = length(bundle),
        )

        if iter >= iter_limit ||
           (dual_gap <= dual_tolerance &&
            (!serious_result.success || primal_gap <= primal_tolerance))
            return (
                cutInfo = (best_point.intercept, best_point.π, 1.0),
                iter = iter,
                diagnostics = diagnostics,
            )
        end

        trial_level = isfinite(dual_upper) ?
            dual_lower + λ * (dual_upper - dual_lower) :
            dual_lower
    end
end
