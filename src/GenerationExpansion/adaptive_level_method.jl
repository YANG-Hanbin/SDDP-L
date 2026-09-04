#############################################################################################
######################## Adaptive-level PLC/SMC cut selection ###############################
#############################################################################################

"""
Adaptive-level cut selection for the PLC and SMC Lagrangian duals.

The routine maintains certified lower and upper bounds on the best dual
value while retaining every evaluated oracle point in a bundle. AdaptivePLC
selects a near-optimal point by its Pareto score; AdaptiveSMC selects one by
minimum squared dual norm.
"""
function _adaptive_level_bound(
    bound::Real,
    fallback::Real,
)::Float64
    candidate = Float64(bound)
    if !isfinite(candidate) || candidate <= 0.0
        candidate = Float64(fallback)
    end
    return (!isfinite(candidate) || candidate <= 0.0) ? 1e8 : candidate
end

function _adaptive_status_ok(model::Model)::Bool
    st = termination_status(model)
    return st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
end

function _adaptive_min_oracle_bounds(
    model::Model,
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

function _gep_dual_dot(
    π::StageInfo,
    state::StageInfo;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::Float64
    value = if param.algorithm == :SDDiP
        sum(π.IntVarBinaries[i] * state.IntVarBinaries[i] for i in eachindex(state.IntVarBinaries))
    else
        sum(π.IntVar[i] * state.IntVar[i] for i in eachindex(state.IntVar))
    end

    if param.algorithm == :SDDPL
        value += sum(
            sum(
                π.IntVarLeaf[g][k] * state.IntVarLeaf[g][k]
                for k in keys(state.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    end

    return Float64(value)
end

function _gep_dual_dot_difference(
    π::StageInfo,
    left::StageInfo,
    right::StageInfo;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::Float64
    value = if param.algorithm == :SDDiP
        sum(
            π.IntVarBinaries[i] * (left.IntVarBinaries[i] - right.IntVarBinaries[i])
            for i in eachindex(right.IntVarBinaries)
        )
    else
        sum(
            π.IntVar[i] * (left.IntVar[i] - right.IntVar[i])
            for i in eachindex(right.IntVar)
        )
    end

    if param.algorithm == :SDDPL
        value += sum(
            sum(
                π.IntVarLeaf[g][k] * (left.IntVarLeaf[g][k] - right.IntVarLeaf[g][k])
                for k in keys(right.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    end

    return Float64(value)
end

function _gep_dual_norm2(
    π::StageInfo;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::Float64
    value = if param.algorithm == :SDDiP
        sum(v * v for v in π.IntVarBinaries)
    else
        sum(v * v for v in π.IntVar)
    end

    if param.algorithm == :SDDPL
        value += sum(
            sum(
                π.IntVarLeaf[g][k] * π.IntVarLeaf[g][k]
                for k in keys(π.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    end

    return Float64(value)
end

function _gep_master_dot_difference(
    model::Model,
    left::StageInfo,
    right::StageInfo;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)
    expr = if param.algorithm == :SDDiP
        sum(
            model[:x][i] * (left.IntVarBinaries[i] - right.IntVarBinaries[i])
            for i in eachindex(right.IntVarBinaries)
        )
    else
        sum(
            model[:x][i] * (left.IntVar[i] - right.IntVar[i])
            for i in eachindex(right.IntVar)
        )
    end

    if param.algorithm == :SDDPL
        expr += sum(
            sum(
                model[:x_sur][g, k] * (left.IntVarLeaf[g][k] - right.IntVarLeaf[g][k])
                for k in keys(right.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    end

    return expr
end

function _gep_master_norm2(
    model::Model,
    stateInfo::StageInfo;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)
    expr = if param.algorithm == :SDDiP
        sum(model[:x][i]^2 for i in eachindex(stateInfo.IntVarBinaries))
    else
        sum(model[:x][i]^2 for i in eachindex(stateInfo.IntVar))
    end

    if param.algorithm == :SDDPL
        expr += sum(
            sum(
                model[:x_sur][g, k]^2 for k in keys(stateInfo.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    end

    return expr
end

function _gep_bundle_cut_expr(
    model::Model,
    point;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)
    gradient = point.gradient
    center = point.π
    expr = if param.algorithm == :SDDiP
        sum(
            gradient[:Lt][i] * (model[:x][i] - center.IntVarBinaries[i])
            for i in eachindex(center.IntVarBinaries)
        )
    else
        sum(
            gradient[:St][i] * (model[:x][i] - center.IntVar[i])
            for i in eachindex(center.IntVar)
        )
    end

    if param.algorithm == :SDDPL
        expr += sum(
            sum(
                gradient[:region_indicator][g][k] *
                (model[:x_sur][g, k] - center.IntVarLeaf[g][k])
                for k in keys(center.IntVarLeaf[g]); init = 0.0
            ) for g in 1:binaryInfo.d
        )
    end

    return point.upper_value - expr
end

function _gep_master_point(
    model::Model,
    stateInfo::StageInfo;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::StageInfo
    return StageInfo(
        -1.0,
        nothing,
        param.algorithm == :SDDiP ? nothing : value.(model[:x]),
        param.algorithm == :SDDPL ?
            Dict(
                g => Dict(
                    k => JuMP.value(model[:x_sur][g, k])
                    for k in keys(stateInfo.IntVarLeaf[g])
                ) for g in keys(stateInfo.IntVarLeaf)
            ) :
            nothing,
        param.algorithm == :SDDiP ? value.(model[:x]) : nothing,
    )
end

function _gep_adaptive_master_model(
    stateInfo::StageInfo,
    dual_bound::Float64;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::Model
    model = Model(
        optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV)
        ),
    )
    MOI.set(model, MOI.Silent(), !param.verbose)
    set_optimizer_attribute(model, "MIPGap", param.gapSDDP)
    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "TimeLimit", param.timeSDDP)

    @variable(model, η)
    if param.algorithm == :SDDPL
        @variable(model, -dual_bound <= x[i = 1:binaryInfo.d] <= dual_bound)
        @variable(
            model,
            -dual_bound <=
                x_sur[g in 1:binaryInfo.d, k in keys(stateInfo.IntVarLeaf[g])] <=
                dual_bound
        )
    elseif param.algorithm == :SDDP
        @variable(model, -dual_bound <= x[i = 1:binaryInfo.d] <= dual_bound)
    elseif param.algorithm == :SDDiP
        @variable(model, -dual_bound <= x[i = 1:binaryInfo.n] <= dual_bound)
    end

    return model
end

function _gep_add_bundle_cuts!(
    model::Model,
    bundle;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::Nothing
    for point in bundle
        @constraint(
            model,
            model[:η] <= _gep_bundle_cut_expr(
                model,
                point;
                param = param,
                binaryInfo = binaryInfo,
            )
        )
    end
    return
end

function _gep_solve_dual_upper_master(
    bundle,
    stateInfo::StageInfo,
    dual_bound::Float64;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)
    model = _gep_adaptive_master_model(
        stateInfo,
        dual_bound;
        param = param,
        binaryInfo = binaryInfo,
    )
    _gep_add_bundle_cuts!(model, bundle; param = param, binaryInfo = binaryInfo)
    @objective(model, Max, model[:η])
    optimize!(model)

    if !_adaptive_status_ok(model)
        return (success = false, value = Inf, π = nothing)
    end

    return (
        success = true,
        value = Float64(objective_value(model)),
        π = _gep_master_point(model, stateInfo; param = param, binaryInfo = binaryInfo),
    )
end

function _gep_solve_adaptive_selection_master(
    bundle,
    stateInfo::StageInfo,
    cutTypeInfo::CutGenerationProgram,
    level::Float64,
    dual_bound::Float64,
    use_square_minimization::Bool;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)
    model = _gep_adaptive_master_model(
        stateInfo,
        dual_bound;
        param = param,
        binaryInfo = binaryInfo,
    )
    _gep_add_bundle_cuts!(model, bundle; param = param, binaryInfo = binaryInfo)
    @constraint(model, model[:η] >= level)

    if use_square_minimization
        @objective(
            model,
            Min,
            _gep_master_norm2(model, stateInfo; param = param, binaryInfo = binaryInfo),
        )
    else
        @objective(
            model,
            Max,
            model[:η] +
            _gep_master_dot_difference(
                model,
                cutTypeInfo.CoreState,
                stateInfo;
                param = param,
                binaryInfo = binaryInfo,
            ),
        )
    end

    optimize!(model)
    if !_adaptive_status_ok(model)
        return (success = false, value = NaN, π = nothing)
    end

    return (
        success = true,
        value = Float64(objective_value(model)),
        π = _gep_master_point(model, stateInfo; param = param, binaryInfo = binaryInfo),
    )
end

function _gep_adaptive_bundle_point(
    currentInfo::CurrentInfo,
    currentInfo_f::Float64,
    currentInfo_upper_value::Float64,
    cutTypeInfo::CutGenerationProgram,
    stateInfo::StageInfo,
    use_square_minimization::Bool;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)
    π = currentInfo.var
    score = use_square_minimization ? NaN :
        currentInfo_f + _gep_dual_dot_difference(
            π,
            cutTypeInfo.CoreState,
            stateInfo;
            param = param,
            binaryInfo = binaryInfo,
        )
    norm2 = _gep_dual_norm2(π; param = param, binaryInfo = binaryInfo)

    return (
        π = π,
        f = Float64(currentInfo_f),
        upper_value = Float64(currentInfo_upper_value),
        gradient = currentInfo.d_con[1],
        score = Float64(score),
        norm2 = Float64(norm2),
        intercept = Float64(
            currentInfo_f -
            _gep_dual_dot(π, stateInfo; param = param, binaryInfo = binaryInfo)
        ),
    )
end

function _gep_evaluate_adaptive_oracle!(
    bundle::Vector{Any},
    stageModel::StageModel,
    cutTypeInfo::CutGenerationProgram,
    π::StageInfo,
    stateInfo::StageInfo,
    use_square_minimization::Bool;
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)
    currentInfo, currentInfo_f = solve_inner_minimization_problem(
        cutTypeInfo,
        stageModel.model,
        π,
        stateInfo;
        param = param,
    )
    oracle_values = _adaptive_min_oracle_bounds(
        stageModel.model,
        currentInfo_f,
        cutTypeInfo.primal_bound,
    )
    point = _gep_adaptive_bundle_point(
        currentInfo,
        oracle_values.lower,
        oracle_values.upper,
        cutTypeInfo,
        stateInfo,
        use_square_minimization;
        param = param,
        binaryInfo = binaryInfo,
    )
    push!(bundle, point)
    return point
end

function _gep_best_adaptive_point(
    bundle,
    D_lower::Float64,
    delta::Float64,
    tolerance::Float64,
    use_square_minimization::Bool,
)
    feasible = [
        point for point in bundle
        if point.f >= D_lower - delta - tolerance
    ]
    candidates = isempty(feasible) ? bundle : feasible

    idx = 1
    if use_square_minimization
        best_value = candidates[idx].norm2
        for k in 2:length(candidates)
            if candidates[k].norm2 < best_value
                idx = k
                best_value = candidates[k].norm2
            end
        end
    else
        best_value = candidates[idx].score
        for k in 2:length(candidates)
            if candidates[k].score > best_value
                idx = k
                best_value = candidates[k].score
            end
        end
    end
    return candidates[idx]
end

function _gep_adaptive_cut_info(point)
    return [point.intercept, point.π]
end

function gep_adaptive_level_method!(
    stageModel::StageModel,
    cutGenerationParamInfo::CutGenerationParamInfo,
    cutTypeInfo::CutGenerationProgram;
    stageData::StageData = stageData,
    binaryInfo::BinaryInfo = binaryInfo,
    param::SDDPParam = param,
)::Any
    stateInfo = cutGenerationParamInfo.stateInfo
    use_square_minimization = cutGenerationParamInfo.cutSelection == :AdaptiveSMC
    delta = cutTypeInfo.δ
    threshold = cutGenerationParamInfo.gapBM
    iter_limit = cutGenerationParamInfo.iterBM
    λ = clamp(
        cutGenerationParamInfo.λ === nothing ? 0.5 : cutGenerationParamInfo.λ,
        0.0,
        1.0,
    )
    dual_bound = _adaptive_level_bound(
        cutGenerationParamInfo.nxt_bound,
        param.nxt_bound,
    )

    bundle = Any[]
    iter = 1
    primal_upper = Float64(cutTypeInfo.primal_bound)
    first_point = _gep_evaluate_adaptive_oracle!(
        bundle,
        stageModel,
        cutTypeInfo,
        cutGenerationParamInfo.πₙ,
        stateInfo,
        use_square_minimization;
        param = param,
        binaryInfo = binaryInfo,
    )

    D_lower = first_point.f
    D_upper = isfinite(primal_upper) ? max(D_lower, primal_upper) : Inf
    trial_level = isfinite(D_upper) ? D_upper : D_lower
    best_point = first_point
    p_gap = Inf

    while true
        upper_result = _gep_solve_dual_upper_master(
            bundle,
            stateInfo,
            dual_bound;
            param = param,
            binaryInfo = binaryInfo,
        )
        if upper_result.success && isfinite(upper_result.value)
            D_upper = min(D_upper, upper_result.value)
            if isfinite(primal_upper)
                D_upper = min(D_upper, primal_upper)
            end
            D_upper = max(D_upper, D_lower)

            if iter < iter_limit
                upper_point = _gep_evaluate_adaptive_oracle!(
                    bundle,
                    stageModel,
                    cutTypeInfo,
                    upper_result.π,
                    stateInfo,
                    use_square_minimization;
                    param = param,
                    binaryInfo = binaryInfo,
                )
                iter += 1
                D_lower = max(D_lower, upper_point.f)
                if isfinite(primal_upper)
                    D_upper = min(D_upper, primal_upper)
                end
                D_upper = max(D_upper, D_lower)
            end
        end

        if isfinite(D_upper)
            trial_level = clamp(trial_level, D_lower, D_upper)
        else
            trial_level = max(trial_level, D_lower)
        end

        trial_result = _gep_solve_adaptive_selection_master(
            bundle,
            stateInfo,
            cutTypeInfo,
            trial_level - delta,
            dual_bound,
            use_square_minimization;
            param = param,
            binaryInfo = binaryInfo,
        )

        while !trial_result.success &&
              isfinite(D_upper) &&
              D_upper - D_lower > threshold * max(abs(D_lower), 1.0)
            D_upper = min(D_upper, trial_level - delta)
            D_upper = max(D_upper, D_lower)
            trial_level = D_lower + λ * (D_upper - D_lower)
            trial_result = _gep_solve_adaptive_selection_master(
                bundle,
                stateInfo,
                cutTypeInfo,
                trial_level - delta,
                dual_bound,
                use_square_minimization;
                param = param,
                binaryInfo = binaryInfo,
            )
        end

        if trial_result.success && iter < iter_limit
            trial_point = _gep_evaluate_adaptive_oracle!(
                bundle,
                stageModel,
                cutTypeInfo,
                trial_result.π,
                stateInfo,
                use_square_minimization;
                param = param,
                binaryInfo = binaryInfo,
            )
            iter += 1
            D_lower = max(D_lower, trial_point.f)
        end

        serious_result = _gep_solve_adaptive_selection_master(
            bundle,
            stateInfo,
            cutTypeInfo,
            D_lower - delta,
            dual_bound,
            use_square_minimization;
            param = param,
            binaryInfo = binaryInfo,
        )

        if serious_result.success && iter < iter_limit
            serious_point = _gep_evaluate_adaptive_oracle!(
                bundle,
                stageModel,
                cutTypeInfo,
                serious_result.π,
                stateInfo,
                use_square_minimization;
                param = param,
                binaryInfo = binaryInfo,
            )
            iter += 1
            D_lower = max(D_lower, serious_point.f)
        end

        best_point = _gep_best_adaptive_point(
            bundle,
            D_lower,
            delta,
            1e-8 * max(abs(D_lower), 1.0),
            use_square_minimization,
        )

        if serious_result.success
            p_gap = use_square_minimization ?
                max(0.0, best_point.norm2 - serious_result.value) :
                max(0.0, serious_result.value - best_point.score)
        end

        d_gap = isfinite(D_upper) ? max(0.0, D_upper - D_lower) : Inf
        d_tol = threshold * max(abs(D_lower), 1.0)
        p_ref = use_square_minimization ? best_point.norm2 : best_point.score
        p_tol = threshold * max(abs(p_ref), 1.0)

        if param.cutDiagnostics
            @info "Adaptive enhanced cut diagnostics" cut=cutGenerationParamInfo.cutSelection iter=iter D_lower=D_lower D_upper=D_upper D_gap=d_gap p_gap=p_gap selected_f=best_point.f selected_intercept=best_point.intercept
        end

        diagnostics = (
            D_lower = D_lower,
            D_upper = D_upper,
            d_gap = d_gap,
            p_gap = p_gap,
            selected_f = best_point.f,
            selected_upper_value = best_point.upper_value,
            selected_score = best_point.score,
            selected_norm2 = best_point.norm2,
            bundle_size = length(bundle),
        )

        if iter >= iter_limit ||
           (d_gap <= d_tol && (!serious_result.success || p_gap <= p_tol))
            return (
                cutInfo = _gep_adaptive_cut_info(best_point),
                iter = iter,
                diagnostics = diagnostics,
            )
        end

        trial_level = isfinite(D_upper) ?
            D_lower + λ * (D_upper - D_lower) :
            D_lower
    end
end
