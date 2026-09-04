"""
Run the SDDiP / SDDP / SDDPL algorithm.

Returns a Dict with:
- :solHistory => DataFrame of iterations (LB, UB, gap, times)
- :solution   => first-stage integer state in the original (non-binarized) space
- :gapHistory => Vector of gaps
"""
function first_stage_integer_solution(
    stateInfo::StageInfo,
    binaryInfo::BinaryInfo,
)::Vector{Float64}
    if stateInfo.IntVar !== nothing
        return Float64.(stateInfo.IntVar)
    elseif stateInfo.IntVarBinaries !== nothing
        return Float64.(binaryInfo.A * stateInfo.IntVarBinaries)
    end

    error("The first-stage state contains neither integer nor binarized values.")
end

function add_relu_cut_to_forward_model!(
    model::Model,
    cutInfo::ReLUCutInfo,
    stateInfo::StageInfo,
)
    cut_id = get!(model.ext, :ReLUCutCounter, 0) + 1
    model.ext[:ReLUCutCounter] = cut_id

    dim = length(stateInfo.IntVar)
    πₙ = cutInfo.dualInfo
    πₙ₀ = πₙ.StateValue === nothing ? 1.0 : πₙ.StateValue
    lifted_term =
        πₙ.IntVarLeaf === nothing ||
        stateInfo.IntVarLeaf === nothing ||
        :region_indicator ∉ keys(model.obj_dict) ?
        0.0 :
        sum(
            sum(
                πₙ.IntVarLeaf[g][k] * model[:region_indicator][g][k]
                for k in keys(πₙ.IntVarLeaf[g]);
                init = 0.0,
            )
            for g in keys(πₙ.IntVarLeaf);
            init = 0.0,
        )

    w_plus = @variable(
        model,
        [g = 1:dim],
        lower_bound = 0.0,
        base_name = "relu_cut_w_plus[$cut_id]",
    )
    w_minus = @variable(
        model,
        [g = 1:dim],
        lower_bound = 0.0,
        base_name = "relu_cut_w_minus[$cut_id]",
    )
    r = @variable(
        model,
        [g = 1:dim],
        binary = true,
        base_name = "relu_cut_r[$cut_id]",
    )

    @constraint(
        model,
        [g = 1:dim],
        w_plus[g] - w_minus[g] == model[:St][g] - stateInfo.IntVar[g],
    )
    @constraint(
        model,
        [g = 1:dim],
        w_plus[g] ≤ (upper_bound(model[:St][g]) - stateInfo.IntVar[g]) * r[g],
    )
    @constraint(
        model,
        [g = 1:dim],
        w_minus[g] ≤ stateInfo.IntVar[g] * (1 - r[g]),
    )
    @constraint(
        model,
        πₙ₀ * model[:θ] +
        πₙ.IntVarPlus' * w_plus +
        πₙ.IntVarMinus' * w_minus -
        lifted_term ≥ cutInfo.rhs,
    )

    return
end

function aggregate_relu_leaf_dual(
    backwardPassResult,
    stageProbabilities::AbstractVector{<:Real},
    scale,
)
    first_leaf = backwardPassResult[first(eachindex(stageProbabilities))].dualInfo.IntVarLeaf
    if first_leaf === nothing
        return nothing
    end

    return Dict(
        g => Dict(
            k => sum(
                stageProbabilities[j] *
                backwardPassResult[j].dualInfo.IntVarLeaf[g][k] /
                scale(j)
                for j in eachindex(stageProbabilities)
            )
            for k in keys(first_leaf[g])
        )
        for g in keys(first_leaf)
    )
end

"""
    aggregate_relu_cut_info(
        backwardPassResult,
        stageProbabilities,
        cutType,
    )::ReLUCutInfo

Aggregate the scenario cuts returned by the backward pass into the single-cut
SDDP master.

- For regular ReLU cuts, we take the probability-weighted average of the raw
  cut coefficients.
- For normalized ReLU cuts, we first rescale every scenario cut so that the
  coefficient of the epigraph variable `θ` is equal to one, and only then take
  the expectation. This normalization is required because the single-cut SDDP
  master carries one variable `θ` for the expected downstream cost, rather than
  separate scenario-wise variables.
"""
function aggregate_relu_cut_info(
    backwardPassResult,
    stageProbabilities::AbstractVector{<:Real},
    cutType::Symbol,
)::ReLUCutInfo
    if cutType == :ReLUC
        return ReLUCutInfo(
            sum(stageProbabilities[j] * backwardPassResult[j].rhs for j in eachindex(stageProbabilities)),
            ReLUDualStageInfo(
                nothing,
                sum(stageProbabilities[j] * backwardPassResult[j].dualInfo.IntVarPlus for j in eachindex(stageProbabilities)),
                sum(stageProbabilities[j] * backwardPassResult[j].dualInfo.IntVarMinus for j in eachindex(stageProbabilities)),
                aggregate_relu_leaf_dual(
                    backwardPassResult,
                    stageProbabilities,
                    _ -> 1.0,
                ),
            ),
        )
    elseif cutType == :NormalizedReLUC
        for j in eachindex(stageProbabilities)
            π₀ = backwardPassResult[j].dualInfo.StateValue
            if π₀ === nothing || !isfinite(π₀) || π₀ ≤ RELU_MIN_NORMALIZED_PI0
                error(
                    "Normalized ReLU aggregation requires a strictly positive " *
                    "scenario coefficient on θ. Encountered π₀ = $(π₀) for " *
                    "scenario $j.",
                )
            end
        end

        return ReLUCutInfo(
            sum(
                stageProbabilities[j] *
                backwardPassResult[j].rhs /
                backwardPassResult[j].dualInfo.StateValue
                for j in eachindex(stageProbabilities)
            ),
            ReLUDualStageInfo(
                1.0,
                sum(
                    stageProbabilities[j] *
                    backwardPassResult[j].dualInfo.IntVarPlus /
                    backwardPassResult[j].dualInfo.StateValue
                    for j in eachindex(stageProbabilities)
                ),
                sum(
                    stageProbabilities[j] *
                    backwardPassResult[j].dualInfo.IntVarMinus /
                    backwardPassResult[j].dualInfo.StateValue
                    for j in eachindex(stageProbabilities)
                ),
                aggregate_relu_leaf_dual(
                    backwardPassResult,
                    stageProbabilities,
                    j -> backwardPassResult[j].dualInfo.StateValue,
                ),
            ),
        )
    else
        error("Unsupported ReLU cut type: $cutType")
    end
end

"""
Create the per-iteration runtime table used by diagnostic experiments.
"""
function initialize_runtime_history()::DataFrame
    return DataFrame(
        Iter = Int[],
        Algorithm = Symbol[],
        Cut = Symbol[],
        forward_time = Float64[],
        lifting_time = Float64[],
        backward_time = Float64[],
        other_time = Float64[],
        iteration_time = Float64[],
        total_time = Float64[],
        completed_backward = Bool[],
        min_lnc_scale = Float64[],
        max_lnc_tightness_gap = Float64[],
        num_lnc_fallback = Int[],
        num_lnc_core_theta_fallback = Int[],
    )
end

"""
Return elapsed wall-clock seconds from a `time_ns()` timestamp.
"""
elapsed_seconds(start_time::UInt64)::Float64 = (time_ns() - start_time) / 1e9

"""
Append one per-iteration runtime observation.
"""
function push_runtime_history!(
    runtimeHistory::DataFrame;
    iter::Int,
    algorithm::Symbol,
    cut::Symbol,
    forward_time::Float64,
    lifting_time::Float64,
    backward_time::Float64,
    iteration_time::Float64,
    total_time::Float64,
    completed_backward::Bool,
    min_lnc_scale::Float64 = NaN,
    max_lnc_tightness_gap::Float64 = NaN,
    num_lnc_fallback::Int = 0,
    num_lnc_core_theta_fallback::Int = 0,
)::Nothing
    measured_time = forward_time + lifting_time + backward_time
    other_time = max(iteration_time - measured_time, 0.0)

    push!(
        runtimeHistory,
        (
            Iter = iter,
            Algorithm = algorithm,
            Cut = cut,
            forward_time = forward_time,
            lifting_time = lifting_time,
            backward_time = backward_time,
            other_time = other_time,
            iteration_time = iteration_time,
            total_time = total_time,
            completed_backward = completed_backward,
            min_lnc_scale = min_lnc_scale,
            max_lnc_tightness_gap = max_lnc_tightness_gap,
            num_lnc_fallback = num_lnc_fallback,
            num_lnc_core_theta_fallback = num_lnc_core_theta_fallback,
        ),
    )

    return nothing
end

function stochastic_dual_dynamic_programming_algorithm(
    Ω::Dict{Int, Dict{Int, RandomVariables}},
    probList::Dict{Int, Vector{Float64}},
    stageDataList::Dict{Int, StageData};
    binaryInfo::BinaryInfo = binaryInfo,
    param::SDDPParam = param,
)::Dict
    param.sample_size_SDDP >= 2 || throw(ArgumentError(
        "sample_size_SDDP must be at least 2 because the statistical upper " *
        "bound uses a corrected sample variance.",
    ))
    1 <= param.M <= param.sample_size_SDDP || throw(ArgumentError(
        "M must be between 1 and sample_size_SDDP.",
    ))
    param.iterSDDP >= 1 || throw(ArgumentError("iterSDDP must be positive."))
    0.0 <= param.gapSDDP <= 1.0 || throw(ArgumentError(
        "gapSDDP is a relative tolerance and must be in [0, 1].",
    ))
    0.0 <= param.solverGap <= 1.0 || throw(ArgumentError(
        "solverGap must be in [0, 1].",
    ))
    param.timeSDDP >= 0.0 || throw(ArgumentError("timeSDDP must be nonnegative."))
    param.solverTime > 0.0 || throw(ArgumentError("solverTime must be positive."))
    is_supported_configuration(param.algorithm, param.cutType) || throw(ArgumentError(
        "Unsupported GEP algorithm/cut configuration: " *
        "$(param.algorithm)/$(param.cutType).",
    ))
    param.levelMethodMaxIter >= 1 || throw(ArgumentError(
        "levelMethodMaxIter must be positive.",
    ))
    param.partitionRule in (:Bisection, :Incumbent) || throw(ArgumentError(
        "partitionRule must be :Bisection or :Incumbent.",
    ))
    param.T == length(stageDataList) == length(Ω) == length(probList) ||
        throw(ArgumentError("T is inconsistent with the supplied stage data."))

    # iteration counter and bounds
    i  = 1
    LB = -Inf
    ub_type = :meanvariance
    UB = Inf
    LM_iter = 0

    solCollection = Dict()
    u         = 0.0
    Scenarios = 0

    # result DataFrame: columns (iter, LB, UB, gap, iter_time, LM_iter, total_time)
    col_names = [:iter, :LB, :UB, :gap, :time, :LM_iter, :Time]
    col_types = [Int, Float64, Float64, String, Float64, Int, Float64]
    named_tuple = (; zip(col_names, T[] for T in col_types)...)
    sddipResult = DataFrame(named_tuple)
    gapList     = Float64[]
    runtimeHistory = initialize_runtime_history()

    # build forward models on all workers
    @everywhere begin
        forwardInfoList = Dict{Int, StageModel}()
        for t in 1:param.T
            forwardInfoList[t] = forwardModel!(stageDataList[t], binaryInfo = binaryInfo, param = param)
        end
    end

    initial    = now()
    total_Time = 0.0
    while true
        iteration_timer = time_ns()
        forward_time = 0.0
        lifting_time = 0.0
        backward_time = 0.0
        active_cut_type = get_cutType(param.cutType, i)
        min_lnc_scale = Inf
        max_lnc_tightness_gap = 0.0
        num_lnc_fallback = 0
        num_lnc_core_theta_fallback = 0
        LM_iter = 0

        # container for this iteration
        solCollection = Dict()
        u = Vector{Float64}(undef, param.sample_size_SDDP)

        # sample scenarios
        Random.seed!(i)
        Scenarios = SampleScenarios(
            Ω,
            probList;
            M = param.sample_size_SDDP,
        )

        ########################################
        ### Forward pass
        ########################################
        forward_timer = time_ns()
        forwardPassResult = pmap(1:param.sample_size_SDDP) do k
            forwardPass(k, Scenarios)
        end
        forward_time = elapsed_seconds(forward_timer)

        for k in 1:param.sample_size_SDDP
            for t in 1:param.T
                solCollection[t, k] = forwardPassResult[k][t, k]
            end
            u[k] = sum(solCollection[t, k].StageValue for t in 1:param.T)
        end

        ########################################
        ### Upper / lower bounds and gap
        ########################################
        LB = solCollection[1, 1].StateValue

        μ̄  = mean(u)
        if  ub_type == :mean
            UB  = μ̄
        elseif ub_type == :meanvariance
            σ̂² = Statistics.var(u)
            UB  = μ̄ + 1.96 * sqrt(σ̂² / param.sample_size_SDDP)
        end
        relative_gap = (UB - LB) / UB
        gap = round(relative_gap * 100, digits = 2)
        gapString = string(gap, "%")
        gap_converged = UB >= LB && relative_gap ≤ param.gapSDDP

        # Add the bound observation now and fill in this iteration's runtime
        # fields after either the stopping check or the backward pass.
        current_total_time = (now() - initial).value / 1000
        push!(sddipResult, [i, LB, UB, gapString, forward_time, 0, current_total_time])
        push!(gapList, gap)

        # stopping condition: time limit, target gap, or iteration budget
        if current_total_time > param.timeSDDP ||
           gap_converged ||
           i ≥ param.iterSDDP
            iteration_time = elapsed_seconds(iteration_timer)
            total_Time = (now() - initial).value / 1000
            sddipResult.time[end] = iteration_time
            sddipResult.LM_iter[end] = 0
            sddipResult.Time[end] = total_Time

            if i == 1
                print_iteration_info_bar()
            end
            print_iteration_info(i, LB, UB, gap, iteration_time, 0, total_Time)

            push_runtime_history!(
                runtimeHistory;
                iter = i,
                algorithm = param.algorithm,
                cut = active_cut_type,
                forward_time = forward_time,
                lifting_time = lifting_time,
                backward_time = backward_time,
                iteration_time = iteration_time,
                total_time = total_Time,
                completed_backward = false,
                min_lnc_scale = isfinite(min_lnc_scale) ? min_lnc_scale : NaN,
                max_lnc_tightness_gap = active_cut_type == :LNC ? max_lnc_tightness_gap : NaN,
                num_lnc_fallback = num_lnc_fallback,
                num_lnc_core_theta_fallback = num_lnc_core_theta_fallback,
            )
            save_results_info(
                param,
                Dict(
                    :solHistory => sddipResult,
                    :gapHistory => gapList,
                    :runtimeHistory => runtimeHistory,
                )
            )
            return Dict(
                :solHistory => sddipResult,
                :solution   => first_stage_integer_solution(
                    solCollection[1, 1],
                    binaryInfo,
                ),
                :gapHistory => gapList,
                :runtimeHistory => runtimeHistory,
            )
        end

        if param.algorithm == :SDDPL
            lifting_timer = time_ns()
            if i ≥ param.branchingStart
                for t in 1:param.T-1
                    for ω in [1]  # replace with the full node set if node-wise branching is enabled
                        dev = Dict{Int, Float64}()

                        # compute deviation for each generator
                        for g in 1:binaryInfo.d
                            k_leaf = maximum(
                                [
                                    k for (k, v) in solCollection[t, ω].IntVarLeaf[g]
                                    if v == maximum(values(solCollection[t, ω].IntVarLeaf[g]))
                                ],
                            )
                            info = forwardInfoList[t].IntVarLeaf[:St][g][k_leaf]
                            dev[g] = min(
                                (info[:ub] - solCollection[t, ω].IntVar[g]),
                                (solCollection[t, ω].IntVar[g] - info[:lb]),
                            )
                            if param.cutSparsity
                                for k in collect(keys(solCollection[t, ω].IntVarLeaf[g]))
                                    solCollection[t, ω].IntVarLeaf[g][k] == 1.0 || delete!(solCollection[t, ω].IntVarLeaf[g], k)
                                end
                            end
                        end

                        # choose generator with largest deviation
                        g = [k for (k, v) in dev if v == maximum(values(dev))][1]

                        if dev[g] ≥ 1e-8
                            # find active leaf node
                            keys_with_value_1 = maximum(
                                [k for (k, v) in solCollection[t, ω].IntVarLeaf[g] if v == 1],
                            )

                            # current interval [lb, ub] and split point med
                            lb  = forwardInfoList[t].IntVarLeaf[:St][g][keys_with_value_1][:lb]
                            ub  = forwardInfoList[t].IntVarLeaf[:St][g][keys_with_value_1][:ub]
                            if param.partitionRule == :Bisection
                                med = floor(Int, (lb + ub) / 2)
                                left_lb = lb
                                left_ub = med
                                right_lb = med + 1
                                right_ub = ub
                            elseif param.partitionRule == :Incumbent
                                med = round(solCollection[t, ω].IntVar[g], digits = 2)
                                left_lb = lb
                                left_ub = med
                                right_lb = med
                                right_ub = ub
                            end

                            # create two new leaf nodes and update info
                            left  = length(forwardInfoList[t].model[:region_indicator][g]) + 1
                            right = left + 1

                            @everywhere begin
                                t_local               = $t
                                left_local            = $left
                                right_local           = $right
                                g_local               = $g
                                left_lb_local         = $left_lb
                                left_ub_local         = $left_ub
                                right_lb_local        = $right_lb
                                right_ub_local        = $right_ub
                                keys_with_value_1_loc = $keys_with_value_1

                                # forward model: new region indicators
                                forwardInfoList[t_local].model[:region_indicator][g_local][left_local] = @variable(
                                    forwardInfoList[t_local].model,
                                    base_name = "region_indicator[$g_local, $left_local]",
                                    binary    = true,
                                )
                                forwardInfoList[t_local].model[:region_indicator][g_local][right_local] = @variable(
                                    forwardInfoList[t_local].model,
                                    base_name = "region_indicator[$g_local, $right_local]",
                                    binary    = true,
                                )

                                forwardInfoList[t_local].IntVarLeaf[:St][g_local][left_local] = Dict(
                                    :lb      => left_lb_local,
                                    :ub      => left_ub_local,
                                    :parent  => keys_with_value_1_loc,
                                    :sibling => right_local,
                                    :var     => forwardInfoList[t_local].model[:region_indicator][g_local][left_local],
                                )
                                forwardInfoList[t_local].IntVarLeaf[:St][g_local][right_local] = Dict(
                                    :lb      => right_lb_local,
                                    :ub      => right_ub_local,
                                    :parent  => keys_with_value_1_loc,
                                    :sibling => left_local,
                                    :var     => forwardInfoList[t_local].model[:region_indicator][g_local][right_local],
                                )

                                # remove old leaf node
                                delete!(
                                    forwardInfoList[t_local].IntVarLeaf[:St][g_local],
                                    keys_with_value_1_loc,
                                )

                                # logic constraints (forward models)
                                @constraint(
                                    forwardInfoList[t_local].model,
                                    forwardInfoList[t_local].model[:region_indicator][g_local][left_local] +
                                    forwardInfoList[t_local].model[:region_indicator][g_local][right_local] ==
                                    forwardInfoList[t_local].model[:region_indicator][g_local][keys_with_value_1_loc],
                                )
                                delete(
                                    forwardInfoList[t_local].model,
                                    forwardInfoList[t_local].model[:partition_lower_bound][g_local]
                                )
                                forwardInfoList[t_local].model[:partition_lower_bound][g_local] = @constraint(
                                    forwardInfoList[t_local].model,
                                    forwardInfoList[t_local].model[:St][g_local] ≥ sum(
                                        forwardInfoList[t_local].IntVarLeaf[:St][g_local][k][:lb] *
                                        forwardInfoList[t_local].model[:region_indicator][g_local][k]
                                        for k in keys(forwardInfoList[t_local].IntVarLeaf[:St][g_local])
                                    ),
                                )
                                delete(
                                    forwardInfoList[t_local].model,
                                    forwardInfoList[t_local].model[:partition_upper_bound][g_local]
                                )
                                forwardInfoList[t_local].model[:partition_upper_bound][g_local] = @constraint(
                                    forwardInfoList[t_local].model,
                                    forwardInfoList[t_local].model[:St][g_local] ≤ sum(
                                        forwardInfoList[t_local].IntVarLeaf[:St][g_local][k][:ub] *
                                        forwardInfoList[t_local].model[:region_indicator][g_local][k]
                                        for k in keys(forwardInfoList[t_local].IntVarLeaf[:St][g_local])
                                    ),
                                )

                                # backward-side region_indicator_copy
                                if param.discreteZ
                                    forwardInfoList[t_local + 1].model[:region_indicator_copy][g_local][left_local] =
                                        @variable(
                                            forwardInfoList[t_local + 1].model,
                                            base_name = "region_indicator_copy[$g_local, $left_local]",
                                            binary    = true,
                                        )
                                    forwardInfoList[t_local + 1].model[:region_indicator_copy][g_local][right_local] =
                                        @variable(
                                            forwardInfoList[t_local + 1].model,
                                            base_name = "region_indicator_copy[$g_local, $right_local]",
                                            binary    = true,
                                        )
                                else
                                    forwardInfoList[t_local + 1].model[:region_indicator_copy][g_local][left_local] =
                                        @variable(
                                            forwardInfoList[t_local + 1].model,
                                            base_name   = "region_indicator_copy[$g_local, $left_local]",
                                            lower_bound = 0,
                                            upper_bound = 1,
                                        )
                                    forwardInfoList[t_local + 1].model[:region_indicator_copy][g_local][right_local] =
                                        @variable(
                                            forwardInfoList[t_local + 1].model,
                                            base_name   = "region_indicator_copy[$g_local, $right_local]",
                                            lower_bound = 0,
                                            upper_bound = 1,
                                        )
                                end

                                @constraint(
                                    forwardInfoList[t_local + 1].model,
                                    forwardInfoList[t_local + 1].model[:region_indicator_copy][g_local][left_local] +
                                    forwardInfoList[t_local + 1].model[:region_indicator_copy][g_local][right_local] ==
                                    forwardInfoList[t_local + 1].model[:region_indicator_copy][g_local][keys_with_value_1_loc],
                                )
                                delete(
                                    forwardInfoList[t_local + 1].model,
                                    forwardInfoList[t_local + 1].model[:partition_lower_bound_copy][g_local]
                                )
                                forwardInfoList[t_local + 1].model[:partition_lower_bound_copy][g_local] = @constraint(
                                    forwardInfoList[t_local + 1].model,
                                    forwardInfoList[t_local + 1].model[:Sc][g_local] ≥ sum(
                                        forwardInfoList[t_local].IntVarLeaf[:St][g_local][k][:lb] *
                                        forwardInfoList[t_local + 1].model[:region_indicator_copy][g_local][k]
                                        for k in keys(forwardInfoList[t_local].IntVarLeaf[:St][g_local])
                                    ),
                                )
                                delete(
                                    forwardInfoList[t_local + 1].model,
                                    forwardInfoList[t_local + 1].model[:partition_upper_bound_copy][g_local]
                                )
                                forwardInfoList[t_local + 1].model[:partition_upper_bound_copy][g_local] = @constraint(
                                    forwardInfoList[t_local + 1].model,
                                    forwardInfoList[t_local + 1].model[:Sc][g_local] ≤ sum(
                                        forwardInfoList[t_local].IntVarLeaf[:St][g_local][k][:ub] *
                                        forwardInfoList[t_local + 1].model[:region_indicator_copy][g_local][k]
                                        for k in keys(forwardInfoList[t_local].IntVarLeaf[:St][g_local])
                                    ),
                                )
                            end # @everywhere

                            # update stageDecision's IntVarLeaf
                            stageDecision = deepcopy(solCollection[t, ω])
                            stageDecision.IntVarLeaf[g] = Dict{Int, Float64}()

                            if param.cutSparsity
                                # only keep one active leaf
                                if solCollection[t, ω].IntVar[g] ≤ left_ub
                                    stageDecision.IntVarLeaf[g][left] = 1.0
                                else
                                    stageDecision.IntVarLeaf[g][right] = 1.0
                                end
                            else
                                # dense representation: all leaves present with 0/1
                                for k_leaf in keys(forwardInfoList[t].IntVarLeaf[:St][g])
                                    stageDecision.IntVarLeaf[g][k_leaf] = 0.0
                                end
                                if solCollection[t, ω].IntVar[g] ≤ left_ub
                                    stageDecision.IntVarLeaf[g][left] = 1.0
                                else
                                    stageDecision.IntVarLeaf[g][right] = 1.0
                                end
                            end

                            solCollection[t, ω] = deepcopy(stageDecision)
                        end
                    end
                end
            end
            lifting_time = elapsed_seconds(lifting_timer)
        end


        num_backward_subproblems = 0
        backward_timer = time_ns()
        for t in reverse(2:param.T)
            for ω in 1:param.M
                node_ids = sort(collect(keys(Ω[t])))
                backwardPassResultList = pmap(node_ids) do j
                    backwardPass((i, t, j, ω), solCollection)
                end
                backwardPassResult = Dict(
                    node_ids[j] => backwardPassResultList[j]
                    for j in eachindex(node_ids)
                )

                if active_cut_type == :LNC
                    for j in node_ids
                        min_lnc_scale = min(min_lnc_scale, backwardPassResult[j].lnc_scale)
                        max_lnc_tightness_gap = max(
                            max_lnc_tightness_gap,
                            backwardPassResult[j].lnc_tightness_gap,
                        )
                        num_lnc_fallback += backwardPassResult[j].lnc_fallback ? 1 : 0
                        num_lnc_core_theta_fallback +=
                            backwardPassResult[j].lnc_core_theta_fallback ? 1 : 0
                    end
                end

                LM_iter += sum(backwardPassResult[j].iter for j in node_ids)
                num_backward_subproblems += length(node_ids)

                if active_cut_type == :ReLUC || active_cut_type == :NormalizedReLUC
                    reluCutInfo = aggregate_relu_cut_info(
                        backwardPassResult,
                        probList[t],
                        active_cut_type,
                    )

                    @everywhere begin
                        t_local = $t
                        relu_cut_loc = $reluCutInfo
                        state_info_loc = $(solCollection[t-1, ω])
                        add_relu_cut_to_forward_model!(
                            forwardInfoList[t_local - 1].model,
                            relu_cut_loc,
                            state_info_loc,
                        )
                    end
                else
                    # Initialize the aggregated cut coefficients in the
                    # representation required by the active algorithm.
                    cut_scale(j) =
                        active_cut_type == :LNC ? -backwardPassResult[j].dualInfo.StateValue : 1.0

                    if active_cut_type == :LNC
                        for j in node_ids
                            scale = cut_scale(j)
                            if !isfinite(scale) || scale < param.lncMinScale
                                error(
                                    "LNC aggregation requires -pi0 >= $(param.lncMinScale) before " *
                                    "normalization. Encountered -pi0 = $(scale) " *
                                    "at stage $t, node $j, sample $ω.",
                                )
                            end
                        end
                    end

                    λ₀ = sum(
                        probList[t][j] * backwardPassResult[j].rhs / cut_scale(j)
                        for j in node_ids
                    )

                    if param.algorithm == :SDDP
                        IntVar_init = sum(
                            probList[t][j] * backwardPassResult[j].dualInfo.IntVar / cut_scale(j)
                            for j in node_ids
                        )
                        IntVarLeaf_init     = nothing
                        IntVarBinaries_init = nothing
                    elseif param.algorithm == :SDDPL
                        IntVar_init = sum(
                            probList[t][j] * backwardPassResult[j].dualInfo.IntVar / cut_scale(j)
                            for j in node_ids
                        )
                        IntVarLeaf_init = Dict(
                            g => Dict(
                                k => sum(
                                    probList[t][j] *
                                    backwardPassResult[j].dualInfo.IntVarLeaf[g][k] /
                                    cut_scale(j)
                                    for j in node_ids
                                ) for k in keys(solCollection[t-1, ω].IntVarLeaf[g])
                            ) for g in 1:binaryInfo.d
                        )
                        IntVarBinaries_init = nothing
                    elseif param.algorithm == :SDDiP
                        IntVar_init         = nothing
                        IntVarLeaf_init     = nothing
                        IntVarBinaries_init = sum(
                            probList[t][j] *
                            backwardPassResult[j].dualInfo.IntVarBinaries /
                            cut_scale(j)
                            for j in node_ids
                        )
                    end

                    λ₁ = StageInfo(
                        active_cut_type == :LNC ?
                            -1.0 :
                            sum(probList[t][j] * backwardPassResult[j].dualInfo.StateValue for j in node_ids),
                        nothing,
                        IntVar_init,
                        IntVarLeaf_init,
                        IntVarBinaries_init,
                    )

                    # add cut to forward models (t-1)
                    @everywhere begin
                        t_local = $t
                        λ₀_loc  = $λ₀
                        λ₁_loc  = $λ₁

                        # State contribution used by SDDP and SDDPL.
                        term_state = 0.0
                        if param.algorithm == :SDDP || param.algorithm == :SDDPL
                            term_state = λ₁_loc.IntVar' * forwardInfoList[t_local - 1].model[:St]
                        end

                        # Surrogate-leaf contribution used only by SDDPL.
                        term_leaf = 0.0
                        if param.algorithm == :SDDPL
                            term_leaf = sum(
                                sum(
                                    λ₁_loc.IntVarLeaf[g][k] *
                                    forwardInfoList[t_local - 1].model[:region_indicator][g][k]
                                    for k in keys(λ₁_loc.IntVarLeaf[g])
                                ) for g in 1:binaryInfo.d
                            )
                        end

                        # Binary-state contribution used only by SDDiP.
                        term_bin = 0.0
                        if param.algorithm == :SDDiP
                            term_bin = λ₁_loc.IntVarBinaries' * forwardInfoList[t_local - 1].model[:Lt]
                        end

                        rhs = term_state + term_leaf + term_bin + λ₀_loc

                        @constraint(
                            forwardInfoList[t_local - 1].model,
                            λ₁_loc.StateValue * forwardInfoList[t_local - 1].model[:θ] + rhs ≤ 0
                        )
                    end
                end
            end
        end
        backward_time = elapsed_seconds(backward_timer)

        LM_iter = num_backward_subproblems == 0 ? 0 : floor(Int, LM_iter / num_backward_subproblems)

        # time info
        total_Time = (now() - initial).value / 1000
        runtime_iteration_time = elapsed_seconds(iteration_timer)
        sddipResult.time[end] = runtime_iteration_time
        sddipResult.LM_iter[end] = LM_iter
        sddipResult.Time[end] = total_Time

        if i == 1
            print_iteration_info_bar()
        end
        print_iteration_info(i, LB, UB, gap, runtime_iteration_time, LM_iter, total_Time)

        push_runtime_history!(
            runtimeHistory;
            iter = i,
            algorithm = param.algorithm,
            cut = active_cut_type,
            forward_time = forward_time,
            lifting_time = lifting_time,
            backward_time = backward_time,
            iteration_time = runtime_iteration_time,
            total_time = total_Time,
            completed_backward = true,
            min_lnc_scale = isfinite(min_lnc_scale) ? min_lnc_scale : NaN,
            max_lnc_tightness_gap = active_cut_type == :LNC ? max_lnc_tightness_gap : NaN,
            num_lnc_fallback = num_lnc_fallback,
            num_lnc_core_theta_fallback = num_lnc_core_theta_fallback,
        )
        save_results_info(
            param,
            Dict(
                :solHistory => sddipResult,
                :gapHistory => gapList,
                :runtimeHistory => runtimeHistory,
            ),
        )
        if total_Time > param.timeSDDP
            return Dict(
                :solHistory => sddipResult,
                :solution => first_stage_integer_solution(
                    solCollection[1, 1],
                    binaryInfo,
                ),
                :gapHistory => gapList,
                :runtimeHistory => runtimeHistory,
            )
        end
        i += 1
    end
end
