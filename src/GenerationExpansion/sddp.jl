"""
Run the SDDiP / SDDP / SDDPL algorithm.

Returns a Dict with:
- :solHistory => DataFrame of iterations (LB, UB, gap, times)
- :solution   => first-stage solution
- :gapHistory => Vector of gaps
"""
function stochastic_dual_dynamic_programming_algorithm(
    Ω::Dict{Int, Dict{Int, RandomVariables}},
    probList::Dict{Int, Vector{Float64}},
    stageDataList::Dict{Int, StageData};
    binaryInfo::BinaryInfo = binaryInfo,
    param::SDDPParam = param,
)::Dict
    # iteration counter and bounds
    i  = 1
    LB = -Inf
    ub_type = :mean
    UB = Inf

    solCollection = Dict()
    u         = 0.0
    Scenarios = 0

    # result DataFrame: columns (iter, LB, UB, gap, iter_time, total_time)
    col_names = [:iter, :LB, :UB, :gap, :time, :Time]
    col_types = [Int, Float64, Float64, String, Float64, Float64]
    named_tuple = (; zip(col_names, T[] for T in col_types)...)
    sddipResult = DataFrame(named_tuple)
    gapList     = Float64[]

    # build forward models on all workers
    @everywhere begin
        forwardInfoList = Dict{Int, StageModel}()
        for t in 1:param.T
            forwardInfoList[t] = forwardModel!(stageDataList[t], binaryInfo = binaryInfo, param = param)
        end
    end

    initial    = now()
    iter_time  = 0.0
    total_Time = 0.0
    t0         = now()

    while true
        t0 = now()

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
        forwardPassResult = pmap(1:param.sample_size_SDDP) do k
            forwardPass(k, Scenarios)
        end

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
        gap = round((UB - LB) / UB * 100, digits = 2)
        gapString = string(gap, "%")

        # time info
        t1         = now()
        iter_time  = (t1 - t0).value / 1000
        total_Time = (t1 - initial).value / 1000

        # log into DataFrame and gap list
        push!(sddipResult, [i, LB, UB, gapString, iter_time, total_Time])
        push!(gapList, gap)

        # pretty printing
        if i == 1
            print_iteration_info_bar()
        end
        print_iteration_info(i, LB, UB, gap, iter_time, 0, total_Time)

        # save (if enabled)
        save_results_info(
            param,
            Dict(
                :solHistory => sddipResult,
                :gapHistory => gapList,
            )
        )

        # stopping condition: 时间 or gap
        if total_Time > param.timeSDDP || gap ≤ param.gapSDDP
            return Dict(
                :solHistory => sddipResult,
                :solution   => solCollection[1, 1].IntVar,
                :gapHistory => gapList,
            )
        end

        if param.algorithm == :SDDPL && i ≥ param.branchingStart
            for t in 1:param.T-1
                for ω in [1]  # 这里你可以换成真正的节点集合 keys(Ξ̃)
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
                        elseif param.partitionRule == :Incumbent
                            med = round(solCollection[t, ω].IntVar[g], digits = 2)
                        end

                        # create two new leaf nodes and update info
                        left  = length(forwardInfoList[t].model[:region_indicator][g]) + 1
                        right = left + 1

                        @everywhere begin
                            t_local               = $t
                            left_local            = $left
                            right_local           = $right
                            g_local               = $g
                            lb_local              = $lb
                            ub_local              = $ub
                            med_local             = $med
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
                                :lb      => lb_local,
                                :ub      => med_local,
                                :parent  => keys_with_value_1_loc,
                                :sibling => right_local,
                                :var     => forwardInfoList[t_local].model[:region_indicator][g_local][left_local],
                            )
                            forwardInfoList[t_local].IntVarLeaf[:St][g_local][right_local] = Dict(
                                :lb      => med_local,
                                :ub      => ub_local,
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
                            if solCollection[t, ω].IntVar[g] ≤ med
                                stageDecision.IntVarLeaf[g][left] = 1.0
                            else
                                stageDecision.IntVarLeaf[g][right] = 1.0
                            end
                        else
                            # dense representation: all leaves present with 0/1
                            for k_leaf in keys(forwardInfoList[t].IntVarLeaf[:St][g])
                                stageDecision.IntVarLeaf[g][k_leaf] = 0.0
                            end
                            if solCollection[t, ω].IntVar[g] ≤ med
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
        
        
        for t in reverse(2:param.T)
            for ω in 1:param.M
                backwardNodeInfoList = Dict{Int, Tuple}()

                for j in keys(Ω[t])
                    backwardNodeInfoList[j] = (i, t, j, ω)
                end

                backwardPassResult = pmap(values(backwardNodeInfoList)) do backwardNodeInfo
                    backwardPass(backwardNodeInfo, solCollection)
                end

                # 初始化 λ₀, λ₁（按 algorithm 设置结构）
                λ₀ = sum(probList[t][j] * backwardPassResult[j][1] for j in keys(Ω[t]))

                if param.algorithm == :SDDP
                    IntVar_init         = sum(probList[t][j] * backwardPassResult[j][2].IntVar for j in keys(Ω[t]))
                    IntVarLeaf_init     = nothing
                    IntVarBinaries_init = nothing
                elseif param.algorithm == :SDDPL
                    IntVar_init         = sum(probList[t][j] * backwardPassResult[j][2].IntVar for j in keys(Ω[t]))
                    IntVarLeaf_init = Dict(
                        g => Dict(
                            k => sum(
                                sum(probList[t][j] * backwardPassResult[j][2].IntVarLeaf[g][k] for j in keys(Ω[t]))
                            ) for k in keys(solCollection[t-1, ω].IntVarLeaf[g])
                        ) for g in 1:binaryInfo.d
                    )
                    IntVarBinaries_init = nothing
                elseif param.algorithm == :SDDiP
                    IntVar_init         = nothing
                    IntVarLeaf_init     = nothing
                    IntVarBinaries_init = sum(probList[t][j] * backwardPassResult[j][2].IntVarBinaries for j in keys(Ω[t]))
                end

                λ₁ = StageInfo(
                    sum(probList[t][j] * backwardPassResult[j][2].StateValue for j in keys(Ω[t])),
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

                    # state 部分 (SDDP / SDDPL)
                    term_state = 0.0
                    if param.algorithm == :SDDP || param.algorithm == :SDDPL
                        term_state = λ₁_loc.IntVar' * forwardInfoList[t_local - 1].model[:St]
                    end

                    # leaf 部分 (只有 SDDPL)
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

                    # binary 部分 (只有 SDDiP)
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

        # advance iteration counter
        i += 1
    end
end