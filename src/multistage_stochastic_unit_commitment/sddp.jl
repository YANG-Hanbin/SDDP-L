"""
    function stochastic_dual_dynamic_programming_algorithm(
            scenarioTree::ScenarioTree,                   
            indexSets::IndexSets,                        
            paramDemand::ParamDemand,
            paramOPF::ParamOPF;
            initialStateInfo::StateInfo = initialStateInfo,
            param_cut::NamedTuple = param_cut, 
            param_levelsetmethod::NamedTuple = param_levelsetmethod, 
            param::NamedTuple = param
    )

# Arguments
  1. `scenarioTree::ScenarioTree`: A scenario tree
  2. `indexSets::IndexSets`: A set of indices
  3. `paramDemand::ParamDemand`: Demand parameters
  4. `paramOPF::ParamOPF`: OPF parameters
  5. `initialStateInfo::StateInfo`: Initial state information
  6. `param_cut::NamedTuple`: Named tuple of parameters for the PLC algorithm
  7. `param_levelsetmethod::NamedTuple`: Named tuple of parameters for the level set method
  8. `param::NamedTuple`: Named tuple of parameters for the SDDiP algorithm
"""
function stochastic_dual_dynamic_programming_algorithm(
        scenarioTree::ScenarioTree,                   
        indexSets::IndexSets,                        
        paramDemand::ParamDemand,
        paramOPF::ParamOPF;
        initialStateInfo::StateInfo = initialStateInfo,
        param_cut::NamedTuple = param_cut, 
        param_levelsetmethod::NamedTuple = param_levelsetmethod, 
        param::NamedTuple = param
)::Dict
    ## d: x dim
    initial = now(); i = 1; LB = - Inf; ub_type = :mean; UB = Inf; 
    iter_time::Float64 = 0; total_Time::Float64 = 0; t0 = 0.0; LMiter::Int64 = 0; LM_iter::Int64 = 0; gap::Float64 = 100.0; gapString = "100%"; branchDecision = false;

    col_names = [:Iter, :LB, :OPT, :UB, :gap, :time, :LM_iter, :Time, :Branch];                                 # needs to be a vector Symbols
    col_types = [Int64, Float64, Union{Float64,Nothing}, Float64, String, Float64, Int64, Float64, Bool];       # needs to be a vector of types
    named_tuple = (; zip(col_names, type[] for type in col_types )...);
    solHistory = DataFrame(named_tuple); # 0×7 DataFrame
    gapList = [];
    
    @everywhere begin
        if param.algorithm == :SDDiP
            for g in indexSets.G
                if paramOPF.smax[g] ≥ param.ε
                    param.κ[g] = ceil(Int, log2(paramOPF.smax[g] / param.ε)) # floor(Int, log2(paramOPF.smax[g] / param.ε)) + 1
                else
                    param.κ[g] = 1
                end
            end

            contStateBin = Dict(
                g => binarize_continuous_variable(initialStateInfo.ContVar[:s][g], paramOPF.smax[g], param) for g in indexSets.G
            );
            initialStateInfo.ContStateBin = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                :s => Dict{Any, Dict{Any, Any}}(
                    g => Dict{Any, Any}(
                        i => contStateBin[g][i] for i in 1:param.κ[g]
                    ) for g in indexSets.G
                )
            );
        end
        ModelList = Dict{Int, SDDPModel}();
        for t in 1:indexSets.T 
            ModelList[t] = forwardModel!( 
                paramDemand, 
                paramOPF, 
                scenarioTree.tree[t];
                indexSets = indexSets, 
                param = param
            );
        end 
        stateInfoCollection = Dict();  # to store every iteration results
    end

    ####################################################### Main Loop ###########################################################
    while true
        t0 = now(); 
        Ξ̃ = sample_scenarios(; scenarioTree = scenarioTree, numScenarios = param.numScenarios);
        u = Dict{Int64, Float64}();  # to store the value of each scenario
        
        ####################################################### Forward Steps ###########################################################
        forwardPassResult = pmap(values(Ξ̃)) do ξ
            forwardPass(ξ;
                # ModelList = ModelList,
                paramDemand = paramDemand,
                paramOPF = paramOPF,
                indexSets = indexSets,
                initialStateInfo = initialStateInfo,
                param = param
            )
        end;

        for j in 1:param.numScenarios
            ω = collect(keys(Ξ̃))[j]
            for t in 1:indexSets.T
                stateInfoCollection[i, t, ω] = forwardPassResult[ω][t]
            end
            u[ω] = sum(stateInfoCollection[i, t, ω].StageValue for t in 1:indexSets.T);
        end
        @everywhere stateInfoCollection = $stateInfoCollection;
        ####################################################### Record Info ###########################################################
        LB = maximum([stateInfoCollection[i, 1, 1].StateValue, LB]);
        μ̄ = mean(values(u));
        if ub_type == :mean
            UB = μ̄;
        elseif ub_type == :meanvariance
            σ̂² = Statistics.var(values(u));
            UB = μ̄ + 1.96 * sqrt(σ̂²/param.numScenarios); # minimum([μ̄ + 1.96 * sqrt(σ̂²/numScenarios), UB]);
        end
        gap = round((UB-LB)/UB * 100 ,digits = 2); 
        gapString = string(gap,"%"); 
        push!(solHistory, [i, LB, param.OPT, UB, gapString, iter_time, LM_iter, total_Time, branchDecision]); 
        push!(gapList, gap); 
        branchDecision = false;
        if i == 1
            print_iteration_info_bar();
        end
        print_iteration_info(i, LB, UB, gap, iter_time, LM_iter, total_Time); 

        save_results_info(
            param, 
            param_cut, 
            Dict(
                :solHistory => solHistory, 
                # :solution => stateInfoCollection, 
                :gapHistory => gapList
            )
        );

        LM_iter = 0;
        if total_Time > param.terminate_time || i ≥ param.MaxIter || UB-LB ≤ param.terminate_threshold * UB  
            return Dict(
                :solHistory => solHistory, 
                # :solution => stateInfoCollection, 
                :gapHistory => gapList
            ) 
        end

        ####################################################### Partition Tree ###########################################################
        ## the first rule:: for branching: current convex envelope is good enough
        if i ≥ param.LiftIterThreshold && param.algorithm == :SDDPL                                                                 
            for t in reverse(1:indexSets.T-1) 
                for ω in [1] #keys(Ξ̃)
                    dev = Dict();
                    for g in indexSets.G 
                        if stateInfoCollection[i, t, ω].BinVar[:y][g] > .5
                            k = maximum(
                                [k for (k, v) in stateInfoCollection[i, t, ω].ContVarLeaf[:s][g] if v[:var] > 0.5]
                            );
                            info = ModelList[t].ContVarLeaf[:s][g][k];
                            dev[g] = round(
                                minimum(
                                    [(info[:ub] - stateInfoCollection[i, t, ω].ContVar[:s][g])/(info[:ub] - info[:lb] + 1e-6), 
                                        (stateInfoCollection[i, t, ω].ContVar[:s][g] - info[:lb])/(info[:ub] - info[:lb] + 1e-6)]
                                ), digits = 5
                            );
                        end

                        if param.sparse_cut == :sparse
                            for k in collect(keys(stateInfoCollection[i, t, ω].ContAugState[:s][g]))   
                                stateInfoCollection[i, t, ω].ContAugState[:s][g][k] == 1.0 || delete!(stateInfoCollection[i, t, ω].ContAugState[:s][g], k)
                            end
                        end
                    end
                    ##TODO: the second rule: current point is far from being an extreme point
                    if param.branch_variable == :MFV
                        large_dev = [g for (g, v) in dev if v == maximum(values(dev))] 
                    elseif param.branch_variable == :ALL
                        large_dev = [g for (g, g_dev) in dev if g_dev ≥ param.branch_threshold]
                    end
                    for g in large_dev
                        branchDecision = true;
                        @everywhere begin
                            t = $t; ω = $ω; g = $g; i = $i; 
                            update_partition_tree!(
                                ModelList, 
                                stateInfoCollection[i, t, ω],
                                t, g; param = param
                            );
                        end
                    end            
                end
            end
        end

        ####################################################### Backward Steps ###########################################################
        cut = get_cut_selection(param.cutSelection, i);
        for t = reverse(2:indexSets.T)
            for ω in 1:param.M #keys(Ξ̃)
                backwardNodeInfoList = Dict{Int64, Tuple}(); 
                for n in keys(scenarioTree.tree[t].nodes) 
                    backwardNodeInfoList[n] = (i, t, n, ω, cut, param_cut.core_point_strategy) 
                end

                # backwardPassResult = pmap(backwardPass, values(backwardNodeInfoList))

                backwardPassResult = pmap(values(backwardNodeInfoList)) do backwardNodeInfo
                    backwardPass(
                        backwardNodeInfo; 
                        # ModelList = ModelList,
                        indexSets = indexSets, 
                        paramDemand = paramDemand, 
                        paramOPF = paramOPF,
                        scenarioTree = scenarioTree, 
                        stateInfoCollection = stateInfoCollection,
                        param = param, param_cut = param_cut, param_levelsetmethod = param_levelsetmethod
                    )
                end

                for n in keys(scenarioTree.tree[t].nodes)
                    @everywhere begin
                        n = $n; t = $t; i = $i; ω = $ω; (λ₀, λ₁, πₙ₀) = $backwardPassResult[n][1]; # (λ₀, λ₁) = backwardPassResult[n][1]
                        @constraint(
                            ModelList[t-1].model, 
                            πₙ₀ * ModelList[t-1].model[:θ][n]/scenarioTree.tree[t-1].prob[n] ≥ λ₀ + 
                            sum(
                                (
                                    param.algorithm == :SDDiP ? 
                                    sum(λ₁.ContStateBin[:s][g][i] * ModelList[t-1].model[:λ][g, i] for i in 1:param.κ[g]; init = 0.0) 
                                    : λ₁.ContVar[:s][g] * ModelList[t-1].model[:s][g]
                                ) + 
                                λ₁.BinVar[:y][g] * ModelList[t-1].model[:y][g] + 
                                λ₁.BinVar[:v][g] * ModelList[t-1].model[:v][g] + 
                                λ₁.BinVar[:w][g] * ModelList[t-1].model[:w][g] + 
                                (
                                    param.algorithm == :SDDPL ? 
                                    sum(λ₁.ContAugState[:s][g][k] * ModelList[t-1].model[:augmentVar][g, k] for k in keys(stateInfoCollection[i, t-1, ω].ContAugState[:s][g]); init = 0.0) 
                                    : 0.0
                                ) for g in indexSets.G
                            )
                        );                 
                    end
                end

                LM_iter = LM_iter + sum(backwardPassResult[n][2] for n in keys(scenarioTree.tree[t].nodes))
            end
        end
        LM_iter = floor(Int64, LM_iter/sum(length(scenarioTree.tree[t].nodes) for t in 2:indexSets.T));
        
        t1 = now(); iter_time = (t1 - t0).value/1000; total_Time = (t1 - initial).value/1000; i += 1; 

    end
end