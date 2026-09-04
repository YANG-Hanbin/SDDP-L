"""
    add_relu_cut_to_model!(
        model::Model,
        n::Int,
        reluCutInfo::ReLUCutInfoUC,
        stateInfo::StateInfo,
        node_probability::Float64;
        indexSets::IndexSets = indexSets,
        paramOPF::ParamOPF = paramOPF,
    )::Nothing

Append a scenario cut generated in the ReLU-dual space to the parent-stage
model. The helper introduces a local set of deviation variables `(w⁺, w⁻, r)`
that encode the lifted ReLU representation around the incumbent parent state.

For regular ReLU cuts the coefficient on `θ[n]` is one. For normalized ReLU
cuts the coefficient is `π₀`, stored in `reluCutInfo.dualInfo.StateValue`.
"""
function add_relu_cut_to_model!(
    model::Model,
    n::Int,
    reluCutInfo::ReLUCutInfoUC,
    stateInfo::StateInfo,
    node_probability::Float64;
    indexSets::IndexSets = indexSets,
    paramOPF::ParamOPF = paramOPF,
)
    cut_id = get!(model.ext, :ReLUCutCounter, 0) + 1
    model.ext[:ReLUCutCounter] = cut_id

    πₙ = reluCutInfo.dualInfo
    πₙ₀ = πₙ.StateValue === nothing ? 1.0 : πₙ.StateValue
    aug_term =
        πₙ.ContAugState === nothing ||
        stateInfo.ContAugState === nothing ||
        :augmentVar ∉ keys(model.obj_dict) ?
        0.0 :
        sum(
            sum(
                πₙ.ContAugState[:s][g][k] * model[:augmentVar][g, k]
                for k in keys(πₙ.ContAugState[:s][g]);
                init = 0.0,
            )
            for g in keys(πₙ.ContAugState[:s]);
            init = 0.0,
        )

    w_plus_s = @variable(model, [g in indexSets.G], lower_bound = 0.0, base_name = "relu_cut_w_plus_s[$cut_id]")
    w_minus_s = @variable(model, [g in indexSets.G], lower_bound = 0.0, base_name = "relu_cut_w_minus_s[$cut_id]")
    r_s = @variable(model, [g in indexSets.G], binary = true, base_name = "relu_cut_r_s[$cut_id]")

    w_plus_y = @variable(model, [g in indexSets.G], lower_bound = 0.0, base_name = "relu_cut_w_plus_y[$cut_id]")
    w_minus_y = @variable(model, [g in indexSets.G], lower_bound = 0.0, base_name = "relu_cut_w_minus_y[$cut_id]")
    r_y = @variable(model, [g in indexSets.G], binary = true, base_name = "relu_cut_r_y[$cut_id]")

    w_plus_v = @variable(model, [g in indexSets.G], lower_bound = 0.0, base_name = "relu_cut_w_plus_v[$cut_id]")
    w_minus_v = @variable(model, [g in indexSets.G], lower_bound = 0.0, base_name = "relu_cut_w_minus_v[$cut_id]")
    r_v = @variable(model, [g in indexSets.G], binary = true, base_name = "relu_cut_r_v[$cut_id]")

    w_plus_w = @variable(model, [g in indexSets.G], lower_bound = 0.0, base_name = "relu_cut_w_plus_w[$cut_id]")
    w_minus_w = @variable(model, [g in indexSets.G], lower_bound = 0.0, base_name = "relu_cut_w_minus_w[$cut_id]")
    r_w = @variable(model, [g in indexSets.G], binary = true, base_name = "relu_cut_r_w[$cut_id]")

    @constraint(model, [g in indexSets.G], w_plus_s[g] - w_minus_s[g] == model[:s][g] - stateInfo.ContVar[:s][g])
    @constraint(model, [g in indexSets.G], w_plus_s[g] <= (paramOPF.smax[g] - stateInfo.ContVar[:s][g]) * r_s[g])
    @constraint(model, [g in indexSets.G], w_minus_s[g] <= stateInfo.ContVar[:s][g] * (1 - r_s[g]))

    @constraint(model, [g in indexSets.G], w_plus_y[g] - w_minus_y[g] == model[:y][g] - stateInfo.BinVar[:y][g])
    @constraint(model, [g in indexSets.G], w_plus_y[g] <= (1.0 - stateInfo.BinVar[:y][g]) * r_y[g])
    @constraint(model, [g in indexSets.G], w_minus_y[g] <= stateInfo.BinVar[:y][g] * (1 - r_y[g]))

    @constraint(model, [g in indexSets.G], w_plus_v[g] - w_minus_v[g] == model[:v][g] - stateInfo.BinVar[:v][g])
    @constraint(model, [g in indexSets.G], w_plus_v[g] <= (1.0 - stateInfo.BinVar[:v][g]) * r_v[g])
    @constraint(model, [g in indexSets.G], w_minus_v[g] <= stateInfo.BinVar[:v][g] * (1 - r_v[g]))

    @constraint(model, [g in indexSets.G], w_plus_w[g] - w_minus_w[g] == model[:w][g] - stateInfo.BinVar[:w][g])
    @constraint(model, [g in indexSets.G], w_plus_w[g] <= (1.0 - stateInfo.BinVar[:w][g]) * r_w[g])
    @constraint(model, [g in indexSets.G], w_minus_w[g] <= stateInfo.BinVar[:w][g] * (1 - r_w[g]))

    @constraint(
        model,
        πₙ₀ * model[:θ][n] / node_probability +
        sum(
            πₙ.ContVarPlus[:s][g] * w_plus_s[g] +
            πₙ.ContVarMinus[:s][g] * w_minus_s[g] +
            πₙ.BinVarPlus[:y][g] * w_plus_y[g] +
            πₙ.BinVarMinus[:y][g] * w_minus_y[g] +
            πₙ.BinVarPlus[:v][g] * w_plus_v[g] +
            πₙ.BinVarMinus[:v][g] * w_minus_v[g] +
            πₙ.BinVarPlus[:w][g] * w_plus_w[g] +
            πₙ.BinVarMinus[:w][g] * w_minus_w[g]
            for g in indexSets.G
        ) -
        aug_term ≥ reluCutInfo.rhs,
    )

    return
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

"""
    stochastic_dual_dynamic_programming_algorithm(
        scenarioTree::ScenarioTree,
        indexSets::IndexSets,
        paramDemand::ParamDemand,
        paramOPF::ParamOPF;
        initialStateInfo::StateInfo = initialStateInfo,
        param_cut::NamedTuple = param_cut,
        param_levelsetmethod::NamedTuple = param_levelsetmethod,
        param::NamedTuple = param,
    )::Dict

Run the multistage stochastic dynamic programming algorithm for the MSUC test
bed under the selected algorithm/cut configuration.
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
    runtimeHistory = initialize_runtime_history();

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
        stateInfoCollection = Dict();  # Stores forward-pass solutions by (iter, stage, sample)
        thetaValueCollection = Dict(); # Stores child-node θ values by (iter, stage, sample)
    end

    ####################################################### Main Loop ###########################################################
    while true
        t0 = now();
        iteration_timer = time_ns();
        forward_time::Float64 = 0.0;
        lifting_time::Float64 = 0.0;
        backward_time::Float64 = 0.0;
        cut = get_cut_selection(param.cutSelection, i);
        Ξ̃ = sample_scenarios(; scenarioTree = scenarioTree, numScenarios = param.numScenarios);
        u = Dict{Int64, Float64}();  # to store the value of each scenario

        ####################################################### Forward Steps ###########################################################
        scenario_ids = sort(collect(keys(Ξ̃)))
        forward_timer = time_ns();
        forwardPassResultList = pmap(scenario_ids) do ω
            forwardPass(Ξ̃[ω];
                # ModelList = ModelList,
                paramDemand = paramDemand,
                paramOPF = paramOPF,
                indexSets = indexSets,
                initialStateInfo = initialStateInfo,
                param = param,
                sample_id = ω,
            )
        end;
        forward_time = elapsed_seconds(forward_timer);
        forwardPassResult = Dict(
            scenario_ids[j] => forwardPassResultList[j]
            for j in eachindex(scenario_ids)
        )

        for ω in scenario_ids
            for t in 1:indexSets.T
                stateInfoCollection[i, t, ω] = forwardPassResult[ω].stateInfoList[t]
                thetaValueCollection[i, t, ω] = forwardPassResult[ω].thetaValueList[t]
            end
            u[ω] = sum(stateInfoCollection[i, t, ω].StageValue for t in 1:indexSets.T);
        end
        @everywhere stateInfoCollection = $stateInfoCollection;
        @everywhere thetaValueCollection = $thetaValueCollection;
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
                :gapHistory => gapList,
                :runtimeHistory => runtimeHistory,
            )
        );

        LM_iter = 0;
        if total_Time > param.terminate_time || i ≥ param.MaxIter || UB-LB ≤ param.terminate_threshold * UB
            iteration_time = elapsed_seconds(iteration_timer);
            total_Time = (now() - initial).value / 1000;
            push_runtime_history!(
                runtimeHistory;
                iter = i,
                algorithm = param.algorithm,
                cut = cut,
                forward_time = forward_time,
                lifting_time = lifting_time,
                backward_time = backward_time,
                iteration_time = iteration_time,
                total_time = total_Time,
                completed_backward = false,
            );
            save_results_info(
                param,
                param_cut,
                Dict(
                    :solHistory => solHistory,
                    :gapHistory => gapList,
                    :runtimeHistory => runtimeHistory,
                )
            );
            return Dict(
                :solHistory => solHistory,
                # :solution => stateInfoCollection,
                :gapHistory => gapList,
                :runtimeHistory => runtimeHistory,
            )
        end

        ####################################################### Partition Tree ###########################################################
        # First branching rule: branch only after the lifted envelope has
        # become informative enough.
        if param.algorithm == :SDDPL
            lifting_timer = time_ns();
            if i ≥ param.LiftIterThreshold
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
                        # The implementation currently branches either on the most
                        # fractional variable or on all variables above the
                        # prescribed deviation threshold.
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
            lifting_time = elapsed_seconds(lifting_timer);
        end

        if param.algorithm == :SDDPL && (cut == :ReLUC || cut == :NormalizedReLUC)
            @everywhere begin
                i_sync = $i
                for t_sync in 1:(indexSets.T - 1)
                    for ω_sync in 1:param.M
                        if haskey(stateInfoCollection, (i_sync, t_sync, ω_sync))
                            sync_cont_aug_state!(
                                stateInfoCollection[i_sync, t_sync, ω_sync],
                                ModelList[t_sync];
                                indexSets = indexSets,
                                param = param,
                            )
                        end
                    end
                end
            end
        end

        ####################################################### Backward Steps ###########################################################
        backward_timer = time_ns();
        min_lnc_scale = Inf
        max_lnc_tightness_gap = 0.0
        has_lnc_tightness_gap = false
        num_lnc_fallback = 0
        num_lnc_core_theta_fallback = 0
        for t = reverse(2:indexSets.T)
            for ω in 1:param.M #keys(Ξ̃)
                node_ids = sort(collect(keys(scenarioTree.tree[t].nodes)))
                backwardPassResultList = pmap(node_ids) do n
                    backwardPass(
                        (i, t, n, ω, cut);
                        # ModelList = ModelList,
                        indexSets = indexSets,
                        paramDemand = paramDemand,
                        paramOPF = paramOPF,
                        scenarioTree = scenarioTree,
                        stateInfoCollection = stateInfoCollection,
                        param = param, param_cut = param_cut, param_levelsetmethod = param_levelsetmethod
                    )
                end
                backwardPassResult = Dict(
                    node_ids[j] => backwardPassResultList[j]
                    for j in eachindex(node_ids)
                )

                if cut == :LNC
                    for n in node_ids
                        diagnostics = backwardPassResult[n][3]
                        if isfinite(diagnostics.scale)
                            min_lnc_scale = min(min_lnc_scale, diagnostics.scale)
                        end
                        if isfinite(diagnostics.tightness_gap)
                            has_lnc_tightness_gap = true
                            max_lnc_tightness_gap = max(
                                max_lnc_tightness_gap,
                                diagnostics.tightness_gap,
                            )
                        end
                        num_lnc_fallback += diagnostics.fallback ? 1 : 0
                        num_lnc_core_theta_fallback +=
                            diagnostics.core_theta_fallback ? 1 : 0
                    end
                end

                for n in node_ids
                    backwardCutInfo = backwardPassResult[n][1]
                    @everywhere begin
                        n = $n; t = $t; i = $i; ω = $ω; cut_local = $(QuoteNode(cut)); (λ₀, λ₁, πₙ₀) = $backwardCutInfo;
                        if cut_local == :ReLUC || cut_local == :NormalizedReLUC
                            add_relu_cut_to_model!(
                                ModelList[t-1].model,
                                n,
                                ReLUCutInfoUC(λ₀, λ₁),
                                stateInfoCollection[i, t-1, ω],
                                scenarioTree.tree[t-1].prob[n];
                                indexSets = indexSets,
                                paramOPF = paramOPF,
                            )
                        else
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
                end

                LM_iter = LM_iter + sum(backwardPassResult[n][2] for n in node_ids)
            end
        end
        backward_time = elapsed_seconds(backward_timer);
        LM_iter = floor(Int64, LM_iter/sum(length(scenarioTree.tree[t].nodes) for t in 2:indexSets.T));

        t1 = now(); iter_time = (t1 - t0).value/1000; total_Time = (t1 - initial).value/1000; i += 1;
        runtime_iteration_time = elapsed_seconds(iteration_timer);
        push_runtime_history!(
            runtimeHistory;
            iter = i - 1,
            algorithm = param.algorithm,
            cut = cut,
            forward_time = forward_time,
            lifting_time = lifting_time,
            backward_time = backward_time,
            iteration_time = runtime_iteration_time,
            total_time = total_Time,
            completed_backward = true,
            min_lnc_scale = cut == :LNC && isfinite(min_lnc_scale) ? min_lnc_scale : NaN,
            max_lnc_tightness_gap = cut == :LNC && has_lnc_tightness_gap ?
                max_lnc_tightness_gap :
                NaN,
            num_lnc_fallback = cut == :LNC ? num_lnc_fallback : 0,
            num_lnc_core_theta_fallback = cut == :LNC ? num_lnc_core_theta_fallback : 0,
        );

    end
end
