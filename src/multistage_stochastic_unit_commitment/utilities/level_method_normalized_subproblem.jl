"""
    function add_constraint(
        currentInfo::CurrentInfo,
        modelInfo::ModelInfo;
        indexSets::IndexSets = indexSets
    )::Nothing

# Arguments

    1. `currentInfo::CurrentInfo` : the current information
    2. `modelInfo::ModelInfo` : the model information
    3. `indexSets::IndexSets` : the index sets
    
# Returns
    This function is to add constraints for the model f_star and nxt pt.
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
    F  = JuMP.objective_value(model);
    
    normalization_function =  CutGenerationInfo.uₙ₀ * πₙ₀ + sum(
        πₙ.BinVar[:w][g] * CutGenerationInfo.uₙ.BinVar[:w][g] + 
        πₙ.BinVar[:y][g] * CutGenerationInfo.uₙ.BinVar[:y][g] + 
        πₙ.BinVar[:v][g] * CutGenerationInfo.uₙ.BinVar[:v][g] + 
        πₙ.ContVar[:s][g] * CutGenerationInfo.uₙ.ContVar[:s][g] for g in indexSets.G
    );

    currentInfo = NormalizationCurrentInfo(  
        (πₙ, πₙ₀),  
        - F,                                                                                                                                                            ## obj function value
        Dict(
            1 => normalization_function - 1, 
            2 => πₙ₀
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

    return (currentInfo = currentInfo, currentInfo_f = F)
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
        - 1.0,
        levelsetmethodOracleParam.x₀, 
        stateInfo;
        indexSets = indexSets
    );

    functionHistory = FunctionHistory(  
        Dict(1 => currentInfo.f), 
        Dict(1 => maximum(currentInfo.G[k] for k in keys(currentInfo.G)) )
    );

    cutInfo = [
        currentInfo_f - 
        currentInfo.x[2] * CutGenerationInfo.primal_bound - 
        sum(
            (
                param.algorithm == :SDDiP ?
                sum(
                    currentInfo.x[1].ContStateBin[:s][g][i] * stateInfo.ContStateBin[:s][g][i] for i in 1:param.κ[g]
                ) : currentInfo.x[1].ContVar[:s][g] * stateInfo.ContVar[:s][g]
            ) 
            + 
            currentInfo.x[1].BinVar[:y][g] * stateInfo.BinVar[:y][g] + 
            currentInfo.x[1].BinVar[:v][g] * stateInfo.BinVar[:v][g] + 
            currentInfo.x[1].BinVar[:w][g] * stateInfo.BinVar[:w][g]
            for g in G
        ) - (
            param.algorithm == :SDDPL ? 
            sum(
                sum(
                    currentInfo.x[1].ContAugState[:s][g][k] * stateInfo.ContAugState[:s][g][k] 
                    for k in keys(stateInfo.ContAugState[:s][g]); init = 0.0
                ) for g in G
            ) : 0.0
        ),
        currentInfo.x[1],
        - currentInfo.x[2]
    ];

    # model for oracle
    oracleModel = Model(optimizer_with_attributes(
        ()->Gurobi.Optimizer(GRB_ENV))
    ); 
    MOI.set(oracleModel, MOI.Silent(), true);
    set_optimizer_attribute(oracleModel, "Threads", 1);
    set_optimizer_attribute(oracleModel, "MIPGap", param.MIPGap);
    set_optimizer_attribute(oracleModel, "TimeLimit", param.TimeLimit);

    ## ==================================================== Levelset Method ============================================== ##
    para_oracle_bound = abs(currentInfo.f);
    z_rhs = 10 * 10^(ceil(log10(para_oracle_bound)));
    @variable(oracleModel, z ≥ - 1e8);
    @variable(oracleModel, xθ);
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
    @variable(nxtModel, xθ);
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
    @variable(nxtModel, y1);
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
        result = Δ_model_formulation(functionHistory, f_star, iter);
        previousΔ = copy.(Δ);
        Δ, a_min, a_max = result[1], result[2], result[3];

        if param_levelsetmethod.verbose # && (iter % 30 == 0)
            if iter == 1
                println("------------------------------------ Iteration Info --------------------------------------")
                println("Iter |   Gap                              Objective                             Constraint")
            end
            @printf("%3d  |   %5.3g                         %5.3g                              %5.3g\n", iter, Δ, - currentInfo.f, currentInfo.G[1])
        end

        x₀ = currentInfo.x[1]; 
        if (round(previousΔ) > round(Δ)) || ((currentInfo.G[1] ≤ 0.0))
            cutInfo = [
                currentInfo_f - 
                currentInfo.x[2] * CutGenerationInfo.primal_bound - 
                sum(
                    (param.algorithm == :SDDiP ?
                        sum(
                            currentInfo.x[1].ContStateBin[:s][g][i] * stateInfo.ContStateBin[:s][g][i] for i in 1:param.κ[g]
                        ) : currentInfo.x[1].ContVar[:s][g] * stateInfo.ContVar[:s][g]) + 
                    currentInfo.x[1].BinVar[:y][g] * stateInfo.BinVar[:y][g] + 
                    currentInfo.x[1].BinVar[:v][g] * stateInfo.BinVar[:v][g] + 
                    currentInfo.x[1].BinVar[:w][g] * stateInfo.BinVar[:w][g]
                    for g in G 
                ) - 
                (
                    param.algorithm == :SDDPL ? 
                    sum(
                        sum(
                            currentInfo.x[1].ContAugState[:s][g][k] * stateInfo.ContAugState[:s][g][k] 
                            for k in keys(stateInfo.ContAugState[:s][g]); init = 0.0
                        ) for g in G
                    ) : 0.0
                ),
                currentInfo.x[1],
                - currentInfo.x[2]
            ];
        end

        # update α
        if param_levelsetmethod.μ/2 ≤ (α-a_min)/(a_max-a_min) .≤ 1 - param_levelsetmethod.μ/2
            α = α;
        else
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
            ) 
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
            x0_nxt = JuMP.value(xθ) ≤ 0 ? JuMP.value(xθ) : 0.;
        elseif st == MOI.NUMERICAL_ERROR ## need to figure out why this case happened and fix it
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
                x0_nxt = JuMP.value(xθ) ≤ 0 ? JuMP.value(xθ) : 0.;
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
            indexSets = indexSets
        );
        iter = iter + 1;
        functionHistory.f_his[iter] = currentInfo.f;
        functionHistory.G_max_his[iter] = maximum(currentInfo.G[k] for k in keys(currentInfo.G));
    end

end
