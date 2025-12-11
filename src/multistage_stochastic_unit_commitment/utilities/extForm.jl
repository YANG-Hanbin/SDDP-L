"""
nonanticipativity()

    An auxiliary function to add nonanticipativity constraints to the extensive form model.
  
"""
function nonanticipativity(; 
    model::Model = model, 
    scenarioList = scenarioList, 
    t = t
)::Nothing
    if t < indexSets.T
        for n in keys(scenarioTree.tree[t].nodes)
            Ξ̃ = []
            for ω in scenarioList
                if n == Ξ[ω][:path][t]
                    push!(Ξ̃, ω)
                end
            end
            @constraint(model, [j in 2:length(Ξ̃)], model[:s][:, t, Ξ̃[1]] .== model[:s][:, t, Ξ̃[j]])
            @constraint(model, [j in 2:length(Ξ̃)], model[:y][:, t, Ξ̃[1]] .== model[:y][:, t, Ξ̃[j]])
            @constraint(model, [j in 2:length(Ξ̃)], model[:v][:, t, Ξ̃[1]] .== model[:v][:, t, Ξ̃[j]])
            @constraint(model, [j in 2:length(Ξ̃)], model[:w][:, t, Ξ̃[1]] .== model[:w][:, t, Ξ̃[j]])
            nonanticipativity(model = model, scenarioList = Ξ̃, t = t + 1)
        end
    end
    return
end

"""
    extensive_form()
  
"""
function extensive_form(; 
    indexSets::IndexSets = indexSets, 
    paramDemand::ParamDemand = paramDemand, 
    paramOPF::ParamOPF = paramOPF, 
    initialStageDecision::Dict{Symbol, Dict{Int64, Float64}} = initialStageDecision,
    scenarioTree::ScenarioTree = scenarioTree, Ξ::Dict{Any, Any} = Ξ,                             
    silent::Bool = true, 
    timelimit::Real = 1e3, 
    mipGap::Float64 = 1e-3
)::NamedTuple{(:OPT, :statevariable_s, :statevariable_y), Tuple{Float64, Vector{Float64}, Vector{Float64}}}
    (D, G, L, B, T) = (indexSets.D, indexSets.G, indexSets.L, indexSets.B, indexSets.T);
    (Dᵢ, Gᵢ, in_L, out_L) = (indexSets.Dᵢ, indexSets.Gᵢ, indexSets.in_L, indexSets.out_L); 
    W = length(Ξ); ## number of scenarios

    model = Model(optimizer_with_attributes(()->Gurobi.Optimizer(GRB_ENV),
                                                    "Threads" => 0)); 
    MOI.set(model, MOI.Silent(), silent);
    set_optimizer_attribute(model, "MIPGap", mipGap);
    set_optimizer_attribute(model, "TimeLimit", timelimit);

    @variable(model, θ_angle[B, 1:T, ω in 1:W])                ## phase angle of the bus i
    @variable(model, P[L, 1:T, ω in 1:W])                      ## real power flow on line l; elements in L is Tuple (i, j)
    @variable(model, s[G, 1:T, ω in 1:W])                      ## real power generation at generator g
    @variable(model, 0 ≤ x[D, 1:T, ω in 1:W] ≤ 1)              ## load shedding

    @variable(model, h[G, 1:T, ω in 1:W] ≥ 0);                 ## production cost at generator g

    @variable(model, y[G, 1:T, ω in 1:W], Bin)                 ## binary variable for generator commitment status
    @variable(model, v[G, 1:T, ω in 1:W], Bin)                 ## binary variable for generator startup decision
    @variable(model, w[G, 1:T, ω in 1:W], Bin)                 ## binary variable for generator shutdowm decision
    # power flow constraints
    for l in L
        i = l[1]
        j = l[2]
        @constraint(model, [t in 1:T, ω in 1:W], P[l, t, ω] ≤ - paramOPF.b[l] * (θ_angle[i, t, ω] - θ_angle[j, t, ω]))
        @constraint(model, [t in 1:T, ω in 1:W], P[l, t, ω] ≥ - paramOPF.b[l] * (θ_angle[i, t, ω] - θ_angle[j, t, ω]))
    end
    
    # power flow limitation
    @constraint(model, [t in 1:T, ω in 1:W, l in L], P[l, t, ω] ≥ - paramOPF.W[l])
    @constraint(model, [t in 1:T, ω in 1:W, l in L], P[l, t, ω] ≤   paramOPF.W[l])
    # genertor limitation
    @constraint(model, [t in 1:T, ω in 1:W, g in G], s[g, t, ω] ≥ paramOPF.smin[g] * y[g, t, ω])
    @constraint(model, [t in 1:T, ω in 1:W, g in G], s[g, t, ω] ≤ paramOPF.smax[g] * y[g, t, ω])

    # power balance constriant
    @constraint(
        model, 
        [t in 1:T, ω in 1:W, i in B], 
        sum(s[g, t, ω] for g in Gᵢ[i]) -
        sum(P[(i, j), t, ω] for j in out_L[i]) + 
        sum(P[(j, i), t, ω] for j in in_L[i]) .== 
        sum(paramDemand.demand[d] * scenarioTree.tree[t].nodes[Ξ[ω][:path][t]].deviation[d] * x[d, t, ω] for d in Dᵢ[i]) 
    );

    # on/off status with startup and shutdown decision
    ## t = 1
    @constraint(model, [ω in 1:W, g in indexSets.G], v[g, 1, ω] - w[g, 1, ω] == y[g, 1, ω] - initialStageDecision[:y][g])
    @constraint(model, [ω in 1:W, g in indexSets.G], s[g, 1, ω] - initialStageDecision[:s][g] ≤ paramOPF.M[g] * initialStageDecision[:y][g] + paramOPF.smin[g] * v[g, 1, ω])
    @constraint(model, [ω in 1:W, g in indexSets.G], s[g, 1, ω] - initialStageDecision[:s][g] ≥ - paramOPF.M[g] * y[g, 1, ω] - paramOPF.smin[g] * w[g, 1, ω])
    # t ≥ 2
    @constraint(model, [t in 2:T, ω in 1:W, g in indexSets.G], v[g, t, ω] - w[g, t, ω] == y[g, t, ω] - y[g, t-1, ω])
    @constraint(model, [t in 2:T, ω in 1:W, g in indexSets.G], s[g, t, ω] - s[g, t-1, ω] ≤ paramOPF.M[g] * y[g, t-1, ω] + paramOPF.smin[g] * v[g, t, ω])
    @constraint(model, [t in 2:T, ω in 1:W, g in indexSets.G], s[g, t, ω] - s[g, t-1, ω] ≥ - paramOPF.M[g] * y[g, t, ω] - paramOPF.smin[g] * w[g, t, ω])

    # minimum up and down constraint
    ## t = 1
    @constraint(model, [ω in 1:W, g in indexSets.G], s[g, 1, ω] - initialStageDecision[:s][g] ≤ paramOPF.M[g] * initialStageDecision[:y][g] + paramOPF.smin[g] * v[g, 1, ω])
    @constraint(model, [ω in 1:W, g in indexSets.G], s[g, 1, ω] - initialStageDecision[:s][g] ≤ - paramOPF.M[g] * y[g, 1, ω] - paramOPF.smin[g] * w[g, 1, ω])
    @constraint(model, [ω in 1:W, g in indexSets.G], v[g, 1, ω] + initialStageDecision[:v][g] ≤ y[g, 1, ω])
    @constraint(model, [ω in 1:W, g in indexSets.G], w[g, 1, ω] + initialStageDecision[:w][g] ≤ 1 - y[g, 1, ω])
    # t ≥ 2
    @constraint(model, [t in 2:T, ω in 1:W, g in indexSets.G], v[g, t, ω] + v[g, t-1, ω] ≤ y[g, t, ω])
    @constraint(model, [t in 2:T, ω in 1:W, g in indexSets.G], w[g, t, ω] + w[g, t-1, ω] ≤ 1 - y[g, t, ω])

    # production cost
    @constraint(model, production[t in 1:T, ω in 1:W, g in indexSets.G, o in keys(paramOPF.slope[g])], h[g, t, ω] ≥ paramOPF.slope[g][o] * s[g, t, ω] + paramOPF.intercept[g][o] * y[g, t, ω])

    # objective function
    @objective(
        model, 
        Min, 
        sum(Ξ[ω][:prob] * sum(
            sum(h[g, t, ω] + 
            paramOPF.C_start[g] * v[g, t, ω] + 
            paramOPF.C_down[g] * w[g, t, ω] for g in G) + 
            sum(paramDemand.w[d] * (1 - x[d, t, ω]) for d in D) for t in 1:T 
            ) for ω in 1:W
        )
    );
    
    # nonanticipativity constraints
    nonanticipativity(model = model, scenarioList = keys(Ξ), t = 1)
    optimize!(model)

    return (
        OPT = JuMP.objective_value(model), 
        statevariable_s = JuMP.value.(s[:, 1, 1]), 
        statevariable_y = JuMP.value.(y[:, 1, 1])
    )
end
