"""
Build the forward model for one stage.

Returns a `StageModel` that contains:
- `model`          : JuMP model
- `IntVar`         : (for SDDP/SDDPL) mapping :St => {g => St[g]}
- `IntVarLeaf`     : (for SDDPL) leaf-region structure for surrogate states
- `IntVarBinaries` : (for SDDiP) mapping :Lt => {i => Lt[i]}
"""
function forwardModel!(
    stageData::StageData;
    binaryInfo::BinaryInfo = binaryInfo,
    param::SDDPParam = param,
)::StageModel

    # construct forward problem (3.1)
    model = Model(
        optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV)
        ),
    )
    MOI.set(model, MOI.Silent(), !param.verbose)
    set_optimizer_attribute(model, "Threads", 1)
    set_optimizer_attribute(model, "MIPGap", param.solverGap)
    set_optimizer_attribute(model, "TimeLimit", param.solverTime)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-8);
    set_optimizer_attribute(model, "MIPFocus", 3);           

    # decision variables
    @variable(model, x[g = 1:binaryInfo.d] ≥ 0, Int)   # number of generators built in this stage
    @variable(model, St[g = 1:binaryInfo.d] ≥ 0, Int)  # total generators after investment
    @variable(model, y[g = 1:binaryInfo.d] ≥ 0)        # generation
    @variable(model, slack ≥ 0)
    @variable(model, θ ≥ 0.0)

    # no more than max num of generators
    @constraint(
        model,
        limitationConstraint,
        St .≤ stageData.ū,
    )

    # initial demand satisfaction placeholder, will be modified later
    @constraint(
        model,
        demandConstraint,
        sum(y) + slack ≥ 0,
    )

    # capacity limitation
    @constraint(
        model,
        capacityConstraint,
        stageData.h * stageData.N * (St + stageData.s₀) .≥ y,
    )

    # a cap constraint for two generators
    # @constraint(
    #     model,
    #     cap[i = 4:5],
    #     y[i] ≤ sum(y)/5
    # )

    @expression(
        model,
        primal_objective_expression,
        stageData.c1' * x +
        stageData.c2' * y +
        θ +
        stageData.penalty * slack
    )

    # containers for integer-related structures
    IntVar         = nothing
    IntVarLeaf     = nothing
    IntVarBinaries = nothing

    # algorithm-specific structures
    if param.algorithm == :SDDPL
        # region indicator for surrogate leaves
        region_indicator = Dict( 
            g => Dict(
                i => @variable(
                    model, 
                    base_name = "region_indicator[$g, $i]", 
                    binary = true
                ) for i in 1:1
            ) for g in 1:binaryInfo.d
        )

        model[:region_indicator] = region_indicator

        # choose exactly one leaf per generator
        @constraint(model, [g in 1:binaryInfo.d], region_indicator[g][1] == 1)

        # copy variable Sc for previous state
        if param.discreteZ
            @variable(model, Sc[g = 1:binaryInfo.d] ≥ 0, Int)
        else
            @variable(model, Sc[g = 1:binaryInfo.d] ≥ 0)
        end
        @constraint(model, Sc .≤ stageData.ū)
        model[:Sc] = Sc

        # copy of region indicator (for cuts or auxiliary use)
        if param.discreteZ
            region_indicator_copy = Dict(
                g => Dict(
                    i => @variable(
                        model, 
                        base_name = "region_indicator_copy[$g, $i]", 
                        binary = true
                    ) for i in 1:1
                ) for g in 1:binaryInfo.d
            )
        else
            region_indicator_copy = Dict(
                g => Dict(
                    i => @variable(
                        model,
                        base_name   = "region_indicator_copy[$g, $i]",
                        lower_bound = 0,
                        upper_bound = 1,
                    ) for i in 1:1
                ) for g in 1:binaryInfo.d
            )
        end
        model[:region_indicator_copy] = region_indicator_copy
        @constraint(model, [g in 1:binaryInfo.d], region_indicator_copy[g][1] == 1)

        # state transition and initial condition (Ŝ = 0 at t = 1)
        @constraint(model, Sc + x .== St)
        @constraint(model, NonAnticipativity, Sc .== 0.0)

        # integer state mapping
        IntVar = Dict{Any, Dict{Any, VariableRef}}(
            :St => Dict{Any, VariableRef}(g => St[g] for g in 1:binaryInfo.d),
        )
        IntVarLeaf = Dict{Symbol, Dict{Int, Dict{Int, Dict{Symbol, Any}}}}(
            :St => Dict(
                g => Dict(
                    k => Dict(
                        :lb     => 0.0,
                        :ub     => stageData.ū[g],
                        :parent => nothing,
                        :sibling => nothing,
                        :var    => region_indicator[g][k],
                    ) for k in 1:1
                ) for g in 1:binaryInfo.d
            ),
        )

        @constraint(
            model,
            partition_lower_bound[g in 1:binaryInfo.d],
            St[g] ≥ sum(
                IntVarLeaf[:St][g][k][:lb] *
                region_indicator[g][k]
                for k in keys(IntVarLeaf[:St][g])
            )
        )
        @constraint(
            model,
            partition_upper_bound[g in 1:binaryInfo.d],
            St[g] ≤ sum(
                IntVarLeaf[:St][g][k][:ub] *
                region_indicator[g][k]
                for k in keys(IntVarLeaf[:St][g])
            )
        )
        @constraint(
            model,
            partition_lower_bound_copy[g in 1:binaryInfo.d],
            Sc[g] ≥ sum(
                IntVarLeaf[:St][g][k][:lb] *
                region_indicator_copy[g][k]
                for k in keys(IntVarLeaf[:St][g])
            )
        )
        @constraint(
            model,
            partition_upper_bound_copy[g in 1:binaryInfo.d],
            Sc[g] ≤ sum(
                IntVarLeaf[:St][g][k][:ub] *
                region_indicator_copy[g][k]
                for k in keys(IntVarLeaf[:St][g])
            )
        )

    elseif param.algorithm == :SDDiP
        # binary variables for state representation
        @variable(model, Lt[i = 1:binaryInfo.n], Bin)
        model[:Lt] = Lt

        if param.discreteZ
            @variable(model, Lc[i = 1:binaryInfo.n], Bin)
        else
            @variable(model, 0 ≤ Lc[i = 1:binaryInfo.n] ≤ 1)
        end
        model[:Lc] = Lc

        # link between current and previous binary states
        @constraint(model, binaryInfo.A * Lc + x .== binaryInfo.A * Lt)
        @constraint(model, binaryInfo.A * Lt .== St)

        # initial condition: L̂ = 0 at t = 1
        @constraint(model, NonAnticipativity, Lc .== 0.0)

        IntVarBinaries = Dict{Any, Vector{VariableRef}}(
            :Lt => Lt,
        )

    elseif param.algorithm == :SDDP
        # explicit previous state Sc
        if param.discreteZ
            @variable(model, Sc[g = 1:binaryInfo.d] ≥ 0, Int)
        else
            @variable(model, Sc[g = 1:binaryInfo.d] ≥ 0)
        end
        @constraint(model, Sc .≤ stageData.ū)
        model[:Sc] = Sc

        @constraint(model, Sc + x .== St)
        @constraint(model, NonAnticipativity, Sc .== 0.0)

        IntVar = Dict{Any, Dict{Any, VariableRef}}(
            :St => Dict{Any, VariableRef}(g => St[g] for g in 1:binaryInfo.d),
        )
    end

    return StageModel(model, IntVar, IntVarLeaf, IntVarBinaries)
end

"""
Modify the forward model for a given realization (demand, previous state).

- For all algorithms: update demand constraint.
- For SDDiP      : fix Lc to previous binary state `L̂`.
- For SDDP/SDDPL : fix Sc to previous integer state `Ŝ`.
"""
function model_modification!(
    model::Model,
    demand::Vector{Float64},
    L̂::Vector{Float64},
    Ŝ::Vector{Float64};
    param::SDDPParam = param,
    binaryInfo::BinaryInfo = binaryInfo,
)::Nothing

    # remove old constraints
    delete(model, model[:demandConstraint])
    unregister(model, :demandConstraint)

    if :NonAnticipativity ∈ keys(model.obj_dict) 
        delete(model, model[:NonAnticipativity])
        unregister(model, :NonAnticipativity)
    end

    # new demand constraint: sum(y) + slack ≥ sum(demand)
    @constraint(
        model,
        demandConstraint,
        sum(model[:y]) + model[:slack] ≥ sum(demand),
    )

    # state / binary state constraint
    if param.algorithm == :SDDiP
        @constraint(
            model,
            NonAnticipativity,
            model[:Lc] .== L̂,
        )
    else
        @constraint(
            model,
            NonAnticipativity,
            model[:Sc] .== Ŝ,
        )
    end
    @objective(model, Min, model[:primal_objective_expression])

    return
end

"""
    forwardPass(k, Scenarios; ...)

Run the forward pass along scenario path `k`.

Arguments
---------
- `k`          : scenario index
- `Scenarios`  : Dict(k => Vector{Int}), where Scenarios[k][t] is the node index
                 at stage t for scenario k.

Returns
-------
A dictionary `solCollection` such that `solCollection[(t, k)]` is a `StageInfo`
containing:
- state objective value
- stage cost (without θ)
- IntVar / IntVarLeaf / IntVarBinaries depending on the algorithm.
"""
function forwardPass(
    k::Int,
    Scenarios::Dict{Int, Vector{Int}};
    param::SDDPParam = param,
    Ω::Dict{Int, Dict{Int, RandomVariables}} = Ω,
    binaryInfo::BinaryInfo = binaryInfo,
    stageDataList::Dict{Int, StageData} = stageDataList,
    forwardInfoList::Dict{Int, StageModel} = forwardInfoList,
)::Dict
    # store forward solutions for each stage on path k
    solCollection = Dict()

    # for the first-stage subproblem, we start from zero state
    Ŝ = zeros(Float64, binaryInfo.d)
    L̂ = zeros(Float64, binaryInfo.n)

    for t in 1:param.T
        forwardInfo = forwardInfoList[t]

        # realization of k-th scenario at stage t
        ω = Scenarios[k][t]

        # modify problem coefficients for this node
        model_modification!(
            forwardInfo.model,
            Ω[t][ω].d,
            L̂,
            Ŝ;
            param = param,
            binaryInfo = binaryInfo,
        )

        optimize!(forwardInfo.model)

        stateValue = objective_value(forwardInfo.model)
        stageValue = stateValue - value(forwardInfo.model[:θ])

        # IntVar: only for SDDP / SDDPL
        IntVar = (param.algorithm == :SDDP || param.algorithm == :SDDPL) ?
            round.(value.(forwardInfo.model[:St]), digits = 3) :
            nothing

        # IntVarLeaf: only for SDDPL
        IntVarLeaf = param.algorithm == :SDDPL ? Dict(
            g => Dict(
                i => round.(value(forwardInfo.model[:region_indicator][g][i]), digits = 3)
                for i in keys(forwardInfo.IntVarLeaf[:St][g])
            ) for g in 1:binaryInfo.d
        ) : nothing

        # IntVarBinaries: only for SDDiP
        IntVarBinaries = param.algorithm == :SDDiP ?
            round.(value.(forwardInfo.model[:Lt]), digits = 3) :
            nothing

        solCollection[t, k] = StageInfo(
            stateValue,
            stageValue,
            IntVar,
            IntVarLeaf,
            IntVarBinaries,
        )

        # update previous state or binary state for next stage
        if param.algorithm != :SDDiP
            Ŝ = solCollection[t, k].IntVar
        else
            L̂ = solCollection[t, k].IntVarBinaries
        end
    end

    return solCollection
end