"""
    function update_partition_tree!(
        ModelList, 
        stateInfo, 
        i, 
        t, 
        ω, 
        g, 
        param
    )
# Arguments

    1. `ModelList`: A dictionary of `SDDPModel` objects
    2. `stateInfo`: StateInfo
    3. `t`: the stage index
    4. `g`: the generator index

    # Returns
    1. `Nothing`
"""
function update_partition_tree!(
    ModelList::Dict{Int64, SDDPModel}, 
    stateInfo::StateInfo,
    t, 
    g;
    param::NamedTuple = param
)::Nothing
    # find the active leaf node 
    keys_with_value_1 = maximum([k for (k, v) in stateInfo.ContVarLeaf[:s][g] if v[:var] > 0.5]); ## find the active leaf node: maximum(values(stateInfo.stageSolution[:sur][g]))
    # find the lb and ub of this leaf node 
    info = ModelList[t].ContVarLeaf[:s][g][keys_with_value_1]; (lb, ub) = info[:lb], info[:ub]; 
    if param.partitionRule == :Bisection
        med = (lb + ub)/2; 
    elseif param.partitionRule == :Incumbent
        med = stateInfo.ContVar[:s][g] == nothing ? (lb + ub)/2 : stateInfo.ContVar[:s][g]; # round(stateInfo.ContVar[:s][g], digits = 3); 
    end
    
    # create two new leaf nodes, and update their info (lb, ub)
    left = maximum(keys(ModelList[t].ContVarLeaf[:s][g])) + 1; 
    right = left + 1; 
    ModelList[t].model[:augmentVar][g, left] = @variable(
        ModelList[t].model, 
        base_name = "augmentVar[$g, $left]", 
        binary = true
    ); 
    ModelList[t].model[:augmentVar][g, right] = @variable(
        ModelList[t].model, 
        base_name = "augmentVar[$g, $right]", 
        binary = true
    );
    # delete the parent node and create new leaf nodes
    ModelList[t].ContVarLeaf[:s][g][left] = Dict{Symbol, Any}(
        :lb => lb, 
        :ub => med, 
        :var => ModelList[t].model[:augmentVar][g, left], 
        :sibling => right, 
        :parent => keys_with_value_1
    );
    ModelList[t].ContVarLeaf[:s][g][right] = Dict{Symbol, Any}(
        :lb => med, 
        :ub => ub, 
        :var => ModelList[t].model[:augmentVar][g, right], 
        :sibling => left, 
        :parent => keys_with_value_1
    );
    delete!(ModelList[t].ContVarLeaf[:s][g], keys_with_value_1);

    # add logic constraints
    ## for forward models
    ### Parent-Child relationship
    @constraint(
        ModelList[t].model, 
        ModelList[t].model[:augmentVar][g, left] + 
        ModelList[t].model[:augmentVar][g, right] == 
        ModelList[t].model[:augmentVar][g, keys_with_value_1]
    );

    ### bounding constraints
    delete(
        ModelList[t].model, 
        ModelList[t].model[:partition_lower_bound][g]
    )
    delete(
        ModelList[t].model, 
        ModelList[t].model[:partition_upper_bound][g]
    )
    ModelList[t].model[:partition_upper_bound][g] = @constraint(
        ModelList[t].model, 
        ModelList[t].ContVar[:s][g] ≤ 
        sum(ModelList[t].ContVarLeaf[:s][g][k][:ub] * ModelList[t].ContVarLeaf[:s][g][k][:var] 
            for k in keys(ModelList[t].ContVarLeaf[:s][g]))
    );
    ModelList[t].model[:partition_lower_bound][g] = @constraint(
        ModelList[t].model, 
        ModelList[t].ContVar[:s][g] ≥ 
        sum(
            ModelList[t].ContVarLeaf[:s][g][k][:lb] * ModelList[t].ContVarLeaf[:s][g][k][:var] 
            for k in keys(ModelList[t].ContVarLeaf[:s][g])
        )
    );

    ## for backward pass, we need to add the logic constraints and bounding constraints for the copy variables in the next stage
    ### Parent-Child relationship
    if param.tightness
        ModelList[t+1].model[:augmentVar_copy][g, left] = @variable(
            ModelList[t+1].model, 
            base_name = "augmentVar_copy[$g, $left]", 
            binary = true
        ); 
        ModelList[t+1].model[:augmentVar_copy][g, right] = @variable(
            ModelList[t+1].model, 
            base_name = "augmentVar_copy[$g, $right]", 
            binary = true
        );
    else
        ModelList[t+1].model[:augmentVar_copy][g, left] = @variable(
            ModelList[t+1].model, 
            base_name = "augmentVar_copy[$g, $left]", 
            lower_bound = 0, 
            upper_bound = 1
        ); 
        ModelList[t+1].model[:augmentVar_copy][g, right] = @variable(
            ModelList[t+1].model, 
            base_name = "augmentVar_copy[$g, $right]", 
            lower_bound = 0, 
            upper_bound = 1
        );
    end

    @constraint(
        ModelList[t+1].model, 
        ModelList[t+1].model[:augmentVar_copy][g, left] + 
        ModelList[t+1].model[:augmentVar_copy][g, right] == 
        ModelList[t+1].model[:augmentVar_copy][g, keys_with_value_1]
    );
    delete(
        ModelList[t+1].model, 
        ModelList[t+1].model[:partition_lower_bound_copy][g]
    )
    delete(
        ModelList[t+1].model, 
        ModelList[t+1].model[:partition_upper_bound_copy][g]
    )
    ModelList[t+1].model[:partition_upper_bound_copy][g] = @constraint(
        ModelList[t+1].model, 
        ModelList[t+1].model[:s_copy][g] ≤ 
        sum(ModelList[t].ContVarLeaf[:s][g][k][:ub] * ModelList[t+1].model[:augmentVar_copy][g, k] for k in keys(ModelList[t].ContVarLeaf[:s][g]))
    );
    ModelList[t+1].model[:partition_lower_bound_copy][g] = @constraint(
        ModelList[t+1].model, 
        ModelList[t+1].model[:s_copy][g] ≥ 
        sum(ModelList[t].ContVarLeaf[:s][g][k][:lb] * ModelList[t+1].model[:augmentVar_copy][g, k] for k in keys(ModelList[t].ContVarLeaf[:s][g]))
    );

    ## inherit and update the stageDecision
    if param.sparse_cut == :sparse
        stateInfo.ContAugState[:s][g] = Dict{Any, Any}();
    elseif param.sparse_cut == :dense
        stateInfo.ContAugState[:s][g] = Dict{Any, Any}(
            k => 0.0 for k in keys(ModelList[t].ContVarLeaf[:s][g])
        );
    end

    if stateInfo.ContVar[:s][g] ≤ med
        stateInfo.ContAugState[:s][g][left] = 1.0;
    else
        stateInfo.ContAugState[:s][g][right] = 1.0;
    end
    return
end