################################################################################
########################  Binary encoding of integer state  ####################
################################################################################

"""
Binarize the integer state variable: x = A * L, where L ∈ {0, 1}ⁿ.

Given the maximum number of each type of generator `ū[g]`, this constructs
a matrix `A` such that the integer vector `x` can be represented as

    x = A * L,  L ∈ {0, 1}ⁿ.

Returns a `BinaryInfo` object with fields:
- `A` : binary expansion matrix
- `n` : number of binary variables
- `d` : number of generator types
"""
function integer_binarization(ū::AbstractVector{<:Real})::BinaryInfo
    row_num = length(ū)

    # number of bits for each component
    var_num = floor.(Int, log.(2, ū)) .+ 1

    col_num = sum(var_num)
    A = zeros(Int, row_num, col_num)

    offset = 0
    for i in 1:row_num
        for j in 1:var_num[i]
            A[i, offset + j] = 2^(j - 1)
        end
        offset += var_num[i]
    end

    return BinaryInfo(A, col_num, row_num)
end

"Backward-compatible alias (old misspelled name)."
intergerBinarization(ū::Vector{Float64}) = integer_binarization(ū)


################################################################################
###################  Non-anticipativity: scenario tree recursion ###############
################################################################################

"""
Build a full scenario sequence by recursively expanding the scenario tree.

Each entry of `scenario_sequence` has the form

    scenario_sequence[s] = Dict(1 => path, 2 => prob)

where
- `path` is a `Vector{Int}` with the node indices across stages,
- `prob` is the cumulative probability of that path.

Arguments
---------
- `pathList`           : current path (stage indices visited so far)
- `P`                  : cumulative probability along `pathList`
- `scenario_sequence`  : dictionary that stores all complete paths
- `t`                  : current stage (next stage to expand)
- `Ω`                  : scenario tree, Ω[t][ω] is the random variable at node ω of stage t
- `prob`               : transition probabilities, prob[t][ω] is prob of going to node ω at stage t
- `T`                  : total number of stages
"""
function recursion_scenario_tree(
    pathList::Vector{Int},
    P::Float64,
    scenario_sequence::Dict{Int, Dict{Int, Any}},
    t::Int;
    Ω::Dict{Int, Dict{Int, RandomVariables}},
    prob::Dict{Int, Vector{Float64}},
    T::Int = 2,
)
    if t ≤ T
        for ω_key in keys(Ω[t])
            new_path = copy(pathList)
            new_P = P * prob[t][ω_key]
            push!(new_path, ω_key)

            recursion_scenario_tree(
                new_path,
                new_P,
                scenario_sequence,
                t + 1;
                Ω = Ω,
                prob = prob,
                T = T,
            )
        end
    else
        # We reached stage T, store this complete scenario.
        next_idx = isempty(scenario_sequence) ? 1 : maximum(keys(scenario_sequence)) + 1
        scenario_sequence[next_idx] = Dict(1 => copy(pathList), 2 => P)
    end

    return scenario_sequence
end

# Example usage (if you want to rebuild the full set of paths)
# scenario_sequence = Dict{Int, Dict{Int, Any}}()
# pathList = Int[1]   # root node
# P = 1.0
# recursion_scenario_tree(pathList, P, scenario_sequence, 2; Ω = Ω, prob = prob, T = T)


################################################################################
###########################  Sampling from scenario tree  ######################
################################################################################

"""
Draw one scenario index from a scenario sequence.

`scenario_sequence[s][2]` is the probability mass of scenario `s`.
"""
function DrawSamples(scenario_sequence::Dict{Int, Dict{Int, Any}})::Int
    probs = [scenario_sequence[k][2] for k in keys(scenario_sequence)]
    items = collect(keys(scenario_sequence))
    weights = Weights(probs)
    return sample(items, weights)
end

"""
Sample `M` scenarios from an explicit scenario sequence built by
`recursion_scenario_tree`.

Returns a dictionary `scenarios` such that `scenarios[k]` is the path
(a `Vector{Int}`) for scenario `k`.
"""
function SampleScenarios(
    scenario_sequence::Dict{Int, Dict{Int, Any}};
    T::Int = 5,
    M::Int = 30,
)::Dict{Int, Vector{Int}}
    scenarios = Dict{Int, Vector{Int}}()
    for k in 1:M
        idx = DrawSamples(scenario_sequence)
        # scenario_sequence[idx][1] is the path list
        scenarios[k] = scenario_sequence[idx][1]
    end
    return scenarios
end

"""
    SampleScenarios(Ω, probList; M = 30)

Sample `M` scenario paths directly from the stage-wise scenario tree.

Arguments
---------
- `Ω`        :: Dict{Int, Dict{Int, RandomVariables}}
    Scenario tree. For each stage `t`, `Ω[t]` is a dictionary of nodes, and
    its keys are the node indices at stage `t`.

- `probList` :: Dict{Int, Vector{Float64}}
    Transition probabilities. For each stage `t ≥ 2`, `probList[t][j]` is the
    probability of moving from the (unique) node at stage `t - 1` along the
    sampled path to node `j` at stage `t`.

Keyword arguments
-----------------
- `M` :: Int = 30
    Number of scenario paths to sample.

Returns
-------
A dictionary `scenarios::Dict{Int, Vector{Int}}` such that
`scenarios[k][t]` is the node index at stage `t` for scenario `k`.

Notes
-----
- This implementation assumes that at stage 1 there is a single root node,
  and we fix the path to start at node index `1`.
- Probabilities in `probList[t]` are used as sampling weights and are not
  required to sum exactly to 1 (normalization is handled by `Weights`).
"""
function SampleScenarios(
    Ω::Dict{Int, Dict{Int, RandomVariables}},
    probList::Dict{Int, Vector{Float64}};
    M::Int = 30,
)::Dict{Int, Vector{Int}}

    # number of stages
    T = length(Ω)

    # basic consistency check
    @assert all(haskey(probList, t) for t in 2:T) "probList missing entries for some stages."

    scenarios = Dict{Int, Vector{Int}}()

    for k in 1:M
        # initialize path: root node is assumed to be index 1 at stage 1
        path = Vector{Int}(undef, T)
        path[1] = 1

        # sample a node for each subsequent stage
        for t in 2:T
            nodes   = collect(keys(Ω[t]))
            weights = Weights(probList[t])
            path[t] = sample(nodes, weights)
        end

        scenarios[k] = path
    end

    return scenarios
end


################################################################################
############################  Rounding helper (scaling) ########################
################################################################################

"""
Scale a positive number `a` into a rough scientific notation-like form.

Returns a vector `[b, c, d]` where:
- `b` is the (float) exponent (log10),
- `c` is the mantissa rounded to 2 digits,
- `d = c * 10^b` is the scaled value.

Example:
    a = 1.3333e10
    → [10.0, 1.33, 1.33e10]
"""
function round!(a::Float64)
    @assert a > 0.0 "round! is defined only for positive numbers."
    b = floor(log10(a))
    c = round(a / 10.0^b, digits = 2)
    d = c * 10.0^b
    return [b, c, d]
end


################################################################################
###############################  Data generation  ###############################
################################################################################

"""
Generate stage data, random variables, probabilities, and binary encoding.

Returns a NamedTuple with fields:
- `probList`      : Dict{Int, Vector{Float64}}
- `stageDataList` : Dict{Int, StageData}
- `Ω`             : Dict{Int, Dict{Int, RandomVariables}}
- `binaryInfo`    : BinaryInfo
"""
function dataGeneration(;
    T::Int = 2,
    r::Float64 = 0.08,                # annualized interest rate
    N::Matrix{Float64} = N,           # generator rating
    ū::Vector{Float64} = ū,          # max number of each type of generators
    c::Vector{Float64} = c,           # build cost per MW
    mg::Vector{Int} = mg,
    fuel_price::Vector{Float64} = fuel_price,
    heat_rate::Vector{Int} = heat_rate,
    eff::Vector{Float64} = eff,
    om_cost::Vector{Float64} = om_cost,
    s₀::Vector{Int} = s₀,
    penalty::Float64 = 1e10,
    initial_demand::Float64 = 1e7,
    seed::Int = 1234,
    num_Ω::Int = 10,
)::NamedTuple

    # binary encoding of the state
    binaryInfo = integer_binarization(ū)
    (A, n, d) = (binaryInfo.A, binaryInfo.n, binaryInfo.d)   # if needed later

    # compute c1 (investment cost per stage)
    c1 = [[c[i] * mg[i] / (1 + r)^j for i in 1:length(c)] for j in 1:T]

    # compute c2 (operating cost per stage)
    c2 = [
        [fuel_price[i] * heat_rate[i] * 1e-3 * eff[i] for i in 1:length(c)] * (1.02)^j +
        om_cost * (1.03)^j
        for j in 1:T
    ]

    # stage data
    stageDataList = Dict{Int, StageData}()
    for t in 1:T
        stageDataList[t] = StageData(c1[t], c2[t], ū, 8760.0, N, s₀, penalty)
    end

    ################################################################################
    ##########################  Random variable generation  ########################
    ################################################################################

    # number of realizations at each stage
    N_rv = [num_Ω for _ in 1:T]

    Random.seed!(seed)

    # Ω[t][i] = RandomVariables at stage t, node i
    Ω = Dict{Int, Dict{Int, RandomVariables}}()

    for t in 1:T
        Ω[t] = Dict{Int, RandomVariables}()
        for i in 1:N_rv[t]
            if t == 1
                Ω[t][i] = RandomVariables([initial_demand])
            else
                Ω[t][i] = RandomVariables(rand(Uniform(1, 1.5)) * Ω[t - 1][i].d)
            end
        end
    end

    # transition probabilities: here we use uniform distribution at each stage
    probList = Dict{Int, Vector{Float64}}()
    for t in 1:T
        probList[t] = fill(1.0 / N_rv[t], N_rv[t])
    end

    return (
        probList      = probList,
        stageDataList = stageDataList,
        Ω             = Ω,
        binaryInfo    = binaryInfo,
    )
end


################################################################################
###############################  Printing helpers  ##############################
################################################################################

"""
Print one line of iteration information.

Columns:
- iteration index
- LB, UB
- relative gap (%)
- iteration time (seconds)
- number of level-set iterations (LM_iter)
- total elapsed time (seconds)
"""
function print_iteration_info(
    i::Int,
    LB::Float64,
    UB::Float64,
    gap::Float64,
    iter_time::Float64,
    LM_iter::Int,
    total_Time::Float64,
)::Nothing
    @printf(
        "%4d | %12.2f     | %12.2f     | %9.2f%%     | %9.2f s     | %6d     | %10.2f s\n",
        i,
        LB,
        UB,
        gap,
        iter_time,
        LM_iter,
        total_Time,
    )
    return
end

"""
Print the header/bar for iteration information.
"""
function print_iteration_info_bar()::Nothing
    println("------------------------------------------------- Iteration Info ------------------------------------------------")
    println("Iter |        LB        |        UB        |       Gap      |      i-time     |    #D.     |     T-Time")
    println("-----------------------------------------------------------------------------------------------------------------")
    return
end


################################################################################
###############################  Logging / saving  #############################
################################################################################

"""
Save SDDP/SDDiP results to disk if `logger_save` is true.

The file name encodes `(T, num, algorithm, cutSelection, tightness)`.
"""
# function save_info(
#     param::SDDPParam,
#     sddpResults::Dict;
#     logger_save::Bool = true,
# )::Nothing
#     if !logger_save
#         return
#     end

#     cutSelection = param.cutSelection
#     num          = param.num
#     T            = param.T
#     tightness    = param.tightness
#     algorithm    = param.algorithm

#     filedir = "/Users/aaron/SDDiP_with_EnhancedCut/src/GenerationExpansion/new_logger/Periods$(T)-Real$(num)/"
#     filename = "$algorithm-$cutSelection-$tightness.jld2"
#     filepath = filedir * filename

#     save(filepath, "sddpResults", sddpResults)
#     return
# end


################################################################################
###############################  Parameter setup  ##############################
################################################################################
"""
Create and return an `SDDPParam` instance with reasonable defaults.

You can override any field via keyword arguments, e.g.:

    param = param_setup(T = 10, num = 5, cutType = :LNC)
"""
function param_setup(;
    timeSDDP::Real           = 3600.0,
    gapSDDP::Float64         = 1e-3,
    iterSDDP::Int            = 100,
    sample_size_SDDP::Int    = 5,
    solverGap:: Float64      = 1e-4,
    solverTime:: Float64     = 10.0,
    ε::Float64               = 0.125,
    discreteZ::Bool          = true,
    cutType::Symbol          = :SMC,
    cutSparsity::Bool        = true,
    partitionRule::Symbol    = :Incumbent,
    branchingStart::Int      = 3,
    M::Int                   = 1,
    T::Int                   = 12,
    num::Int                 = 10,
    verbose::Bool            = false,
    ℓ1::Float64              = 1.0,
    ℓ2::Float64              = 1.0,
    nxt_bound::Float64       = 1e8,
    logger_save::Bool        = true,
    algorithm::Symbol        = :SDDiP,
)::SDDPParam
    return SDDPParam(
        float(timeSDDP),
        gapSDDP,
        iterSDDP,
        solverGap,
        solverTime,
        sample_size_SDDP,
        ε,
        discreteZ,
        cutType,
        cutSparsity,
        partitionRule,
        branchingStart,
        M,
        T,
        num,
        verbose,
        ℓ1,
        ℓ2,
        nxt_bound,
        logger_save,
        algorithm,
    )
end

"Default parameter set for SDDP/SDDiP."
const DEFAULT_PARAM = param_setup()