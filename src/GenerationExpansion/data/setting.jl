using Random
using Distributions
using StatsBase

# === Utility / Types note ===
# This code assumes you have these user-defined types defined elsewhere:
# - BinaryInfo(A::Matrix{Int64}, col_num::Int, row_num::Int)
# - RandomVariables(d::Vector{Float64})  # or adjust to your concrete type
# - StageData(...)                       # adjust constructor as needed

# ---------------------------
# Integer (binary) binarization
# ---------------------------
"""
    integerBinarization(u_bar::Vector{Float64}) -> BinaryInfo

Given an upper bound vector `u_bar` (length n) that specifies the maximum
integer value for each original integer variable, produce a binary encoding
matrix `A` and related sizes such that x = A * L where L ∈ {0,1}^m.

Returns a BinaryInfo containing A, total number of binary columns, and number of rows.
"""
function integerBinarization(u_bar::Vector{Float64})
    # number of original integer variables
    row_num = length(u_bar)

    # For each variable i, compute how many binary digits are needed.
    # If u_bar[i] <= 0 we treat it as needing 0 bits (or 1 bit set to zero)
    var_num = [u_bar[i] <= 0.0 ? 0 : floor(Int, log2(u_bar[i])) + 1 for i in 1:row_num]

    col_num = sum(var_num)

    # A maps binary vector L (length col_num) to integer vector x (length row_num).
    A = zeros(Int64, row_num, col_num)

    # Fill A with powers of two for each variable's block.
    offset = 0
    for i in 1:row_num
        bits = var_num[i]
        for j in 0:(bits - 1)
            # column index in A is offset + j + 1 (1-based)
            A[i, offset + j + 1] = Int(2)^j
        end
        offset += bits
    end

    return BinaryInfo(A, col_num, row_num)
end

# ---------------------------
# Scenario tree recursion (no hidden globals)
# ---------------------------
"""
    recursion_scenario_tree(pathList, P, scenario_sequence, t, Ω, prob, T)

Recursively traverse the scenario tree stored in `Ω` (Ω[t] is a Dict of nodes at stage t)
and probabilities `prob` (prob[t] is a vector of probabilities for nodes at stage t),
collecting complete paths and their scenario probabilities into `scenario_sequence`.

- pathList: vector of node indices collected so far (should start with root index).
- P: current path probability (start with 1.0).
- scenario_sequence: Dict mapping scenario index -> Dict(1 => pathList, 2 => P)
- t: current stage index (1-based)
- Ω: Dict{Int, Dict{Int, RandomVariables}}
- prob: Dict{Int, Vector{Float64}}
- T: final stage (inclusive)
"""
function recursion_scenario_tree(
    pathList::Vector{Int},
    P::Float64,
    scenario_sequence::Dict{Int, Dict{Int, Any}},
    t::Int;
    Ω::Dict{Int, Dict{Int, Any}},
    prob::Dict{Int, Vector{Float64}},
    T::Int = 2
)
    if t <= T
        # iterate through nodes at stage t in deterministic order
        for ω_key in sort(collect(keys(Ω[t])))
            # copy path and probability for the recursive call
            path_copy = copy(pathList)
            push!(path_copy, ω_key)
            P_copy = P * prob[t][ω_key]

            recursion_scenario_tree(path_copy, P_copy, scenario_sequence, t+1;
                                   Ω = Ω, prob = prob, T = T)
        end
    else
        # reached a leaf; append to scenario_sequence with next index
        next_index = isempty(scenario_sequence) ? 1 : maximum(keys(scenario_sequence)) + 1
        scenario_sequence[next_index] = Dict(1 => copy(pathList), 2 => P)
    end

    return scenario_sequence
end

# ---------------------------
# Sampling helpers
# ---------------------------
"""
    DrawSamples(scenario_sequence)

Randomly sample a scenario index from the given `scenario_sequence` keyed by indices,
weighted by each scenario's probability (stored at scenario_sequence[k][2]).
Returns the sampled scenario index.
"""
function DrawSamples(scenario_sequence::Dict{Int, Dict{Int, Any}})
    # keep deterministic alignment between items and weights by sorting keys
    items = sort(collect(keys(scenario_sequence)))
    weights = Float64[]
    for k in items
        push!(weights, scenario_sequence[k][2])
    end

    # construct Weights and sample one index
    j = sample(items, Weights(weights))
    return j
end

"""
    SampleScenarios(scenario_sequence; T=5, M=30)

Draw M independent scenario indices (according to their probabilities) and return a Dict
mapping sample index (1..M) -> scenario index (key from scenario_sequence).
"""
function SampleScenarios(scenario_sequence::Dict{Int, Dict{Int, Any}}; T::Int = 5, M::Int = 30)
    scenarios = Dict{Int, Int}()
    for k in 1:M
        scenarios[k] = DrawSamples(scenario_sequence)
    end
    return scenarios
end

# ---------------------------
# Rounding helper (don't override Base.round!)
# ---------------------------
"""
    round_scientific(a::Float64)

Return a tuple (exponent b, mantissa c, rounded_value d) where:
- b = floor(log10(a))
- c = round(a / 10^b, digits=2)
- d = c * 10^b
Useful for producing a shorter human-readable representation of very large numbers.
"""
function round_scientific(a::Float64)
    if a == 0.0
        return (0, 0.0, 0.0)
    end
    b = floor(Int, log10(abs(a)))
    c = round(a / 10.0^b, digits = 2)
    d = c * 10.0^b
    return (b, c, d)
end

# ---------------------------
# Data generation
# ---------------------------
"""
    dataGeneration(; kwargs...)

Create stage data, random variable realizations Ω, and a probability list for each stage.

Keyword arguments (examples; adapt defaults or pass your own):
- T::Int = number of stages
- num_Ω::Int = number of realizations per stage
- seed::Int = RNG seed
- r::Float64 = annual interest rate
- N::Matrix{Float64} = generator rating matrix
- u_bar::Vector{Float64} = upper bounds on generator counts
- c::Vector{Float64} = capital cost per MW per generator type
- mg::Vector{Int} = capacity multipliers or sizes
- fuel_price, heat_rate, eff, om_cost, s0, penalty, total_hours, initial_demand
"""
function dataGeneration(;   
    T::Int = 2,
    num_Ω::Int = 3,
    seed::Int = 1234,
    r::Float64 = 0.05,
    N::Matrix{Float64} = zeros(0,0),
    u_bar::Vector{Float64} = Float64[],
    c::Vector{Float64} = Float64[],
    mg::Vector{Int} = Int[],
    fuel_price::Vector{Float64} = Float64[],
    heat_rate::Vector{Float64} = Float64[],
    eff::Vector{Float64} = Float64[],
    om_cost::Vector{Float64} = Float64[],
    s0::Vector{Int} = Int[],
    penalty::Float64 = 1e6,
    total_hours::Float64 = 8760.0,
    initial_demand::Float64 = 100.0
)
    # compute binary encoding info
    binaryInfo = integerBinarization(u_bar)

    G = length(c)  # number of generator types, adapt if necessary

    # Compute c1 (investment cost per MW) and c2 (generation cost per MWh) for each stage
    # c1 and c2 are vectors of length G for each stage t
    c1 = [ [ c[i] * mg[i] / (1 + r)^j for i in 1:G ] for j in 1:T ] ./ 1e5
    c2 = [ [ (fuel_price[i] * heat_rate[i] * 1e-3 / eff[i]) * (1.02)^j + om_cost[i] * (1.03)^j for i in 1:G ] for j in 1:T ] ./ 1e5

    # Build stage data list with StageData objects (adjust constructor as needed)
    stageDataList = Dict{Int, Any}()
    for t in 1:T 
        stageDataList[t] = StageData(c1[t], c2[t], u_bar, total_hours, N, s0, penalty/1e5)
    end

    # Random seed
    Random.seed!(seed)

    # number of realizations per stage (N_rv)
    N_rv = [num_Ω for _ in 1:T]

    # generate Ω: a Dict mapping stage t => Dict(node_index => RandomVariables)
    Ω = Dict{Int, Dict{Int, Any}}()
    for t in 1:T
        Ω[t] = Dict{Int, Any}()
        for i in 1:N_rv[t]
            if t == 1
                # stage 1 base demand is initial_demand (stored in a vector if you need multiple dims)
                Ω[t][i] = RandomVariables([initial_demand])
            else
                # for later stages generate demand as scaled version of initial (or previous stage mean)
                # here we use a random multiplier in [1.0, 1.2], scaled by 1.05^t
                multiplier = rand(Uniform(1.0, 1.2))
                Ω[t][i] = RandomVariables([1.05^t * multiplier * initial_demand])
            end
        end
    end

    # set equal probabilities for nodes at each stage (uniform)
    probList = Dict{Int, Vector{Float64}}()
    for t in 1:T
        probList[t] = fill(1.0 / N_rv[t], N_rv[t])
    end

    return (
        probList = probList,
        stageDataList = stageDataList,
        Ω = Ω,
        binaryInfo = binaryInfo
    )
end