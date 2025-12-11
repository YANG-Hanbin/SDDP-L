## binarize stage variable, x = A * L, where L ∈ {0, 1}ⁿ
function intergerBinarization(ū::Vector{Float64})
    row_num = size(ū)[1];

    var_num = floor.(Int, log.(2,ū)) .+ 1; 

    col_num = sum(Int, var_num);

    A = zeros(Int64, row_num, col_num)
 
    for i in 1:row_num
        l = sum(var_num[l] for l in 1:i)
        for j in (l+1-var_num[i]):(l+var_num[i]-var_num[i])
            A[i,j] = 2^(j-(l+1-var_num[i]))
        end
    end
    return BinaryInfo(A, col_num, row_num)
end






################### nonanticipativity for multistage problem #######################

function recursion_scenario_tree(pathList::Vector{Int64}, 
                                P::Float64, 
                                scenario_sequence::Dict{Int64, Dict{Int64, Any}}, 
                                t::Int64;   
                                Ω::Dict{Int64,Dict{Int64,RandomVariables}} = Ω, prob::Dict{Int64,Vector{Float64}} = prob, T::Int64 = 2)

    if t ≤ T
        for ω_key in keys(Ω[t])

            pathList_copy = copy(pathList)
            P_copy = copy(P)

            push!(pathList_copy, ω_key)
            P_copy = P_copy * prob[t][ω_key]

            recursion_scenario_tree(pathList_copy, P_copy, scenario_sequence, t+1, Ω = Ω, prob = prob, T = T)
        end
    else
        if haskey(scenario_sequence, 1)
            scenario_sequence[maximum(keys(scenario_sequence))+1] = Dict(1 => pathList, 2 => P)
        else
            scenario_sequence[1] = Dict(1 => pathList, 2 => P)
        end
        return scenario_sequence
    end

end

# scenario_sequence = Dict{Int64, Dict{Int64, Any}}()  ## the first index is for scenario index, the second one is for stage
# pathList = Vector{Int64}()
# push!(pathList, 1)
# P = 1.0

# recursion_scenario_tree(pathList, P, scenario_sequence, 2, T = T)
# scenario_tree = scenario_sequence



## sampling function 
function DrawSamples(scenario_sequence::Dict{Int64, Dict{Int64, Any}})
    # draw f, A, B, C, b from Ωₜ according to distribution P
    P = Vector{Float64}()
    for key in keys(scenario_sequence)
        push!(P, scenario_sequence[key][2])
    end
    items = [i for i in keys(scenario_sequence)]
    weights = Weights(P)
    j = sample(items, weights)
    return j
end


## form scenarios
function SampleScenarios(scenario_sequence::Dict{Int64, Dict{Int64, Any}}; T::Int64 = 5, M::Int64 = 30)
    ## a dict to store realization for each stage t in scenario k
    scenarios = Dict{Int64, Int64}()
    for k in 1:M
          scenarios[k] = DrawSamples(scenario_sequence)
    end
    return scenarios
end

## rounding data
function round!(a::Float64)               ## a = 1.3333e10
    b = floor(log10(a))                   ## b = 10
    c = round(a/10^b,digits = 2)          ## c = 1.33
    d = c * 10^b                          ## d = 1.33e10
    return [b, c, d]
end


## setup coefficients
function dataGeneration(;   
    T::Int64 = 2, num_Ω::Int64 = num_Ω, seed::Int64 = 1234,
    r::Float64 = r, ## the annualized interest rate
    N::Matrix{Float64} = N, ## Generator rating
    ū::Vector{Float64} = ū, ## maximum number of each type of generators
    c::Vector{Float64} = c, # c_g from table 4, cost/MW to build a generator of type g
    mg::Vector{Int64} = mg,
    fuel_price::Vector{Float64} = fuel_price,
    heat_rate::Vector{Int64} = heat_rate,
    eff::Vector{Float64} = eff,
    om_cost::Vector{Float64} = om_cost, 
    s₀::Vector{Int64} = s₀,
    penalty::Float64 = penalty, 
    total_hours::Float64 = 8760., ## total hours in a year
    initial_demand::Float64 = initial_demand
)::NamedTuple

    binaryInfo = intergerBinarization(ū)

    # Compute c1 (investment cost per MW)
    c1 = [[c[i] * mg[i] / (1 + r)^j for i in 1:6] for j in 1:T]./1e5                                                            # 1e5 is used to scale the cost to a reasonable range

    # Compute c2 (generation cost per MWh)
    c2 = [[(fuel_price[i] * heat_rate[i] * 1e-3 / eff[i]) * (1.02)^j + om_cost[i] * (1.03)^j for i in 1:6] for j in 1:T]./1e5

    stageDataList = Dict{Int64,StageData}()
    for t in 1:T 
        stageDataList[t] = StageData(c1[t], c2[t], ū, total_hours, N, s₀, penalty/1e5)
    end

    ##########################################################################################
    ############################  To generate random variable  ###############################
    ##########################################################################################
    N_rv = Vector{Int64}()  # the number of realization of each stage
    N_rv = [num_Ω for t in 1:T] 
    # N_rv = round.(rand(T) * 10) 

    Random.seed!(seed)

    Ω = Dict{Int64,Dict{Int64,RandomVariables}}()   # each stage t, its node are included in Ω[t]

    for t in 1:T 
        Ω[t] = Dict{Int64,RandomVariables}()
        for i in 1:N_rv[t]
            if t == 1
                Ω[t][i]= RandomVariables([initial_demand])
            else
                # Ω[t][i]= RandomVariables( rand(Uniform(1.05, 1.2))*Ω[t-1][i].d )
                Ω[t][i]= RandomVariables( 1.05^t * rand(Uniform(1.0, 1.2))*Ω[1][1].d )
            end
        end
    end


    probList = Dict{Int64,Vector{Float64}}()  # P(node in t-1 --> node in t ) = prob[t]
    for t in 1:T 
        # randomVector = round.(rand(N_rv[t]),digits = 2)
        # probList[t] = round.(randomVector/sum(randomVector),digits = 2)
        probList[t] = [1/N_rv[t] for i in 1:N_rv[t]]
    end

    return (
        probList = probList, 
        stageDataList = stageDataList, 
        Ω = Ω, 
        binaryInfo = binaryInfo
    )
end