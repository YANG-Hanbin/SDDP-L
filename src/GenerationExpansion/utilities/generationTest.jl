"""
    The data is provided by Jin2011.

"""

##########################################################################################
############################  To generate Stage Data  ####################################
##########################################################################################


r = 0.08
T = 3

N = [
    1130.0 0 0 0 0 0;
    0 390 0 0 0 0; 
    0 0 380 0 0 0; 
    0 0 0 1180 0 0;
    0 0 0 0 175 0; 
    0 0 0 0 0 560]

ū = [4.0, 10, 10, 1, 45, 4]

function intergerBinarization(
    ū::Vector{Float64}
)::BinaryInfo
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


binaryInfo = intergerBinarization(ū)
(A, n, d) = (binaryInfo.A, binaryInfo.n, binaryInfo.d)




c = [1.445, 0.795, 0.575, 1.613, 1.650, 1.671] * 10^6  # c_g from table 4, cost/MW to build a generator of type g
mg = [1200, 400, 400, 1200, 500, 600]

#  compute c1
c1 = [[c[i]*mg[i]/(1+r)^j for i in 1:6 ]  for j in 1:T ] 


#  compute c2
fuel_price = [3.37, 9.11, 9.11, 9.3e-4, 0, 3.7]
heat_rate = [8844, 7196, 10842, 10400, 1, 8613]
eff = [0.4, 0.56, 0.4, 0.45, 0, 0.48]
om_cost = [4.7, 2.11, 3.66, 0.51, 5.00, 2.98]

c2 = [[fuel_price[i]*heat_rate[i]*1e-3*eff[i] for i in 1:6]*(1.02)^j + om_cost*(1.03)^j for j in 1:T]


stageDataList = Dict{Int64,StageData}()
s₀ = [1,2,3,4,5,6]
penalty = 1e5


for t in 1:T 
    stageDataList[t] = StageData(c1[t], c2[t], ū, 8760., N, s₀, penalty)
end




##########################################################################################
############################  To generate random variable  ###############################
##########################################################################################
N_rv = Vector{Int64}()  # the number of realization of each stage
num_Ω = 10
N_rv = [num_Ω for t in 1:T]  ## xxxx 需要update


Random.seed!(1234)

Ω = Dict{Int64,Dict{Int64,RandomVariables}}()   # each stage t, its node are included in Ω[t]
initial_demand = 2e6  #  5.685e8

for t in 1:T 
    Ω[t] = Dict{Int64,RandomVariables}()
    for i in 1:N_rv[t]
        if t == 1
            Ω[t][i]= RandomVariables([initial_demand])
        else
            Ω[t][i]= RandomVariables( (rand(1)[1]/5+1)*Ω[t-1][i].d )
        end
    end
end




probList = Dict{Int64,Vector{Float64}}()  # P(node in t-1 --> node in t ) = prob[t]
for t in 1:T 
    probList[t] = [0.1 for i in 1:N_rv[t]]
end






function recursion_scenario_tree(
    pathList::Vector{Int64}, 
    P::Float64, 
    scenario_sequence::Dict{Int64, Dict{Int64, Any}}, 
    t::Int64;   
    Ω::Dict{Int64,Dict{Int64,RandomVariables}} = Ω, 
    prob::Dict{Int64,Vector{Float64}} = prob, 
    T::Int64 = 2
)
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


scenario_sequence = Dict{Int64, Dict{Int64, Any}}()  ## the first index is for scenario index, the second one is for stage
pathList = Vector{Int64}()
push!(pathList, 1)
P = 1.0

recursion_scenario_tree(pathList, P, scenario_sequence, 2, T = T, prob = probList)
scenario_tree = scenario_sequence



































