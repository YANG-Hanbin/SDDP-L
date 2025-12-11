#############################################################################################
####################################   Data Structure   #####################################
#############################################################################################
struct CutCoefficient
    v               ::Dict{Int64,Dict{Int64, Float64}} # [i][k] where i is the iteration index, and k is the scenario index
    π               ::Dict{Int64,Dict{Int64, Vector{Float64}}}  # [[1.,2.,3.],[1.,2.,3.]]  -- push!(Π, π) to add a new element
end




struct RandomVariables
    d   ::Vector{Float64}
end



struct StageData ## with the assumption that only b has stochasticity
    c1       ::Vector{Float64}
    c2       ::Vector{Float64}
    ū        ::Vector{Float64}
    h        ::Float64
    N        ::Matrix{Float64}
    s₀       ::Vector{Float64}
    penalty  ::Float64
end



struct ForwardModelInfo
    model           :: Model
    x               :: Vector{VariableRef} ## for current state, x is the number of generators
    Lt              :: Vector{VariableRef} ## stage variable, A * Lt is total number of generators built
    y               :: Vector{VariableRef} ## amount of electricity
    θ               :: VariableRef
    # demand          :: Vector{Float64}
    slack           :: VariableRef
    # sum_generator   :: Vector{Float64}
end



struct BackwardModelInfo
    model           :: Model
    x               :: Vector{VariableRef} ## for current state, x is the number of generators
    Lt              :: Vector{VariableRef} ## stage variable, A * Lt is total number of generators built
    Lc              :: Vector{VariableRef} ## local cooy variable
    y               :: Vector{VariableRef} ## amount of electricity
    θ               :: VariableRef
    # demand          :: Vector{Float64}
    slack           :: VariableRef
    # sum_generator   :: Vector{Float64}
end



struct GurobiModelInfo
    model           :: Model
    x               :: Array{VariableRef} ## for current state, x is the number of generators
    y               :: Array{VariableRef} ## amount of electricity
    slack           :: Matrix{VariableRef}
    num_Ω           :: Int64
end


## data structure for levelset method
## data structure for levelset method
mutable struct FunctionHistory
    f_his        :: Dict{Int64, Float64}          ## record f(x_j)     
    G_max_his    :: Dict{Int64, Float64}          ## record max(g[k] for k in 1:m)(x_j)   
end


mutable struct CurrentInfo
    x            :: Vector{Float64}                 ## record x point
    f            :: Float64                         ## record f(x_j)
    G            :: Dict{Int64, Float64} 
    df           :: Vector{Float64}
    dG           :: Dict{Int64, Vector{Float64}}    ## actually is a matrix.  But we use dict to store it
    L_at_x̂       :: Float64                         ## only for solving dual problem
end




struct ModelInfo
    model :: Model
    x     :: Vector{VariableRef}
    y     :: VariableRef
    z     :: VariableRef
end




struct BinaryInfo
    A     ::Matrix{Int64}
    n     ::Int64
    d     ::Int64
end






struct LevelSetMethodParam
    μ             ::Float64                     ## param for adjust α
    λ             ::Union{Float64, Nothing}     ## param for adjust level
    threshold     ::Float64                     ## threshold for Δ
    nxt_bound     ::Float64                     ## lower bound for solving next iteration point π
    max_iter      ::Int64     
    Output        ::Int64                       ## Gurobi Output parameter
    Output_Gap    ::Bool                        ## if True will print Δ info
    ## the following parameters are problem-specific
    L̂             ::Vector{Float64}                 ## first stage solution
    cutSelection  ::String
    L̃             ::Union{Vector{Float64}, Nothing} ## interior point
    f_star_value  ::Union{Float64, Nothing}         ## subproblem optimal value
end


