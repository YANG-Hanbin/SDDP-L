## ====================================================================================== ##
## =================================== DC-OPF Problem =================================== ##
## ====================================================================================== ##
"""
    Structs for the security-constrained unit commitment (SCUC) problem
    
    1. Define the power grid network system
        mutable struct IndexSets
            D       :: Vector{Int64}                    ## set of load demand
            G       :: Vector{Int64}                    ## set of generators
            L       :: Vector{Tuple{Int64, Int64}}      ## set of transmission lines
            B       :: Vector{Int64}                    ## set of buses
            T       :: Int64                            ## set of time periods  1:T
            Dᵢ      :: Dict{Int64,Vector{Int64}}       
            Gᵢ      :: Dict{Int64,Vector{Int64}}
            out_L   :: Dict{Int64,Vector{Int64}} 
            in_L    :: Dict{Int64,Vector{Int64}}
        end
    
    2. Define the parameters of the demand
        mutable struct ParamDemand
            demand        :: Dict{Int64, Float64}                    ## mean of load demand for d ∈ D
            w             :: Dict{Int64, Float64}                    ## priority level of load D
            cb            :: Dict{Int64, Float64}                    ## set of fire damage cost cᵢ at :b ∈ B
            cg            :: Dict{Int64, Float64}                    ## set of fire damage cost cᵢ at :g ∈ G
            cl            :: Dict{Tuple{Int64, Int64}, Float64}      ## set of fire damage cost cᵢ at :l ∈ L
            penalty       :: Float64                                 ## penalty parameter for constraints b and c
        end
    
    3. Define the parameters of the optimal power flow (OPF)
        struct ParamOPF  ## included in a period dict
            b           :: Dict{Tuple{Int64, Int64}, Float64}       ## :l ∈ L  total line charging susceptance 
            θmax        :: Float64                                  ## angle difference
            θmin        :: Float64
            W           :: Dict{Tuple{Int64, Int64}, Float64}       ## :l ∈ L
            smax        :: Dict{Int64, Float64}                     ## :g ∈ G  Pmax
            smin        :: Dict{Int64, Float64}                     ## :g ∈ G
            M           :: Dict{Int64, Float64}                     ## :g ∈ G  multi-period ramping limit 
            slope       :: Dict{Int64, Dict{Int64, Float64}}        ## :g ∈ G, :o ∈ O  cost function slope
            intercept   :: Dict{Int64, Dict{Int64, Float64}}        ## :g ∈ G, :o ∈ O  cost function intercept
            C_start     :: Dict{Int64, Float64}                     ## :g ∈ G
            C_down      :: Dict{Int64, Float64}                     ## :g ∈ G
        end

    4. Define the random variables
        struct RandomVariables  
            deviation        :: Dict{Int64, Float64}                   ## :d ∈ D, realization deviation from the mean demand
        end

    5. Define the stage realization
        struct StageRealization
            nodes            :: Dict{Int64, RandomVariables}           ## :n ∈ N_t, nodes of stage t
            prob             :: Dict{Int64, Float64}                   ## :n ∈ N_{t+1}, probabilities for next stage nodes
        end
    
    6. Define the scenario tree
        struct ScenarioTree
            tree      :: Dict{Int64, StageRealization}                ## :t ∈ [1, ..., T], realization percentage of the demand
        end
"""
mutable struct IndexSets
    D       :: Vector{Int64}                    ## set of load demand
    G       :: Vector{Int64}                    ## set of generators
    L       :: Vector{Tuple{Int64, Int64}}      ## set of transmission lines
    B       :: Vector{Int64}                    ## set of buses
    T       :: Int64                            ## set of time periods  1:T
    Dᵢ      :: Dict{Int64,Vector{Int64}}       
    Gᵢ      :: Dict{Int64,Vector{Int64}}
    out_L   :: Dict{Int64,Vector{Int64}} 
    in_L    :: Dict{Int64,Vector{Int64}}
end

mutable struct ParamDemand
    demand        :: Dict{Int64, Float64}                    ## mean of load demand for d ∈ D
    w             :: Dict{Int64, Float64}                    ## priority level of load D
    cb            :: Dict{Int64, Float64}                    ## set of fire damage cost cᵢ at :b ∈ B
    cg            :: Dict{Int64, Float64}                    ## set of fire damage cost cᵢ at :g ∈ G
    cl            :: Dict{Tuple{Int64, Int64}, Float64}      ## set of fire damage cost cᵢ at :l ∈ L
    penalty       :: Float64                                 ## penalty parameter for constraints b and c
end

struct ParamOPF  ## included in a period dict
    b           :: Dict{Tuple{Int64, Int64}, Float64}       ## :l ∈ L  total line charging susceptance 
    θmax        :: Float64                                  ## angle difference
    θmin        :: Float64
    W           :: Dict{Tuple{Int64, Int64}, Float64}       ## :l ∈ L
    smax        :: Dict{Int64, Float64}                     ## :g ∈ G  Pmax
    smin        :: Dict{Int64, Float64}                     ## :g ∈ G
    M           :: Dict{Int64, Float64}                     ## :g ∈ G  multi-period ramping limit 
    slope       :: Dict{Int64, Dict{Int64, Float64}}        ## :g ∈ G, :o ∈ O  cost function slope
    intercept   :: Dict{Int64, Dict{Int64, Float64}}        ## :g ∈ G, :o ∈ O  cost function intercept
    C_start     :: Dict{Int64, Float64}                     ## :g ∈ G
    C_down      :: Dict{Int64, Float64}                     ## :g ∈ G
end

struct RandomVariables  
    deviation        :: Dict{Int64, Float64}                   ## :d ∈ D, realization deviation from the mean demand
end

struct StageRealization
    nodes            :: Dict{Int64, RandomVariables}           ## :n ∈ N_t, nodes of stage t
    prob             :: Dict{Int64, Float64}                   ## :n ∈ N_{t+1}, probabilities for next stage nodes
end

struct ScenarioTree
    tree      :: Dict{Int64, StageRealization}                ## :t ∈ [1, ..., T], realization percentage of the demand
end

## ====================================================================================== ##
## ============ Stochastic Dual Dynamic Programming with Lifting Algorithm ============== ##
## ====================================================================================== ##
abstract type SequentialModels end

abstract type SolutionMethod end
"""
    mutable struct SDDPModel  <: ForwardModel 
        model           ::Model
        BinVar          ::Dict{Any, Dict{Any, VariableRef}}
        IntVar          ::Dict{Any, Dict{Any, Dict{Symbol, Any}}}
        ContVar         ::Dict{Any, Dict{Any, Dict{Symbol, Any}}}
        IntVarLeaf      ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}}
        ContVarLeaf     ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}}
        IntVarBinaries  ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, VariableRef}}}}
        ContVarBinaries ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, VariableRef}}}}
    end

Forward nodal problem.

Fields
------

- `model`:
    the forward model is used to solve each sub-problem.
- `BinVar`: 
    the binary state variables of the problem, the first `Any` is the variable name, the second `Any` is the index.
- `IntVar`: 
    the general integer state variables of the problem, the first `Any` is the variable name, the second `Any` is the index.
- `ContVar`:
    the continuous state variables of the problem, the first `Any` is the variable name, the second `Any` is the index.
- `IntVarLeaf` and `ContVarLeaf`:
    the partition tree info of Int/Cont variables of the problem:
    the first `Any` is the variable name, the second `Any` is the index, 
    the third `Any` is the leaf node index,
    `Symbol ∈ {:lb, :ub, :parent, :sibling, :var}` is the key to record the information.
- `IntVarBinaries` and `ContVarBinaries`:
    the binarization variables of the integer/continuous state variables of the problem, the first `Any` is the variable name, the second `Any` is the index.
"""
mutable struct SDDPModel  <: SequentialModels 
    model           ::Model
    BinVar          ::Union{Nothing, Dict{Any, Dict{Any, VariableRef}}}
    IntVar          ::Union{Nothing, Dict{Any, Dict{Any, VariableRef}}}
    ContVar         ::Union{Nothing, Dict{Any, Dict{Any, VariableRef}}}
    IntVarLeaf      ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}}
    ContVarLeaf     ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}}
    IntVarBinaries  ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, VariableRef}}}}
    ContVarBinaries ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, VariableRef}}}}
end

"""
    mutable struct StateInfo
        BinVar          ::Union{Nothing, Dict{Any, Dict{Any, Any}}}
        IntVar          ::Union{Nothing, Dict{Any, Dict{Any, Any}}}
        ContVar         ::Union{Nothing, Dict{Any, Dict{Any, Any}}}
        IntVarLeaf      ::Union{Nothing, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}
        ContVarLeaf     ::Union{Nothing, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}
        StageValue      ::Union{Nothing, Float64}
        StateValue      ::Union{Nothing, Float64}
        IntAugState     ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Any}}}}
        ContAugState    ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Any}}}}
        IntStateBin     ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Any}}}}
        ContStateBin    ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Any}}}}
    end

Record the state variables of the problem.

Fields
------

- `BinVar`: 
    the binary state variables of the problem, the first `Any` is the variable name, the second `Any` is the index.
- `IntVar`: 
    the general integer state variables of the problem, the first `Any` is the variable name, the second `Any` is the index.
- `ContVar`:
    the continuous state variables of the problem, the first `Any` is the variable name, the second `Any` is the index.
- `IntVarLeaf` and `ContVarLeaf`:
    the partition tree info of Int/Cont variables of the problem:
    the first `Any` is the variable name, the second `Any` is the index, 
    the third `Any` is the leaf node index,
    `Symbol ∈ {:lb, :ub, :parent, :sibling, :var}` is the key to record the information.
- `StageValue`:
    the current stage cost
- `StateValue`:
    the objective value of the state t, i.e., stage cost + future cost
- `IntAugState`:
    the augmented variables of the integer state variables are used as the augmented state variable
- `ContAugState`:
    the augmented variables of the continuous state variables are used as the augmented state variable
- `IntStateBin` and `ContStateBin`:
    the binarization variables of the integer/continuous state variables of the problem, the first `Any` is the variable name, the second `Any` is the index of the variable, and the third `Any` is the index of the binarization variables.
"""
mutable struct StateInfo
    BinVar              ::Union{Nothing, Dict{Any, Dict{Any, Any}}}
    IntVar              ::Union{Nothing, Dict{Any, Dict{Any, Any}}}
    ContVar             ::Union{Nothing, Dict{Any, Dict{Any, Any}}}
    IntVarLeaf          ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}}
    ContVarLeaf         ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}}
    StageValue          ::Union{Nothing, Float64}
    StateValue          ::Union{Nothing, Float64}
    IntAugState         ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Any}}}}
    ContAugState        ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Any}}}}
    IntStateBin         ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Any}}}}
    ContStateBin        ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Any}}}}
end

## ====================================================================================== ##
## ================================= Level-set Method =================================== ##
## ====================================================================================== ##

"""
    mutable struct LevelSetMethodOracleParam
        μ             ::Union{Nothing, Float64}                     ## param for adjust α
        λ             ::Union{Nothing, Float64}                     ## param for adjust level
        threshold     ::Union{Nothing, Float64}                     ## threshold for Δ
        nxt_bound     ::Union{Nothing, Float64}                     ## lower bound for solving next iteration point π
        MaxIter       ::Union{Nothing, Any}     
        verbose       ::Union{Nothing, Bool}                        ## if True will print Δ info
        x₀            ::Union{Nothing, StateInfo}                   ## initial point for the level-set method
    end

    fields
"""
mutable struct LevelSetMethodOracleParam
    μ             ::Union{Nothing, Float64}                     ## param for adjust α
    λ             ::Union{Nothing, Float64}                     ## param for adjust level
    threshold     ::Union{Nothing, Float64}                     ## threshold for Δ
    nxt_bound     ::Union{Nothing, Float64}                     ## lower bound for solving next iteration point π
    MaxIter       ::Union{Nothing, Any}     
    verbose       ::Union{Nothing, Bool}                        ## if True will print Δ info
    x₀            ::Union{Nothing, StateInfo}
end

## data structure for level-set method
mutable struct FunctionHistory
    f_his        :: Dict{Int64, Float64}          ## record f(x_j)     
    G_max_his    :: Dict{Int64, Float64}          ## record max(g[k] for k in 1:m)(x_j)   
end


mutable struct CurrentInfo
    x            :: StateInfo                                       ## record x point
    f            :: Float64                                         ## record f(x_j)
    G            :: Dict{Int64, Float64} 
    df           :: Dict{Symbol, Dict{Int64, Any}}
    dG           :: Dict{Int64, StateInfo}                          ## actually is a matrix.  But we use dict to store it
end

mutable struct NormalizationCurrentInfo
    x            :: Tuple{StateInfo, Float64}                                       ## record x point
    f            :: Float64                                         ## record f(x_j)
    G            :: Dict{Int64, Float64} 
    df           :: Dict{Symbol, Any}
    dG           :: Dict{Int64, Tuple{Union{Nothing, StateInfo}, Union{Nothing, Float64}}}                          ## actually is a matrix.  But we use dict to store it
end


struct ModelInfo
    model :: Model
    xs    :: Any
    xy    :: Any
    xv    :: Any
    xw    :: Any
    sur   :: Any
    y     :: VariableRef
    z     :: VariableRef
end


## ====================================================================================== ##
## ================================== Lagrangian Cuts =================================== ##
## ====================================================================================== ##
"""
    mutable struct ParetoLagrangianCutGeneration  <: CutGeneration 
        core_point_strategy ::Symbol
        core_point          ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}}
        δ                   ::Float64
    end

Lagrangian Cut information.
Fields
------

- `core_point_strategy`:
- `core_point`:
- `δ`:
- `primal_bound`:

"""
abstract type CutGeneration end
mutable struct ParetoLagrangianCutGeneration{T} <: CutGeneration 
    core_point_strategy ::String
    core_point          ::Union{Nothing, StateInfo}
    δ                   ::Float64
    primal_bound        ::Union{Nothing, T}
end

mutable struct LagrangianCutGeneration{T} <: CutGeneration 
    primal_bound        ::Union{Nothing, T}
end

mutable struct SquareMinimizationCutGeneration{T} <: CutGeneration 
    δ                   ::Float64
    primal_bound        ::Union{Nothing, T}
end
mutable struct StrengthenedBendersCutGeneration{T} <: CutGeneration 

end
mutable struct LinearNormalizationLagrangianCutGeneration{T} <: CutGeneration 
    uₙ                  ::Union{Nothing, StateInfo}
    uₙ₀                 ::Union{Nothing, Float64}
    primal_bound        ::Union{Nothing, T}
end