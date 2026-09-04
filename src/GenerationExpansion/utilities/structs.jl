"""
Container for all algorithmic parameters used in SDDP / SDDiP.

Fields:
- timeSDDP         : time limit (in seconds)
- gapSDDP          : relative optimality gap tolerance
- iterSDDP         : maximum number of iterations
- levelMethodMaxIter : maximum number of inner level-method iterations used
  during cut generation
- sample_size_SDDP : number of sampled scenarios per iteration
- ε                : step size / risk parameter (depending on context)
- discreteZ        : whether to use a discrete approximation for Z
- cutType          : type of cuts (e.g., :SMC, :MMC, :LNC, etc.)
- cutSparsity      : whether to enforce sparse cuts
- T                : number of stages
- num              : number of random realizations per stage (data-related)
- verbose          : if true, print detailed logs
- ℓ1, ℓ2           : regularization weights (if used in the algorithm)
- nxt_bound        : lower bound for the next subproblem (e.g., Δ model)
- logger_save      : whether to save logs / results to disk
- algorithm        : symbol describing the algorithm variant (e.g. :SDDiP)
- corePointStrategy : core point rule used by PLC/LNC (`:Mid`, `:Eps`, `:Conv`)
- corePointWeight   : convex weight used by the `:Conv` rule
- corePointEpsilon  : relative interior margin used by the `:Eps` rule
- lncMinScale       : minimum accepted value of `-π₀` for LNC cuts
- lncCoreThetaMargin: relative fallback margin for the LNC core epigraph value
- cutDiagnostics    : whether to print cut-generation diagnostics
"""
mutable struct SDDPParam
    timeSDDP         :: Float64
    gapSDDP          :: Float64
    iterSDDP         :: Int
    levelMethodMaxIter :: Int
    solverGap        :: Float64
    solverTime       :: Float64
    sample_size_SDDP :: Int
    ε                :: Float64
    discreteZ        :: Bool
    cutType          :: Symbol
    cutSparsity      :: Bool
    partitionRule    :: Symbol
    branchingStart   :: Int
    M                :: Int
    T                :: Int
    num              :: Int
    verbose          :: Bool
    ℓ1               :: Float64
    ℓ2               :: Float64
    nxt_bound        :: Float64
    logger_save      :: Bool
    algorithm        :: Symbol
    corePointStrategy :: Symbol
    corePointWeight   :: Float64
    corePointEpsilon  :: Float64
    lncMinScale       :: Float64
    lncCoreThetaMargin :: Float64
    cutDiagnostics    :: Bool
end

"""
Cut coefficients.

- `v[i][k]`: constant term for iteration i, scenario k 
- `πₙ[i][k]`: x coefficient term for iteration i, scenario k
- `πₙ₀[i][k]`: θ coefficient term for iteration i, scenario k
"""
struct CutCoefficient
    vₙ               ::Dict{Int64,Dict{Int64, Float64}} 
    πₙ               ::Dict{Int64,Dict{Int64, Vector{Float64}}}  
    πₙ₀              ::Dict{Int64,Dict{Int64, Float64}} 
end

"""
static data for each stage 
"""
struct StageData 
    c1       ::Vector{Float64}
    c2       ::Vector{Float64}
    ū        ::Vector{Float64}
    h        ::Float64
    N        ::Matrix{Float64}
    s₀       ::Vector{Float64}
    penalty  ::Float64
end

"""
Random Variables
"""
struct RandomVariables
    d   ::Vector{Float64}
end

"""
Extensive form JuMP Model
"""
struct GurobiModelInfo
    model           :: Model
    x               :: Array{VariableRef} 
    y               :: Array{VariableRef}
    slack           :: Matrix{VariableRef}
    num_Ω           :: Int64
end

########################### Level-set method data ################################
"""
Store the level-method history at a bundle point, namely `f(x_j)` and
`max_k g_k(x_j)`.
"""
mutable struct FunctionHistory
    obj_history        :: Dict{Int64, Float64}
    max_con_history    :: Dict{Int64, Float64}
end

"""
    mutable struct StageInfo
        StateValue          ::Union{Nothing, Float64}
        StageValue          ::Union{Nothing, Float64}
        IntVar              ::Union{Nothing, Vector}
        IntVarLeaf          ::Union{Nothing, Dict}
        IntVarBinaries      ::Union{Nothing, Dict}
    end

Record the state variables of the problem.

Fields
------
- `StateValue`:
    the objective value of the state t, i.e., stage cost + future cost
- `StageValue`:
    the current stage cost
- `IntVar`: 
    the general integer state variables of the problem, the first `Any` is the variable name, the second `Any` is the index.
- `IntVarLeaf`:
    the partition tree info of Int variables of the problem:
    the first `Any` is the variable name, the second `Any` is the index, 
    the third `Any` is the leaf node index,
    `Symbol ∈ {:lb, :ub, :parent, :sibling, :var}` is the key to record the information.
- `IntVarBinaries`:
    the binarization variables of the integer state variables of the problem, the first `Any` is the variable name, the second `Any` is the index of the variable, and the third `Any` is the index of the binarization variables.
"""
mutable struct StageInfo
    StateValue          ::Union{Nothing, Float64}
    StageValue          ::Union{Nothing, Float64}
    IntVar              ::Union{Nothing, Vector}
    IntVarLeaf          ::Union{Nothing, Dict}
    IntVarBinaries      ::Union{Nothing, Vector}
end

"""
Dual iterate information for ReLU-based cut generation in the SDDP benchmark.

Fields
------
- `StateValue`:
    coefficient of the epigraph variable `θ` in the normalized ReLU cut.
    It is `nothing` for the non-normalized ReLU cut.
- `IntVarPlus`:
    coefficients associated with positive state deviations.
- `IntVarMinus`:
    coefficients associated with negative state deviations.
- `IntVarLeaf`:
    linear lifted-state coefficients used only by SDDP-L.
"""
mutable struct ReLUDualStageInfo
    StateValue          ::Union{Nothing, Float64}
    IntVarPlus          ::Union{Nothing, Vector}
    IntVarMinus         ::Union{Nothing, Vector}
    IntVarLeaf          ::Union{Nothing, Dict}
end

ReLUDualStageInfo(
    StateValue::Union{Nothing, Float64},
    IntVarPlus::Union{Nothing, Vector},
    IntVarMinus::Union{Nothing, Vector},
) = ReLUDualStageInfo(StateValue, IntVarPlus, IntVarMinus, nothing)

"""
Current iteration information:
- var:   decision variable
- obj:   objective function f(x)
- con:   constraint functions {gₖ : k = 1,...,m} and let G = maxₖ gₖ
- d_obj: the derivative of f(x)
- d_con: the derivative of gₖ(k)
"""
mutable struct CurrentInfo
    var             :: StageInfo                  
    obj             :: Float64                                                                      
    con             :: Dict{Int64, Float64}                                                         
    d_obj           :: Dict{Symbol, Any}                                                            
    d_con           :: Dict{Int, Any}      
end

"""
Current iteration information for ReLU-based cut generation.

Fields
------
- `var`:
    current ReLU-dual iterate.
- `obj`:
    oracle objective value used by the level-set method.
- `con`:
    linearized feasibility constraints.
- `d_obj`:
    subgradients of the oracle objective with respect to
    `(:plus, :minus, :pi0)`.
- `d_con`:
    subgradients of the feasibility constraints.
"""
mutable struct ReLUCurrentInfo
    var             :: ReLUDualStageInfo
    obj             :: Float64
    con             :: Dict{Int64, Float64}
    d_obj           :: Dict{Symbol, Any}
    d_con           :: Dict{Int, Any}
end
"""
Binary expansion information: x = A * L, L ∈ {0, 1}ⁿ。
- A: binary expansion matrhix
- n: dim for L 
- d: generator type
"""
struct BinaryInfo
    A     ::Matrix{Int64}
    n     ::Int64
    d     ::Int64
end

## ====================================================================================== ##
## ============ Stochastic Dual Dynamic Programming with Lifting Algorithm ============== ##
## ====================================================================================== ##
abstract type SequentialModels end

abstract type SolutionMethod end
"""
    mutable struct StageModel  <: ForwardModel 
        model           ::Model
        IntVar          ::Dict{Any, Dict{Any, Dict{Symbol, Any}}}
        IntVarLeaf      ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}}
        IntVarBinaries  ::Union{Nothing, Dict{Any, Dict{Any, Dict{Any, VariableRef}}}}
    end

Forward nodal problem.

Fields
------

- `model`:
    the forward model is used to solve each sub-problem.
- `IntVar`: 
    the general integer state variables of the problem, the first `Any` is the variable name, the second `Any` is the index.
- `IntVarLeaf`:
    the partition tree info of Int variables of the problem:
    the first `Any` is the variable name, the second `Any` is the index, 
    the third `Any` is the leaf node index,
    `Symbol ∈ {:lb, :ub, :parent, :sibling, :var}` is the key to record the information.
- `IntVarBinaries`:
    the binarization variables of the integer state variables of the problem, the first `Any` is the variable name, the second `Any` is the index.
"""
mutable struct StageModel  <: SequentialModels 
    model           ::Model
    IntVar          ::Union{Nothing, Dict{Any, Dict{Any, VariableRef}}}
    IntVarLeaf      ::Union{Nothing, Dict{Symbol, Dict{Int64, Dict{Int64, Dict{Symbol, Any}}}}}
    IntVarBinaries  ::Union{Nothing, Dict{Any, Vector{VariableRef}}}
end

## ====================================================================================== ##
## ================================== Lagrangian Cuts =================================== ##
## ====================================================================================== ##
mutable struct CutGenerationParamInfo
    μ           :: Float64
    λ           :: Union{Float64, Nothing}
    gapBM       :: Float64
    iterBM      :: Int
    nxt_bound   :: Float64
    verbose     :: Bool
    stateInfo   :: Union{StageInfo, Nothing}
    cutSelection:: Symbol
    πₙ          :: Union{StageInfo, ReLUDualStageInfo}
end

abstract type CutGenerationProgram end

mutable struct LagrangianCutGenerationProgram{T} <: CutGenerationProgram 
    primal_bound        ::Union{Nothing, T}
end

"""
    mutable struct ParetoLagrangianCutGenerationProgram  <: CutGenerationProgram 
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
mutable struct ParetoLagrangianCutGenerationProgram{T} <: CutGenerationProgram
    CoreState           :: Union{StageInfo, Nothing}
    primal_bound        ::Union{Nothing, T}
    δ                   ::Float64
end

mutable struct SquareMinimizationCutGenerationProgram{T} <: CutGenerationProgram 
    δ                   ::Float64
    primal_bound        ::Union{Nothing, T}
end
mutable struct StrengthenedBendersCutGenerationProgram{T} <: CutGenerationProgram 
    πₙ                  :: StageInfo
end
mutable struct LinearNormalizationLagrangianCutGenerationProgram{T} <: CutGenerationProgram 
    CoreState           ::Union{StageInfo, Nothing}
    primal_bound        ::Union{Nothing, T}
    theta_anchor        ::Union{Nothing, T}
    min_scale           ::Float64
    core_theta_fallback ::Bool
end

mutable struct ReLULagrangianCutGenerationProgram{T} <: CutGenerationProgram
    primal_bound        ::Union{Nothing, T}
end

mutable struct NormalizedReLULagrangianCutGenerationProgram{T} <: CutGenerationProgram
    NormalizationInfo   ::Union{ReLUDualStageInfo, Nothing}
    # Incumbent epigraph value θ̂_n appearing in the normalized dual objective.
    incumbent_theta     ::Union{Nothing, T}
    # Incumbent node primal value Q_n(x̂), retained separately for scaling and
    # validity checks.
    primal_bound        ::Union{Nothing, T}
end

"""
ReLU cut coefficients returned by the backward pass.

Fields
------
- `rhs`:
    right-hand side value `𝓛(π)` of the ReLU cut.
- `dualInfo`:
    ReLU-dual coefficients.
"""
mutable struct ReLUCutInfo{T}
    rhs                 ::T
    dualInfo            ::ReLUDualStageInfo
end
