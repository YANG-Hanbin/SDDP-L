using JLD2

const RESULTS_ROOT = abspath(joinpath(@__DIR__, "..", "new_logger"))

function relu_lifted_leaf_dot(
    dualInfo::ReLUDualStageInfo,
    stateInfo::StageInfo,
)::Float64
    if dualInfo.IntVarLeaf === nothing || stateInfo.IntVarLeaf === nothing
        return 0.0
    end

    return sum(
        sum(
            dualInfo.IntVarLeaf[g][k] * stateInfo.IntVarLeaf[g][k]
            for k in keys(dualInfo.IntVarLeaf[g]);
            init = 0.0,
        )
        for g in keys(dualInfo.IntVarLeaf);
        init = 0.0,
    )
end

function relu_lifted_leaf_dot(
    left::ReLUDualStageInfo,
    right::ReLUDualStageInfo,
)::Float64
    if left.IntVarLeaf === nothing || right.IntVarLeaf === nothing
        return 0.0
    end

    return sum(
        sum(
            left.IntVarLeaf[g][k] * right.IntVarLeaf[g][k]
            for k in keys(left.IntVarLeaf[g]);
            init = 0.0,
        )
        for g in keys(left.IntVarLeaf);
        init = 0.0,
    )
end

function relu_cut_rhs(
    oracle_value::Real,
    dualInfo::ReLUDualStageInfo,
    stateInfo::StageInfo,
)::Float64
    return Float64(oracle_value) - relu_lifted_leaf_dot(dualInfo, stateInfo)
end

function relu_lifted_leaf_lagrangian_term(
    dualInfo::ReLUDualStageInfo,
    model::Model,
    stateInfo::StageInfo,
)
    if dualInfo.IntVarLeaf === nothing || stateInfo.IntVarLeaf === nothing
        return 0.0
    end

    return sum(
        sum(
            dualInfo.IntVarLeaf[g][k] *
            (stateInfo.IntVarLeaf[g][k] - model[:region_indicator_copy][g][k])
            for k in keys(dualInfo.IntVarLeaf[g]);
            init = 0.0,
        )
        for g in keys(dualInfo.IntVarLeaf);
        init = 0.0,
    )
end

function relu_lifted_leaf_value_gradient(
    model::Model,
    stateInfo::StageInfo,
)
    if stateInfo.IntVarLeaf === nothing
        return nothing
    end

    return Dict(
        g => Dict(
            k => JuMP.value(model[:region_indicator_copy][g][k]) -
                 stateInfo.IntVarLeaf[g][k]
            for k in keys(stateInfo.IntVarLeaf[g])
        )
        for g in keys(stateInfo.IntVarLeaf)
    )
end

"""
Return a zero lifted-leaf block with the same support as `stateInfo`.
"""
function zero_relu_lifted_leaf_state(stateInfo::StageInfo)
    if stateInfo.IntVarLeaf === nothing
        return nothing
    end

    return Dict(
        g => Dict(k => 0.0 for k in keys(stateInfo.IntVarLeaf[g]))
        for g in keys(stateInfo.IntVarLeaf)
    )
end

zero_relu_lifted_leaf_gradient(stateInfo::StageInfo) =
    zero_relu_lifted_leaf_state(stateInfo)

"""
    build_results_path(param; root = RESULTS_ROOT, run_id = nothing) -> String

Build the JLD2 results path for Generation Expansion experiments.

Directory layout:
    {root}/
        case={case}/
            alg={algorithm}/
                T={T}/
                    Real={num}/
                        cut=...__core=...__M=...__sparsity=...__discZ=....jld2
"""
function build_results_path(
    param;
    root::AbstractString = RESULTS_ROOT,
    run_id = nothing,
)::String
    # ---------- basic fields ----------
    algorithm = getproperty(param, :algorithm)
    T         = getproperty(param, :T)
    num       = getproperty(param, :num)

    # Use the case name when available; otherwise fall back to a generic label.
    case = if Base.hasproperty(param, :case)
        getproperty(param, :case)
    else
        "GenerationExpansion"
    end

    # ---------- directory hierarchy ----------
    dir = joinpath(
        root,
        "case=$(case)",
        "alg=$(algorithm)",
        "T=$(T)",
        "Real=$(num)",
    )

    # ---------- filename tags ----------
    tags = String[]

    # Cut family tag, e.g., SMC / PLC / LC / ReLUC.
    if Base.hasproperty(param, :cutType)
        cutType = getproperty(param, :cutType)
        push!(tags, "cut=$(cutType)")
    end

    # partitionRule
    if algorithm == :SDDPL
        if Base.hasproperty(param, :partitionRule)
            partitionRule = getproperty(param, :partitionRule)
            push!(tags, "partitionRule=$(partitionRule)")
        end
    end

    if Base.hasproperty(param, :corePointStrategy)
        strategy = getproperty(param, :corePointStrategy)
        push!(tags, "core=$(strategy)")

        if strategy == :Conv && Base.hasproperty(param, :corePointWeight)
            push!(tags, "cpw=$(getproperty(param, :corePointWeight))")
        elseif strategy == :Eps && Base.hasproperty(param, :corePointEpsilon)
            push!(tags, "cpeps=$(getproperty(param, :corePointEpsilon))")
        end
    end

    # Optional norm parameters can be re-enabled here if they are needed in the
    # saved filename convention.
    # if Base.hasproperty(param, :ℓ1)
    #     ℓ1 = getproperty(param, :ℓ1)
    #     push!(tags, "ell1=$(ℓ1)")
    # end

    if Base.hasproperty(param, :M)
        M = getproperty(param, :M)
        push!(tags, "M=$(M)")
    end

    # Sparsity tag.
    if Base.hasproperty(param, :cutSparsity)
        sparse_cut = getproperty(param, :cutSparsity)
        push!(tags, "sparsity=$(sparse_cut)")
    elseif Base.hasproperty(param, :sparse_cut)
        sparse_cut = getproperty(param, :sparse_cut)
        push!(tags, "sparsity=$(sparse_cut)")
    end

    # Discrete/continuous `z` flag.
    if Base.hasproperty(param, :discreteZ)
        discZ = getproperty(param, :discreteZ)
        push!(tags, "discZ=$(discZ)")
    end

    if run_id !== nothing
        push!(tags, "run=$(run_id)")
    end

    filename = join(tags, "__") * ".jld2"

    return joinpath(dir, filename)
end

"""
    save_results_info(param, sddpResults)

Save SDDP/SDDiP results to a JLD2 file, using the
GenerationExpansion-style path-building logic.
"""
function save_results_info(
    param::SDDPParam,
    sddpResults::Dict,
)::Nothing
    # Respect `logger_save` when the parameter object exposes it.
    if Base.hasproperty(param, :logger_save)
        getproperty(param, :logger_save) || return nothing
    end

    # Allow the caller to override the results root and run identifier.
    root = if Base.hasproperty(param, :results_root)
        getproperty(param, :results_root)
    else
        RESULTS_ROOT
    end

    run_id = Base.hasproperty(param, :run_id) ? getproperty(param, :run_id) : nothing

    filepath = build_results_path(
        param;
        root   = root,
        run_id = run_id,
    )

    dir = dirname(filepath)
    isdir(dir) || mkpath(dir)

    @save filepath sddpResults

    return nothing
end

"""
    get_cut_selection(cutSelection::Symbol, i::Int)

Determine the actual cut family used at iteration `i` when the experiment
schedule switches from a warm-start family (typically `:SBC`) to another cut.
"""
function get_cutType(
    cutType::Symbol, 
    i::Int;
    threshold::Int = 3)
    if cutType == :SBCLC
        return i <= threshold ? :SBC : :LC
    elseif cutType == :SBCSMC
        return i <= threshold ? :SBC : :SMC
    elseif cutType == :SBCPLC
        return i <= threshold ? :SBC : :PLC
    elseif cutType == :SBCLNC
        return i <= threshold ? :SBC : :LNC
    elseif cutType == :SBCReLUC
        return i <= threshold ? :SBC : :ReLUC
    elseif cutType == :SBCNormalizedReLUC
        return i <= threshold ? :SBC : :NormalizedReLUC
    else
        return cutType
    end
end
