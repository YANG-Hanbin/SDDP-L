using Dates
using JLD2

const RESULTS_ROOT = joinpath(@__DIR__, "..", "new_logger")  # 如果外面已有，可以删掉这一行

"""
    build_results_path(param; root = RESULTS_ROOT, run_id = nothing) -> String

Build the JLD2 results path for Generation Expansion experiments.

Directory layout:
    {root}/
        case={case}/
            alg={algorithm}/
                T={T}/
                    Real={num}/
                        cut=...__eps=...__ell1=...__ell2=...__sparsity=...__discZ=...__run=....jld2
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

    # case: 如果 param 里有 case 就用，没有就默认 "GenerationExpansion"
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

    # cut 类型：SMC / PLC / LC 等
    if Base.hasproperty(param, :cutType)
        cutType = getproperty(param, :cutType)
        push!(tags, "cut=$(cutType)")
    end

    # # ε
    # if Base.hasproperty(param, :ε)
    #     ε = getproperty(param, :ε)
    #     # 比如 eps=8 表示 ε = 1/8
    #     eps_int = Int(round(1 / ε))
    #     push!(tags, "eps=$(eps_int)")
    # end

    # # ℓ1 / ℓ2（如果你只用其中一个，也可以只留一个）
    # if Base.hasproperty(param, :ℓ1)
    #     ℓ1 = getproperty(param, :ℓ1)
    #     push!(tags, "ell1=$(ℓ1)")
    # end

    if Base.hasproperty(param, :M)
        M = getproperty(param, :M)
        push!(tags, "M=$(M)")
    end

    # 稀疏 cut 标记
    if Base.hasproperty(param, :cutSparsity)
        sparse_cut = getproperty(param, :cutSparsity)
        push!(tags, "sparsity=$(sparse_cut)")
    elseif Base.hasproperty(param, :sparse_cut)
        sparse_cut = getproperty(param, :sparse_cut)
        push!(tags, "sparsity=$(sparse_cut)")
    end

    # binary z / continuous z 信息
    if Base.hasproperty(param, :discreteZ)
        discZ = getproperty(param, :discreteZ)
        push!(tags, "discZ=$(discZ)")
    end

    # run_id：外面没给就用日期
    if run_id === nothing
        run_id = Dates.format(now(), "yyyymmdd")
    end
    push!(tags, "run=$(run_id)")

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
    # respect logger_save if present
    if Base.hasproperty(param, :logger_save)
        getproperty(param, :logger_save) || return nothing
    end

    # 外部可以在 param 里给 results_root / run_id（可选）
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

    Base on cutSelection and iteration index i, determine the actual cut selection strategy to use.
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
    else
        return cutType
    end
end