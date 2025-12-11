using Pkg
Pkg.activate(".")
# using BenchmarkTools;
# BenchmarkTools.DEFAULT_PARAMETERS.samples = 1000;
# BenchmarkTools.DEFAULT_PARAMETERS.evals = 5;
using Distributed; addprocs(5); 
@everywhere begin
    using JuMP, Gurobi, PowerModels;
    using Statistics, StatsBase, Random, Dates, Distributions;
    using Distributed, ParallelDataTransfer;
    using CSV, DataFrames, Printf;
    using JLD2, FileIO;
    using Base.Filesystem: mkpath, dirname, isdir;

    const GRB_ENV = Gurobi.Env();

    project_root = @__DIR__;

    include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "utilities", "structs.jl"))
    include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "utilities", "auxiliary.jl"))
    include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "utilities", "level_method_regular_subproblem.jl"))
    include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "utilities", "level_method_normalized_subproblem.jl"))
    include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "utilities", "cut_variants.jl"))
    include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "utils.jl"))
    include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "forward_pass.jl"))
    include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "backward_pass.jl"))
    include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "partition_tree.jl"))
    include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "sddp.jl"))
    # include(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "utilities", "extForm.jl"))
end

#############################################################################################
###################################### Parameter Setup ######################################
#############################################################################################
case                = "case30"; # "case30pwl", "case30", "case30pwl",
algorithm           = :SDDPL; # :SDDPL, :SDDP, :SDDiP
cut                 = :LNC; # :PLC, :SMC, :LC, :SBC, :SBCLC, :SBCSMC, :SBCPLC, :LNC
numScenarios        = 500;
M                   = 1; # number of samples for cut generation
logger_save         = true;
med_method          = :IntervalMed; # :IntervalMed, :ExactPoint
ε                   = 1/2^8;
ℓ                   = 0.5; 
δ                   = 1e-2;
sparse_cut          = :sparse; # :sparse, :dense
tightness           = false;
branch_variable     = :ALL; # :ALL, :MFV
LiftIterThreshold   = 2;
num = 5; T = 12;

for algorithm in [:SDDPL, :SDDP, :SDDiP]
    for cut in [:PLC, :SMC, :LC, :SBC, :BC]
        for num in [5, 10]
            for T in [6, 8, 12] 
                param = param_setup(
                    terminate_time = 3600,
                    TimeLimit = 10,
                    ε = ε; # 1/32, 1/64, 1/128, 1/256
                    MIPGap = 1e-4,
                    numScenarios = numScenarios,
                    M = M,
                    tightness = tightness,
                    LiftIterThreshold = LiftIterThreshold,
                    cutSelection = cut, 
                    algorithm = algorithm,
                    terminate_threshold = 1e-3,
                    branch_threshold = 1e-6,
                    branch_variable = branch_variable, # :ALL, :MFV
                    sparse_cut = sparse_cut, # true, false
                    T = T,
                    num = num,
                    case = case,
                    med_method = med_method, 
                    logger_save = logger_save
                );
                param_cut = param_cut_setup(
                    core_point_strategy = "Eps", # "Mid", "Eps"
                    δ = δ,
                    ℓ = ℓ,
                );
                param_levelsetmethod = param_levelsetmethod_setup(
                    μ = 0.9,
                    λ = 0.5,
                    threshold = 1e-4,
                    nxt_bound = 1e10,
                    MaxIter = 200,
                    verbose = false
                );
                indexSets        = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "indexSets.jld2"))["indexSets"];
                paramOPF         = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "paramOPF.jld2"))["paramOPF"];
                paramDemand      = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "paramDemand.jld2"))["paramDemand"];
                scenarioTree     = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "scenarioTree.jld2"))["scenarioTree"];
                initialStateInfo = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "initialStateInfo.jld2"))["initialStateInfo"];
                # Ξ = load("src/alg/experiment_$case/stage($T)real($num)/Ξ.jld2")["Ξ"];
                # @time extResult = extensive_form(
                #     indexSets = indexSets, 
                #     paramDemand = paramDemand, 
                #     paramOPF = paramOPF, 
                #     scenarioTree = scenarioTree, 
                #     Ξ = Ξ, 
                #     silent = false, 
                #     initialStageDecision = initialStageDecision
                #     ); 
                # OPT = extResult.OPT;

                
                @everywhere begin
                    indexSets = $indexSets; paramOPF = $paramOPF; paramDemand = $paramDemand; scenarioTree = $scenarioTree; initialStateInfo = $initialStateInfo;
                    param_cut = $param_cut; param_levelsetmethod = $param_levelsetmethod; param = $param;
                end

                sddpResults = stochastic_dual_dynamic_programming_algorithm(
                        scenarioTree,                   
                        indexSets,                        
                        paramDemand,
                        paramOPF;
                        initialStateInfo = initialStateInfo,
                        param_cut = param_cut, 
                        param_levelsetmethod = param_levelsetmethod, 
                        param = param
                );
                @everywhere GC.gc(); # garbage collection
            end
        end
    end
end

for ε in [1/32, 1/128, 1/256]
    param = param_setup(
        terminate_time = 7200,
        TimeLimit = 10,
        ε = ε; # 1/32, 1/64, 1/128, 1/256
        numScenarios = 500,
        LiftIterThreshold = LiftIterThreshold,
        cutSelection = cut, 
        algorithm = algorithm,
        terminate_threshold = 1e-3,
        branch_threshold = 1e-6,
        branch_variable = branch_variable,
        T = T,
        num = num,
        case = case,
        med_method = med_method, 
        logger_save = logger_save
    );
    param_cut = param_cut_setup(
        core_point_strategy = "Eps", # "Mid", "Eps"
        δ = 1e2,
        ℓ = .0,
    );
    param_levelsetmethod = param_levelsetmethod_setup(
        μ = 0.9,
        λ = 0.5,
        threshold = 1e-4,
        nxt_bound = 1e10,
        MaxIter = 200,
        verbose = false
    );
    indexSets        = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "indexSets.jld2"))["indexSets"];
    paramOPF         = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "paramOPF.jld2"))["paramOPF"];
    paramDemand      = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "paramDemand.jld2"))["paramDemand"];
    scenarioTree     = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "scenarioTree.jld2"))["scenarioTree"];
    initialStateInfo = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "initialStateInfo.jld2"))["initialStateInfo"];
    # Ξ = load("src/alg/experiment_$case/stage($T)real($num)/Ξ.jld2")["Ξ"];
    # @time extResult = extensive_form(
    #     indexSets = indexSets, 
    #     paramDemand = paramDemand, 
    #     paramOPF = paramOPF, 
    #     scenarioTree = scenarioTree, 
    #     Ξ = Ξ, 
    #     silent = false, 
    #     initialStageDecision = initialStageDecision
    #     ); 
    # OPT = extResult.OPT;

    
    @everywhere begin
        indexSets = $indexSets; paramOPF = $paramOPF; paramDemand = $paramDemand; scenarioTree = $scenarioTree; initialStateInfo = $initialStateInfo;
        param_cut = $param_cut; param_levelsetmethod = $param_levelsetmethod; param = $param;
    end


    sddpResults = stochastic_dual_dynamic_programming_algorithm(
            scenarioTree,                   
            indexSets,                        
            paramDemand,
            paramOPF;
            initialStateInfo = initialStateInfo,
            param_cut = param_cut, 
            param_levelsetmethod = param_levelsetmethod, 
            param = param
    );
    @everywhere GC.gc(); # garbage collection
end


case = "case30"; # "case_RTS_GMLC", "case30", "case30pwl",
algorithm = :SDDPL; 
cut = :PLC; 
num = 5; T = 8;
logger_save = true;
med_method = :IntervalMed; # :IntervalMed, :ExactPoint
ε = 1/2^8;
ℓ = .0; δ = 1e-2;
branch_variable = :ALL; # :ALL, :MFV
LiftIterThreshold = 15;
tightness = false;
for ℓ in [0.2, 0.4, 0.6, 0.8]
    param = param_setup(
        terminate_time = 3600,
        TimeLimit = 10,
        ε = ε; # 1/32, 1/64, 1/128, 1/256
        MIPGap = 1e-4,
        numScenarios = 500,
        tightness = tightness,
        LiftIterThreshold = LiftIterThreshold,
        cutSelection = cut, 
        algorithm = algorithm,
        terminate_threshold = 1e-3,
        branch_threshold = 1e-6,
        branch_variable = branch_variable, # :ALL, :MFV
        T = T,
        num = num,
        case = case,
        med_method = med_method, 
        logger_save = logger_save
    );

    param_cut = param_cut_setup(
        core_point_strategy = "Eps", # "Mid", "Eps"
        δ = 1e2,
        ℓ = ℓ,
    );
    param_levelsetmethod = param_levelsetmethod_setup(
        μ = 0.9,
        λ = 0.5,
        threshold = 1e-4,
        nxt_bound = 1e10,
        MaxIter = 200,
        verbose = false
    );
    indexSets        = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "indexSets.jld2"))["indexSets"];
    paramOPF         = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "paramOPF.jld2"))["paramOPF"];
    paramDemand      = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "paramDemand.jld2"))["paramDemand"];
    scenarioTree     = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "stage($T)real($num)", "scenarioTree.jld2"))["scenarioTree"];
    initialStateInfo = load(joinpath(project_root, "src", "multistage_stochastic_unit_commitment", "experiment_$case", "initialStateInfo.jld2"))["initialStateInfo"];

    @everywhere begin
        indexSets = $indexSets; paramOPF = $paramOPF; paramDemand = $paramDemand; scenarioTree = $scenarioTree; initialStateInfo = $initialStateInfo;
        param_cut = $param_cut; param_levelsetmethod = $param_levelsetmethod; param = $param;
    end


    sddpResults = stochastic_dual_dynamic_programming_algorithm(
            scenarioTree,                   
            indexSets,                        
            paramDemand,
            paramOPF;
            initialStateInfo = initialStateInfo,
            param_cut = param_cut, 
            param_levelsetmethod = param_levelsetmethod, 
            param = param
    );
    @everywhere GC.gc(); # garbage collection
end