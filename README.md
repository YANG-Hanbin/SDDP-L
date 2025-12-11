# Globally Converging Algorithm for Multistage Stochastic Mixed-Integer Programs via Enhanced Lagrangian Cuts




## Description

The goal of this software is to demonstrate different algorithms for identifying socially optimal Nash equilibria in bimatrix games.

This repository is made up of the following directories and subdirectories

1. `src`: Julia implementations of two test cases on different algorithms, as well as other necessary utility scripts
    1. `GenerationExpansion`: Julia implementations of the generation expansion planning (GEP) problem
    2. `multistage_stochastic_unit_commitment`: Julia implementations of the multistage stochastic unit commitment (MSUC) problem

## Building

### Requirements

1. Julia v1.11
2. Gurobi v11.1

### Solver Remarks

We implemented our proposed algorithms in `src/GenerationExpansion/test/runTest.jl` for GEP and `src/multistage_stochastic_unit_commitment/test/runTest.jl` for MSUC. 
The extensive form is given in `src/GenerationExpansion/utilities/extFormGurobi.jl` and `src/multistage_stochastic_unit_commitment/utilities/extForm.jl`

The parameters used for all tests include

1. `algorithm:` : SDDPL
2. `cut:`:LC, :LNC, :PLC, :SMC
3. `med_method`: the approach to branch continuous variables, bisection or incumbent solution

