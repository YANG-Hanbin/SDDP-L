# SDDP-L

This repository provides the Julia implementation used in the computational
study of globally convergent algorithms for multistage stochastic mixed-integer
programs via enhanced Lagrangian cuts.

The implementation contains three algorithmic frameworks:

- stochastic dual dynamic programming (`SDDP`),
- stochastic dual dynamic integer programming (`SDDiP`), and
- the lifted-state method (`SDDP-L`).

It supports Lagrangian (`LC`), Pareto Lagrangian (`PLC`), square-minimization
(`SMC`), linear-normalization (`LNC`), strengthened Benders (`SBC`), ReLU
(`ReLUC`), and normalized ReLU (`NormalizedReLUC`) cuts. ReLU-based cuts are
implemented for `SDDP` and `SDDP-L`. Adaptive-level variants of the PLC and
SMC cut-generation problems are available as `AdaptivePLC` and `AdaptiveSMC`.

## Repository layout

```text
src/
|-- GenerationExpansion/
|   |-- data/                 # benchmark model definitions
|   |-- numerical_data/       # benchmark instances used by the loaders
|   |-- test/                 # experiment entry points
|   |-- utilities/            # data structures and helper routines
|   `-- *.jl                  # forward/backward passes and cut generation
`-- multistage_stochastic_unit_commitment/
    |-- data/                 # benchmark source data
    |-- experiment_case30/    # benchmark instances used by the loaders
    |-- test/                 # experiment entry points
    |-- utilities/            # data structures and helper routines
    `-- *.jl                  # forward/backward passes and cut generation
experiments/
`-- normalized_relu/          # normalized-ReLU batch drivers
```

## Requirements

- Julia 1.11
- Gurobi 11.x or another release supported by the installed Gurobi.jl package
- A working Gurobi license

The checked-in `Manifest.toml` records the package environment used for the
experiments. Install it from the repository root with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The examples below use the Juliaup channel syntax. If the active `julia`
executable is already Julia 1.11, omit `+1.11`.

## Running experiments

Run commands from the repository root. The primary entry points are:

```bash
# Generation expansion planning
julia +1.11 --project=. src/GenerationExpansion/test/runTest.jl

# Multistage stochastic unit commitment
julia +1.11 --project=. \
  src/multistage_stochastic_unit_commitment/test/runTest.jl
```

Additional experiment drivers are available in each `test/` directory:

- `adaptiveCutTest.jl` runs compact adaptive-level PLC/SMC regressions;
- `cutTest.jl` compares cut families;
- `corepointTest.jl` runs the core-point sensitivity study;
- `partitionTest.jl` compares partition rules;
- `sparsityTest.jl` compares sparse and dense cuts; and
- `timeTest.jl` records runtime components.

Run the adaptive-level regressions for the two applications with:

```bash
julia +1.11 --project=. src/GenerationExpansion/test/adaptiveCutTest.jl
julia +1.11 --project=. \
  src/multistage_stochastic_unit_commitment/test/adaptiveCutTest.jl
```

Both regression drivers use two SDDP-L iterations and disable result-file
output by default.

The normalized-ReLU batch for both applications can be launched with:

```bash
experiments/normalized_relu/run_all.sh
```

The batch script uses the Juliaup `+1.11` channel by default. Set
`JULIA_VERSION=` to use the active Julia executable directly, and set
`JULIA_BIN` if Julia is not available as `julia` on `PATH`.

The supplied drivers provide the configurations used in the computational
study and can be long-running. Parameters can be changed in their
configuration blocks, or
experiments can be launched programmatically through
`run_generation_expansion_experiments` and `run_experiment_grid` in the
corresponding `loadMod.jl` files. Each loader starts five Julia worker
processes.

Generated solver output is written beneath `src/results/` or the application
result directories. These directories are excluded from version control.

## Benchmark data

The repository includes the serialized benchmark instances required by the
experiment loaders. The MATPOWER case file retains its upstream attribution
and source notes in the file header. Third-party benchmark data remains subject
to its original terms and attribution.

## License

The implementation is released under the MIT License; see `LICENSE`.
