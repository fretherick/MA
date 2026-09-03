# SIGSOM

Code for experiments with signature features, self-organising maps, hidden
Markov models, and jump models.

## Repository layout

```text
SIGSOM/                 SIGSOM implementation
other_models/           HMM helpers and comparison models
jumpmodels/             Jump-model implementation and attribution
src/                    Reproducible experiment entry points
Notebooks (examples)/   Exploratory notebooks and paper analyses
helpers/                Experiment utilities
df_utils/               DataFrame utilities
params.yaml             Experiment parameters
dvc.yaml                DVC experiment pipeline
data/                   Generated data (ignored by Git)
artifacts/              Generated model outputs (ignored by Git)
```

## Quick start

The checked-in `environment.yaml` contains the current Conda environment.
Create it with:

```bash
conda env create -f environment.yaml
conda activate SIGSOM_v1
```

Run the simulation pipeline from the repository root:

```bash
python -m src.simulation_experiments
```

To reproduce the DVC stage:

```bash
dvc repro
```

Generated data, artifacts, metrics, caches, and local serialized data are
ignored by Git. Keep large reproducibility inputs in a configured DVC remote
instead of committing them to the source repository.

## Notebooks

The notebooks in `Notebooks (examples)/` contain exploratory analyses and
paper-specific workflows. Scratch files under its `tmp/` directory are local
working files and should not be published.

## Attribution

The `jumpmodels/` directory contains code based on the
[jump-models project](https://github.com/Yizhan-Oliver-Shu/jump-models).
Its original license and attribution files must remain with that code.

## Before publishing

- Replace the paper citation and author details with the final information.
- Add a repository license at the root.
- Configure a DVC remote if generated data must be reproducible by others.
- Review downloaded datasets and serialized files for size, licensing, and
  redistribution permissions.
