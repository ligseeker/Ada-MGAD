# Agent entry point

Before inspecting or changing this repository, read the canonical current
research context:

- [`docs/GAIA_P5_CURRENT_CONTEXT.md`](docs/GAIA_P5_CURRENT_CONTEXT.md)

That file is the single source for the active GAIA P5 experiment, artifact
locations, current results, protocol constraints, and known limitations. The
longer GAIA audit/runbook documents linked from it contain historical decisions
and implementation detail; do not assume that an old status section is current.

Do not overwrite an existing experiment directory or shared preprocessing
artifact. Use a new run directory for a new experiment, keep all preprocessing
decisions Train-only, and do not start a full preprocessing/training/evaluation
run unless the user explicitly asks for it.
