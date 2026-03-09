# NN-GA (BRKGA for CartPole)

Train a neural-network policy for `CartPole-v1` using a Biased Random-Key Genetic Algorithm (BRKGA), then automatically save artifacts (agents, plots, videos, and network visualizations) per run.

## What this project does

- Evolves policy parameters with BRKGA.
- Evaluates individuals with Gymnasium rollouts.
- Saves best and progress agent bundles as `.npz`.
- Records gameplay videos from saved genomes.
- Renders neural-network diagrams for saved agents.

## Project layout

- `run.py`: main pipeline (train + save + record + visualize).
- `config.py`: all experiment settings.
- `algorithm.py`: BRKGA evolution loop and run directory setup.
- `fitness_function.py`: policy model, genome mapping, fitness rollout.
- `agent_store.py`: save/load agent bundles.
- `video_utils.py`: Gymnasium video recording helper.
- `network_visualizer.py`: render network image from one agent.
- `visualize_network_progress.py`: render progression frames from checkpoint agents.
- `render_all_saved_agents.py`: batch-render all `.npz` agents, optional MP4 progression.
- `validate_agent.py`: record one validation episode from latest best agent.
- `record_saved_agent.py`: record the latest best agent without retraining.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies from `requirements.txt`:

- `numpy`
- `torch`
- `gymnasium`
- `matplotlib`

Note:
- `render_all_saved_agents.py --make-video` requires `ffmpeg` available on PATH.

## Quick start

```bash
python3 run.py
```

After a run, check:

- `runs/<run-id>/agents/`
- `runs/<run-id>/plots/`
- `runs/<run-id>/videos/`

`<run-id>` defaults to timestamp format: `YYYY-MM-DD-HH:MM`.

## Default behavior (current config)

From `BRKGAConfig` defaults in `config.py`:

- Population: `pop_size=10`
- Generations: `generations=50`
- Hidden layers: `(8, 8)`
- Evaluation: `episodes_per_individual=3`, `max_steps=1000`
- Early stop: `early_stop_patience=8`, `early_stop_min_delta=1e-6`
- Progress recording: enabled (`rec_progress=True`)
- Progress checkpoints: every generation (`record_every_generation=True`)
- Save progress agents: every generation (`save_every_generation_agent_bundle=True`)
- Final best validation video: enabled (`record_best=True`)

## Output structure

Each `python3 run.py` creates a new run folder:

```text
runs/<run-id>/
  agents/
    best_agent.npz
    progress_best_gen_000.npz
    ...
    visualizer/
      *_network.png
      network_progression.gif
      network_progression.mp4   # only if ffmpeg succeeded
  plots/
    fitness_history.png
    fitness_history.csv
  videos/
    gen_000-episode-0.mp4
    ...
    best_global_validated_000-episode-0.mp4
```

## Common commands

Train full pipeline:

```bash
python3 run.py
```

Validate latest best agent (single episode recording):

```bash
python3 validate_agent.py
```

Record latest best agent without retraining:

```bash
python3 record_saved_agent.py
```

Render latest best network:

```bash
python3 network_visualizer.py
```

Render a specific agent and save to custom path:

```bash
python3 network_visualizer.py --agent-path runs/<run-id>/agents/best_agent.npz --save-path runs/<run-id>/plots/custom_network.png
```

Render network progression frames from checkpoint agents:

```bash
python3 visualize_network_progress.py
```

Batch render all agents in one run:

```bash
python3 render_all_saved_agents.py --agents-dir runs/<run-id>/agents --out-dir runs/<run-id>/agents/visualizer
```

Batch render recursively and optionally build MP4:

```bash
python3 render_all_saved_agents.py --agents-dir runs --recursive --out-dir runs/all_visualizer --make-video
```

## Configuration workflow

1. Edit `config.py`.
2. Run `python3 run.py`.
3. Compare run folders under `runs/`.

Useful settings to tune first:

- `pop_size`, `generations`
- `hidden_layers`
- `episodes_per_individual`, `max_steps`
- `elite_frac`, `mutant_frac`, `bias`
- `processes` (parallel workers)
- `record_every_generation` and `save_every_generation_agent_bundle`

## Minimal experiment preset

Use this preset when you want a quick local sanity check (few minutes instead of full runs).  
Edit `config.py` and temporarily set:

```python
pop_size = 8
generations = 10
episodes_per_individual = 1
max_steps = 300
processes = 1

rec_progress = False
save_progress_agent_bundles = False
record_best = False
```

Why this is faster:

- Smaller population and fewer generations reduce total evaluations.
- One episode per individual cuts rollout cost.
- Lower `max_steps` caps long episodes.
- Disabling progress saves/recording removes extra I/O and rendering work.

## Reproducibility notes

- `seed` controls global RNG seeding.
- With `fixed_eval_seeds=True`, each genome gets a deterministic evaluation seed derived from genome content, improving cross-generation comparability.
- Results can still vary by Python/OS/library versions.

## Troubleshooting

- No videos generated:
  - confirm Gymnasium recording works in your environment and write permissions exist in the run folder.
- `network_progression.mp4` missing:
  - `ffmpeg` is likely not installed or not on PATH.
- Slower training than expected:
  - reduce `episodes_per_individual`/`max_steps` or tune `processes` in `config.py`.
