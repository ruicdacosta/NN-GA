# BRKGA CartPole Workflow

This project trains a CartPole policy with a BRKGA (Biased Random-Key Genetic Algorithm), saves agents and artifacts per run, records selected policy videos, and visualizes network parameters over time.

This README is intended as the single source of truth for how to run and reuse the project.

## 1. Project Structure

- `config.py`: all parameters and flags (training, recording, outputs).
- `run.py`: main entrypoint (train + save + optional validation + visualizations).
- `algorithm.py`: BRKGA core loop and population evolution.
- `fitness_function.py`: policy model + rollout fitness.
- `agent_store.py`: save/load `.npz` agent bundles.
- `video_utils.py`: MP4 recording utilities.
- `visualizer_utils.py`: fitness plots and CSV history export.
- `network_visualizer.py`: draw NN graph from saved genome.
- `visualize_network_progress.py`: render NN images across saved progress generations.
- `render_all_saved_agents.py`: render NN image for every saved agent bundle.
- `validate_agent.py`: run and record one full validation episode from saved best agent.

## 2. One Run = One Folder

Each training run creates a timestamped folder:

- `runs/YYYY-MM-DD-HH:MM/`
  - `agents/`
  - `plots/`
  - `videos/`

Typical files produced:

- `agents/best_agent.npz`: global best at end of run.
- `agents/progress_best_gen_XXX.npz`: best agents at progress checkpoint generations.
- `agents/visualizer/*.png`: NN renders auto-generated for all agents in this run.
- `plots/fitness_history.png`: best fitness progression plot.
- `plots/fitness_history.csv`: generation stats.
- `plots/best_network.png`: NN render for global best.
- `videos/gen_XXX-episode-0.mp4`: progress videos.
- `videos/best_global_validated_000-episode-0.mp4`: optional final validation video.

## 3. Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training:

```bash
python3 run.py
```

3. Inspect latest run outputs under `runs/`.

## 4. Main Workflow (What Happens in `run.py`)

1. Load config from `BRKGAConfig()` in `config.py`.
2. Create run folder (`runs/<timestamp>/...`) and wire output directories.
3. Determine progress generation checkpoints (5 total).
4. Run BRKGA evolution (`algorithm.run_brkga`):
   - evaluate population,
   - preserve elites,
   - crossover + mutants,
   - early stop by plateau only.
5. During selected progress generations:
   - record a video,
   - optionally save that generation's best agent bundle.
6. After training:
   - save global best agent (`best_agent.npz`),
   - save fitness plot and CSV,
   - render `best_network.png` (if enabled),
   - auto-render NN images for all agents into `agents/visualizer/`,
   - optionally run final validation recording.

## 5. Core Behavior You Configured

### Elitism consistency
Elites are copied unchanged, and evaluation seeding is genome-stable, so copied elites are evaluated under identical conditions across generations (`fixed_eval_seeds=True` case).

### Early stopping
No max-fitness constraint is used anymore. Stopping is plateau-based only:

- `early_stop_patience`
- `early_stop_min_delta`

### Progress recording
When `rec_progress=True`, exactly 5 generations are selected:

- with `rec_progress_include_first=True`: first + 3 middle + last,
- with `rec_progress_include_first=False`: 5 checkpoints excluding generation 0.

## 6. Important Config Flags (`config.py`)

### Training
- `seed`: global seed.
- `pop_size`, `generations`: BRKGA scale.
- `elite_frac`, `mutant_frac`, `bias`: BRKGA dynamics.
- `episodes_per_individual`, `max_steps`: fitness evaluation setup.
- `fixed_eval_seeds`: reproducible per-genome evaluation.

### Recording
- `rec_progress`: enable progress checkpoint recording.
- `rec_progress_include_first`: include generation 0 in the 5 checkpoints.
- `record_max_steps`: max steps in recorded episode.
- `record_best`: validate and record the global best at end.
- `video_prefix`: prefix for progress videos.

### Agent persistence
- `save_best_agent_bundle`: save final global best.
- `save_progress_agent_bundles`: save progress checkpoint best agents.
- `agent_prefix`: file prefix for global best.
- `progress_agent_prefix`: file prefix for progress generation best agents.

### Network visualization
- `draw_best_network`: render global best NN after run.
- `network_plot_name`: filename in `plots/`.

### Output layout
- `run_root_dir`: parent output folder (default `runs`).
- `run_id`: optional fixed run id; if `None`, timestamp is used.

## 7. Commands You’ll Reuse

### A) Train + full pipeline

```bash
python3 run.py
```

### B) Validate best agent from latest run (one full episode)

```bash
python3 validate_agent.py
```

### C) Render best network from latest run

```bash
python3 network_visualizer.py
```

Open directly in a window instead of saving:

```bash
python3 network_visualizer.py --show
```

### D) Render progression of saved checkpoint agents

```bash
python3 visualize_network_progress.py
```

Outputs to:

- `runs/<latest>/plots/network_progress/network_gen_XXX.png`

### E) Render NN for all saved agents

```bash
python3 render_all_saved_agents.py --agents-dir runs/<run-id>/agents
```

Default output:

- `runs/<run-id>/agents/visualizer/*.png`

Across all runs recursively:

```bash
python3 render_all_saved_agents.py --agents-dir runs --recursive --out-dir runs/all_visualizer
```

## 8. Typical Troubleshooting

### "Import ... could not be resolved"
Your IDE interpreter does not have project deps. Use the same Python environment where `pip install -r requirements.txt` was run.

### Progress video shows resets
A reset means episode ended (terminated/truncated). Current setup records single episodes for progress and validation.

### Best per generation appears to decrease
That can happen with stochastic eval if seeds vary. With `fixed_eval_seeds=True` and genome-stable seeding, copied elites remain comparable.

### Can't find generated images
Look in latest run folder:

- `runs/<latest>/plots/best_network.png`
- `runs/<latest>/agents/visualizer/*.png`
- `runs/<latest>/plots/network_progress/*.png`

## 9. Recommended Daily Routine

1. Edit `config.py` for your experiment settings.
2. Run `python3 run.py`.
3. Review:
   - `plots/fitness_history.png`
   - `videos/`
   - `agents/visualizer/`
4. If needed, run:
   - `python3 visualize_network_progress.py`
5. Keep each run folder as immutable experiment history.

## 10. Notes on Reproducibility

- Reproducibility depends on fixed seeds and same environment.
- Different CPU/OS/library versions can still slightly change outcomes.
- Save run folder + config snapshot (`training_config` is stored in agent metadata) for traceability.
