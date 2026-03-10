"""
Central configuration for BRKGA training and recording.
"""

import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BRKGAConfig:
    # Environment
    env_id: str = "MountainCarContinuous-v0"

    # Reproducibility
    seed: int = 123

    # Population
    pop_size: int = 100
    generations: int = 200

    # BRKGA parameters
    elite_frac: float = 0.20
    mutant_frac: float = 0.30
    bias: float = 0.60

    # Evaluation
    episodes_per_individual: int = 5
    max_steps: int = 999
    fixed_eval_seeds: bool = True #If True, each individual gets the same eval seeds
    use_shaped_fitness: bool = True
    mountaincar_progress_scale: float = 1000.0
    mountaincar_velocity_scale: float = 50.0
    mountaincar_goal_bonus: float = 1000.0

    # Model
    hidden_layers: Tuple[int, ...] = (8, 8)

    # Parallelism
    processes: int = max(1, mp.cpu_count() - 1)

    # Decode bounds: decoded_gene = low + rk * (high - low)
    gene_low: float = -1.0
    gene_high: float = 1.0

    # Video output
    video_dir: str = "videos"
    rec_progress: bool = True
    rec_progress_include_first: bool = True
    record_every_generation: bool = True
    record_episodes: int = 5 
    record_max_steps: int = 999
    record_fps: int = 50
    record_min_seconds: float = 30.0
    video_prefix: str = "gen"
    record_best: bool = True

    # Plot output (best-fitness progression only)
    plot_dir: str = "plots"
    draw_best_network: bool = True
    network_plot_name: str = "best_network.png"

    # Agent artifact output
    agent_dir: str = "agents"
    agent_prefix: str = "best_agent"
    save_best_agent_bundle: bool = True
    save_progress_agent_bundles: bool = True
    save_every_generation_agent_bundle: bool = True
    progress_agent_prefix: str = "progress_best_gen"

    # Run output layout
    run_root_dir: str = "runs"
    run_id: Optional[str] = None

    # Early stopping
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-6

    def __post_init__(self) -> None:
        if self.env_id != "MountainCarContinuous-v0":
            raise ValueError("This project is currently configured for MountainCarContinuous-v0 only.")
        if isinstance(self.hidden_layers, int):
            self.hidden_layers = (self.hidden_layers,)
        elif isinstance(self.hidden_layers, list):
            self.hidden_layers = tuple(self.hidden_layers)
        elif not isinstance(self.hidden_layers, tuple):
            raise ValueError("hidden_layers must be an int, list[int], or tuple[int, ...].")
        if self.pop_size < 2:
            raise ValueError("pop_size must be >= 2.")
        if self.generations < 1:
            raise ValueError("generations must be >= 1.")
        if self.episodes_per_individual < 1:
            raise ValueError("episodes_per_individual must be >= 1.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if not isinstance(self.use_shaped_fitness, bool):
            raise ValueError("use_shaped_fitness must be a bool.")
        if self.mountaincar_progress_scale < 0.0:
            raise ValueError("mountaincar_progress_scale must be >= 0.")
        if self.mountaincar_velocity_scale < 0.0:
            raise ValueError("mountaincar_velocity_scale must be >= 0.")
        if self.mountaincar_goal_bonus < 0.0:
            raise ValueError("mountaincar_goal_bonus must be >= 0.")
        if not isinstance(self.fixed_eval_seeds, bool):
            raise ValueError("fixed_eval_seeds must be a bool.")
        if not self.hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer.")
        if any(h < 1 for h in self.hidden_layers):
            raise ValueError("all hidden layer sizes must be >= 1.")
        if self.processes < 1:
            raise ValueError("processes must be >= 1.")
        if not (0.0 <= self.elite_frac <= 1.0):
            raise ValueError("elite_frac must be in [0, 1].")
        if not (0.0 <= self.mutant_frac <= 1.0):
            raise ValueError("mutant_frac must be in [0, 1].")
        if self.elite_frac + self.mutant_frac > 1.0:
            raise ValueError("elite_frac + mutant_frac must be <= 1.")
        if not (0.0 <= self.bias <= 1.0):
            raise ValueError("bias must be in [0, 1].")
        if self.gene_low >= self.gene_high:
            raise ValueError("gene_low must be smaller than gene_high.")
        if not isinstance(self.rec_progress_include_first, bool):
            raise ValueError("rec_progress_include_first must be a bool.")
        if not isinstance(self.record_every_generation, bool):
            raise ValueError("record_every_generation must be a bool.")
        if self.record_episodes < 1:
            raise ValueError("record_episodes must be >= 1.")
        if self.record_max_steps < 1:
            raise ValueError("record_max_steps must be >= 1.")
        if self.record_fps < 1:
            raise ValueError("record_fps must be >= 1.")
        if self.record_min_seconds <= 0.0:
            raise ValueError("record_min_seconds must be > 0.")
        if self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be >= 1.")
        if self.early_stop_min_delta < 0.0:
            raise ValueError("early_stop_min_delta must be >= 0.")
        if not self.agent_prefix:
            raise ValueError("agent_prefix must be non-empty.")
        if not self.progress_agent_prefix:
            raise ValueError("progress_agent_prefix must be non-empty.")
        if not isinstance(self.save_every_generation_agent_bundle, bool):
            raise ValueError("save_every_generation_agent_bundle must be a bool.")
        if not self.network_plot_name:
            raise ValueError("network_plot_name must be non-empty.")
