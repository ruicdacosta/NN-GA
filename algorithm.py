# algorithm.py
"""
BRKGA core logic for MountainCar policy search.
"""

import os
import random
import multiprocessing as mp
import hashlib
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from config import BRKGAConfig
from fitness_function import flatten_params, make_template_model, rollout_fitness
from visualizer_utils import (
    BRKGAHistory,
    save_fitness_history_plot,
    save_history_csv,
)


# ---------------------------------------------------
# Random-key utilities
# ---------------------------------------------------
def decode_random_keys(
    rk: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Decode random keys from [0, 1] to real-valued genome."""
    return low + rk * (high - low)


def make_gene_bounds(
    genome_len: int,
    cfg: BRKGAConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same lower/upper bound for all genes."""
    low = np.full(genome_len, cfg.gene_low, dtype=np.float64)
    high = np.full(genome_len, cfg.gene_high, dtype=np.float64)
    return low, high


# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------
def _worker_eval_one(args) -> float:
    """Worker function for multiprocessing."""
    rk, cfg, low, high, seed = args
    genome = decode_random_keys(rk, low, high)

    return rollout_fitness(
        genome=genome,
        hidden_layers=cfg.hidden_layers,
        episodes=cfg.episodes_per_individual,
        max_steps=cfg.max_steps,
        seed=seed,
        env_id=cfg.env_id,
        use_shaped_fitness=cfg.use_shaped_fitness,
        mountaincar_progress_scale=cfg.mountaincar_progress_scale,
        mountaincar_velocity_scale=cfg.mountaincar_velocity_scale,
        mountaincar_goal_bonus=cfg.mountaincar_goal_bonus,
    )


def evaluate_population_parallel(
    pop_rk: List[np.ndarray],
    cfg: BRKGAConfig,
    low: np.ndarray,
    high: np.ndarray,
    base_seed: int,
) -> np.ndarray:
    """Evaluate the whole population, optionally in parallel."""
    args = []
    for rk in pop_rk:
        indiv_seed = stable_seed_from_rk(rk, base_seed)
        args.append((rk, cfg, low, high, indiv_seed))

    if cfg.processes <= 1:
        scores = [_worker_eval_one(a) for a in args]
    else:
        try:
            with mp.Pool(processes=cfg.processes) as pool:
                scores = pool.map(_worker_eval_one, args)
        except (PermissionError, OSError) as e:
            print(
                f"[WARN] Multiprocessing unavailable ({e}). "
                "Falling back to single-process evaluation."
            )
            scores = [_worker_eval_one(a) for a in args]

    return np.asarray(scores, dtype=np.float64)


def stable_seed_from_rk(rk: np.ndarray, base_seed: int) -> int:
    """
    Deterministic seed tied to genome content instead of population index.
    Copied elites then keep identical evaluation conditions across generations.
    """
    digest = hashlib.blake2b(rk.tobytes(), digest_size=8).digest()
    genome_key = int.from_bytes(digest, byteorder="little", signed=False)
    return int((base_seed ^ genome_key) % (2**31 - 1))


# ---------------------------------------------------
# BRKGA core
# ---------------------------------------------------
def biased_crossover(
    elite_parent: np.ndarray,
    non_elite_parent: np.ndarray,
    bias: float,
) -> np.ndarray:
    """
    For each gene:
    - choose elite gene with probability `bias`
    - otherwise choose non-elite gene
    """
    mask = np.random.rand(elite_parent.size) < bias
    return np.where(mask, elite_parent, non_elite_parent)


def run_brkga(
    cfg: BRKGAConfig,
    seed: int = 123,
    on_progress_best: Optional[Callable[[int, np.ndarray, float], None]] = None,
) -> Tuple[np.ndarray, float, BRKGAHistory]:
    """
    Run BRKGA and return:
    - best decoded genome
    - best fitness
    - fitness history
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(cfg.video_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)
    os.makedirs(cfg.agent_dir, exist_ok=True)

    template = make_template_model(hidden_layers=cfg.hidden_layers, env_id=cfg.env_id)
    genome_len = int(flatten_params(template).size)

    low, high = make_gene_bounds(genome_len, cfg)

    pop_rk: List[np.ndarray] = [
        np.random.rand(genome_len).astype(np.float64)
        for _ in range(cfg.pop_size)
    ]

    elite_n = max(1, int(round(cfg.pop_size * cfg.elite_frac)))
    mutant_n = max(0, int(round(cfg.pop_size * cfg.mutant_frac)))
    offspring_n = cfg.pop_size - elite_n - mutant_n

    if offspring_n < 0:
        raise ValueError("Invalid fractions: elite_frac + mutant_frac must be <= 1.")

    best_fit = -1e18
    best_rk: Optional[np.ndarray] = None

    history = BRKGAHistory()
    generations_without_improvement = 0

    print(
        f"BRKGA | Pop={cfg.pop_size} Gens={cfg.generations} "
        f"Elite={elite_n} Mutants={mutant_n} Bias={cfg.bias}"
    )
    print(f"Decode bounds: [{cfg.gene_low}, {cfg.gene_high}] GenomeLen={genome_len}")
    print(f"Videos: {os.path.abspath(cfg.video_dir)}")
    print(f"Plots:  {os.path.abspath(cfg.plot_dir)}")
    print(f"Agents: {os.path.abspath(cfg.agent_dir)}")
    print()

    progress_gens = []
    if cfg.rec_progress:
        progress_gens = (
            list(range(cfg.generations))
            if cfg.record_every_generation
            else choose_progress_generations(
                generations=cfg.generations,
                include_first=cfg.rec_progress_include_first,
            )
        )

    for gen in range(cfg.generations):
        eval_base_seed = seed if cfg.fixed_eval_seeds else seed + gen * 100000
        fitness = evaluate_population_parallel(
            pop_rk=pop_rk,
            cfg=cfg,
            low=low,
            high=high,
            base_seed=eval_base_seed,
        )

        order = np.argsort(fitness)[::-1]
        pop_rk = [pop_rk[i] for i in order]
        fitness = fitness[order]

        history.append(fitness)
        current_best = float(fitness[0])

        if current_best > best_fit + cfg.early_stop_min_delta:
            best_fit = current_best
            best_rk = pop_rk[0].copy()
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        print(
            f"[Gen {gen:03d}] best={current_best:.1f} "
            f"mean={fitness.mean():.1f} std={fitness.std():.1f} "
            f"globalbest={best_fit:.1f} "
            f"stuck={generations_without_improvement}"
        )

        if cfg.rec_progress and gen in progress_gens:
            best_genome_gen = decode_random_keys(pop_rk[0], low, high)
            if on_progress_best is not None:
                on_progress_best(gen, best_genome_gen, current_best)

        if generations_without_improvement >= cfg.early_stop_patience:
            print(
                f"\n[EARLY STOP] No improvement greater than "
                f"{cfg.early_stop_min_delta} for "
                f"{cfg.early_stop_patience} generations."
            )
            break

        elites = pop_rk[:elite_n]
        non_elites = pop_rk[elite_n:]

        new_pop: List[np.ndarray] = []
        new_pop.extend([e.copy() for e in elites])

        for _ in range(offspring_n):
            p_e = elites[np.random.randint(0, elite_n)]

            if len(non_elites) > 0:
                p_n = non_elites[np.random.randint(0, len(non_elites))]
            else:
                p_n = elites[np.random.randint(0, elite_n)]

            child = biased_crossover(p_e, p_n, cfg.bias)
            new_pop.append(child)

        for _ in range(mutant_n):
            new_pop.append(np.random.rand(genome_len).astype(np.float64))

        pop_rk = new_pop

    if best_rk is None:
        raise RuntimeError("BRKGA produced no best solution (unexpected).")

    plot_path = save_fitness_history_plot(history, out_dir=cfg.plot_dir)
    csv_path = save_history_csv(history, out_dir=cfg.plot_dir)

    print(f"\n[PLOT] Fitness history: {plot_path}")
    print(f"[CSV]  Fitness history: {csv_path}")

    best_genome = decode_random_keys(best_rk, low, high)
    return best_genome, best_fit, history


def prepare_run_dirs(cfg: BRKGAConfig) -> str:
    """Create a unique run directory and point output folders into it."""
    run_id = cfg.run_id or datetime.now().strftime("%Y-%m-%d-%H:%M")
    run_dir = os.path.join(cfg.run_root_dir, run_id)
    cfg.video_dir = os.path.join(run_dir, "videos")
    cfg.plot_dir = os.path.join(run_dir, "plots")
    cfg.agent_dir = os.path.join(run_dir, "agents")

    os.makedirs(cfg.video_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)
    os.makedirs(cfg.agent_dir, exist_ok=True)
    return run_dir


def choose_progress_generations(
    generations: int,
    include_first: bool,
) -> List[int]:
    """
    Select exactly 5 generations.
    - include_first=True: first + last + 3 between
    - include_first=False: 5 checkpoints excluding generation 0
    """
    start = 0 if include_first else 1
    available = list(range(start, generations))
    if len(available) < 5:
        need = 5 if include_first else 6
        raise ValueError(f"generations must be >= {need} for rec_progress.")

    idx = np.linspace(0, len(available) - 1, num=5)
    points = [available[int(round(i))] for i in idx]

    unique = []
    for p in points:
        if p not in unique:
            unique.append(p)

    if len(unique) < 5:
        for p in available:
            if p not in unique:
                unique.append(p)
            if len(unique) == 5:
                break

    return sorted(unique[:5])
