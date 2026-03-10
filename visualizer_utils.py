# visualizer_utils.py
"""
Visualization helpers for BRKGA MountainCar.

These plots/logs reflect what the evolutionary algorithm is doing:
- best fitness per generation
- running global-best fitness (monotonic)
- CSV export of the history
"""

import csv
import os
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------
# History container
# ---------------------------------------------------
@dataclass
class BRKGAHistory:
    best_per_gen: List[float] = field(default_factory=list)
    mean_per_gen: List[float] = field(default_factory=list)
    std_per_gen: List[float] = field(default_factory=list)
    worst_per_gen: List[float] = field(default_factory=list)

    def append(self, fitness: np.ndarray) -> None:
        """
        Append summary statistics from one generation.
        """
        self.best_per_gen.append(float(np.max(fitness)))
        self.mean_per_gen.append(float(np.mean(fitness)))
        self.std_per_gen.append(float(np.std(fitness)))
        self.worst_per_gen.append(float(np.min(fitness)))


# ---------------------------------------------------
# Plot: fitness history
# ---------------------------------------------------
def save_fitness_history_plot(
    history: BRKGAHistory,
    out_dir: str = "plots",
) -> str:
    """
    Save a compact line plot focused on best-fitness progression.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fitness_history.png")

    gens = np.arange(len(history.best_per_gen))

    running_best = np.maximum.accumulate(np.array(history.best_per_gen))

    plt.figure(figsize=(9, 5))
    plt.plot(gens, history.best_per_gen, label="Best (Generation)")
    plt.plot(gens, running_best, label="Best (Global)", linewidth=2.0)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("BRKGA Best Fitness Progression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


# ---------------------------------------------------
# Plot: histogram for one generation
# ---------------------------------------------------
def save_population_histogram(
    fitness: np.ndarray,
    gen: int,
    out_dir: str = "plots",
) -> str:
    """
    Save a histogram of population fitness values for a specific generation.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"population_hist_gen_{gen:03d}.png")

    plt.figure(figsize=(7, 4.5))
    plt.hist(fitness, bins=15)
    plt.xlabel("Fitness")
    plt.ylabel("Count")
    plt.title(f"Population Fitness Distribution - Gen {gen}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


# ---------------------------------------------------
# CSV export
# ---------------------------------------------------
def save_history_csv(
    history: BRKGAHistory,
    out_dir: str = "plots",
) -> str:
    """
    Save generation statistics to CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fitness_history.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best", "mean", "std", "worst"])

        for i, (b, m, s, w) in enumerate(
            zip(
                history.best_per_gen,
                history.mean_per_gen,
                history.std_per_gen,
                history.worst_per_gen,
            )
        ):
            writer.writerow([i, b, m, s, w])

    return path
