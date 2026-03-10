"""
Draw policy network structure using trained weights and biases.
"""

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from agent_store import load_agent_bundle
from fitness_function import PolicyNet, get_env_dims, is_continuous_env, unflatten_params


def _layer_positions(
    layer_sizes: List[int],
    layer_x: Optional[List[float]] = None,
    x_min: float = 0.05,
    x_max: float = 0.95,
    y_min: float = 0.05,
    y_max: float = 0.95,
) -> List[List[Tuple[float, float]]]:
    if layer_x is not None:
        if len(layer_x) != len(layer_sizes):
            raise ValueError("layer_x must match number of layers.")
        xs = np.array(layer_x, dtype=float)
    else:
        xs = np.linspace(x_min, x_max, num=len(layer_sizes))
    positions: List[List[Tuple[float, float]]] = []
    for x, n in zip(xs, layer_sizes):
        ys = np.linspace(y_min, y_max, num=n) if n > 1 else np.array([(y_min + y_max) * 0.5])
        positions.append([(float(x), float(y)) for y in ys])
    return positions


def _signed_color(val: float) -> str:
    return "#66bb6a" if val >= 0 else "#ef9a9a"


def _bias_rgba(b: float, max_abs: float) -> Tuple[float, float, float, float]:
    strength = 0.0 if max_abs <= 0.0 else min(1.0, abs(b) / max_abs)
    alpha = 0.20 + 0.55 * strength
    if b >= 0:
        return (0.60, 0.82, 0.62, alpha)  # soft green tint
    return (0.93, 0.72, 0.72, alpha)  # soft red tint


def save_policy_network_plot(
    genome: np.ndarray,
    hidden_layers: Sequence[int],
    out_path: str,
    env_id: str = "MountainCarContinuous-v0",
    generation: Optional[int] = None,
) -> str:
    """
    Save network graph for PolicyNet:
    input(obs_dim) -> hidden... -> output(act_dim)
    Edge color: green (+) / orange-red (-), alpha by |weight|.
    Node fill (hidden/output): bias sign + magnitude.
    """
    obs_dim, act_dim = get_env_dims(env_id)
    model = PolicyNet(
        obs_dim=obs_dim,
        hidden_layers=hidden_layers,
        act_dim=act_dim,
        continuous_action=is_continuous_env(env_id),
    )
    unflatten_params(model, genome)

    linear_layers = list(model.layers)
    weights = [layer.weight.detach().cpu().numpy() for layer in linear_layers]
    biases = [layer.bias.detach().cpu().numpy() for layer in linear_layers]

    layer_sizes = [obs_dim, *list(hidden_layers), act_dim]
    x_positions = np.linspace(0.08, 0.68, num=len(layer_sizes)).tolist()
    pos = _layer_positions(layer_sizes, layer_x=x_positions, y_min=0.12, y_max=0.80)

    # Keep output nodes close together vertically.
    if act_dim > 1:
        out_ys = np.linspace(0.43, 0.53, num=act_dim)
    else:
        out_ys = np.array([0.48])
    pos[-1] = [(pos[-1][i][0], float(out_ys[i])) for i in range(act_dim)]

    max_w = float(max(max(np.max(np.abs(w)) for w in weights), 1e-12))
    max_b = float(max(max(np.max(np.abs(b)) for b in biases), 1e-12))

    fig, ax = plt.subplots(figsize=(13, 8), facecolor="white")
    ax.set_facecolor("white")

    # Draw all layer-to-layer connections.
    for layer_idx, w_mat in enumerate(weights):
        out_n, in_n = w_mat.shape
        for j in range(out_n):
            for i in range(in_n):
                w = float(w_mat[j, i])
                x0, y0 = pos[layer_idx][i]
                x1, y1 = pos[layer_idx + 1][j]
                norm = abs(w) / max_w
                alpha = 0.18 + 0.70 * norm
                lw = 0.6 + 1.8 * norm
                ax.plot([x0, x1], [y0, y1], color=_signed_color(w), alpha=alpha, linewidth=lw)

    # input nodes (no bias)
    for (x, y) in pos[0]:
        circ = plt.Circle((x, y), 0.0175, facecolor="white", edgecolor="#4a4a4a", linewidth=1.2)
        ax.add_patch(circ)

    # Hidden + output nodes with bias tint from corresponding linear layers.
    for layer_idx in range(1, len(layer_sizes)):
        b_vec = biases[layer_idx - 1]
        radius = 0.020 if layer_idx == len(layer_sizes) - 1 else 0.0175
        line_w = 1.3 if layer_idx == len(layer_sizes) - 1 else 1.2
        for node_idx, (x, y) in enumerate(pos[layer_idx]):
            circ = plt.Circle(
                (x, y),
                radius,
                facecolor=_bias_rgba(float(b_vec[node_idx]), max_b),
                edgecolor="#4a4a4a",
                linewidth=line_w,
            )
            ax.add_patch(circ)

    # Weight scale legend (top-right)
    lx0, ly0 = 0.79, 0.95
    ax.text(lx0, ly0, "Weights", color="#2f2f2f", fontsize=9, fontweight="bold")
    ax.plot([lx0, lx0 + 0.07], [ly0 - 0.03, ly0 - 0.03], color="#66bb6a", linewidth=2.2)
    ax.text(lx0 + 0.08, ly0 - 0.03, "+w", color="#2f2f2f", fontsize=8, va="center")
    ax.plot([lx0, lx0 + 0.07], [ly0 - 0.06, ly0 - 0.06], color="#ef9a9a", linewidth=2.2)
    ax.text(lx0 + 0.08, ly0 - 0.06, "-w", color="#2f2f2f", fontsize=8, va="center")

    # Bias scale legend (top-right)
    bx0, by0 = 0.79, 0.86
    ax.text(bx0, by0, "Biases", color="#2f2f2f", fontsize=9, fontweight="bold")
    plus_bias = plt.Circle((bx0 + 0.02, by0 - 0.04), 0.012, facecolor=_bias_rgba(max_b, max_b), edgecolor="#4a4a4a", linewidth=0.9)
    minus_bias = plt.Circle((bx0 + 0.02, by0 - 0.08), 0.012, facecolor=_bias_rgba(-max_b, max_b), edgecolor="#4a4a4a", linewidth=0.9)
    ax.add_patch(plus_bias)
    ax.add_patch(minus_bias)
    ax.text(bx0 + 0.04, by0 - 0.04, "+b (strong)", color="#2f2f2f", fontsize=8, va="center")
    ax.text(bx0 + 0.04, by0 - 0.08, "-b (strong)", color="#2f2f2f", fontsize=8, va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if generation is not None:
        ax.text(0.02, 0.98, f"gen={generation:03d}", transform=ax.transAxes, fontsize=12, color="#2f2f2f", va="top")


    # Just open the plot in a window if no output path is given
    if not out_path:
        plt.show()
        return ""

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    return out_path


def find_latest_run_dir(run_root: str = "runs") -> str:
    if not os.path.isdir(run_root):
        raise FileNotFoundError(f"Run root not found: {run_root}")
    candidates = [
        os.path.join(run_root, d)
        for d in os.listdir(run_root)
        if os.path.isdir(os.path.join(run_root, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"No run folders found in: {run_root}")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved best agent network.")
    parser.add_argument("--run-root", default="runs", help="Root directory containing runs.")
    parser.add_argument("--agent-path", default=None, help="Path to a saved .npz agent bundle.")
    parser.add_argument(
        "--save-path",
        default=None,
        help="Output png path. If omitted, uses latest run plots folder.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive window instead of only saving image.",
    )
    args = parser.parse_args()

    if args.agent_path:
        agent_path = args.agent_path
        default_out = "best_network.png"
    else:
        latest_run = find_latest_run_dir(args.run_root)
        agent_path = os.path.join(latest_run, "agents", "best_agent.npz")
        default_out = os.path.join(latest_run, "plots", "best_network.png")

    genome, hidden_layers, meta = load_agent_bundle(agent_path)
    env_id = "MountainCarContinuous-v0"
    if isinstance(meta, dict):
        training_cfg = meta.get("training_config")
        if isinstance(training_cfg, dict) and isinstance(training_cfg.get("env_id"), str):
            env_id = training_cfg["env_id"]
    out_path = "" if args.show else (args.save_path or default_out)
    rendered = save_policy_network_plot(
        genome=genome,
        hidden_layers=hidden_layers,
        out_path=out_path,
        env_id=env_id,
    )
    if rendered:
        print(f"[NN] Rendered: {rendered}")


if __name__ == "__main__":
    main()
