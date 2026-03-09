"""
Utilities to persist and load trained agent decision variables.
"""

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def save_agent_bundle(
    genome: np.ndarray,
    hidden_layers: Sequence[int],
    out_dir: str,
    prefix: str,
    metadata: Optional[Dict[str, object]] = None,
) -> str:
    """
    Save an agent bundle as .npz:
    - genome: flattened model parameters
    - hidden_layers: hidden layer widths
    - metadata_json: free-form JSON metadata
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}.npz")

    meta = metadata or {}
    np.savez(
        path,
        genome=np.asarray(genome, dtype=np.float64),
        hidden_layers=np.asarray(list(hidden_layers), dtype=np.int64),
        metadata_json=np.array(json.dumps(meta)),
    )
    return path


def load_agent_bundle(path: str) -> Tuple[np.ndarray, Tuple[int, ...], Dict[str, object]]:
    """
    Load an agent bundle created by save_agent_bundle.
    """
    with np.load(path, allow_pickle=False) as data:
        genome = np.asarray(data["genome"], dtype=np.float64)
        # Backward compatibility with older bundles storing only `hidden`.
        if "hidden_layers" in data:
            hidden_layers = tuple(int(v) for v in np.asarray(data["hidden_layers"]).tolist())
        elif "hidden" in data:
            hidden_layers = (int(data["hidden"]),)
        else:
            raise KeyError("Agent bundle must contain `hidden_layers` or `hidden`.")
        raw_meta = str(data["metadata_json"])

    metadata = json.loads(raw_meta) if raw_meta else {}
    return genome, hidden_layers, metadata
