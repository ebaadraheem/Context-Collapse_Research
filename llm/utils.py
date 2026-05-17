"""Shared utilities for the context-collapse benchmark."""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "results", level: int = logging.INFO) -> None:
    """Configure root logger to write to stdout and a timestamped log file."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"benchmark_{ts}.log")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.getLogger(__name__).info("Logging to %s", log_path)


# ---------------------------------------------------------------------------
# Script loading
# ---------------------------------------------------------------------------

def load_scripts(scripts_dir: str) -> list[dict]:
    """
    Load all JSON test scripts from scripts_dir.

    Each file may contain a single script (dict) or a list of scripts.
    Files are sorted for reproducibility (os.listdir order is OS-dependent).
    """
    scripts: list[dict] = []
    path = Path(scripts_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Scripts directory not found: {scripts_dir}")

    for fname in sorted(path.iterdir()):
        if fname.suffix != ".json":
            continue
        with fname.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                logging.getLogger(__name__).warning(
                    "Skipping %s: JSON parse error: %s", fname.name, exc
                )
                continue

        if isinstance(data, list):
            scripts.extend(data)
        elif isinstance(data, dict):
            scripts.append(data)
        else:
            logging.getLogger(__name__).warning(
                "Skipping %s: unexpected top-level type %s", fname.name, type(data)
            )

    logging.getLogger(__name__).info("Loaded %d scripts from %s", len(scripts), scripts_dir)
    return scripts


# ---------------------------------------------------------------------------
# Script validation
# ---------------------------------------------------------------------------

def validate_script(script: dict) -> list[str]:
    """
    Validate a single script dict. Returns a list of warning strings.
    Empty list means the script is clean.
    """
    warnings: list[str] = []
    sid = script.get("script_id", "<unknown>")

    if "script_id" not in script:
        warnings.append(f"{sid}: missing 'script_id'")
    if "turns" not in script or not isinstance(script["turns"], list):
        warnings.append(f"{sid}: missing or invalid 'turns' list")
        return warnings

    recall_turns: list[int] = []
    for turn in script["turns"]:
        if not isinstance(turn, dict):
            warnings.append(f"{sid}: turn is not a dict: {turn!r}")
            continue
        if "turn" not in turn:
            warnings.append(f"{sid}: turn missing 'turn' field")
        if "role" not in turn:
            warnings.append(f"{sid}: turn {turn.get('turn')} missing 'role'")
        if "content" not in turn:
            warnings.append(f"{sid}: turn {turn.get('turn')} missing 'content'")
        if turn.get("is_recall"):
            t = turn.get("turn")
            if t not in {5, 10, 15, 20, 25}:
                warnings.append(f"{sid}: recall turn {t} is not in {{5,10,15,20,25}}")
            if not turn.get("ground_truth"):
                warnings.append(f"{sid}: recall turn {t} missing 'ground_truth'")
            if not turn.get("target_fact_id"):
                warnings.append(f"{sid}: recall turn {t} missing 'target_fact_id'")
            recall_turns.append(t)

    if not recall_turns:
        warnings.append(f"{sid}: no recall turns found")

    return warnings


# ---------------------------------------------------------------------------
# History saving
# ---------------------------------------------------------------------------

def save_conversation_history(
    history: list[dict],
    out_dir: str,
    script_id: str,
    strategy_name: str,
    rep: int,
) -> str:
    """
    Save full conversation history for one run to a JSONL file.
    Each line: {"turn": int, "role": str, "content": str, "is_recall": bool}
    Returns the path to the saved file.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = f"history_{script_id}_{strategy_name}_rep{rep}.jsonl"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        for entry in history:
            f.write(json.dumps(entry) + "\n")
    return fpath


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed Python random and numpy (if available)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    logging.getLogger(__name__).info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def filter_none_keys(api_keys: list[str | None]) -> list[str]:
    """Filter out None / empty strings from a list of API key candidates."""
    return [k for k in api_keys if k]