"""Persistence helpers for fast optimizer runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _sanitize(obj: Any) -> Any:
    """Convert objects to JSON-serialisable primitives."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, tuple):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    if hasattr(obj, '__dict__') and not isinstance(obj, (str, bytes)):
        return _sanitize(obj.__dict__)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


class TrialStore:
    """Manage JSONL trial logs and checkpoints for optimizer runs."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.run_dir / 'trials.jsonl'
        self.checkpoint_path = self.run_dir / 'checkpoint.json'
        self.config_path = self.run_dir / 'config.json'

    # ---------------------------- configuration
    def write_config(self, config: Dict[str, Any]) -> None:
        payload = _sanitize(config)
        tmp_path = self.config_path.with_suffix('.tmp')
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(self.config_path)

    # ---------------------------- trial records
    def load_trials(self) -> List[Dict[str, Any]]:
        if not self.trials_path.exists():
            return []
        trials: List[Dict[str, Any]] = []
        with self.trials_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    trials.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return trials

    def append_trial(self, record: Dict[str, Any]) -> None:
        payload = _sanitize(record)
        line = json.dumps(payload, separators=(',', ':'))
        with self.trials_path.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    # ---------------------------- checkpoints
    def write_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        payload = _sanitize(checkpoint)
        tmp_path = self.checkpoint_path.with_suffix('.tmp')
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(self.checkpoint_path)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        if not self.checkpoint_path.exists():
            return None
        try:
            return json.loads(self.checkpoint_path.read_text())
        except json.JSONDecodeError:
            return None

