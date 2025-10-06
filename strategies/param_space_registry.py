#!/usr/bin/env python3
"""
Parameter Space Registry

Builds per-strategy parameter spaces for all strategies under
strategies/lazybear/technical_indicators/technical_indicators by instantiating
each class and deriving tuned bounds from its default parameters using
BaseStrategy heuristics, enhanced with common indicator rules.

This central registry allows optimizer runners to fetch explicit, strategy-
specific parameter spaces without modifying each file individually.
"""

import os
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader
import re
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
"""Ensure repo root on path for imports."""

from strategies.base_strategy import BaseStrategy


def _discover_strategy_files() -> Dict[str, str]:
    """Return mapping of class_name -> file path for direct BaseStrategy subclasses.

    Note: We load modules to locate classes robustly.
    """
    mapping: Dict[str, str] = {}
    root = REPO_ROOT / 'strategies' / 'lazybear' / 'technical_indicators'
    if not root.exists():
        return mapping
    for py in root.rglob('*.py'):
        try:
            mod = SourceFileLoader(f"ps_{py.stem}", str(py)).load_module()
            # scan module attributes
            for name in dir(mod):
                obj = getattr(mod, name)
                try:
                    if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                        mapping[name] = str(py)
                except Exception:
                    continue
        except Exception:
            continue
    return mapping


def _build_param_space_for_class(cls) -> Dict[str, Tuple[float, float]]:
    try:
        if hasattr(cls, 'parameter_space') and cls.parameter_space.__qualname__.split('.')[0] != 'BaseStrategy':
            # Strategy overrides BaseStrategy.parameter_space
            space = cls.parameter_space()
            if isinstance(space, dict) and space:
                return space
    except Exception:
        pass
    # fallback: instantiate and derive from defaults
    try:
        inst = cls()
        params = getattr(inst, 'parameters', {}) or {}
        return BaseStrategy._derive_param_space_from_defaults(params)
    except Exception:
        return {}


def build_registry() -> Dict[str, Dict[str, Tuple[float, float]]]:
    reg: Dict[str, Dict[str, Tuple[float, float]]] = {}
    files = _discover_strategy_files()
    for class_name, _ in files.items():
        try:
            # reload via mapping to get the class object
            # Note: class object was already found; reuse via eval
            # Safer: re-import the module and fetch attribute
            py = Path(files[class_name])
            mod = SourceFileLoader(f"ps2_{py.stem}", str(py)).load_module()
            cls = getattr(mod, class_name, None)
            if cls is None:
                continue
            space = _build_param_space_for_class(cls)
            if space:
                reg[class_name] = space
        except Exception:
            continue
    return reg


# Build at import
PARAMETER_SPACES: Dict[str, Dict[str, Tuple[float, float]]] = build_registry()


def get_param_space_for(class_name: str) -> Dict[str, Tuple[float, float]]:
    return PARAMETER_SPACES.get(class_name, {})
