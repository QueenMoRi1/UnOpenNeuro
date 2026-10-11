"""Loads config.yaml on top of built-in defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "llm": {
        "base_url": "http://localhost:11434",
        "model": "qwen3:4b",
        "temperature": 0.9,
        "timeout": 120,
    },
    "persona": {"name": "Neuro", "creator": "Vedal", "extra": ""},
    "neuro_api": {
        "host": "127.0.0.1",
        "port": 8000,
        "max_retries": 3,
        "idle_action_interval": 0,
        "context_window": 40,
    },
    "voice": {
        "tts": "none",
        "kokoro_voice": "af_heart",
        "pitch": 1.2,
        "whisper_model": "base.en",
    },
    "vtube_studio": {
        "enabled": False,
        "url": "ws://localhost:8001",
        "token": "",
        "mouth_parameter": "MouthOpen",
    },
}


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = "config.yaml") -> dict[str, Any]:
    config = copy.deepcopy(DEFAULTS)
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            _merge(config, yaml.safe_load(f) or {})
    return config


def save_value(path: str | Path, section: str, key: str, value: Any) -> None:
    """Write one value back into config.yaml, creating the file if needed."""
    path = Path(path)
    data = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    data.setdefault(section, {})[key] = value
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
