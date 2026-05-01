from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_PROFILE_DIR = Path("profiles")


class ProfileVerification(BaseModel):
    required_cache_capability: str | None = None
    expected_context_length: int | None = None
    startup_timeout_seconds: int = 180


class LlamaServerProfile(BaseModel):
    name: str
    description: str = ""
    model_path: str
    alias: str = "local"
    context_size: int = Field(gt=0)
    cache_type_k: str
    cache_type_v: str
    gpu_layers: int = 99
    extra_args: list[str] = Field(default_factory=list)
    verification: ProfileVerification = Field(default_factory=ProfileVerification)

    @property
    def is_turboquant(self) -> bool:
        return self.cache_type_k.startswith("turbo") or self.cache_type_v.startswith("turbo")

    def resolved_model_path(self, model_root: Path) -> Path:
        path = Path(self.model_path)
        if path.is_absolute() or str(path).startswith("/"):
            return path
        return model_root / path


def _profile_path(name: str, profiles_dir: Path) -> Path:
    return profiles_dir / f"{name}.yaml"


def load_profile(name: str, profiles_dir: Path | str = DEFAULT_PROFILE_DIR) -> LlamaServerProfile:
    base_dir = Path(profiles_dir)
    path = _profile_path(name, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Unknown qllama profile '{name}' at {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}

    if "name" not in payload:
        payload["name"] = name

    return LlamaServerProfile.model_validate(payload)


def list_profiles(profiles_dir: Path | str = DEFAULT_PROFILE_DIR) -> list[str]:
    base_dir = Path(profiles_dir)
    if not base_dir.exists():
        return []

    return sorted(path.stem for path in base_dir.glob("*.yaml"))
