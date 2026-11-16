"""Load config.yaml into the dataclasses used by the rest of the code."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .kinematics import ArmGeometry, JointLimits
from .mapping import MappingConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class SmoothingConfig:
    min_cutoff: float = 1.2
    beta: float = 4.0


@dataclass
class ControlConfig:
    hand_prefer: str = "any"
    lost_timeout_s: float = 0.6
    max_speed_deg_s: dict = field(
        default_factory=lambda: {
            "shoulder_pan": 150.0,
            "shoulder_lift": 150.0,
            "elbow_flex": 150.0,
            "wrist_flex": 200.0,
            "wrist_roll": 250.0,
            "gripper": 300.0,
        }
    )
    max_relative_target: float | None = 25.0


@dataclass
class Config:
    geometry: ArmGeometry = field(default_factory=ArmGeometry)
    limits: JointLimits = field(default_factory=JointLimits)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)


def _build(cls, data: dict | None):
    obj = cls()
    if not data:
        return obj
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        default = getattr(obj, key)
        if isinstance(default, tuple):
            value = tuple(float(v) for v in value)
        elif isinstance(default, dict):
            value = {**default, **value}
        elif isinstance(default, float) and value is not None:
            value = float(value)
        setattr(obj, key, value)
    return obj


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    data = data or {}
    return Config(
        geometry=_build(ArmGeometry, data.get("geometry")),
        limits=_build(JointLimits, data.get("limits")),
        mapping=_build(MappingConfig, data.get("mapping")),
        smoothing=_build(SmoothingConfig, data.get("smoothing")),
        control=_build(ControlConfig, data.get("control")),
    )
