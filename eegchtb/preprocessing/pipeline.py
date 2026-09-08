"""Config-driven preprocessing pipelines, loadable from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .steps import PreprocessingStep, build_step

if TYPE_CHECKING:
    from mne.io import BaseRaw


@dataclass
class StepConfig:
    """A pipeline step: a registered name and its parameters."""

    name: str
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "StepConfig":
        return cls(
            name=data["name"],
            params=dict(data.get("params") or {}),
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "params": self.params}


@dataclass
class PipelineConfig:
    """A pipeline described as data."""

    name: str
    steps: list[StepConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        steps = [StepConfig.from_dict(s) for s in data.get("steps") or []]
        return cls(name=data["name"], steps=steps)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
        }


class Pipeline:
    """Builds steps from a PipelineConfig (unknown names fail at construction) and applies them in order."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.steps: list[PreprocessingStep] = [
            build_step(s.name, s.params) for s in config.steps
        ]

    @property
    def name(self) -> str:
        return self.config.name

    def resolved_config(self) -> dict:
        """The pipeline as actually built: every step's full effective params() (including design defaults), unlike config.to_dict(); the faithful provenance record and fingerprint key."""
        return {
            "name": self.name,
            "steps": [
                {"name": step.name, "params": step.params()} for step in self.steps
            ],
        }

    def apply(self, raw: BaseRaw, copy: bool = True) -> BaseRaw:
        """Apply every step in order to a continuous mne.Raw (copied unless copy=False)."""
        working = raw.copy() if copy else raw
        for step in self.steps:
            working = step.apply(working)
        return working

    @classmethod
    def from_dict(cls, data: dict) -> "Pipeline":
        return cls(PipelineConfig.from_dict(data))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Pipeline":
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        step_names = ", ".join(s.name for s in self.config.steps)
        return f"Pipeline(name={self.config.name!r}, steps=[{step_names}])"
