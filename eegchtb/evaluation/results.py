"""Typed per-run result record with a stable JSON round-trip and config fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def config_fingerprint(config: dict) -> str:
    """A short deterministic hash of a run configuration, for provenance and dedup."""
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunResult:
    """One (dataset, preprocessing, model, protocol, fold, seed) run: metrics, sizes, timings, fingerprint."""

    run_id: str
    dataset: str
    model: str
    preprocessing: str
    protocol: str
    seed: int
    fold: str
    metrics: dict  # metric name -> float
    n_trials: dict  # "train"/"val"/"test" -> int
    timings: dict  # phase -> seconds
    config_fingerprint: str
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "RunResult":
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "RunResult":
        return cls.from_dict(json.loads(text))
