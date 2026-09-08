"""Continuous-signal preprocessing: an ordered Pipeline of named steps run before epoching."""

from __future__ import annotations

from .pipeline import Pipeline, PipelineConfig, StepConfig
from .steps import (
    STEP_REGISTRY,
    BandpassFilter,
    ClipOutliers,
    CurrentSourceDensity,
    HjorthLaplacian,
    PreprocessingStep,
    Rereference,
    Resample,
    Standardize,
    build_step,
    hjorth_laplacian_operator,
    register_step,
)

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "StepConfig",
    "PreprocessingStep",
    "STEP_REGISTRY",
    "build_step",
    "register_step",
    "BandpassFilter",
    "Rereference",
    "Resample",
    "Standardize",
    "ClipOutliers",
    "CurrentSourceDensity",
    "HjorthLaplacian",
    "hjorth_laplacian_operator",
]
