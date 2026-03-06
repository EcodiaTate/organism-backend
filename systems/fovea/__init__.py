"""
Fovea -- Prediction Error as Attention.

EOS's predictive coding layer. Instead of computing salience via weighted
scoring, Fovea computes prediction errors against the world model.
Attention IS prediction error: what the model got wrong is what gets attended to.

Fovea provides prediction error decomposition as the salience signal.
It does not replace Atune -- it provides the error signals that Atune's
workspace broadcasts.
"""

from .habituation import HabituationCompleteInfo, HabituationEngine, SignatureStats
from .integration import DynamicIgnitionThreshold, FoveaAtuneBridge
from .internal import InternalPredictionEngine
from .learning import AttentionWeightLearner
from .precision import PrecisionWeightComputer
from .prediction import FoveaPredictionEngine
from .protocols import LogosWorldModel, LogosWorldModelAdapter, StubWorldModel
from .service import FoveaService
from .types import (
    ActivePrediction,
    ErrorRoute,
    ErrorType,
    FoveaMetrics,
    FoveaPredictionError,
    InternalErrorType,
    InternalPredictionError,
    PerceptContext,
    WorldModelUpdate,
)

__all__ = [
    # Service
    "FoveaService",
    # Core engines (Phase A+B)
    "FoveaPredictionEngine",
    "PrecisionWeightComputer",
    "HabituationEngine",
    # Phase C: Weight learning
    "AttentionWeightLearner",
    "HabituationCompleteInfo",
    "SignatureStats",
    # Phase D: Self-attention
    "InternalPredictionEngine",
    # Integration
    "FoveaAtuneBridge",
    "DynamicIgnitionThreshold",
    # Protocols
    "LogosWorldModel",
    "LogosWorldModelAdapter",
    "StubWorldModel",
    # Types
    "ActivePrediction",
    "ErrorRoute",
    "ErrorType",
    "FoveaMetrics",
    "FoveaPredictionError",
    "InternalErrorType",
    "InternalPredictionError",
    "PerceptContext",
    "WorldModelUpdate",
]
