"""Strong Models Experiment Package - Modular negotiation experiments between state-of-the-art LLMs."""

from .data_models import ExperimentResults, BatchResults
from .configs import STRONG_MODELS_CONFIG

__all__ = [
    'ExperimentResults',
    'BatchResults', 
    'STRONG_MODELS_CONFIG',
    'StrongModelsExperiment'
]


def __getattr__(name):
    """Lazy import heavy runtime dependencies only when requested."""
    if name == "StrongModelsExperiment":
        from .experiment import StrongModelsExperiment
        return StrongModelsExperiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
