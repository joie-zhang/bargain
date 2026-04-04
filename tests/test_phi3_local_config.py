#!/usr/bin/env python3
"""Regression checks for the local Phi-3 Princeton-cluster route."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strong_models_experiment.configs import STRONG_MODELS_CONFIG


def test_phi3_uses_local_cluster_weights():
    config = STRONG_MODELS_CONFIG["Phi-3-mini-128k-instruct"]

    assert config["api_type"] == "princeton_cluster"
    assert config["model_id"] == "microsoft/phi-3-mini-128k-instruct"
    assert config["local_path"] == "/scratch/gpfs/DANQIC/models/Phi-3-mini-128k-instruct"
