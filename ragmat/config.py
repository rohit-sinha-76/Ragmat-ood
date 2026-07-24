"""Config loading, validation, and integrity enforcement for RAGMat-OOD.

All experiment hyperparameters live in YAML config files under configs/.
This module loads them into typed dataclasses and enforces all integrity
rules at load time — no hardcoded hyperparameters anywhere else.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


class ConfigIntegrityError(ValueError):
    """Raised when a config violates a hard integrity rule."""


@dataclass
class CGCNNConfig:
    """Hyperparameters for the Tier-1 CGCNN encoder."""

    hidden_dim: int = 64
    n_conv_layers: int = 3
    cutoff_radius: float = 8.0
    n_gaussian_basis: int = 40
    dropout_rate: float = 0.1
    weight_decay: float = 0.0
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    n_epochs: int = 200
    lr: float = 1e-3
    lr_scheduler: str = "cosine"
    early_stopping_patience: int = 30
    checkpoint_every_epoch: bool = True

    def __post_init__(self) -> None:
        """Coerce numeric values to correct types to handle YAML parsing edge cases."""
        self.hidden_dim = int(self.hidden_dim)
        self.n_conv_layers = int(self.n_conv_layers)
        self.cutoff_radius = float(self.cutoff_radius)
        self.n_gaussian_basis = int(self.n_gaussian_basis)
        self.dropout_rate = float(self.dropout_rate)
        self.weight_decay = float(self.weight_decay)
        self.batch_size = int(self.batch_size)
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        self.n_epochs = int(self.n_epochs)
        self.lr = float(self.lr)
        self.early_stopping_patience = int(self.early_stopping_patience)


@dataclass
class Tier0Config:
    """Hyperparameters for the Tier-0 matminer + sklearn pipeline."""

    composition_featurizer: str = "ElementProperty_magpie"
    structure_featurizer: str = "CrystalNNFingerprint_SiteStatsFingerprint"
    downstream_model: Literal["random_forest", "xgboost"] = "random_forest"
    n_estimators: int = 200
    max_features: float | str = 1.0
    min_samples_leaf: int = 1
    min_samples_split: int = 2


@dataclass
class UQConfig:
    """Uncertainty quantification hyperparameters."""

    mc_dropout_n_passes: int = 30
    conformal_coverage: float = 0.9


@dataclass
class WandbConfig:
    """Weights & Biases logging settings."""

    project: str = "ragmat-ood-tier0"
    entity: str = "YOUR_WANDB_USERNAME"


@dataclass
class ExperimentConfig:
    """Full experiment configuration loaded from a YAML file.

    All integrity rules are enforced during ``from_yaml`` — any violation
    raises ``ConfigIntegrityError`` immediately.
    """

    experiment_name: str = "unnamed"
    tier: Literal[0, 1] = 0
    target_property: Literal["formation_energy", "band_gap"] = "formation_energy"
    split_type: Literal["iid", "family_out", "element_out"] = "iid"
    representation: Literal["matminer", "cgcnn"] = "matminer"
    retrieval_mode: Literal["none", "true_neighbor", "random_control"] = "none"
    fusion_method: Literal["concat", "cross_attention"] = "concat"
    top_k: int = 10
    encoder_property: str = "formation_energy"
    retrieval_index_property: str = "formation_energy"
    gating: bool = False
    cgcnn: CGCNNConfig = field(default_factory=CGCNNConfig)
    tier0: Tier0Config = field(default_factory=Tier0Config)
    uq: UQConfig = field(default_factory=UQConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 42
    device: str = "cuda"

    # Internal — set after loading
    _source_path: Optional[str] = field(default=None, repr=False)
    _config_hash: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ExperimentConfig":
        """Load and validate an experiment config from a YAML file.

        Args:
            config_path: Path to the experiment YAML file.

        Returns:
            Validated ``ExperimentConfig`` instance.

        Raises:
            ConfigIntegrityError: If any integrity rule is violated.
            FileNotFoundError: If the config file does not exist.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Load base defaults first, then override with experiment-specific values
        base_path = config_path.parent / "base.yaml"
        raw: dict = {}
        if base_path.exists():
            with open(base_path) as f:
                raw = yaml.safe_load(f) or {}
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        _deep_merge(raw, overrides)

        # Nested sub-configs
        cgcnn_cfg = CGCNNConfig(**{k: v for k, v in raw.pop("cgcnn", {}).items()})
        tier0_cfg = Tier0Config(**{k: v for k, v in raw.pop("tier0", {}).items()})
        uq_cfg = UQConfig(**{k: v for k, v in raw.pop("uq", {}).items()})
        wandb_cfg = WandbConfig(**{k: v for k, v in raw.pop("wandb", {}).items()})

        # Filter to known top-level fields
        known_fields = {
            "experiment_name",
            "tier",
            "target_property",
            "split_type",
            "representation",
            "retrieval_mode",
            "fusion_method",
            "top_k",
            "gating",
            "encoder_property",
            "retrieval_index_property",
            "seed",
            "device",
        }
        filtered = {k: v for k, v in raw.items() if k in known_fields}

        cfg = cls(
            cgcnn=cgcnn_cfg,
            tier0=tier0_cfg,
            uq=uq_cfg,
            wandb=wandb_cfg,
            **filtered,
        )
        cfg._source_path = str(config_path)
        cfg._config_hash = _hash_config(overrides)

        # ── Integrity rules ───────────────────────────────────────────────
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        """Enforce all hard integrity rules.

        Raises:
            ConfigIntegrityError: On any violation.
        """
        # Rule 1: encoder_property must equal target_property
        if self.encoder_property != self.target_property:
            raise ConfigIntegrityError(
                f"encoder_property '{self.encoder_property}' must equal "
                f"target_property '{self.target_property}'. "
                "Each encoder is trained on one property only."
            )
        # Rule 2: retrieval_index_property must equal target_property
        if self.retrieval_index_property != self.target_property:
            raise ConfigIntegrityError(
                f"retrieval_index_property '{self.retrieval_index_property}' must "
                f"equal target_property '{self.target_property}'. "
                "Never share a FAISS index across property types."
            )
        # Rule 3: tier/representation consistency
        if self.tier == 1 and self.representation != "cgcnn":
            raise ConfigIntegrityError(
                "Tier 1 experiments must use representation='cgcnn'."
            )
        if self.tier == 0 and self.representation not in ("matminer",):
            raise ConfigIntegrityError(
                "Tier 0 experiments must use representation='matminer'."
            )

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary for logging."""
        import dataclasses

        d = dataclasses.asdict(self)
        d.pop("_source_path", None)
        d.pop("_config_hash", None)
        return d

    @property
    def config_hash(self) -> str:
        """MD5 hash of the experiment-specific overrides."""
        return self._config_hash or ""


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge ``overrides`` into ``base`` in-place."""
    for key, val in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _hash_config(cfg_dict: dict) -> str:
    """Return MD5 hex digest of the JSON-serialised config dictionary."""
    canonical = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()
