"""Planning utilities for trace-level embedding-model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ViewGeneratorSpec:
    name: str
    token_mask_ratio: float
    token_dropout_ratio: float
    crop_ratio: float
    shuffle: bool


@dataclass
class PositivePairSamplerSpec:
    name: str
    positive_strategy: str
    negative_strategy: str
    use_labels: bool


@dataclass
class SequenceEncoderSpec:
    name: str
    architecture: str
    input_type: str
    token_column: str
    max_len: int
    hidden_dim: int
    embedding_dim: int
    pooling: str


@dataclass
class TrainingObjectiveSpec:
    name: str
    objective_type: str
    contrastive_weight: float
    reconstruction_weight: float
    reconstruction_mask_ratio: float
    temperature: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float


def build_hdfs_component_specs(cfg: dict) -> dict[str, dict]:
    sample_mode = str(cfg["sample_mode"])
    input_type = str(cfg["input_type"])
    token_column = str(cfg["token_column"])
    training_rule = cfg["training_rule"]
    augmentation = cfg["augmentation"]
    encoder = cfg["encoder"]
    optimization = cfg["optimization"]
    objective = cfg.get("objective", {})
    objective_type = str(objective.get("type", "contrastive"))

    if sample_mode != "trace":
        raise ValueError(f"HDFS currently only supports sample_mode='trace', got {sample_mode!r}.")
    if input_type != "event_sequence":
        raise ValueError(f"HDFS currently only supports input_type='event_sequence', got {input_type!r}.")
    if token_column != "sequence":
        raise ValueError(f"HDFS currently expects token_column='sequence', got {token_column!r}.")

    if objective_type not in {"contrastive", "reconstruction", "hybrid"}:
        raise ValueError(
            "HDFS implementation currently supports objective.type in "
            "{'contrastive', 'reconstruction', 'hybrid'}."
        )

    contrastive_weight = float(
        objective.get("contrastive_weight", 1.0 if objective_type in {"contrastive", "hybrid"} else 0.0)
    )
    reconstruction_weight = float(
        objective.get("reconstruction_weight", 1.0 if objective_type in {"reconstruction", "hybrid"} else 0.0)
    )
    if contrastive_weight < 0.0 or reconstruction_weight < 0.0:
        raise ValueError("Objective weights must be non-negative.")
    if contrastive_weight == 0.0 and reconstruction_weight == 0.0:
        raise ValueError("At least one objective weight must be positive.")

    positive_strategy = str(training_rule.get("positive_strategy", "same_sample_two_views"))
    negative_strategy = str(training_rule.get("negative_strategy", "in_batch_different_sample"))
    use_labels = bool(training_rule.get("use_labels", False))
    if contrastive_weight > 0.0:
        if positive_strategy != "same_sample_two_views":
            raise ValueError(
                "HDFS contrastive implementation requires "
                "training_rule.positive_strategy='same_sample_two_views'."
            )
        if negative_strategy != "in_batch_different_sample":
            raise ValueError(
                "HDFS contrastive implementation requires "
                "training_rule.negative_strategy='in_batch_different_sample'."
            )
        if use_labels:
            raise ValueError("HDFS contrastive implementation requires training_rule.use_labels=false.")
    elif use_labels:
        raise ValueError("HDFS SSL implementation does not use labels during training.")

    crop_ratio = float(augmentation.get("crop_ratio", 0.0))
    shuffle = bool(augmentation.get("shuffle", False))
    if crop_ratio > 0.0:
        raise ValueError("HDFS implementation currently expects augmentation.crop_ratio=0.0.")
    if shuffle:
        raise ValueError("HDFS implementation currently expects augmentation.shuffle=false.")

    view_generator = ViewGeneratorSpec(
        name="token_sequence_augmenter",
        token_mask_ratio=float(augmentation.get("token_mask_ratio", 0.15)),
        token_dropout_ratio=float(augmentation.get("token_dropout_ratio", 0.05)),
        crop_ratio=crop_ratio,
        shuffle=shuffle,
    )
    positive_pair_sampler = PositivePairSamplerSpec(
        name="same_sample_two_views_sampler",
        positive_strategy=positive_strategy,
        negative_strategy=negative_strategy,
        use_labels=use_labels,
    )
    encoder_architecture = str(encoder["architecture"])
    if encoder_architecture not in {"tcn", "bigru"}:
        raise ValueError("HDFS SSL implementation currently supports encoder.architecture='tcn' or 'bigru'.")

    sequence_encoder = SequenceEncoderSpec(
        name="trace_sequence_encoder",
        architecture=encoder_architecture,
        input_type=input_type,
        token_column=token_column,
        max_len=int(cfg["max_len"]),
        hidden_dim=int(encoder["hidden_dim"]),
        embedding_dim=int(encoder["embedding_dim"]),
        pooling=str(encoder["pooling"]),
    )
    objective_name_by_type = {
        "contrastive": "in_batch_contrastive_objective",
        "reconstruction": "masked_event_reconstruction_objective",
        "hybrid": "hybrid_ssl_objective",
    }
    objective_type_by_type = {
        "contrastive": "contrastive_nt_xent",
        "reconstruction": "masked_event_cross_entropy",
        "hybrid": "contrastive_nt_xent_plus_masked_event_cross_entropy",
    }
    training_objective = TrainingObjectiveSpec(
        name=objective_name_by_type[objective_type],
        objective_type=objective_type_by_type[objective_type],
        contrastive_weight=contrastive_weight,
        reconstruction_weight=reconstruction_weight,
        reconstruction_mask_ratio=float(
            augmentation.get("reconstruction_mask_ratio", augmentation.get("token_mask_ratio", 0.15))
        ),
        temperature=float(optimization["temperature"]),
        batch_size=int(optimization["batch_size"]),
        epochs=int(optimization["epochs"]),
        learning_rate=float(optimization["lr"]),
        weight_decay=float(optimization.get("weight_decay", 0.0)),
    )

    return {
        "view_generator": asdict(view_generator),
        "positive_pair_sampler": asdict(positive_pair_sampler),
        "sequence_encoder": asdict(sequence_encoder),
        "training_objective": asdict(training_objective),
    }
