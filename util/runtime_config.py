import argparse

DATASET_PROFILES = {
    "msds": {
        "data_module": "util.MSDS.data_MSDS",
        "process_mode": "kwargs",
        "description": "MSDS microservice benchmark",
        "random_seed": 42,
        "gpu": True,
        "epochs": 200,
        "patience": 10,
        "learning_rate": 1e-3,
        "weight_decay": 5e-4,
        "learning_change": 500,
        "learning_gamma": 0.5,
        "eval_interval": 1,
        "train_eval_interval": 1,
        "label_weight": 1e-2,
        "label_percent": 0.5,
        "abnormal_weight": 96,
        "rec_down": 1,
        "para_low": 1e-2,
        "feature_node": 4,
        "feature_edge": 4,
        "feature_log": 16,
        "raw_node": 3,
        "raw_edge": 7,
        "log_len": 256,
        "num_heads_edge": 4,
        "num_heads_node": 4,
        "num_heads_log": 4,
        "num_heads_n2e": 4,
        "num_heads_e2n": 2,
        "num_layer": 2,
        "dropout": 0.2,
        "graph_hidden": 16,
        "graph_sparse_weight": 1e-3,
        "graph_update_steps": 2,
        "graph_summary_mode": "last",
        "contrast_weight": 0.1,
        "contrast_temp": 0.1,
        "contrast_proj_dim": 32,
        "contrast_summary_mode": "last",
        "contrast_start_epoch": 1,
        "contrast_warmup": 2,
        "score_fusion_alpha": 0.7,
        "batch_size": 50,
        "window": 10,
        "step": 1,
        "num_nodes": 5,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "max_timesteps": 0,
        "data_path": "./data/MSDS-pre",
        "dataset_path": "./data/MSDS-save",
        "result_dir": "./result",
        "main_model": "Ada-MGAD",
        "evaluate": False,
        "model_path": None,
    },
    "gaia": {
        "data_module": "util.GAIA.data_GAIA",
        "process_mode": "dict",
        "description": "GAIA microservice benchmark",
        "random_seed": 42,
        "gpu": True,
        "epochs": 120,
        "patience": 7,
        "learning_rate": 1e-3,
        "weight_decay": 5e-4,
        "learning_change": 30,
        "learning_gamma": 0.5,
        "eval_interval": 1,
        "train_eval_interval": 1,
        "label_weight": 1e-2,
        "label_percent": 0.5,
        "abnormal_weight": 96,
        "rec_down": 1,
        "para_low": 1e-2,
        "feature_node": 16,
        "feature_edge": 4,
        "feature_log": 8,
        "raw_node": 0,
        "raw_edge": 0,
        "log_len": 0,
        "num_heads_edge": 4,
        "num_heads_node": 4,
        "num_heads_log": 4,
        "num_heads_n2e": 4,
        "num_heads_e2n": 2,
        "num_layer": 2,
        "dropout": 0.2,
        "graph_hidden": 16,
        "graph_sparse_weight": 1e-3,
        "graph_update_steps": 2,
        "graph_summary_mode": "last",
        "contrast_weight": 0.1,
        "contrast_temp": 0.1,
        "contrast_proj_dim": 32,
        "contrast_summary_mode": "last",
        "contrast_start_epoch": 1,
        "contrast_warmup": 2,
        "score_fusion_alpha": 0.7,
        "batch_size": 32,
        "window": 10,
        "step": 1,
        "num_nodes": 10,
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": False,
        "max_timesteps": 0,
        "data_path": "./data/GAIA-pre",
        "dataset_path": "./data/GAIA-save",
        "result_dir": "./result",
        "main_model": "Ada-MGAD",
        "evaluate": False,
        "model_path": None,
    },
}

RUNTIME_OVERRIDE_KEYS = {"model_path", "evaluate", "result_dir", "data_path", "dataset_path"}

ARG_SPECS = [
    ("random_seed", int, "Random seed."),
    ("gpu", "bool", "Use GPU when available."),
    ("epochs", int, "Number of training epochs."),
    ("patience", float, "Early-stop patience."),
    ("learning_rate", float, "Optimizer learning rate."),
    ("weight_decay", float, "Optimizer weight decay."),
    ("learning_change", int, "StepLR change interval."),
    ("learning_gamma", float, "StepLR decay factor."),
    ("eval_interval", int, "Evaluate every N epochs."),
    ("train_eval_interval", int, "Evaluate train metrics every N epochs."),
    ("label_weight", float, "Unknown label weight in reconstruction loss."),
    ("label_percent", float, "Ratio of labeled anomalies."),
    ("abnormal_weight", int, "Positive-class weight."),
    ("rec_down", int, "Reconstruction loss schedule."),
    ("para_low", float, "Minimum reconstruction loss weight."),
    ("feature_node", int, "Embedded node feature dimension."),
    ("feature_edge", int, "Embedded edge feature dimension."),
    ("feature_log", int, "Embedded log feature dimension."),
    ("raw_node", int, "Raw metric feature dimension."),
    ("raw_edge", int, "Raw trace feature dimension."),
    ("log_len", int, "Log feature dimension."),
    ("num_heads_edge", int, "Trace attention heads."),
    ("num_heads_node", int, "Metric attention heads."),
    ("num_heads_log", int, "Log attention heads."),
    ("num_heads_n2e", int, "Node-to-edge attention heads."),
    ("num_heads_e2n", int, "Edge-to-node attention heads."),
    ("num_layer", int, "Number of model layers."),
    ("dropout", float, "Dropout ratio."),
    ("graph_hidden", int, "Dynamic graph learner hidden size."),
    ("graph_sparse_weight", float, "Graph regularization weight."),
    ("graph_update_steps", int, "Refresh dynamic graph every N steps."),
    ("graph_summary_mode", str, "Summary mode for dynamic graph learner."),
    ("contrast_weight", float, "Contrastive loss weight."),
    ("contrast_temp", float, "Contrastive temperature."),
    ("contrast_proj_dim", int, "Projection dimension for contrastive head."),
    ("contrast_summary_mode", str, "Summary mode for contrastive learning."),
    ("contrast_start_epoch", int, "Start contrastive learning at this epoch."),
    ("contrast_warmup", int, "Contrastive warmup epochs."),
    ("score_fusion_alpha", float, "Classification score weight in score fusion."),
    ("batch_size", int, "Batch size."),
    ("window", int, "Sliding-window size."),
    ("step", int, "Sliding-window stride."),
    ("num_nodes", int, "Number of service nodes."),
    ("num_workers", int, "Number of dataloader workers."),
    ("pin_memory", "bool", "Pin host memory in dataloader."),
    ("persistent_workers", "bool", "Keep dataloader workers alive."),
    ("max_timesteps", int, "Limit loaded timesteps or windows; primarily used by GAIA."),
    ("data_path", str, "Path to preprocessed raw data."),
    ("dataset_path", str, "Path to cached windowed data."),
    ("result_dir", str, "Directory for outputs."),
    ("main_model", str, "Model name used for saved artifacts."),
    ("evaluate", "bool", "Evaluate an existing checkpoint directory."),
    ("model_path", str, "Path to an existing checkpoint directory."),
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified training/evaluation entrypoint for the anonymous Ada-MGAD artifact."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_PROFILES.keys()),
        help="Select which dataset profile to run.",
    )
    for name, value_type, help_text in ARG_SPECS:
        parser.add_argument(
            f"--{name}",
            default=argparse.SUPPRESS,
            type=str2bool if value_type == "bool" else value_type,
            help=help_text,
        )
    return parser


def resolve_args():
    parser = build_parser()
    cli_args = vars(parser.parse_args())
    dataset = cli_args["dataset"]
    args = dict(DATASET_PROFILES[dataset])
    args.update(cli_args)
    return args
