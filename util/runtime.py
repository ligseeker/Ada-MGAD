import importlib
import logging
import os
import warnings

import util.train as train
import util.util as util
from torch.utils.data import DataLoader

from util.runtime_config import DATASET_PROFILES, RUNTIME_OVERRIDE_KEYS, resolve_args

warnings.filterwarnings("ignore")


def load_process(args):
    module = importlib.import_module(args["data_module"])
    if args["process_mode"] == "dict":
        return module.Process(args)
    return module.Process(**args)


def prepare_args(args):
    if args["evaluate"]:
        stored_args = util.read_params(args)
        if "dataset" not in stored_args:
            dataset_hint = str(stored_args.get("dataset_path", "")).lower()
            stored_args["dataset"] = "gaia" if "gaia" in dataset_hint else args["dataset"]
        for key, value in stored_args.items():
            if key not in RUNTIME_OVERRIDE_KEYS:
                args[key] = value
        args["result_dir"] = args["model_path"]
        args["hash_id"] = os.path.basename(args["model_path"])
    else:
        args["hash_id"], args["result_dir"] = util.dump_params(args)
        util.json_pretty_dump(args, os.path.join(args["result_dir"], "params.json"))
        args["model_path"] = args["result_dir"]

    dataset_profile = DATASET_PROFILES[args["dataset"]]
    args["data_module"] = dataset_profile["data_module"]
    args["process_mode"] = dataset_profile["process_mode"]
    util.seed_everything(args["random_seed"])
    return args


def build_dataloaders(processed, args):
    loader_kwargs = {
        "batch_size": args["batch_size"],
        "num_workers": args["num_workers"],
        "pin_memory": bool(args["pin_memory"] and args["gpu"]),
        "drop_last": True,
    }
    if args["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = args["persistent_workers"]

    split_idx = int(len(processed.dataset) * 0.7)
    train_dl = DataLoader(processed.dataset[:split_idx], shuffle=True, **loader_kwargs)
    test_dl = DataLoader(processed.dataset[split_idx:], shuffle=False, **loader_kwargs)
    return train_dl, test_dl


def evaluate_and_log(system, test_dl, args):
    logging.info("calculate scores...")
    summary_path = os.path.join(args["result_dir"], "result_summary.log")
    with open(summary_path, "a+", encoding="utf-8") as file:
        file.writelines(
            f"\n{args['main_model']}-{args['hash_id']} --dataset:{args['dataset']} "
            f"--weight_decay:{args['weight_decay']} --learning_change:{args['learning_change']}\n"
        )
        for statue in ["loss", "f1"]:
            logging.info(f"calculate label with {statue}...")
            system.load_model(args["model_path"], name=statue)
            info = system.evaluate(test_dl, isFinall=True)
            file.writelines(statue + "   " + info + "\n")


def run(args):
    args = prepare_args(args)

    logging.info(
        "---- Model: ----%s-%s----train : %s----evaluate : %s----dataset : %s",
        args["main_model"],
        args["hash_id"],
        not args["evaluate"],
        args["evaluate"],
        args["dataset"],
    )

    processed = load_process(args)
    if args["dataset"] == "gaia":
        logging.info(
            "Actual dimensions: raw_node=%s, log_len=%s, raw_edge=%s",
            args["raw_node"],
            args["log_len"],
            args["raw_edge"],
        )
    train_dl, test_dl = build_dataloaders(processed, args)

    import src.model as model

    models = model.MyModel(processed.graph, **args)
    system = train.MY(models, **args)

    if not args["evaluate"]:
        system.fit(train_loader=train_dl, test_loader=test_dl)

    evaluate_and_log(system, test_dl, args)
    logging.info(
        "^^^^^^ Current Model: ----%s----%s ^^^^^",
        args["main_model"],
        args["hash_id"],
    )


def main():
    run(resolve_args())
