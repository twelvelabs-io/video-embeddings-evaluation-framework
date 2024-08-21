# python imports
import argparse
import os
import time
import datetime

# torch imports
import json
from torch import nn
import logging

# our code
from libs.core import load_config
from libs.core.config import _update_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (
    train_one_epoch,
    valid_one_epoch,
    ANETdetection,
    make_optimizer,
    make_scheduler,
    fix_random_seed,
    ModelEma,
)

# arg parser
parser = argparse.ArgumentParser(description="Temporal Action Localization Evaluation")

# Embedding Selection
parser.add_argument("--embeddings_dir", type=str, default="../embeddings/tl_embedding_large/ActivityNet_1.3", help="embeddings directory")
parser.add_argument("--dataset_name", type=str, default="ActivityNet_1.3", help="dataset name", choices=["ActivityNet_1.3", "THUMOS14"])
parser.add_argument("--embed_dim", type=int, default=1024, help="embedding dimension")

# Training Rule
parser.add_argument("--train_batch_size", type=int, default=None, help="train batch size")
parser.add_argument("--num_workers", type=int, default=None, help="number of workers")
parser.add_argument("--base_lr", type=float, default=None, help="learning rate")
parser.add_argument("--epochs", type=int, default=None, help="number of epochs")
parser.add_argument("--warmup_epochs", type=int, default=5, help="number of warmup epochs")

# TAD Setting
parser.add_argument("--print_freq", default=10, type=int, help="print frequency (default: 10 iterations)")
parser.add_argument("--with_ext_classifier", action="store_true", help="train classification head")
parser.add_argument("--save_predictions", action="store_true", help="save result json for analysis")

# etc
parser.add_argument("--seed", type=int, default=1212, help="random seed")

args = parser.parse_args()


def setup_logging(log_file, level, include_host=False):
    if include_host:
        import socket

        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f"%(asctime)s |  {hostname} | %(levelname)s | %(message)s", datefmt="%Y-%m-%d,%H:%M:%S"
        )
    else:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d,%H:%M:%S")

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # Logging
    log_file_path = (
        f'../results/temporal_action_localization--{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/main.log'
    )
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_file_path, logging.INFO)

    # Log args line by line
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    # parse args
    args.start_epoch = 0
    config_path = f"configs/{args.dataset_name}.yaml"
    if os.path.isfile(config_path):
        cfg = load_config(config_path)
    else:
        logging.error("Config file does not exist.")
        raise ValueError("Config file does not exist.")

    cfg["init_rand_seed"] = args.seed
    cfg["dataset"]["feat_folder"] = args.embeddings_dir
    cfg["dataset"]["feature_type"] = "c"

    if args.base_lr is not None:
        cfg["opt"]["learning_rate"] = args.base_lr
    if args.epochs is not None:
        cfg["opt"]["epochs"] = args.epochs
    if args.warmup_epochs is not None:
        cfg["opt"]["warmup_epochs"] = args.warmup_epochs
    if args.train_batch_size is not None:
        cfg["loader"]["batch_size"] = args.train_batch_size
    if args.num_workers is not None:
        cfg["loader"]["num_workers"] = args.num_workers

    if args.with_ext_classifier:
        logging.info("Use classification head ...")
        cfg["test_cfg"]["ext_score_file"] = None
        cfg["test_cfg"]["multiclass_nms"] = True
    else:
        logging.info("Regression only ...")
        cfg["dataset"]["num_classes"] = 1
    cfg = _update_config(cfg)

    logging.info(cfg)

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg["init_rand_seed"], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg["opt"]["learning_rate"] *= len(cfg["devices"])
    cfg["loader"]["num_workers"] *= len(cfg["devices"])

    """2. create dataset / dataloader"""
    train_dataset, embed_dim = make_dataset(cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"])
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg["model"]["train_cfg"]["head_empty_cls"] = train_db_vars["empty_label_ids"]
    cfg["model"]["input_dim"] = embed_dim

    # data loaders
    train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg["loader"])

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg["devices"])
    # optimizer
    optimizer = make_optimizer(model, cfg["opt"])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg["opt"], num_iters_per_epoch)

    # enable model EMA
    logging.info("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. training / validation loop"""
    logging.info("Start training model {:s} ...".format(cfg["model_name"]))

    # start training
    max_epochs = cfg["opt"].get("early_stop_epochs", cfg["opt"]["epochs"] + cfg["opt"]["warmup_epochs"])
    assert max_epochs > 5, "Too few epochs for training!"

    """2. create dataset / dataloader"""
    val_dataset, _ = make_dataset(cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"])
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(val_dataset, False, None, 1, cfg["loader"]["num_workers"])

    best_mAP = -1
    best_APs = None
    best_epoch = None
    best_results = None
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg["train_cfg"]["clip_grad_l2norm"],
            tb_writer=None,
            print_freq=args.print_freq,
        )

        if epoch >= 5:
            # model
            model_eval = make_meta_arch(cfg["model_name"], **cfg["model"])
            # not ideal for multi GPU training, ok for now
            model_eval = nn.DataParallel(model_eval, device_ids=cfg["devices"])

            model_eval.load_state_dict(model_ema.module.state_dict())

            # set up evaluator
            det_eval, output_file = None, None
            val_db_vars = val_dataset.get_attributes()
            det_eval = ANETdetection(
                val_dataset.json_file, val_dataset.split[0], tiou_thresholds=val_db_vars["tiou_thresholds"]
            )

            """5. Test the model"""
            logging.info("Start testing model {:s} ...".format(cfg["model_name"]))
            start = time.time()
            APs, results = valid_one_epoch(
                val_loader,
                model_eval,
                -1,
                evaluator=det_eval,
                output_file=output_file,
                ext_score_file=cfg["test_cfg"]["ext_score_file"],
                tb_writer=None,
                print_freq=args.print_freq,
            )
            mAP = APs.mean()
            end = time.time()

            logging.info("All done! Total time: {:0.2f} sec".format(end - start))

            if mAP >= best_mAP:
                best_mAP = mAP
                best_APs = APs
                best_epoch = epoch
                best_results = results

    # wrap up
    logging.info("All done!")

    if args.save_predictions:
        logging.info("Save result json ...")
        result_dict = dict(
            {
                "version": "VERSION 1.3",
                "results": dict(),
                "external_data": {
                    "used": True,
                    "details": "3D-CNN for feature extracting is pre-trained on Kinetics-400",
                },
            }
        )

        for r_i in range(len(best_results["video-id"])):
            video_id = best_results["video-id"][r_i]
            start_time = best_results["t-start"][r_i].item()
            end_time = best_results["t-end"][r_i].item()
            label = best_results["label"][r_i].item()
            label = val_dataset.label_id_dict[label]
            score = best_results["score"][r_i].item()

            if video_id not in result_dict["results"].keys():
                result_dict["results"][video_id] = list()

            result_dict["results"][video_id].append({"label": label, "score": score, "segment": (start_time, end_time)})

        if args.with_ext_classifier:
            result_json_path = os.path.join(
                "results", args.dataset_name, args.encoder_name, "results_with_ext_classifier.json"
            )
        else:
            result_json_path = os.path.join("results", args.dataset_name, args.encoder_name, "results_reg_only.json")
        if not os.path.exists(os.path.dirname(result_json_path)):
            os.makedirs(os.path.dirname(result_json_path))

        with open(result_json_path, "w") as fp:
            json.dump(result_dict, fp, indent=4, sort_keys=True)

    # print the best results
    logging.info("[RESULTS] Action detection results on {:s} at {:d} Epoch.".format(cfg["dataset_name"], best_epoch))
    block = ""
    for tiou, tiou_mAP in zip(det_eval.tiou_thresholds, best_APs):
        block += "\n|tIoU = {:.2f}: mAP = {:.2f} (%)".format(tiou, tiou_mAP * 100)
    logging.info(block)
    logging.info("Avearge mAP: {:.2f} (%)".format(best_mAP * 100))

    return


if __name__ == "__main__":
    main(args)
