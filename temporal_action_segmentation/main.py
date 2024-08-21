import os
import argparse
import copy
import random
import logging
import datetime
import torch

from modeling.model import MyTransformer, Trainer
from utils.batch_gen import BatchGenerator, SegDataset
from utils.eval import eval

# arg parser
parser = argparse.ArgumentParser("Temporal Action Segmentation Evaluation")

# Embedding Selection
parser.add_argument("--embeddings_dir", type=str, default="../embeddings/tl_embedding_large/GTEA", help="embeddings directory")
parser.add_argument("--dataset_name", type=str, default="GTEA", help="dataset name", choices=["50Salads", "GTEA", "Breakfast"])

# TAS Setting
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate for training")
parser.add_argument("--batch_size", type=int, default=1, help="batch size for training")
parser.add_argument("--epochs", type=int, default=120, help="number of epochs for training")
parser.add_argument("--num_layers", type=int, default=8, help="number of layers in transformer")
parser.add_argument("--num_f_maps", type=int, default=256, help="number of feature maps in transformer")
parser.add_argument("--num_workers", type=int, default=8, help="number of workers for data loader")
parser.add_argument("--sample_rate", type=int, default=1, help="sample rate for features")
parser.add_argument("--channel_mask_rate", type=float, default=0.3, help="channel mask rate for transformer")
parser.add_argument("--save_vis", action="store_true", help="save visualization")

# etc
parser.add_argument("--cuda", type=(lambda x: x.lower() in ["yes", "true", "y"]), default=True, help="use cuda")
parser.add_argument("--seed", default=1212)

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


def main():
    # Logging
    log_file_path = f'../results/action-segmentation--{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/main.log'
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_file_path, logging.INFO)

    # Log args line by line
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.dataset_name == "50Salads":
        # sample input features @ 15fps instead of 30 fps
        # for 50salads, and up-sample the output to 30 fps
        args.sample_rate = 2

    if args.dataset_name == "Breakfast":
        args.lr = 0.0001

    logging.info(args)

    cnt_split_dict = {"50Salads": 5, "GTEA": 4, "Breakfast": 4}

    recognition = list()
    for split in range(1, cnt_split_dict[args.dataset_name] + 1):
        gt_path = os.path.join("annotations", args.dataset_name, "groundTruth")
        mapping_file = os.path.join("annotations", args.dataset_name, "mapping.txt")
        vid_list_file = os.path.join("annotations", args.dataset_name, "splits/train.split" + str(split) + ".bundle")
        vid_list_file_tst = os.path.join("annotations", args.dataset_name, "splits/test.split" + str(split) + ".bundle")
        features_path = os.path.join(os.path.abspath(args.embeddings_dir))

        with open(mapping_file, "r") as fp:
            actions = fp.read().split("\n")[:-1]
        actions_dict = dict()
        for a in actions:
            actions_dict[a.split()[1]] = int(a.split()[0])
        index2label = dict()
        for k, v in actions_dict.items():
            index2label[v] = k
        num_classes = len(actions_dict)

        # Training
        batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, args.sample_rate)
        batch_gen.read_data(vid_list_file)
        seg_dataset = SegDataset(batch_gen)
        data_loader = torch.utils.data.DataLoader(
            seg_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=seg_dataset.collate_fn,
            pin_memory=True,
        )

        embed_dim = seg_dataset[0][0].shape[0]
        model = MyTransformer(
            3,
            args.num_layers,
            2,
            2,
            args.num_f_maps,
            embed_dim,
            num_classes,
            args.channel_mask_rate,
            attn_type="sliding_att" if args.dataset_name != "Breakfast" else "normal_att",
        )
        device = torch.device("cuda" if args.cuda else "cpu")
        model.to(device)

        trainer = Trainer(model, num_classes)

        trainer.train(data_loader, args.epochs, args.batch_size, args.lr, split)

        # Inference
        batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, args.sample_rate)
        batch_gen_tst.read_data(vid_list_file_tst)
        seg_dataset_tst = SegDataset(batch_gen_tst)
        data_loader_tst = torch.utils.data.DataLoader(
            seg_dataset_tst,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=seg_dataset_tst.collate_fn,
            pin_memory=True,
        )
        if args.save_vis:
            results_dir = os.path.join(log_dir, args.dataset_name, "split-" + str(split))
            os.makedirs(results_dir, exist_ok=True)
        else:
            results_dir = None
        trainer.predict(data_loader_tst, actions_dict, args.sample_rate, split, results_dir=results_dir)
        recognition.append(copy.deepcopy(trainer.recognition))

    # Evaluation
    eval(recognition, args.dataset_name, cnt_split_dict)


if __name__ == "__main__":
    main()
