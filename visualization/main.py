import os
from dataset import Embeddings
import torch
from torch.utils.data import DataLoader
import argparse
import random
import logging
import datetime
import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import cm

dict_class_num = {
    "Kinetics400": 400,
    "sthsth-v2": 174,
    "Diving48": 48,
    "MomentsInTime": 305,
}

# arg parser
parser = argparse.ArgumentParser("Visualization of Pre-computed Embeddings")

# Embedding Selection
parser.add_argument("--embeddings_dir", type=str, default="../embeddings/tl_embedding_large/Kinetics400/uniform", help="embeddings directory")
parser.add_argument("--dataset_name", type=str, default="Kinetics400", help="dataset name", choices=["Kinetics400", "sthsth-v2", "Diving48", "MomentsInTime"])
parser.add_argument("--embed_type", type=str, default="video", help="embed type", choices=["clip", "video"])
parser.add_argument("--split", type=str, default="val", help="split: [train, val, test]")

# visualize rule
parser.add_argument("--method", type=str, default="pca", help="visualization method", choices=["pca", "lda", "tsne"])
parser.add_argument("--num_workers", type=int, default=16, help="number of workers")

# class subsampling
parser.add_argument("--class_random_sample", type=int, default=None, help="number of random classes")
parser.add_argument("--class_include", type=int, nargs="+", default=None, help="included classes. e.g. 0 1 2 3 4")
parser.add_argument("--class_exclude", type=int, nargs="+", default=None, help="excluded classes. e.g. 0 1 2 3 4")

# point subsampling
parser.add_argument("--point_random_sample", type=float, default=None, help="random sample ratio")

# visualization
parser.add_argument("--legend", action="store_true", help="store legend")

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


def main():
    # Logging
    log_file_path = f'../results/visualization--{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/main.log'
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_file_path, logging.INFO)

    # Log args line by line
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    assert (
        sum([args.class_random_sample is not None, args.class_include is not None, args.class_exclude is not None]) <= 1
    ), "Only one class subsampling method is allowed."

    # Load dataset
    embedding_dataset = Embeddings(
        embed_dir=args.embeddings_dir,
        split=args.split,
        embed_type=args.embed_type,
    )

    embed_dim = embedding_dataset.embed_dim
    logging.info(f"Embedding Dimension: {embed_dim}")
    num_classes = dict_class_num[args.dataset_name]

    embedding_loader = DataLoader(
        embedding_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, drop_last=False
    )
    all_embeddings = []
    all_labels = []

    included_classes = []
    if args.class_random_sample is not None:
        included_classes = random.sample(range(num_classes), args.class_random_sample)
    elif args.class_include is not None:
        included_classes = args.class_include
    elif args.class_exclude is not None:
        included_classes = [i for i in range(num_classes) if i not in args.class_exclude]
    else:
        included_classes = list(range(num_classes))

    for samples, labels in tqdm.tqdm(embedding_loader):
        if labels.squeeze().item() not in included_classes:
            continue
        if args.point_random_sample is not None:
            if random.random() > args.point_random_sample:
                continue
        if args.embed_type == "clip":
            samples = samples.squeeze(0)
            labels = labels.repeat(samples.size(0), 1)
        all_embeddings.append(samples)
        all_labels.append(labels)
    vis_num_classes = len(included_classes)

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).squeeze(-1).numpy()

    if args.method == "pca":
        projector = PCA(n_components=2)
        embeddings_proj = projector.fit_transform(all_embeddings)
    elif args.method == "tsne":
        projector = TSNE(n_components=2, random_state=args.seed)
        embeddings_proj = projector.fit_transform(all_embeddings)
    elif args.method == "lda":
        projector = LinearDiscriminantAnalysis(n_components=2)
        embeddings_proj = projector.fit_transform(all_embeddings, all_labels)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    colors = cm.rainbow(np.linspace(0, 1, vis_num_classes))

    plt.figure(figsize=(15, 10))
    for class_idx in range(vis_num_classes):
        indices = all_labels == included_classes[class_idx]
        plt.scatter(
            embeddings_proj[indices, 0],
            embeddings_proj[indices, 1],
            color=colors[class_idx],
            label=f"Class {included_classes[class_idx]}",
            s=5,
        )

    plt.title(f"{args.method} Visualization of {args.dataset_name} {args.embed_type} Embeddings")
    plt.grid(True)
    if args.legend:
        plt.legend()

    plt.savefig(os.path.join(log_dir, f"{args.method}.png"))
    plt.close()


if __name__ == "__main__":
    main()
