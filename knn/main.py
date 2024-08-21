import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import random
import logging
import datetime
from tqdm import tqdm
from dataset import Embeddings

dict_class_num = {
    "Kinetics400": 400,
    "sthsth-v2": 174,
    "Diving48": 48,
    "MomentsInTime": 305,
}

# arg parser
parser = argparse.ArgumentParser("K-Nearest Neighbors with Pre-computed Embeddings")

# Embedding Selection
parser.add_argument("--embeddings_dir", type=str, default="../embeddings/tl_embedding_large/Kinetics400/uniform", help="embeddings directory")
parser.add_argument("--dataset_name", type=str, default="Kinetics400", help="dataset name", choices=["Kinetics400", "sthsth-v2", "Diving48", "MomentsInTime"])
parser.add_argument("--train_embed_type", type=str, default="video", help="training set(=KNN memory) embedding vector type", choices=["clip", "video"])
parser.add_argument("--val_embed_type", type=str, default="video", help="validation set embedding vector type", choices=["clip", "video"])

# training rule
parser.add_argument("--num_neighbors", type=int, default=200, help="number of neighbors")
parser.add_argument("--train_batch_size", type=int, default=1024, help="batch size")
parser.add_argument("--val_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers")

# etc
parser.add_argument("--cuda", type=(lambda x: x.lower() in ["yes", "true", "y"]), default=True, help="use cuda")
parser.add_argument("--seed", type=int, default=12, help="random seed")

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


# https://github.com/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
@torch.no_grad()
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k=200, knn_t=0.1, return_score=False):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    return pred_scores if return_score else pred_scores.argsort(dim=-1, descending=True)


@torch.no_grad()
def main():
    assert os.path.exists(args.embeddings_dir), f"Dataset base directory does not exist: {args.embeddings_dir}"

    # Logging
    log_file_path = f'../results/knn--{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/main.log'
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    setup_logging(log_file_path, logging.INFO)

    # Log args line by line
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    # Set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load dataset
    train_dataset = Embeddings(
        embed_dir=args.embeddings_dir,
        split="train",
        embed_type=args.train_embed_type,
    )

    val_dataset = Embeddings(
        embed_dir=args.embeddings_dir,
        split="val",
        embed_type=args.val_embed_type,
    )

    embed_dim = train_dataset.embed_dim
    logging.info(f"Embedding dimension: {embed_dim}")
    num_classes = dict_class_num[args.dataset_name]

    # for type of "clip", each sample can have different number of clips
    def collate_fn(batch):
        # padding zero frames to match the length of each sample
        samples, labels, num_clips = zip(*batch)
        max_len = max([s.size(0) for s in samples])
        samples = [F.pad(s, (0, 0, 0, max_len - s.size(0)), value=0) for s in samples]
        samples = torch.stack(samples, dim=0)
        labels = torch.stack(labels, dim=0)

        return samples, labels, num_clips

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=(collate_fn if args.train_embed_type == "clip" else None),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=(collate_fn if args.val_embed_type == "clip" else None),
    )

    feature_bank, gt_bank = [], []
    for samples, labels, num_clips in tqdm(train_loader, desc="Building feature bank"):
        if args.cuda:
            samples, labels = samples.cuda(), labels.cuda()

        if args.train_embed_type == "clip":
            for b_idx in range(samples.size(0)):
                for c_idx in range(num_clips[b_idx]):
                    feature_bank.append(F.normalize(samples[b_idx : b_idx + 1, c_idx], dim=1))
                    gt_bank.append(labels[b_idx : b_idx + 1])
        else:
            feature_bank.append(F.normalize(samples, dim=1))
            gt_bank.append(labels)

    feature_bank, gt_bank = torch.cat(feature_bank, dim=0).t().contiguous(), torch.cat(gt_bank, dim=0).contiguous()
    total_top1, total_num = 0, 0

    logging.info("feature bank shape: " + str(feature_bank.shape))
    logging.info("gt bank shape: " + str(gt_bank.shape))

    for samples, labels, num_clips in tqdm(val_loader, desc="Validation"):
        if args.cuda:
            samples = samples.cuda()
            labels = labels.cuda()

        if args.val_embed_type == "clip":
            for b_idx in range(samples.shape[0]):
                clips = F.normalize(samples[b_idx, : num_clips[b_idx]], dim=1)
                pred_scores = knn_predict(
                    clips, feature_bank, gt_bank, num_classes, knn_k=args.num_neighbors, return_score=True
                )
                pred_scores = pred_scores.sum(dim=0, keepdim=True)
                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += (pred_labels[:, 0] == labels[b_idx]).float().sum().item()
            total_num += samples.shape[0]
        else:
            feature = F.normalize(samples, dim=1)
            pred_labels = knn_predict(feature, feature_bank, gt_bank, num_classes, knn_k=args.num_neighbors)
            total_num += feature.size(0)
            total_top1 += (pred_labels[:, 0] == labels).float().sum().item()

    acc = total_top1 / total_num * 100
    logging.info(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
