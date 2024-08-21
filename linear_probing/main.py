import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import random
import numpy as np
import logging
import datetime
from dataset import Embeddings

dict_class_num = {
    "Kinetics400": 400,
    "sthsth-v2": 174,
    "Diving48": 48,
    "MomentsInTime": 305,
}

# arg parser
parser = argparse.ArgumentParser("Linear Probing with Pre-computed Embeddings")

# Embedding Selection
parser.add_argument("--embeddings_dir", type=str, default="../embeddings/tl_embedding_large/Kinetics400/uniform", help="embeddings directory",)
parser.add_argument("--dataset_name", type=str,default="Kinetics400", help="dataset name", choices=["Kinetics400", "sthsth-v2", "Diving48", "MomentsInTime"])
parser.add_argument("--embed_type", type=str, default="video", help="embed type", choices=["clip", "video"])

# Training Rule
parser.add_argument("--train_batch_size", type=int, default=1024, help="train batch size")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
parser.add_argument("--base_lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--epochs", type=int, default=150, help="number of epochs")
parser.add_argument("--warmup_epochs", type=int, default=10, help="number of warmup epochs")
parser.add_argument("--val_interval", type=int, default=10, help="validation interval")

# etc
parser.add_argument("--cuda", type=(lambda x: x.lower() in ["yes", "true", "y"]), default=True, help="use cuda")
parser.add_argument("--seed", type=int, default=1212, help="random seed")

args = parser.parse_args()


# linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


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


def create_scheduler(optimizer, base_lr, warmup_length, steps, min_lr=0.0):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = base_lr * (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    return _lr_adjuster


def main():
    # Logging
    log_file_path = f'../results/linear_probing--{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/main.log'
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
        embed_type=args.embed_type,
    )

    val_dataset = Embeddings(
        embed_dir=args.embeddings_dir,
        split="val",
        embed_type=args.embed_type,
    )

    embed_dim = train_dataset.embed_dim
    logging.info(f"Embedding Dimension: {embed_dim}")
    num_classes = dict_class_num[args.dataset_name]

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # Load model
    model = LinearClassifier(input_dim=embed_dim, num_classes=num_classes)
    if args.cuda:
        model = model.cuda()

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)

    scheduler = create_scheduler(
        optimizer, args.base_lr, args.warmup_epochs * len(train_loader), args.epochs * len(train_loader)
    )

    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (samples, labels) in enumerate(train_loader):
            if args.cuda:
                samples = samples.cuda()
                labels = labels.cuda()

            # Train the model
            model.train()

            # scheduler step
            scheduler(i + epoch * len(train_loader))

            # Forward pass
            outputs = model(samples)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training progress
            if (i + 1) % 10 == 0:
                logging.info(
                    f'Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}'
                )

        # Validation
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for samples, labels in val_loader:
                    if args.cuda:
                        samples = samples.cuda()
                        labels = labels.cuda()

                    outputs = model(samples)
                    if args.embed_type == "clip":
                        outputs = torch.softmax(outputs, dim=-1).mean(dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                logging.info(f"Epoch [{epoch + 1}/{args.epochs}], Accuracy: {100 * correct / total}")


if __name__ == "__main__":
    main()
