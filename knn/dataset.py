import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json


class Embeddings(Dataset):
    def __init__(
        self,
        embed_dir: str,
        split: str,
        embed_type: str,
    ):
        assert embed_type in ["clip", "video"], "encoder_name must be either clip or video"
        self.embed_dir = os.path.join(embed_dir, split)
        self.embed_type = embed_type
        self.file_names = [f[:-5] for f in os.listdir(self.embed_dir) if f.endswith(".json")]
        self.split = split
        self.embed_dim = self.__getitem__(0)[0].shape[-1]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.file_names[idx]
        file_path = os.path.join(self.embed_dir, file_name)
        label_dict = json.load(open(file_path + ".json", "r"))
        if self.embed_type == "clip":
            sample = np.load(file_path + "_c.npy")
            num_clips = sample.shape[0]
        elif self.embed_type == "video":
            sample = np.load(file_path + "_v.npy")
            num_clips = 1
        else:
            raise ValueError("embed_type must be either clip or video")

        # Convert numpy array to torch tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(int(label_dict["annotation"]))

        return sample, label, num_clips
