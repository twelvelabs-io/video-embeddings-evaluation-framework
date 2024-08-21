import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
import mmengine

import pickle

import io
import logging


@register_dataset("thumos_new")
class THUMOS14Dataset(Dataset):
    def __init__(
        self,
        is_training,  # if in training mode
        split,  # split, a tuple/list allowing concat of subsets
        feat_folder,  # folder for features
        json_file,  # json file for annotations
        downsample_rate,  # downsample rate for feats
        max_seq_len,  # maximum sequence length during training
        trunc_thresh,  # threshold for truncate an action segment
        crop_ratio,  # a tuple (e.g., (0.9, 1.0)) for random cropping
        num_classes,  # number of action categories
        file_prefix,  # feature file prefix if any
        file_ext,  # feature file extension if any
        force_upsampling,  # force to upsample to max_seq_len
        feature_type,  # feature type (e.g., 'c', 'v')
        clip_length,  # clip length in seconds
        overlap_ratio,  # overlap ratio between clips
    ):
        # file path
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio is None or len(crop_ratio) == 2
        self.feature_type = feature_type
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ""

        self.file_ext = file_ext
        self.json_file = json_file
        self.clip_length = clip_length
        self.overlap_ratio = overlap_ratio

        # split / training mode
        self.split = split
        self.is_training = is_training

        self.force_upsampling = force_upsampling

        # features meta info
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert (num_classes == 1) or (len(label_dict) == num_classes)
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            "dataset_name": "thumos-14",
            "tiou_thresholds": np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            "empty_label_ids": [],
        }
        self.embed_dim = self.__getitem__(0)["feats"].shape[0]
        logging.info(f"Loaded THUMOS14 dataset {json_file} with {len(self.data_list)} videos. ")

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, "r") as fid:
            json_data = json.load(fid)
        json_db = json_data["database"]

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value["annotations"]:
                    label_dict[act["label"]] = act["label_id"]

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value["subset"].lower() not in self.split:
                continue

            subset_folder = "train" if value["subset"].lower() == "validation" else "val"
            config_path = os.path.join(self.feat_folder, subset_folder, self.file_prefix + key + ".json")
            # skip the video if not in the split
            if value["subset"].lower() not in self.split or not os.path.exists(config_path):
                continue

            # load feature info
            with open(config_path, "r") as fp:
                feature_config = json.load(fp)

            fps = feature_config["num_frames"] / feature_config["duration"]
            num_frames = fps * self.clip_length
            feat_stride = fps * ((1 - self.overlap_ratio) * self.clip_length)

            duration = feature_config["duration"]

            # get annotations if available
            if ("annotations" in value) and (len(value["annotations"]) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value["annotations"]:
                    segments.append(act["segment"])
                    # labels.append([label_dict[act['label']]])
                    if self.num_classes == 1:
                        labels.append(
                            [
                                0,
                            ]
                        )
                    else:
                        labels.append([label_dict[act["label"]]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            dict_db += (
                {
                    "id": key,
                    "fps": fps,
                    "subset": subset_folder,
                    "duration": duration,
                    "segments": segments,
                    "labels": labels,
                    "num_frames": num_frames,
                    "feat_stride": feat_stride,
                },
            )

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(
            self.feat_folder,
            video_item["subset"],
            self.file_prefix + video_item["id"] + "_" + self.feature_type + self.file_ext,
        )
        data = io.BytesIO(mmengine.get(filename))
        if "npy" in self.file_ext:
            feats = np.load(data).astype(np.float32)
        elif "pt" in self.file_ext:
            feats = torch.load(data).numpy().astype(np.float32)
        else:
            raise NotImplementedError

        video_num_frames = video_item["num_frames"]
        video_feat_stride = video_item["feat_stride"]

        # deal with downsampling (= increased feat stride)
        feats = feats[:: self.downsample_rate, :]
        feat_stride = video_feat_stride * self.downsample_rate

        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item["segments"] is not None:
            segments = torch.from_numpy(
                (video_item["segments"] * video_item["fps"] - 0.5 * video_num_frames) / feat_stride
            )
            labels = torch.from_numpy(video_item["labels"])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {
            "video_id": video_item["id"],
            "feats": feats,  # C x T
            "segments": segments,  # N x 2
            "labels": labels,  # N
            "fps": video_item["fps"],
            "duration": video_item["duration"],
            "feat_stride": feat_stride,
            "feat_num_frames": video_num_frames,
        }

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)

        return data_dict
