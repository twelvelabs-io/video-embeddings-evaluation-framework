import os
import json
import h5py
import numpy as np
from json import JSONDecodeError

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations
import mmengine
import io
import logging


def remove_unuseful_annotations(ants):
    # remove duplicate annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e, l = event["segment"][0], event["segment"][1], event["label_id"]
        if s < e:
            valid_events.append(event)
    return valid_events


@register_dataset("anet")
class ActivityNetDataset(Dataset):
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
        assert mmengine.exists(feat_folder) and mmengine.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio is None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.use_hdf5 = ".hdf5" in feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ""
        self.feature_type = feature_type
        self.file_ext = file_ext
        self.json_file = json_file
        self.clip_length = clip_length
        self.overlap_ratio = overlap_ratio

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split
        self.is_training = is_training
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        # proposal vs action categories
        assert (num_classes == 1) or (len(label_dict) == num_classes)
        self.data_list = dict_db
        self.label_dict = label_dict
        self.label_id_dict = {v: k for k, v in label_dict.items()}

        # dataset specific attributes
        self.db_attributes = {
            "dataset_name": "ActivityNet 1.3",
            "tiou_thresholds": np.linspace(0.5, 0.95, 10),
            "empty_label_ids": [],
        }
        self.embed_dim = self.__getitem__(0)["feats"].shape[0]
        logging.info(f"Loaded ActivityNet dataset {json_file} with {len(self.data_list)} videos. ")

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
            if value["subset"].lower() not in self.split:
                continue
            subset_folder = "train" if value["subset"] == "training" else "val"
            config_path = os.path.join(self.feat_folder, subset_folder, self.file_prefix + key + ".json")
            # skip the video if not in the split
            if value["subset"].lower() not in self.split or not os.path.exists(config_path):
                continue

            # load feature info
            try:
                with open(config_path, "r") as fp:
                    feature_config = json.load(fp)
            except JSONDecodeError:
                logging.error(f"Error json loading {config_path}")
                continue

            fps = feature_config["num_frames"] / feature_config["duration"]
            num_frames = fps * self.clip_length
            feat_stride = fps * ((1 - self.overlap_ratio) * self.clip_length)

            duration = feature_config["duration"]

            # get annotations if available
            if ("annotations" in value) and (len(value["annotations"]) > 0):
                valid_acts = remove_duplicate_annotations(value["annotations"])
                valid_acts = remove_unuseful_annotations(valid_acts)
                num_acts = len(valid_acts)
                if num_acts > 0:
                    segments = np.zeros([num_acts, 2], dtype=np.float32)
                    labels = np.zeros(
                        [
                            num_acts,
                        ],
                        dtype=np.int64,
                    )
                    for idx, act in enumerate(valid_acts):
                        segments[idx][0] = act["segment"][0]
                        segments[idx][1] = act["segment"][1]
                        if self.num_classes == 1:
                            labels[idx] = 0
                        else:
                            labels[idx] = label_dict[act["label"]]
                else:
                    segments = None
                    labels = None
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
        # directly return a (truncated) annotations point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the annotations
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

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if video_feat_stride > 0 and (not self.force_upsampling):
            # var length features
            feat_stride, num_frames = video_feat_stride, video_num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                feats = feats[:: self.downsample_rate, :]
                feat_stride = video_feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        elif video_feat_stride > 0 and self.force_upsampling:
            feat_stride = float((feats.shape[0] - 1) * video_feat_stride + video_num_frames) / self.max_seq_len
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:
                # reset to max_seq_len
                seq_len = self.max_seq_len
            feat_stride = video_item["duration"] * video_item["fps"] / seq_len
            # center the features
            num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride

        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        # print(feats.shape,flush=True)

        # resize the features if needed
        if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
            resize_feats = F.interpolate(feats.unsqueeze(0), size=self.max_seq_len, mode="linear", align_corners=False)
            feats = resize_feats.squeeze(0)

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item["segments"] is not None:
            segments = torch.from_numpy(video_item["segments"] * video_item["fps"] / feat_stride - feat_offset)
            labels = torch.from_numpy(video_item["labels"])
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                vid_len = feats.shape[1] + feat_offset
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # skip an action outside of the feature map
                        continue
                    # skip an action that is mostly outside of the feature map
                    ratio = (min(seg[1].item(), vid_len) - seg[0].item()) / (seg[1].item() - seg[0].item())
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None

        # return a annotations dict
        data_dict = {
            "video_id": video_item["id"],
            "feats": feats,  # C x T
            "segments": segments,  # N x 2
            "labels": labels,  # N
            "fps": video_item["fps"],
            "duration": video_item["duration"],
            "feat_stride": feat_stride,
            "feat_num_frames": num_frames,
        }

        # no truncation is needed
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)

        return data_dict
