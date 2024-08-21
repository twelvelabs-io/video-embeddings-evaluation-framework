# Video Foundation Model Evaluation Framework
In this repo, we provide two things:  
1. Pre-extracted feature vectors obtained using Twelve Labs' video foundation model
2. Pytorch evaluation code to evaluate & utilize the embeddings  

We hope that (1) the published embeddings will help to achieve high performance in various downstream tasks and will be valuable for research, and (2) the evluation source code will be a good baseline code for researchers / developers studying the video foundation models.


## Original Paper
Please refer to our technical report for the further details of the evaluation pipeline.

TBD


## Downstream Tasks & Supported Benchmarks
All results will be saved in `./results` directory.

- [Linear Probing](./linear_probing/)
  - Kinetics-400
  - Something-Something-v2
  - Moments in Time
  - Diving 48
- [K-Nearest-Neighbor](./knn/)
  - Kinetics-400
  - Something-Something-v2
  - Moments in Time
  - Diving 48
- [Temporal Action Localization](./temporal_action_localization/)
  - ActivityNet v1.3
  - THUMOS14
- [Temporal Action Segmentation](./temporal_action_segmentation/)
  - 50Salads
  - Breakfast
  - GTEA
- [Embedding Visualization](./visualization/)
  - Kinetics-400
  - Something-Something-v2
  - Moments in Time
  - Diving 48


## Embedding Description
- Some of the benchmark folders are organized according to how they sample frames (`uniform` or `multi-clip`). If you enter the top-level folder of the dataset, or the directory corresponding to each sampling, you will arrive at the location of the `train` or `train`/`val` folder. The current directory in this state is the `--embeddings_dir` in each downstream task.
- There are three files corresponding to a video.
  - `[video_id].json`
    - This `json` file contains the label corresponding to the video, as well as meta data about the duration, number of frames, and the start and end times of each subclip. Exceptionally, the label for temporal action segmentation utilizes external files rather than this `json` file.
  - `[video_id]_c.npy`
    - This file contains embedding vectors for each subclip of the video in the form (number of subclips) x (dimension).
  - `[video_id]_v.npy`
    - This file contains one embedding vector that represents the entire video. Same as `[video_id]_c.npy` for uniform sampling or when only one clip is defined for the entire video.


## Citation
If you think this project is helpful, please feel free to leave a star and cite our paper:

```
@inproceedings{twelvelabs2024twlv,
  title={TWLV-I: Analysis and Insights from Holistic Evaluation on Video Foundation Models},
  author={Twelve Labs},
  year={2024}
}
```