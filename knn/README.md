# K-Neareast Neighbor (Action Recognition)
This source code is for the k-nearest neighbor (knn) evaluation of twelvelabs embeddings

## Embedding Preparation
- Download the embeddings via following links.
- Supported bechmarks
   - Kinetics-400 ([link](https://www.dropbox.com/scl/fi/loo63h944w9kpt2ykfhlp/Kinetics400.tar?rlkey=qhpuyxi6v0rww8y06cetl73cu&st=2uvenn0n&dl=0))
   - Something-Something-v2 ([link](https://www.dropbox.com/scl/fi/oaa53kpu3kw7sr0734737/sthsth-v2.tar?rlkey=so3pwzxe07bg6ymucd6favtoc&st=hx16uuw2&dl=0))
   - Diving-48 ([link](https://www.dropbox.com/scl/fi/mqxmjaffk7vq14j25ie44/Diving48.tar?rlkey=vly2sf22lbhthko9xh1dwrcsh&st=0cqg7snw&dl=0))
   - Moments in Time ([link](https://www.dropbox.com/scl/fi/u0serudpkicaat97nhqvk/MomentsInTime.tar?rlkey=8jfz6scbffm609ap51axz3zrc&st=e44racfr&dl=0))

## Run
After preparing the embeddings, you can run knn by following command.

```shell
python main.py --embeddings_dir [embeddings_dir] --dataset_name [dataset_name] --embed_type [video | clip]
```

- `embeddings_dir` should be the parent directory of 'train' and 'val' folders.
- `dataset_name` should be one from the following list.
  - `Kinetics400` (Kinetics-400)
  - `sthsth-v2` (Something-Something-v2)
  - `Diving48` (Diving-48)
  - `MomentsInTime` (Moments in Time)
- `embed_tpye` is a selection between `video` and `clip`. `video` means that each embedding represent a video, and `clip` means that we split the videos into multiple clips and get embeddings from all clips. For evaluation, the votes from all clips are summed.
- Sampling rule
  - There are two types of frame sampling: `uniform` & `multi-clip`
  - You can choose one of them by just choosing the directory using `embed_dir`. The two types of embeddings are separated into the two folders named as `uniform` & `multi-clip`.
  - `uniform` yields only one clip from a video. Therefore there is no difference between `embed_tpye=video` and `embed_tpye=clip`.