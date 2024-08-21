# Temporal Action Segmentation
This source code is for the temporal action segmentation (TAS) evaluation of twelvelabs embeddings

## Embedding Preparation
- Download the embeddings via following links.
- Supported bechmarks
   - 50Salads ([link](https://www.dropbox.com/scl/fi/hmae2pou23alolctzltum/50Salads.tar?rlkey=qgcegqaan2b0yzwvco5jiy5pm&st=49wzxg54&dl=0))
   - GTEA ([link](https://www.dropbox.com/scl/fi/c4n2n4ha0epwzh242rsgk/GTEA.tar?rlkey=gdog0t8pqp6nwggddvseg92sm&st=pmhlcmgo&dl=0))
   - Breakfast ([link](https://www.dropbox.com/scl/fi/khnxooxy43m1avun130pi/Breakfast.tar?rlkey=zx80zfnb7j6kdg79i5czhec2l&st=nmosj51a&dl=0))


## Run
After preparing the embeddings, you can run knn by following command.

```shell
python main.py --embeddings_dir [embeddings_dir] --dataset_name [dataset_name]
```

- `embeddings_dir` should be the parent directory of 'train' folder.
- `dataset_name` should be one from the following list.
  - `50Salads` (50 Salads Dataset)
  - `GTEA` (GTEA, Georgia Tech Egocentric Activity Datasets)
  - `Breakfast` (Breakfast Action Dataset)

## Reference
This TAS evaluation source code is based on [Video Mamba Suit](https://github.com/OpenGVLab/video-mamba-suite)
