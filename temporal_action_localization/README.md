# Temporal Action Localization
This source code is for the temporal action localization (TAL) evaluation of twelvelabs embeddings

## Embedding Preparation
- Download the embeddings via following links.
- Supported bechmarks
   - ActivityNet v1.3 ([link](https://www.dropbox.com/scl/fi/yfp2k0r3ngmvdg5r7z05n/ActivityNet_1.3.tar?rlkey=zv4xr7tfggwkwqo4bzk63ztab&st=rig2ycvd&dl=0))
   - THUMOS 14 ([link](https://www.dropbox.com/scl/fi/q2ftstt23s4tfshp965t2/THUMOS14.tar?rlkey=hgd2wennvnf9a3cr97awb9xvr&st=16ht5b00&dl=0))

## Run
After preparing the embeddings, you can train and evaluate the model by following command (self-contained).

```shell
python main.py --embeddings_dir [embeddings_dir] --dataset_name [dataset_name]
```

If you want to evaluate the model with external classifier, add `--with_ext_classifier` option.

```shell
python main.py --embeddings_dir [embeddings_dir] --dataset_name [dataset_name] --with_ext_classifier
```

- `embeddings_dir` should be the parent directory of 'train' and 'val' folders.
- `dataset_name` should be one from the following list.
  - `ActivityNet_1.3` (ActivityNet v1.3)
  - `THUMOS14` (THUMOS 14)
- For the training options, the default values are in [libs/core/config.py](/temporal_action_localization/libs/core/config.py). Then they are overrided by the [config files](/temporal_action_localization/configs/) and then overrided by the argparse values.


## Reference
This TAL evaluation source code is based on [Video Mamba Suit](https://github.com/OpenGVLab/video-mamba-suite)
