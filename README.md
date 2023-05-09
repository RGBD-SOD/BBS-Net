# <p align=center>`BBS-Net`</p>

> **Note** > <em>The is not an official implementation of BBS-Net (refer to origin [here](https://github.com/DengPingFan/BBS-Net))</em>

We aim to improve the project by implementing:

- Format the code with **black** formatter
- Reimplementing **BBS-Net** with **Python 3.10** on **Ubuntu 22.04**
- Build the model on **HuggingFace** ([https://huggingface.co/RGBD-SOD/bbsnet](https://huggingface.co/RGBD-SOD/bbsnet)) for ease of integration

## Installation

- Python 3.10 (tested version)
- `pip install -r requirements.txt`

## Use

```python
from typing import Dict

import numpy as np
from datasets import load_dataset
from matplotlib import cm
from PIL import Image
from torch import Tensor
from transformers import AutoImageProcessor, AutoModel

model = AutoModel.from_pretrained("RGBD-SOD/bbsnet", trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(
    "RGBD-SOD/bbsnet", trust_remote_code=True
)
dataset = load_dataset("RGBD-SOD/test", "v1", split="train", cache_dir="data")

index = 0

"""
Get a specific sample from the dataset

sample = {
    'depth': <PIL.PngImagePlugin.PngImageFile image mode=L size=640x360>,
    'rgb': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=640x360>,
    'gt': <PIL.PngImagePlugin.PngImageFile image mode=L size=640x360>,
    'name': 'COME_Train_5'
}
"""
sample = dataset[index]

depth: Image.Image = sample["depth"]
rgb: Image.Image = sample["rgb"]
gt: Image.Image = sample["gt"]
name: str = sample["name"]


"""
1. Preprocessing step

preprocessed_sample = {
    'rgb': tensor([[[[-0.8507, ....0365]]]]),
    'gt': tensor([[[[0., 0., 0...., 0.]]]]),
    'depth': tensor([[[[0.9529, 0....3490]]]])
}
"""
preprocessed_sample: Dict[str, Tensor] = image_processor.preprocess(sample)

"""
2. Prediction step

output = {
    'logits': tensor([[[[-5.1966, ...ackward0>)
}
"""
output: Dict[str, Tensor] = model(
    preprocessed_sample["rgb"], preprocessed_sample["depth"]
)

"""
3. Postprocessing step
"""
postprocessed_sample: np.ndarray = image_processor.postprocess(
    output["logits"], [sample["gt"].size[1], sample["gt"].size[0]]
)
prediction = Image.fromarray(np.uint8(cm.gist_earth(postprocessed_sample) * 255))

"""
Show the predicted salient map and the corresponding ground-truth(GT)
"""
prediction.show()
gt.show()
```

## Train model

```bash
python BBSNet_train.py  \
    --rgb_root /kaggle/input/rgbdsod-set1/train/RGB/ \
    --depth_root /kaggle/input/rgbdsod-set1/train/depths/ \
    --gt_root /kaggle/input/rgbdsod-set1/train/GT/ \
    --test_rgb_root /kaggle/input/rgbdsod-set1/test/COME-E/RGB/ \
    --test_depth_root /kaggle/input/rgbdsod-set1/test/COME-E/depths/ \
    --test_gt_root /kaggle/input/rgbdsod-set1/test/COME-E/GT/ \
    --batchsize 16
```

## Experiment on COME15K

- See [reports](https://wandb.ai/thinh-huynh-re/BBS-Net/reports/Experiment-Train-BBS-Net-on-COME15K--Vmlldzo0MzAyODI3) on Wandb

## Citation

Please cite the following paper if you use BBS-Net in your reseach.

```
@inproceedings{fan2020bbs,
  title={BBS-Net: RGB-D salient object detection with a bifurcated backbone strategy network},
  author={Fan, Deng-Ping and Zhai, Yingjie and Borji, Ali and Yang, Jufeng and Shao, Ling},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part XII},
  pages={275--292},
  year={2020},
  organization={Springer}
}
```
