import numpy as np
from datasets import load_dataset
from matplotlib import cm
from PIL import Image
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
sample["gt"].show()


"""
preprocessed_sample = {
    'rgb': tensor([[[[-0.8507, ....0365]]]]), 
    'gt': tensor([[[[0., 0., 0...., 0.]]]]), 
    'depth': tensor([[[[0.9529, 0....3490]]]])
}
"""
preprocessed_sample = image_processor.preprocess(sample)

"""
output = {
    'logits': tensor([[[[-5.1966, ...ackward0>)
}
"""
output = model(preprocessed_sample["rgb"], preprocessed_sample["depth"])

postprocessed_sample: np.ndarray = image_processor.postprocess(
    output["logits"], [sample["gt"].size[1], sample["gt"].size[0]]
)
rs = Image.fromarray(np.uint8(cm.gist_earth(postprocessed_sample) * 255))
rs.show()
