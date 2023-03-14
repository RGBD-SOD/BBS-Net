from transformers import AutoModel, AutoImageProcessor
from datasets import load_dataset

model = AutoModel.from_pretrained("RGBD-SOD/bbsnet", trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained("RGBD-SOD/bbsnet")
dataset = load_dataset("RGBD-SOD/test", "v1", split="train", cache_dir="data")
