import torch

from bbsnet_model.configuration_bbsnet import BBSNetConfig
from bbsnet_model.image_processor_bbsnet import BBSNetImageProcessor
from bbsnet_model.modeling_bbsnet import BBSNetModel

BBSNetConfig.register_for_auto_class()
BBSNetModel.register_for_auto_class("AutoModel")
BBSNetImageProcessor.register_for_auto_class("AutoImageProcessor")

config = BBSNetConfig()
model = BBSNetModel(config)
model.model.load_state_dict(torch.load("./model_pths/BBSNet.pth", map_location="cpu"))
image_processor = BBSNetImageProcessor()

model.push_to_hub("RGBD-SOD/bbsnet")
image_processor.push_to_hub("RGBD-SOD/bbsnet")
