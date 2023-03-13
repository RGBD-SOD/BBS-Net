import torch

from bbsnet_model.configuration_bbsnet import BBSNetConfig
from bbsnet_model.modeling_bbsnet import BBSNetModel

BBSNetConfig.register_for_auto_class()
BBSNetModel.register_for_auto_class("AutoModel")


config = BBSNetConfig()
model = BBSNetModel(config)
model.model.load_state_dict(torch.load("./model_pths/BBSNet.pth"))

model.push_to_hub("RGBD-SOD/bbsnet")
