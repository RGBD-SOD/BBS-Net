from typing import Dict, Optional

from torch import Tensor, nn
from transformers import PreTrainedModel

from .configuration_bbsnet import BBSNetConfig
from .BBSNet_model import BBSNet


class BBSNetModel(PreTrainedModel):
    """
    The line that sets the config_class is not mandatory,
    unless you want to register your model with the auto classes
    """

    config_class = BBSNetConfig

    def __init__(self, config: BBSNetConfig):
        super().__init__(config)
        self.model = BBSNet()
        self.loss = nn.BCEWithLogitsLoss()

    """
    You can have your model return anything you want, 
    but returning a dictionary with the loss included when labels are passed, 
    will make your model directly usable inside the Trainer class. 
    Using another output format is fine as long as you are planning on 
    using your own training loop or another library for training.
    """

    def forward(
        self, rgbs: Tensor, depths: Tensor, gts: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        _, logits = self.model(rgbs, depths)
        if gts is not None:
            loss = self.loss(logits, gts)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


if __name__ == "__main__":
    resnet50d_config = ResnetConfig.from_pretrained("custom-resnet")
    resnet50d = ResnetModelForImageClassification(resnet50d_config)

    # Load pretrained weights from timm
    pretrained_model: nn.Module = timm.create_model("resnet50d", pretrained=True)
    resnet50d.model.load_state_dict(pretrained_model.state_dict())
