from typing import List

from transformers import PretrainedConfig

"""
The configuration of a model is an object that 
will contain all the necessary information to build the model.

The three important things to remember when writing you own configuration are the following:

- you have to inherit from PretrainedConfig,
- the __init__ of your PretrainedConfig must accept any kwargs,
- those kwargs need to be passed to the superclass __init__.
"""


class BBSNetConfig(PretrainedConfig):

    """
    Defining a model_type for your configuration is not mandatory,
    unless you want to register your model with the auto classes."""

    model_type = "bbsnet"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == "__main__":
    """
    With this done, you can easily create and save your configuration like
    you would do with any other model config of the library.
    Here is how we can create a resnet50d config and save it:
    """
    bbsnet_config = BBSNetConfig()
    bbsnet_config.save_pretrained("custom-bbsnet")

    """
    This will save a file named config.json inside the folder custom-resnet. 
    You can then reload your config with the from_pretrained method:
    """
    bbsnet_config = BBSNetConfig.from_pretrained("custom-bbsnet")

    """
    You can also use any other method of the PretrainedConfig class, 
    like push_to_hub() to directly upload your config to the Hub.
    """
