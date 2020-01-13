from configuration.coco_config.w32_256x192_config import CocoW32Size256x192
import tensorflow as tf


def get_config_params(config_name):
    if config_name == "coco_w32_256x192":
        config_params = CocoW32Size256x192()
        return config_params
    else:
        raise ValueError("Invalid config_name.")

