from core.hrnet import HRNet
from configuration.base_config import Config
from utils.tools import get_config_params


def get_model(cfg):
    model = HRNet(cfg)
    return model


def print_model_summary(network):
    config_params = get_config_params(Config.TRAINING_CONFIG_NAME)
    network.build(input_shape=(None, config_params.IMAGE_HEIGHT, config_params.IMAGE_WIDTH, config_params.CHANNELS))
    network.summary()

