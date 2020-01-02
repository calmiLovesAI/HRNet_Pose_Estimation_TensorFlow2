from core.hrnet import HRNet
from configuration.base_config import Config
from utils.tools import get_config_params
import numpy as np


def get_model(cfg):
    model = HRNet(cfg)
    return model


def print_model_summary(network):
    config_params = get_config_params(Config.TRAINING_CONFIG_NAME)
    network.build(input_shape=(None, config_params.IMAGE_HEIGHT, config_params.IMAGE_WIDTH, config_params.CHANNELS))
    network.summary()


def get_max_preds(heatmap_tensor):
    heatmap = heatmap_tensor.numpy()
    batch_size, _, width, num_of_joints = heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[-1]
    heatmap = heatmap.reshape((batch_size, -1, num_of_joints))
    index = np.argmax(heatmap, axis=1)
    maxval = np.amax(heatmap, axis=1)
    index = index.reshape((batch_size, 1, num_of_joints))
    maxval = maxval.reshape((batch_size, 1, num_of_joints))
    preds = np.tile(index, (1, 2, 1)).astype(np.float32)

    preds[:, 0, :] = (preds[:, 0, :]) / width
    preds[:, 1, :] = np.floor((preds[:, 1, :]) / width)

    pred_mask = np.tile(np.greater(maxval, 0.0), (1, 2, 1))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask

    return preds, maxval
