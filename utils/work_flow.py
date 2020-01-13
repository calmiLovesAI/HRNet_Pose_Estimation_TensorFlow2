from core.hrnet import HRNet
from configuration.base_config import Config
from utils.tools import get_config_params
import numpy as np
import cv2


def get_model(cfg):
    model = HRNet(cfg)
    return model


def print_model_summary(network):
    config_params = get_config_params(Config.TRAINING_CONFIG_NAME)
    network.build(input_shape=(None, config_params.IMAGE_HEIGHT, config_params.IMAGE_WIDTH, config_params.CHANNELS))
    network.summary()


def get_max_preds(heatmap_tensor):
    heatmap = heatmap_tensor.numpy()
    batch_size, height, width, num_of_joints = heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[-1]
    heatmap = heatmap.reshape((batch_size, -1, num_of_joints))
    index = np.argmax(heatmap, axis=1)
    maxval = np.amax(heatmap, axis=1)
    index = index.reshape((batch_size, 1, num_of_joints))
    maxval = maxval.reshape((batch_size, 1, num_of_joints))
    preds = np.tile(index, (1, 2, 1)).astype(np.float32)

    preds[:, 0, :] = (preds[:, 0, :]) / width
    preds[:, 1, :] = np.floor((preds[:, 1, :]) / height)

    pred_mask = np.tile(np.greater(maxval, 0.0), (1, 2, 1))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask

    return preds, maxval


def get_final_preds(batch_heatmaps):
    preds, maxval = get_max_preds(batch_heatmaps)
    num_of_joints = preds.shape[-1]
    batch_size = preds.shape[0]
    # print(preds.shape, preds.dtype)   # (1, 2, 17) float32
    # print(maxval.shape, maxval.dtype)   # (1, 1, 17) float32
    # heatmap_height = batch_heatmaps.shape[1]
    # heatmap_width = batch_heatmaps.shape[2]
    batch_x = []
    batch_y = []
    for b in range(batch_size):
        single_image_x = []
        single_image_y = []
        for j in range(num_of_joints):
            # hm = batch_heatmaps[b, ..., j]   # (heatmap_height, heatmap_width)
            point_x = int(preds[b, 0, j])
            point_y = int(preds[b, 1, j])
            class_prob = np.argmax(maxval, axis=-1)
            single_image_x.append(point_x)
            single_image_y.append(point_y)
        batch_x.append(single_image_x)
        batch_y.append(single_image_y)
    return batch_x, batch_y


def draw_on_image(cfg, image, x, y):
    keypoints_coords = []
    for j in range(len(x)):
        x_coord, y_coord = cfg.IMAGE_WIDTH * x[j], cfg.IMAGE_HEIGHT * y[j]
        keypoints_coords.append([x_coord, y_coord])
        cv2.circle(img=image, center=(x_coord, y_coord), radius=5, color=(0, 0, 255), thickness=-1)
    # draw lines
    for i in range(len(cfg.SKELETON)):
        index_1 = cfg.SKELETON[i][0] - 1
        index_2 = cfg.SKELETON[i][1] - 1
        x1, y1 = cfg.IMAGE_WIDTH * x[index_1], cfg.IMAGE_HEIGHT * y[index_1]
        x2, y2 = cfg.IMAGE_WIDTH * x[index_2], cfg.IMAGE_HEIGHT * y[index_2]
        cv2.line(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=1)
    return image


def inference(cfg, image_tensor, model, image_dir):
    pred_heatmap = model(image_tensor, training=False)
    batch_x_list, batch_y_list = get_final_preds(batch_heatmaps=pred_heatmap)
    keypoints_x = batch_x_list[0]
    keypoints_y = batch_y_list[0]
    return draw_on_image(cfg=cfg, image=cv2.imread(image_dir), x=keypoints_x, y=keypoints_y)
