from core.hrnet import HRNet
from configuration.base_config import Config
from utils.tools import get_config_params
import numpy as np
import cv2

from utils.transforms import KeypointsRescaleToOriginal


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

    preds[:, 0, :] = preds[:, 0, :] % width
    preds[:, 1, :] = np.floor(preds[:, 1, :] / width)

    pred_mask = np.tile(np.greater(maxval, 0.0), (1, 2, 1))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask

    return preds, maxval


def get_final_preds(batch_heatmaps):
    preds, maxval = get_max_preds(batch_heatmaps)
    num_of_joints = preds.shape[-1]
    batch_size = preds.shape[0]
    batch_x = []
    batch_y = []
    for b in range(batch_size):
        single_image_x = []
        single_image_y = []
        for j in range(num_of_joints):
            point_x = int(preds[b, 0, j])
            point_y = int(preds[b, 1, j])
            single_image_x.append(point_x)
            single_image_y.append(point_y)
        batch_x.append(single_image_x)
        batch_y.append(single_image_y)
    return batch_x, batch_y


def draw_on_image(cfg, image, x, y, rescale):
    keypoints_coords = []
    for j in range(len(x)):
        x_coord, y_coord = rescale(x=x[j], y=y[j])
        keypoints_coords.append([x_coord, y_coord])
        cv2.circle(img=image, center=(x_coord, y_coord), radius=8, color=cfg.get_dye_vat_bgr()["Red"], thickness=2)
    # draw lines
    color_list = cfg.color_pool()
    for i in range(len(cfg.SKELETON)):
        index_1 = cfg.SKELETON[i][0] - 1
        index_2 = cfg.SKELETON[i][1] - 1
        x1, y1 = rescale(x=x[index_1], y=y[index_1])
        x2, y2 = rescale(x=x[index_2], y=y[index_2])
        cv2.line(img=image, pt1=(x1, y1), pt2=(x2, y2), color=color_list[i % len(color_list)], thickness=5, lineType=cv2.LINE_AA)
    return image


def inference(cfg, image_tensor, model, image_dir, original_image_size):
    pred_heatmap = model(image_tensor, training=False)
    keypoints_rescale = KeypointsRescaleToOriginal(input_image_height=cfg.IMAGE_HEIGHT,
                                                   input_image_width=cfg.IMAGE_WIDTH,
                                                   heatmap_h=pred_heatmap.shape[1],
                                                   heatmap_w=pred_heatmap.shape[2],
                                                   original_image_size=original_image_size)
    batch_x_list, batch_y_list = get_final_preds(batch_heatmaps=pred_heatmap)
    keypoints_x = batch_x_list[0]
    keypoints_y = batch_y_list[0]
    return draw_on_image(cfg=cfg, image=cv2.imread(image_dir), x=keypoints_x, y=keypoints_y, rescale=keypoints_rescale)
