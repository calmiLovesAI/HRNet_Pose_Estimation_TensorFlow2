import tensorflow as tf
import cv2
from pathlib import Path

from configuration.base_config import Config
from utils.tools import get_config_params
from utils.transforms import read_image
from utils.work_flow import get_model, print_model_summary, inference


def image_preprocess(cfg, picture_dir):
    image_tensor = read_image(image_dir=picture_dir, cfg=cfg)
    if cfg.TRANSFORM_METHOD == "random crop":
        raise NotImplementedError("Not implemented!")
    elif cfg.TRANSFORM_METHOD == "resize":
        resized_image = tf.image.resize(images=image_tensor, size=[cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])
        resized_image = tf.expand_dims(input=resized_image, axis=0)
    else:
        raise ValueError("Invalid TRANSFORM_METHOD.")
    return resized_image


def test_during_training(cfg, epoch, model):
    for image_dir in cfg.TEST_PICTURES_DIRS:
        image_path = Path(image_dir)
        resized_image = image_preprocess(cfg, image_dir)
        image = inference(cfg=cfg, image_tensor=resized_image, model=model, image_dir=image_dir)
        cv2.imwrite(filename=cfg.SAVE_TEST_RESULTS_DIR + "epoch-{}-{}".format(epoch, image_path.name), img=image)


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    cfg = get_config_params(Config.TRAINING_CONFIG_NAME)

    hrnet = get_model(cfg)
    print_model_summary(hrnet)

    for image_dir in cfg.TEST_PICTURES_DIRS:
        resized_image = image_preprocess(cfg, image_dir)
        image = inference(cfg=cfg, image_tensor=resized_image, model=hrnet, image_dir=image_dir)
        cv2.namedWindow("Pose Estimation", flags=cv2.WINDOW_NORMAL)
        cv2.imshow("Pose Estimation", image)
        cv2.waitKey(0)
