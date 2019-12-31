from configuration.coco_config.w32_256x192_config import CocoW32Size256x192
import tensorflow as tf


def get_config_params(config_name):
    if config_name == "coco_w32_256x192":
        config_params = CocoW32Size256x192()
        return config_params
    else:
        raise ValueError("Invalid config_name.")


def read_image(image_dir, cfg):
    image_content = tf.io.read_file(filename=image_dir)
    # The 'image' has been normalized.
    image = tf.io.decode_image(contents=image_content, channels=cfg.CHANNELS, dtype=tf.dtypes.float32)
    return image
