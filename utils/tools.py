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


def random_crop_and_resize_image(image_tensor, bbox, resize_h, resize_w):
    if resize_h != resize_w:
        raise ValueError("The values of resize_h and resize_w should be equal.")
    human_instance = tf.image.crop_to_bounding_box(image=image_tensor,
                                                   offset_height=bbox[1],
                                                   offset_width=bbox[0],
                                                   target_height=bbox[3],
                                                   target_width=bbox[2])
    left_top_of_human_instance = bbox[0:2]
    crop_rect, cropped_image = random_crop_in_roi(image=image_tensor, roi=human_instance, left_top_of_roi=left_top_of_human_instance)
    resize_ratio = resize_h / crop_rect.shape[-1]
    resized_image = tf.image.resize(images=cropped_image, size=[resize_h, resize_w])
    return resized_image, resize_ratio, crop_rect


def random_crop_in_roi(image, roi, left_top_of_roi):
    roi_h = roi.shape[0]
    roi_w = roi.shape[1]
    if roi_h > roi_w:
        longer_border = roi_h
        shorter_border = roi_w
    else:
        longer_border = roi_w
        shorter_border = roi_h
    random_coord = tf.random.uniform(shape=(), minval=0, maxval=longer_border - shorter_border)
    if longer_border == roi_h:
        x_random_crop = left_top_of_roi[0]
        y_random_crop = int(left_top_of_roi[1] + random_coord)
    else:
        x_random_crop = int(left_top_of_roi[0] + random_coord)
        y_random_crop = left_top_of_roi[1]
    crop_rect = tf.convert_to_tensor(value=[x_random_crop, y_random_crop, shorter_border, shorter_border], dtype=tf.dtypes.int32)
    cropped_image = tf.image.crop_to_bounding_box(image=image,
                                                  offset_height=y_random_crop,
                                                  offset_width=x_random_crop,
                                                  target_height=shorter_border,
                                                  target_width=shorter_border)
    return crop_rect, cropped_image


def point_in_rect(point_x, point_y, rect):
    # rect : (x, y, w, h)
    xmin = rect[0]
    ymin = rect[1]
    xmax = xmin + rect[2]
    ymax = ymin + rect[3]
    if xmin <= point_x <= xmax and ymin <= point_y <= ymax:
        is_point_in_rect = True
    else:
        is_point_in_rect = False
    return is_point_in_rect