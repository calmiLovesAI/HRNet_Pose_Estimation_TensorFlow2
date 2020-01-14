import tensorflow as tf


def read_image(image_dir, cfg):
    image_content = tf.io.read_file(filename=image_dir)
    # The 'image' has been normalized.
    image = tf.io.decode_image(contents=image_content, channels=cfg.CHANNELS, dtype=tf.dtypes.float32)
    return image


# Determine whether a point is within a rectangular border.
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


class RandomCropTransform(object):
    def __init__(self, image, keypoints, bbox, resize_h, resize_w, num_of_joints):
        self.image = image
        self.keypoints = keypoints
        self.bbox = bbox
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.num_of_joints = num_of_joints

    def image_transform(self):
        if self.resize_h != self.resize_w:
            raise ValueError("The values of resize_h and resize_w should be equal.")
        human_instance = tf.image.crop_to_bounding_box(image=self.image,
                                                       offset_height=self.bbox[1],
                                                       offset_width=self.bbox[0],
                                                       target_height=self.bbox[3],
                                                       target_width=self.bbox[2])
        left_top_of_human_instance = self.bbox[0:2]
        crop_rect, cropped_image = self.__random_crop_in_roi(image=self.image, roi=human_instance,
                                                             left_top_of_roi=left_top_of_human_instance)
        resize_ratio = self.resize_h / crop_rect.shape[-1]
        resized_image = tf.image.resize(images=cropped_image, size=[self.resize_h, self.resize_w])
        return resized_image, resize_ratio, crop_rect

    def keypoints_transform(self, resize_ratio, crop_rect):
        crop_rect = crop_rect.numpy()
        transformed_keypoints = self.keypoints.numpy()
        # First determine whether the point is inside the crop area.
        for i in range(self.num_of_joints):
            if not point_in_rect(point_x=transformed_keypoints[i, 0], point_y=transformed_keypoints[i, 1], rect=crop_rect):
                transformed_keypoints[i, 2] = 0.0

        for i in range(self.num_of_joints):
            if transformed_keypoints[i, 2] > 0.0:
                # Calculate the coordinates of the keypoints after cropping the original picture.
                transformed_keypoints[i, 0] = transformed_keypoints[i, 0] - crop_rect[0]
                transformed_keypoints[i, 1] = transformed_keypoints[i, 1] - crop_rect[1]
                # Calculate the coordinates of the keypoints after resizing.
                transformed_keypoints[i, 0] = int(transformed_keypoints[i, 0] * resize_ratio)
                transformed_keypoints[i, 1] = int(transformed_keypoints[i, 1] * resize_ratio)
        return transformed_keypoints

    def __random_crop_in_roi(self, image, roi, left_top_of_roi):
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
        crop_rect = tf.convert_to_tensor(value=[x_random_crop, y_random_crop, shorter_border, shorter_border],
                                         dtype=tf.dtypes.int32)
        cropped_image = tf.image.crop_to_bounding_box(image=image,
                                                      offset_height=y_random_crop,
                                                      offset_width=x_random_crop,
                                                      target_height=shorter_border,
                                                      target_width=shorter_border)
        return crop_rect, cropped_image



class ResizeTransform(object):
    def __init__(self, image, keypoints, bbox, resize_h, resize_w, num_of_joints):
        self.image = image
        self.keypoints = keypoints
        self.bbox = bbox
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.num_of_joints = num_of_joints

    def image_transform(self):
        human_instance = tf.image.crop_to_bounding_box(image=self.image,
                                                       offset_height=self.bbox[1],
                                                       offset_width=self.bbox[0],
                                                       target_height=self.bbox[3],
                                                       target_width=self.bbox[2])
        left_top_of_human_instance = self.bbox[0:2]
        resize_ratio = [self.resize_h / human_instance.shape[0], self.resize_w / human_instance.shape[1]]
        resized_image = tf.image.resize(images=human_instance, size=[self.resize_h, self.resize_w])
        return resized_image, resize_ratio, left_top_of_human_instance

    def keypoints_transform(self, resize_ratio, left_top):
        transformed_keypoints = self.keypoints.numpy()
        for i in range(self.num_of_joints):
            if transformed_keypoints[i, 2] > 0.0:
                transformed_keypoints[i, 0] = int((transformed_keypoints[i, 0] - left_top[0]) * resize_ratio[1])
                transformed_keypoints[i, 1] = int((transformed_keypoints[i, 1] - left_top[1]) * resize_ratio[0])
        return transformed_keypoints


class KeypointsRescaleToOriginal(object):
    def __init__(self, input_image_height, input_image_width, heatmap_h, heatmap_w, original_image_size):
        self.scale_ratio = [input_image_height / heatmap_h, input_image_width / heatmap_w]
        self.original_scale_ratio = [original_image_size[0] / input_image_height, original_image_size[1] / input_image_width]

    def __scale_to_input_size(self, x, y):
        return x * self.scale_ratio[1], y * self.scale_ratio[0]

    def __call__(self, x, y):
        temp_x, temp_y = self.__scale_to_input_size(x=x, y=y)
        return int(temp_x * self.original_scale_ratio[1]), int(temp_y * self.original_scale_ratio[0])



