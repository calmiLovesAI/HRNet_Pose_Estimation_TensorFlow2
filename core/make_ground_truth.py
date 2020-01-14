import tensorflow as tf
import numpy as np
from utils.transforms import read_image, RandomCropTransform, ResizeTransform


class GroundTruth(object):
    def __init__(self, config_params, batch_keypoints_info):
        self.config_params = config_params
        self.batch_keypoints_list = self.__tensor2list(batch_keypoints_info)
        self.num_of_joints = config_params.num_of_joints
        self.image_size = np.array([config_params.IMAGE_HEIGHT, config_params.IMAGE_WIDTH])
        self.heatmap_size = np.array([config_params.HEATMAP_HEIGHT, config_params.HEATMAP_WIDTH])
        self.sigma = config_params.SIGMA
        self.transform_method = config_params.TRANSFORM_METHOD

    def __tensor2list(self, tensor_data):
        list_data = []
        length = tensor_data.shape[0]
        for i in range(length):
            list_data.append(bytes.decode(tensor_data[i].numpy(), encoding="utf-8"))
        return list_data

    def __convert_string_to_float_and_int(self, string_list):
        float_list = []
        int_list = []
        for data_string in string_list:
            data_float = float(data_string)
            data_int = int(data_float)
            float_list.append(data_float)
            int_list.append(data_int)
        return float_list, int_list

    def get_ground_truth(self):
        batch_target = []
        batch_target_weight = []
        batch_images = []
        for item in self.batch_keypoints_list:
            image, keypoints_3d, keypoints_3d_exist = self.__get_one_human_instance_keypoints(line_keypoints=item)
            target, target_weight = self.__generate_target(keypoints_3d.numpy(), keypoints_3d_exist.numpy())
            batch_images.append(image)
            batch_target.append(target)
            batch_target_weight.append(target_weight)
        batch_images_tensor = tf.stack(values=batch_images, axis=0)  # (batch_size, image_height, image_width, channels)
        batch_target_tensor = tf.stack(values=batch_target, axis=0)    # (batch_size, heatmap_height, heatmap_width, num_of_joints)
        batch_target_weight_tensor = tf.stack(values=batch_target_weight, axis=0)  # (batch_size, num_of_joints, 1)
        return batch_images_tensor, batch_target_tensor, batch_target_weight_tensor

    def __get_one_human_instance_keypoints(self, line_keypoints):
        line_keypoints = line_keypoints.strip()
        split_line = line_keypoints.split(" ")
        image_file = split_line[0]
        # image_height = int(float(split_line[1]))
        # image_width = int(float(split_line[2]))
        _, bbox = self.__convert_string_to_float_and_int(split_line[3:7])
        keypoints, _ = self.__convert_string_to_float_and_int(split_line[7:])
        keypoints_tensor = tf.convert_to_tensor(value=keypoints, dtype=tf.dtypes.float32)
        keypoints_tensor = tf.reshape(keypoints_tensor, shape=(-1, 3))

        # Resize the image, and change the coordinates of the keypoints accordingly.
        image_tensor, keypoints = self.__image_and_keypoints_process(image_file, keypoints_tensor, bbox)

        keypoints_3d, keypoints_3d_exist = self.__get_keypoints_3d(keypoints)
        return image_tensor, keypoints_3d, keypoints_3d_exist

    def __image_and_keypoints_process(self, image_dir, keypoints, bbox):
        image_tensor = read_image(image_dir, self.config_params)
        if self.transform_method == "random crop":
            raise NotImplementedError("Not available temporarily.")
            # transform = RandomCropTransform(image=image_tensor, keypoints=keypoints, bbox=bbox, resize_h=self.image_size[0], resize_w=self.image_size[1], num_of_joints=self.num_of_joints)
            # resized_image, resize_ratio, crop_rect = transform.image_transform()
            # keypoints = transform.keypoints_transform(resize_ratio, crop_rect)
            # return resized_image, keypoints
        elif self.transform_method == "resize":
            transform = ResizeTransform(image=image_tensor, keypoints=keypoints, bbox=bbox, resize_h=self.image_size[0], resize_w=self.image_size[1], num_of_joints=self.num_of_joints)
            resized_image, resize_ratio, left_top = transform.image_transform()
            keypoints = transform.keypoints_transform(resize_ratio, left_top)
            return resized_image, keypoints
        else:
            raise ValueError("Invalid TRANSFORM_METHOD.")

    def __get_keypoints_3d(self, keypoints):
        keypoints_3d_list = []
        keypoints_3d_exist_list = []
        for i in range(self.num_of_joints):
            keypoints_3d_list.append(tf.convert_to_tensor([keypoints[i, 0], keypoints[i, 1], 0], dtype=tf.dtypes.float32))
            exist_value = keypoints[i, 2]
            if exist_value > 1:
                exist_value = 1
            keypoints_3d_exist_list.append(tf.convert_to_tensor([exist_value, exist_value, 0], dtype=tf.dtypes.float32))
        
        keypoints_3d = tf.stack(values=keypoints_3d_list, axis=0)
        keypoints_3d_exist = tf.stack(values=keypoints_3d_exist_list, axis=0)
        return keypoints_3d, keypoints_3d_exist

    def __generate_target(self, keypoints_3d, keypoints_3d_exist):
        target_weight = np.ones((self.num_of_joints, 1), dtype=np.float32)
        target_weight[:, 0] = keypoints_3d_exist[:, 0]

        target = np.zeros((self.num_of_joints, self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)
        temp_size = self.sigma * 3
        for joint_id in range(self.num_of_joints):
            feature_stride = self.image_size / self.heatmap_size
            mu_x = int(keypoints_3d[joint_id][0] / feature_stride[1] + 0.5)
            mu_y = int(keypoints_3d[joint_id][1] / feature_stride[0] + 0.5)
            upper_left = [int(mu_x - temp_size), int(mu_y - temp_size)]
            bottom_right = [int(mu_x + temp_size + 1), int(mu_y + temp_size + 1)]
            if upper_left[0] > self.heatmap_size[1] or upper_left[1] > self.heatmap_size[0] or bottom_right[0] < 0 or bottom_right[1] < 0:
                # Set the joint invisible.
                target_weight[joint_id] = 0
                continue
            size = 2 * temp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]   # shape : (size, 1)
            x0 = y0 = size // 2
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
            g_x = max(0, -upper_left[0]), min(bottom_right[0], self.heatmap_size[1]) - upper_left[0]
            g_y = max(0, -upper_left[1]), min(bottom_right[1], self.heatmap_size[0]) - upper_left[1]
            img_x = max(0, upper_left[0]), min(bottom_right[0], self.heatmap_size[1])
            img_y = max(0, upper_left[1]), min(bottom_right[1], self.heatmap_size[0])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        target = tf.convert_to_tensor(target, tf.dtypes.float32)
        target = tf.transpose(a=target, perm=[1, 2, 0])    # shape : (self.heatmap_size[0], self.heatmap_size[1], self.num_of_joints)
        target_weight = tf.convert_to_tensor(target_weight, tf.dtypes.float32)    # shape : (self.num_of_joints, 1)
        return target, target_weight



