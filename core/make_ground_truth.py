import tensorflow as tf


class GroundTruth(object):
    def __init__(self, batch_keypoints_info):
        self.batch_keypoints_list = self.__tensor2list(batch_keypoints_info)

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
        for item in self.batch_keypoints_list:
            self.__get_one_human_instance_keypoints(line_keypoints=item)

    def __get_one_human_instance_keypoints(self, line_keypoints):
        line_keypoints = line_keypoints.strip()
        split_line = line_keypoints.split(" ")
        image_name = split_line[0]
        image_height = split_line[1]
        image_width = split_line[2]
        _, bbox = self.__convert_string_to_float_and_int(split_line[3:7])
        keypoints, _ = self.__convert_string_to_float_and_int(split_line[7:])
        keypoints_tensor = tf.convert_to_tensor(value=keypoints, dtype=tf.dtypes.float32)
        keypoints_tensor = tf.reshape(keypoints_tensor, shape=(-1, 3))
        keypoints_3d, keypoints_3d_exist = self.__get_keypoints_3d(keypoints_tensor)

    def __get_keypoints_3d(self, keypoints_tensor):
        keypoints_3d_list = []
        keypoints_3d_exist_list = []
        num_of_joints = keypoints_tensor.shape[0]
        for i in range(num_of_joints):
            keypoints_3d_list.append(tf.convert_to_tensor([keypoints_tensor[i, 0], keypoints_tensor[i, 1], 0], dtype=tf.dtypes.float32))
            exist_value = keypoints_tensor[i, 2]
            if exist_value > 1:
                exist_value = 1
            keypoints_3d_exist_list.append(tf.convert_to_tensor([exist_value, exist_value, 0], dtype=tf.dtypes.float32))
        
        keypoints_3d = tf.stack(values=keypoints_3d_list, axis=0)
        keypoints_3d_exist = tf.stack(values=keypoints_3d_exist_list, axis=0)
        return keypoints_3d, keypoints_3d_exist
