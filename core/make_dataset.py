import tensorflow as tf


class CocoDataset(object):
    def __init__(self, config_params, dataset_type):
        self.config_params = config_params
        if dataset_type == "train":
            self.data_dir = self.config_params.COCO_TRAIN_TXT
        elif dataset_type == "valid":
            self.data_dir = self.config_params.COCO_VALID_TXT
        else:
            raise ValueError("Invalid dataset_type name!")

    @staticmethod
    def __get_length_of_dataset(dataset):
        count = 0
        for _ in dataset:
            count += 1
        return count

    def generate_dataset(self):
        dataset = tf.data.TextLineDataset(filenames=self.data_dir)
        dataset_length = self.__get_length_of_dataset(dataset)
        dataset = dataset.batch(batch_size=self.config_params.BATCH_SIZE)
        return dataset, dataset_length
