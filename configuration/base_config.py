
class StageParams(object):
    def __init__(self, channels, modules, block, num_blocks, fusion_method):
        self.channels = channels
        self.modules = modules
        self.block = block
        self.num_blocks = num_blocks
        self.fusion_method = fusion_method
        self.expansion = self.__get_expansion()

    def __get_expansion(self):
        if self.block == "BASIC":
            return 1
        elif self.block == "BOTTLENECK":
            return 4
        else:
            raise ValueError("Invalid block name.")

    def get_stage_channels(self):
        num_channels = [num_channel * self.expansion for num_channel in self.channels]
        return num_channels

    def get_branch_num(self):
        return len(self.channels)

    def get_modules(self):
        return self.modules

    def get_block(self):
        return self.block

    def get_num_blocks(self):
        return self.num_blocks

    def get_fusion_method(self):
        return self.fusion_method


class Config(object):
    BATCH_SIZE = 8
    EPOCHS = 50
    TRAINING_CONFIG_NAME = "coco_w32_256x192"

    # input image
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    CHANNELS = 3

    # heatmap
    HEATMAP_WIDTH = 64
    HEATMAP_HEIGHT = 64
    SIGMA = 2

    TRANSFORM_METHOD = "resize"   # random_crop, resize

    # dataset
    COCO_ROOT_DIR = "./dataset/COCO/2017/"
    COCO_TRAIN_TXT = "./coco_train.txt"
    COCO_VALID_TXT = "./coco_valid.txt"


    # save model
    save_weights_dir = "saved_model/weights/"

    # test
    TEST_PICTURES_DIRS = ["./experiment/test_image_1.jpg", "./experiment/test_image_2.jpg"]
    TEST_DURING_TRAINING = True
    SAVE_TEST_RESULTS_DIR = "./experiment/"



    def __init__(self):
        pass