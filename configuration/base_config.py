
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
    # train
    BATCH_SIZE = 8
    EPOCHS = 50
    LOAD_WEIGHTS_BEFORE_TRAINING = False
    LOAD_WEIGHTS_FROM_EPOCH = 0

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
    SAVE_FREQUENCY = 5

    # test
    TEST_PICTURES_DIRS = ["", ""]  # "./experiment/xxx.jpg"
    TEST_DURING_TRAINING = True
    SAVE_TEST_RESULTS_DIR = "./experiment/"

    # color (r, g, b)
    DYE_VAT = {"Pink": (255, 192, 203), "MediumVioletRed": (199, 21, 133), "Magenta": (255, 0, 255),
               "Purple": (128, 0, 128), "Blue": (0, 0, 255), "LightSkyBlue": (135, 206, 250),
               "Cyan": (0, 255, 255), "LightGreen": (144, 238, 144), "Green": (0, 128, 0),
               "Yellow": (255, 255, 0), "Gold": (255, 215, 0), "Orange": (255, 165, 0),
               "Red": (255, 0, 0), "LightCoral": (240, 128, 128), "DarkGray": (169, 169, 169)}

    def __init__(self):
        pass

    def get_dye_vat_bgr(self):
        bgr_color = {}
        for k, v in self.DYE_VAT.items():
            r, g, b = v[0], v[1], v[2]
            bgr_color[k] = (b, g, r)
        return bgr_color

    def color_pool(self):
        bgr_color_dict = self.get_dye_vat_bgr()
        bgr_color_pool = []
        for k, v in bgr_color_dict.items():
            bgr_color_pool.append(v)
        return bgr_color_pool
