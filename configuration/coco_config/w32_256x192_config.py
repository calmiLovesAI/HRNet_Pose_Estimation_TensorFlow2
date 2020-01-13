from configuration.base_config import Config
from configuration.base_config import StageParams


class CocoW32Size256x192(Config):
    def __init__(self):
        super(CocoW32Size256x192, self).__init__()
        self.num_of_joints = 17
        self.conv3_kernel = 3
        self.SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        self.stage_2 = StageParams(channels=[32, 64], modules=1, block="BASIC", num_blocks=[4, 4], fusion_method="sum")
        self.stage_3 = StageParams(channels=[32, 64, 128], modules=4, block="BASIC", num_blocks=[4, 4, 4], fusion_method="sum")
        self.stage_4 = StageParams(channels=[32, 64, 128, 256], modules=3, block="BASIC", num_blocks=[4, 4, 4, 4], fusion_method="sum")

    def get_stage(self, stage_name):
        if stage_name == "s2":
            channels = self.stage_2.get_stage_channels()
            num_branches = self.stage_2.get_branch_num()
            num_modules = self.stage_2.get_modules()
            block = self.stage_2.get_block()
            num_blocks = self.stage_2.get_num_blocks()
            fusion_method = self.stage_2.get_fusion_method()
        elif stage_name == "s3":
            channels = self.stage_3.get_stage_channels()
            num_branches = self.stage_3.get_branch_num()
            num_modules = self.stage_3.get_modules()
            block = self.stage_3.get_block()
            num_blocks = self.stage_3.get_num_blocks()
            fusion_method = self.stage_3.get_fusion_method()
        elif stage_name == "s4":
            channels = self.stage_4.get_stage_channels()
            num_branches = self.stage_4.get_branch_num()
            num_modules = self.stage_4.get_modules()
            block = self.stage_4.get_block()
            num_blocks = self.stage_4.get_num_blocks()
            fusion_method = self.stage_4.get_fusion_method()
        else:
            raise ValueError("Invalid stage name.")
        return [channels, num_branches, num_modules, block, num_blocks, fusion_method]






