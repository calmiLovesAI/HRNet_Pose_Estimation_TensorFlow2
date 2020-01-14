# HRNet_Pose_Estimation_TensorFlow2
A tensorflow2 implementation of HRNet for human pose estimation.

## Requirements:
+ Python == 3.7
+ TensorFlow == 2.1.0
+ numpy == 1.17.0
+ opencv-python == 4.1.0


## Usage
### Prepare dataset
1. Download the COCO2017 dataset.
2. Unzip the **train2017.zip**,  **annotations_trainval2017.zip** and place them in the 'dataset' folder, make sure the directory is like this : 
```
|——dataset
    |——COCO
        |——2017
            |——annotations
            |——train2017
```

### Train on COCO2017
1. Open the file *"./configuration/base_config.py"*, change the parameters according to your needs.
2. Run *"write_coco_to_txt.py"* to generate coco annotation files.
3. Run *"train.py"* to train on coco2017 dataset.

### Test
1. Prepare the test pictures and make sure that the **TEST_PICTURES_DIRS** in *"./configuration/base_config.py"* is correct.
2. Run *"test.py"* to test on your test pictures.





## Acknowledgments
1. https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

## Citation
```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {CoRR},
  volume    = {abs/1908.07919},
  year={2019}
}
```

## References
1. Paper: [Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212)