from core.coco_annotation import COCO_keypoints


if __name__ == '__main__':
    coco = COCO_keypoints()
    coco.write_information_to_txt(dataset="train")
    coco.write_information_to_txt(dataset="valid")
