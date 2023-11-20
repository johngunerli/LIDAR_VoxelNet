import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from utils.utils import cal_anchors, process_pointcloud, cal_rpn_target
from utils.custom_collate import default_collate
from aug_data import aug_data
class KittiDataset(Dataset):
    def __init__(self, cfg, params, buffer_size, mode, is_aug_data, label_encoder, create_anchors=False):
        self.cfg = cfg
        self.params = params
        self.mode = mode
        self.buffer_size = buffer_size
        self.is_aug_data = is_aug_data
        self.label_encoder = label_encoder
        data_d = "training" if mode == "train" else "testing" if mode == "test" else "validation"

        if mode != "test":
            label_tags = [os.path.basename(a).split(".")[0] for a in glob.glob(os.path.join(cfg.DATA_DIR, data_d, "label_2/*.txt"))]
        img_tags = [os.path.basename(a).split(".")[0] for a in glob.glob(os.path.join(cfg.DATA_DIR, data_d, "image_2/*.png"))]
        lidar_tags = [os.path.basename(a).split(".")[0] for a in glob.glob(os.path.join(cfg.DATA_DIR, data_d, "velodyne/*.bin"))]

        if mode != "test":
            assert label_tags and img_tags and lidar_tags, "One of the three (label_2, image_2, velodyne) folders is empty, Data folder must not be empty"
            assert not set(label_tags).symmetric_difference(set(img_tags)) and not set(img_tags).symmetric_difference(set(lidar_tags)), "Must have equivalent tags in image_2, label_2, and velodyne dirs, check those files"
        else:
            assert img_tags and lidar_tags, "One of the three (image_2, velodyne) folders is empty, Data folder must not be empty"
            assert not set(img_tags).symmetric_difference(set(lidar_tags)), "Must have equivalent tags in image_2, velodyne dirs, check those files"

        self.tags = lidar_tags
        self.num_examples = len(lidar_tags)
        if create_anchors:
            pass
        self.anchors = cal_anchors(cfg)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        sample = {}  # Create a dictionary to hold data for this item

        data_d = "training" if self.mode == "train" else "testing" if self.mode == "test" else "validation"
        img_dir = os.path.join(self.cfg.DATA_DIR, data_d, "image_2")
        labels_dir = os.path.join(self.cfg.DATA_DIR, data_d, "label_2")
        pc_dir = os.path.join(self.cfg.DATA_DIR, data_d, "velodyne")

        if self.mode in ["train", "sample_test"]:
            random.shuffle(self.tags)
        else:
            self.tags.sort()

        index = self.tags[idx]
        dic = {}

        if self.is_aug_data:
            dic = aug_data(index, os.path.join(self.cfg.DATA_DIR, data_d))
        else:
            pc = np.fromfile(os.path.join(pc_dir, f"{int(index):06d}.bin"), dtype=np.float32).reshape(-1, 4)
            if self.mode == "test":
                dic["lidar"] = pc
                dic["labels"] = []
            elif self.mode == "sample_test" or self.mode == "eval":
                dic["lidar"] = pc
                dic["labels"] = np.array([line.strip() for line in open(os.path.join(labels_dir, f"{int(index):06d}.txt"), 'r').readlines()])
            else:
                dic["lidar"] = 0
                dic["labels"] = np.array([line.strip() for line in open(os.path.join(labels_dir, f"{int(index):06d}.txt"), 'r').readlines()])
            dic["num_points"] = len(pc)

            if self.mode == "train":
                dic["img"] = 0
            else:
                img = read_image(os.path.join(img_dir, f"{int(index):06d}.png"))
                dic["img"] = ToTensor()(img)

            dic["tag"] = f"{int(index):06d}"

            dic.update(process_pointcloud(pc, self.cfg))

        if self.mode in ["train", "eval", "sample_test"]:
            dic["pos_equal_one"], dic["neg_equal_one"], dic["targets"] = cal_rpn_target(dic["labels"][np.newaxis, ...].astype(str), self.cfg.MAP_SHAPE, self.anchors, self.cfg.DETECT_OBJECT, 'lidar')

            dic["pos_equal_one_reg"] = np.concatenate([np.tile(dic["pos_equal_one"][..., [0]], 7), np.tile(dic["pos_equal_one"][..., [1]], 7)], axis=-1)[0]
            dic["pos_equal_one_sum"] = np.clip(np.sum(dic["pos_equal_one"], axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)[0]
            dic["neg_equal_one_sum"] = np.clip(np.sum(dic["neg_equal_one"], axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)[0]

            dic["pos_equal_one"], dic["neg_equal_one"], dic["targets"] = dic["pos_equal_one"][0], dic["neg_equal_one"][0], dic["targets"][0]
        else:
            dic["pos_equal_one"], dic["neg_equal_one"], dic["targets"] = 0, 0, 0
            dic["pos_equal_one_reg"], dic["pos_equal_one_sum"], dic["neg_equal_one_sum"] = 0, 0, 0
        dic["labels"] = self.label_encoder.fit_transform(dic["labels"])
        dic.pop("labels")
        return dic

def create_data_loader(cfg, params, buffer_size, mode, is_aug_data, label_encoder, create_anchors=False):
    train_dataset = KittiDataset(cfg, params, buffer_size, mode, is_aug_data, label_encoder, create_anchors)
    
    from torch.utils.data import random_split
    train_split = 0.8
    train_dataset, val_dataset = random_split(train_dataset, [train_split, 1.0 - train_split])
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_threads"], collate_fn=default_collate)
    val_loader   = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_threads"], collate_fn=default_collate)
    return train_loader, val_loader

# Usage:
# data_loader = create_data_loader(cfg, params, buffer_size, mode, is_aug_data)
# for batch in data_loader:
#     # Process the batch


# JG: The purpose is to create a dataset that can be used with the dataloader
class PointCloudImageDataset(Dataset):
    def __init__(self, point_cloud_dir, image_dir, transform=None):
        """
        Args:
            point_cloud_dir (string): Directory with all the point clouds.
            image_dir (string): Directory with all the corresponding images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.point_cloud_dir = point_cloud_dir
        self.image_dir = image_dir
        self.transform = transform
        self.file_names = [f for f in os.listdir(point_cloud_dir) if os.path.isfile(os.path.join(point_cloud_dir, f))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        pc_file = os.path.join(self.point_cloud_dir, self.file_names[idx])
        img_file = os.path.join(self.image_dir, self.file_names[idx].replace('.pcd', '.png'))  # Assuming image files are .png

        # Load point cloud data
        point_cloud = np.load(pc_file)  # Modify this based on your point cloud data format

        # Load image data
        image = Image.open(img_file)
        image = np.array(image)

        sample = {'point_cloud': point_cloud, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample