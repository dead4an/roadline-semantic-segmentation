import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RoadlineTrainDataset(Dataset):
    def __init__(self, img_dir, ann_dir, orig_shape=(1080, 1920), resize_shape=(416, 416),
                 device='cpu'):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_filenames = [filename for filename in os.listdir(self.img_dir)]
        self.orig_shape = orig_shape
        self.resize_shape = resize_shape
        self.device = device

    def __getitem__(self, idx):
        img = self.get_image(idx)
        mask = self.get_mask(idx)

        return img, mask
    
    def __len__(self):
        return len(self.img_filenames)

    def get_image(self, idx):
        img_filename = os.path.join(self.img_dir, self.img_filenames[idx])
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.orig_shape)   # resize to orig shape
        img = cv2.resize(img, self.resize_shape) # resize to model shape
        img = img.astype(np.float32)
        img /= 255.0

        img = torch.tensor(img, dtype=torch.float32)
        img = torch.permute(img, (2, 0, 1))
        
        return img.to(self.device)
    
    def get_mask(self, idx):
        """Returns binary mask for image."""
        # define img and its annotation paths
        img_filename = self.img_filenames[idx]
        ann_filename = os.path.join(self.ann_dir, f'{img_filename}.json')
        
        # load annotation dict
        with open(ann_filename, 'rb') as f:
            ann = json.load(f)

        polygones = self.parse_mask_from_ann(ann)

        # create and fill binary mask
        # [::-1] so shape is in cv2 format (height, width)
        mask = np.zeros(self.orig_shape[::-1], dtype=np.uint8)
        for polygone in polygones:
            points = np.array(polygone, dtype=np.int32)
            cv2.fillPoly(mask, [points], color=1)

        # resize to img shape
        mask = cv2.resize(mask, self.resize_shape)
        mask = torch.tensor(mask, dtype=torch.float32)

        return mask.to(self.device)

    def parse_mask_from_ann(self, ann):
        """Parses polygones list from json annotation."""
        objects = ann['objects']

        polygones = []
        for object in objects:
            poly_coords = object['points']['exterior']
            polygones.append(poly_coords)

        return polygones
