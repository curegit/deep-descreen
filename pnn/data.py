import torch
from torch.utils.data import Dataset

import os


class CustomImageArrayDataset(Dataset):
    def __init__(self, npy_img_dir):
        super().__init__()
        self.img_dir = npy_img_dir
        files = glob()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class 
