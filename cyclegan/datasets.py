import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, rootA, rootB, transforms_=None, unaligned=True):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        
        with open(rootA) as f1:
            self.files_A = f1.read().splitlines()
        with open(rootB) as f2:
            self.files_B = f2.read().splitlines()

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))