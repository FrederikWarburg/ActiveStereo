import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess 

import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.npy',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):

    BASELINE=0.25
    FOCAL_LENGTH=320

    dataL = np.load(path)
    dataL = BASELINE * FOCAL_LENGTH / dataL

    return dataL



class TartanairLoader(data.Dataset):
    def __init__(self, datapath, normalize, split = 'train', loader=default_loader, dploader= disparity_loader):
 
        train_left_img, train_right_img, train_left_depth = dataloader(datapath, split)
        print(len(train_left_img))

        self.left = train_left_img
        self.right = train_right_img
        self.depth_L = train_left_depth
        self.loader = loader
        self.dploader = dploader
        self.training = True if split == 'train' else False
        self.normalize =normalize

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        depth_L= self.depth_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(depth_L)

        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(left_img)
        plt.subplot(1,3,2)
        plt.imshow(right_img)
        plt.subplot(1,3,3)
        plt.imshow(dataL)
        plt.show()
        """
        if False: #self.training:  
           w, h = left_img.size
           th, tw = 50, 50
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False, normalize=self.normalize)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return left_img, right_img, dataL
        else:
           w, h = left_img.size

           dataL = np.ascontiguousarray(dataL,dtype=np.float32) #/256

           processed = preprocess.get_transform(augment=False, normalize=self.normalize)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)

def dataloader(filepath, split):

    left, right, depth_left = [], [], []

    path = os.path.join(filepath, split)
    for env in os.listdir(path):
        env_path = os.path.join(path, env, 'Easy')
        for seq in os.listdir(env_path):
            seq_path = os.path.join(env_path, seq)
            left.extend( [os.path.join(seq_path, 'cam0', 'data', file) for file in os.listdir(os.path.join(seq_path, 'cam0', 'data'))] )
    
    right = [file.replace("cam0", "cam1") for file in left]
    depth_left = [file.replace("cam0", "ground_truth/depth0").replace(".png",".npy") for file in left]

            #depth_left.extend( [os.path.join(seq_path, 'ground_truth', 'depth0', 'data', file) for file in os.listdir(os.path.join(seq_path, 'ground_truth', 'depth0', 'data'))])
            #depth_right = [os.path.join(seq_path, 'ground_truth', 'depth1', 'data') for file in os.listdir(os.path.join(seq_path, 'ground_truth', 'depth1', 'data'))]


    return left, right, depth_left
