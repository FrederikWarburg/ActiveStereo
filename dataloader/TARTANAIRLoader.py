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
import torchvision.transforms.functional as TF

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

    dataL = np.load(path)
    dataL = 0.25 * 320 / dataL

    return dataL

def custom_transform(im, flip, degree, scale, h_start, w_start, interpolation = Image.BICUBIC):

    # Horizontal flip
    #if flip > 0.5:
    #    TF.hflip(im)

    # Rotation
    #im = TF.rotate(im, angle=degree, resample=interpolation)

    # Resize
    im = TF.resize(im, scale, interpolation)

    # Crop
    #im = TF.crop(im, h_start, w_start, 480, 640)

    return im

def custom_transform_K(K, flip, degree, _scale, h_start, w_start):

    # Horizontal flip
    #if flip > 0.5:
    #    K[2] = 480 - K[2]

    # Scale
    K[0] = K[0] * _scale
    K[1] = K[1] * _scale
    K[2] = K[2] * _scale
    K[3] = K[3] * _scale

    # Crop
    #K[2] = K[2] - w_start
    #K[3] = K[3] - h_start

    return K

def custom_color_transform(rgb):

    # Color jitter
    brightness = np.random.uniform(0.6, 1.4)
    contrast = np.random.uniform(0.6, 1.4)
    saturation = np.random.uniform(0.6, 1.4)

    rgb = TF.adjust_brightness(rgb, brightness)
    rgb = TF.adjust_contrast(rgb, contrast)
    rgb = TF.adjust_saturation(rgb, saturation)

    return rgb

class TartanairLoader(data.Dataset):
    def __init__(self, datapath, normalize, split = 'train', loader=default_loader, dploader= disparity_loader):
 
        left_img, right_img, left_depth = dataloader(datapath, split)

        self.left = left_img
        self.right = right_img
        self.depth_L = left_depth
        self.loader = loader
        self.dploader = dploader
        self.training = True if split == 'train' else False
        self.normalize =normalize
        self.K = np.asarray([320.0,320.0,320.0,240.0])


    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        depth_L= self.depth_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(depth_L)
        K = self.K

        if self.training:  

            #TODO: Hozizontal flipping means left becomes right?
            #TODO: Rotation happens around wrong axis

            #_scale = np.random.uniform(1.0, 1.5)
            #scale = np.int(height * _scale)
            #degree = 0 #np.random.uniform(-5.0, 5.0)
            #flip = 0 #np.random.uniform(0.0, 1.0)

            #h_start = random.randint(0, 80)
            #w_start = random.randint(0, 120)

            #dataL = Image.fromarray(dataL)
            #K = custom_transform_K(K, flip, degree, _scale, h_start, w_start)
            #left_img = custom_transform(left_img, flip, degree, scale, h_start, w_start, Image.BICUBIC)
            #right_img = custom_transform(right_img, flip, degree, scale, h_start, w_start, Image.BICUBIC)
            #dataL = custom_transform(dataL, flip, degree, scale, h_start, w_start, Image.NEAREST)

            left_img_aug = custom_color_transform(left_img)
            right_img_aug = custom_color_transform(right_img)

            processed = preprocess.get_transform(augment=False, normalize=self.normalize)  
            left_img       = processed(left_img)
            right_img      = processed(right_img)
            left_img_aug   = processed(left_img_aug)
            right_img_aug  = processed(right_img_aug)

            dataL = np.ascontiguousarray(dataL, dtype=np.float32)
            #dataL = dataL / _scale

            return left_img, right_img, dataL, K, left_img_aug, right_img_aug
        else:
           dataL = np.ascontiguousarray(dataL,dtype=np.float32) #/256

           processed = preprocess.get_transform(augment=False, normalize=self.normalize)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return left_img, right_img, dataL, K, left_img, right_img

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
