import logging
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2 

def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    handler = logging.StreamHandler()
    handler.setFormatter(file_formatter)
    logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) is not '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def setup_tensorboard(filepath):

    writer = SummaryWriter()

    return writer


def log_images(log, writer, outputs, imgL, disp_L, iter_, save_path, split = 'train'):

    rgb = imgL.cpu()
    img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    rgb.mul_(img_std.type_as(rgb)).add_(img_mean.type_as(rgb))
    rgb = rgb[0,:,:,:].data.cpu().numpy()

    _, H, W = outputs[0].shape

    all_results = np.zeros((len(outputs)+2, 3, H, W))
    for j in range(len(outputs)):
        all_results[j, :, :, :] = visualize(outputs[j][0, :, :].detach().cpu().numpy(), size = (W, H), sizeb = (W, H), normalize=True) 
    all_results[-2, :, :, :] = visualize(disp_L[:, :].detach().cpu().numpy(), size = (W, H), sizeb = (W, H), normalize=True)
    all_results[-1, :, :, :] = rgb
    
    all_results = torch.from_numpy(all_results)

    torchvision.utils.save_image(all_results, os.path.join(save_path, split, "iter-%d.jpg" % iter_))
    writer.add_images(split + "/all-results", all_results, iter_)

def visualize(im, sizeb, size, normalize = False):

    cm = plt.get_cmap('plasma')

    Wb, Hb = sizeb
    W,H = size
    im = im.reshape(Hb, Wb)
    im = cv2.resize(im, (W,H), interpolation=cv2.INTER_LINEAR)

    if normalize:
        im = (im - np.min(im)) / (np.max(im) - np.min(im) + 1e-6)
    im = 255.0 * im 
    im = cm(im.astype('uint8'))
    im = np.transpose(im[:, :, :3], (2, 0, 1))

    return im 