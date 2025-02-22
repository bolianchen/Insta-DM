from __future__ import division
import os
import cv2
import shutil
import numpy as np
from PIL import Image
import torch
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def viz_flow(u, v, logscale=True, scaledown=6, output=False):
    """
    topleft is zero, u is horiz, v is vertical
    red is 3 o'clock, yellow is 6, light blue is 9, blue/purple is 12
    """
    colorwheel = makecolorwheel()
    ncols = colorwheel.shape[0]

    radius = np.sqrt(u**2 + v**2)
    if output:
        print("Maximum flow magnitude: %04f" % np.max(radius))
    if logscale:
        radius = np.log(radius + 1)
        if output:
            print("Maximum flow magnitude (after log): %0.4f" % np.max(radius))
    radius = radius / scaledown    
    if output:
        print("Maximum flow magnitude (after scaledown): %0.4f" % np.max(radius))
    rot = np.arctan2(-v, -u) / np.pi

    fk = (rot+1)/2 * (ncols-1)  # -1~1 maped to 0~ncols
    k0 = fk.astype(np.uint8)       # 0, 1, 2, ..., ncols

    k1 = k0+1
    k1[k1 == ncols] = 0

    f = fk - k0

    ncolors = colorwheel.shape[1]
    img = np.zeros(u.shape+(ncolors,))
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]
        col1 = tmp[k1]
        col = (1-f)*col0 + f*col1
       
        idx = radius <= 1
        # increase saturation with radius
        col[idx] = 1 - radius[idx]*(1-col[idx])
        # out of range    
        col[~idx] *= 0.75
        img[:,:,i] = np.floor(255*col).astype(np.uint8)
    
    return img.astype(np.uint8)
   
    
def makecolorwheel():
    # Create a colorwheel for visualization
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    
    ncols = RY + YG + GC + CB + BM + MR
    
    colorwheel = np.zeros((ncols,3))
    
    col = 0
    # RY
    colorwheel[0:RY,0] = 1
    colorwheel[0:RY,1] = np.arange(0,1,1./RY)
    col += RY
    
    # YG
    colorwheel[col:col+YG,0] = np.arange(1,0,-1./YG)
    colorwheel[col:col+YG,1] = 1
    col += YG
    
    # GC
    colorwheel[col:col+GC,1] = 1
    colorwheel[col:col+GC,2] = np.arange(0,1,1./GC)
    col += GC
    
    # CB
    colorwheel[col:col+CB,1] = np.arange(1,0,-1./CB)
    colorwheel[col:col+CB,2] = 1
    col += CB
    
    # BM
    colorwheel[col:col+BM,2] = 1
    colorwheel[col:col+BM,0] = np.arange(0,1,1./BM)
    col += BM
    
    # MR
    colorwheel[col:col+MR,2] = np.arange(1,0,-1./MR)
    colorwheel[col:col+MR,0] = 1

    return colorwheel


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0,1,low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0,max_value,resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:,i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    return array


def save_checkpoint(epoch, save_freq, save_path, dispnet_state, ego_pose_state, obj_pose_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'ego_pose', 'obj_pose']
    states = [dispnet_state, ego_pose_state, obj_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))

    if save_freq != 0:
        if epoch % save_freq == 0:
            for (prefix, state) in zip(file_prefixes, states):
                torch.save(state, save_path/'{}_{}_{}'.format(prefix, epoch, filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))

def overlay_images_with_masks(path, ext='.png'):
    """Overlay images with segmentation masks

    This function is to find all the *.png(images) and *-fseg.npy(masks) pairs
    and generate new images by overlapping masks over images.
    
    Args:
        path: path to the root folder to search valid pairs from top to bottom
    """
    for entry in os.scandir(path):
        if entry.name.endswith(ext):
            img_path = entry.path
            mask_path = os.path.join(
                    os.path.dirname(img_path),
                    os.path.basename(img_path).replace(ext, '-fseg.npy'))
            if os.path.exists(mask_path):
                img = cv2.imread(img_path)
                mask = np.any(np.load(mask_path), axis=0)
                overlay = np.zeros(mask.shape + (3,), dtype=np.uint8)
                overlay[mask] = [0, 0, 255]
                img_overlaid_with_mask = cv2.addWeighted(
                        img, 0.5, overlay, 0.5, 0)
                cv2.imwrite(mask_path.replace('.npy', '.png'),
                            img_overlaid_with_mask)
        elif entry.is_dir(follow_symlinks=False):
            overlay_images_with_masks(entry, ext)

def load_intrinsics(folder, frame_index):
    """Load 3x3 camera intrinsics from a text file
    
    There are 9 entries in the same line separated by comma
    """
    intrinsics_file = os.path.join(folder, frame_index + '_cam.txt')
    f = open(intrinsics_file, 'r') 
    arr = np.array([ [float(e) for e in l.split(',')] for l in f.readlines() ])
    arr = arr.reshape(3,3)
    return arr.astype(np.float32)

def scale_array(arr, scale_factor):
    arr = arr.copy()
    min_element = np.min(arr)
    max_element = np.max(arr)
    arr -= min_element
    arr /= max_element - min_element
    img = Image.fromarray((arr * 255).astype(np.uint8))
    new_size = int(img.size[0] * scale_factor), int(img.size[1] * scale_factor)
    img = img.resize(new_size)
    arr = np.array(img)
    arr = arr / 255.0
    arr *= max_element - min_element
    arr += min_element
    return np.clip(arr, min_element, max_element)

if __name__ == '__main__':

    test_path = '/home/bryanchen/Desktop/hdd/Dataset/DEPTH/INSTA_DM/kitti'
    overlay_images_with_masks(test_path)
    import pdb; pdb.set_trace()


