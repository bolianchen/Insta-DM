import os
import math
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms as transforms
import flow_vis

def writeFlowFile(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!");
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())

class ProcessedImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, img_ext):
        super().__init__(
            root, 
            torchvision.transforms.ToTensor(),
            is_valid_file=make_image_checker(img_ext)
        )
    
    def __getitem__(self, index):
        # Load processed image
        path, _ = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)

        return image, path

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def convert_file_img_paths(root_dir, filename, ext='.png'):
    lines = readlines(os.path.join(root_dir, filename))
    img_paths = []
    for line in lines:
        folder, img_name = line.split()
        img_path = os.path.join(root_dir, folder, img_name + ext)
        if not os.path.exists(img_path):
            raise RuntimeError(f'{img_path} does not exist')
        img_paths.append(img_path)
    return img_paths

class FlowDataset(Dataset):
    """Dataset to sample image and its path from a given list of image paths
    """
    def __init__(self, img_paths, divisor=64, forward=True):
        super().__init__()
        self.img_paths = img_paths
        self.divisor = divisor
        self.forward = forward
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def get_pair(self, img_path):
        """Get the paths of the image pair for computing optical flow
        """
        shift = 1 if self.forward else -1
        folder = os.path.dirname(img_path)
        idx, ext = os.path.basename(img_path).split('.')
        idx_length = len(idx)
        
        paired_img_path = os.path.join(
                os.path.dirname(img_path),
                str(int(idx)+shift).zfill(idx_length) + '.' + ext)
        if self.forward:
            return [img_path, paired_img_path]
        else:
            return [img_path, paired_img_path]
    
    def process_pair(self, img_paths):
        # read in BGR format
        imgs = [cv2.imread(img_path) for img_path in img_paths]
        self.h, self.w, _ = imgs[0].shape
        h = int(math.ceil(self.h/self.divisor) * self.divisor)
        w = int(math.ceil(self.w/self.divisor) * self.divisor)
        imgs = [cv2.resize(img, (w, h)) for img in imgs]
        imgs = (np.concatenate(imgs, axis=-1)/255.0).astype(np.float32)
        imgs = np.transpose(imgs, (2, 0, 1))
        imgs = torch.from_numpy(imgs)
        return imgs

    def get_rawimg_hw(self):
        return self.h, self.w
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_paths = self.get_pair(img_path)
        imgs = self.process_pair(img_paths)

        return imgs, img_path

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
