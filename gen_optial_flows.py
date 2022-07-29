import os
import sys
import argparse
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.ndimage import imread
import flow_vis
import pwc_net.models as models
from pwc_net.utils import viz_flow, writeFlowFile, convert_file_img_paths, FlowDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default= '/home/bryanchen/Coding/Projects/ss_depth/DATASETS/video_data',
                        help='root directory of data')
    parser.add_argument('--divisor',
                        type=float,
                        default=64.0,
                        help='width and height of the input tensor to '
                             'optical flow models are rescaled to its multiples')
    parser.add_argument('--ckpt', type=str,
                        choices = ['./pwc_net/pwc_net_chairs.pth.tar',
                                  './pwc_net/pwc_net.pth.tar'],
                        default= './pwc_net/pwc_net.pth.tar',
                        help='path to the checkpoint')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of dataloader workers')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to use')

    args = parser.parse_args()
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}')

    net = models.pwc_dc_net(args.ckpt)
    net = net.to(device)
    net.eval()

    for file_name in ['train_files.txt', 'val_files.txt']:
        img_paths = convert_file_img_paths(args.root_dir, file_name)
        
        for orientation in ['forward', 'backward']:
            dataset = FlowDataset(img_paths, divisor=args.divisor,
                                  forward=orientation=='forward')
            dataloader = DataLoader(dataset, args.batch_size, shuffle=False,
                                    pin_memory=True, drop_last=False)

            for data, paths in dataloader:
                data = data.to(device)
                flows = net(data) * 20.0
                flows = flows.cpu().data.numpy()
                flows = np.transpose(flows, (0, 2, 3, 1))
                height, width = data.shape[2:]
                raw_height, raw_width = dataset.get_rawimg_hw()

                for flow, path in zip(flows, paths):
                    u = cv2.resize(flow[:,:,0], (raw_width, raw_height))
                    v = cv2.resize(flow[:,:,1], (raw_width, raw_height))
                    u *= raw_width/width
                    v *= raw_height/height
                    flow = np.dstack((u, v))
                    flow_img = flow_vis.flow_to_color(flow, convert_to_bgr=True)
                    folder = os.path.dirname(path)
                    idx, ext = os.path.basename(path).split('.')
                    if orientation == 'forward':
                        flow_file = idx + 'f.flo'
                        flow_img_file = idx + f'f.{ext}'
                    elif orientation == 'backward':
                        flow_file = idx + 'b.flo'
                        flow_img_file = idx + f'b.{ext}'

                    writeFlowFile(os.path.join(folder, flow_file), flow) 
                    cv2.imwrite(os.path.join(folder, flow_img_file), flow_img)

