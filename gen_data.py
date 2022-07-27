# Copyright All Rights Reserved.

"""Generates data for training/validation and save it to disk."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import multiprocessing
import os
import argparse
import yaml
from absl import app
from absl import flags
from absl import logging
from torch.utils import data
import numpy as np
import imageio
import torch
import torchvision
from tqdm import tqdm

from datasets.auxiliary_datasets import ProcessedImageFolder
from datasets.data_prep.video_loader import Video
from datasets.data_prep.kitti_loader import KittiRaw

parser = argparse.ArgumentParser(
   description='Options for training data generation')
parser.add_argument('--dataset_name',
                    type=str, 
                    choices=['video', 'kitti_raw_eigen',
                             'kitti_raw_stereo'],
                    default='video',
                    help='what raw dataset to convert'
                         'video: videos in mp4 format')
parser.add_argument('--dataset_dir',
                    type=str, default='./raw_data',
                    help='location of the folder containing the '
                         'raw dataset')
parser.add_argument('--save_dir',
                    type=str, default='./generated_data',
                    help='location to save the generated '
                         'training data.')
parser.add_argument('--save_img_ext',
                    type=str, choices=['png', 'jpg'],
                    default='png',
                    help='what image format to save')
parser.add_argument('--seq_length',
                    type=int, default=3,
                    help='number of images of each training '
                         'sequence')
parser.add_argument('--img_height',
                    type=int, default=128,
                    help='height of the generated images')
parser.add_argument('--img_width',
                    type=int, default=416,
                    help='width of the generated images')
parser.add_argument('--data_format',
                    type=str, choices=['mono2', 'struct2depth'],
                    default='mono2',
                    help='mono2: a single generated image is '
                         'converted from a single raw image'
                         'struct2depth: a single generated image'
                         ' is a concatenation of raw images in a '
                         'training sequence')
parser.add_argument('--del_static_frames',
                    action='store_true',
                    help='remove frames when the camera is '
                         'judged as not moving with a heuristic '
                         'algorithm implemented by us')
parser.add_argument('--intrinsics',
                    type=str, default=None,
                    help='a document containing 9 entries '
                         'of the flattened target intrinsics')
parser.add_argument('--trim',
                    nargs=4,
                    type=float, default=[0.0, 0.0, 0.0, 0.0],
                    help='romove the [left, right, top, bottom] '
                         'part of each frame by this proportion'
                         '; this operation WILL NOT induce '
                         'intrinsics adjustment')
parser.add_argument('--crop',
                    nargs=4,
                    type=float, default=[0.0, 0.0, 0.0, 0.0],
                    help='romove the [left, right, top, bottom] '
                         'part of each frame by this proportion'
                         '; this operation WILL induce '
                         'intrinsics adjustment')
parser.add_argument('--augment_strategy',
                    type=str, choices=['none', 'single', 'multi'],
                    default='single',
                    help='multi: augment data with 3 pre-defined '
                         'cropping; '
                         'single: crop images according to '
                         'shift_h '
                         'none: no cropping, for random cropping '
                         'during the training')
parser.add_argument('--augment_shift_h',
                    type=float, default=0.0,
                    help='what proportion from the top to crop '
                         'a frame. this only applies when augment'
                         '_strategy is set to single')
parser.add_argument('--video_start',
                    type=str,
                    default='00:00:00',
                    help='set a start time for the video '
                         'conversion; the format should be '
                         'hh:mm:ss')
parser.add_argument('--video_end',
                    type=str,
                    default='00:00:00',
                    help='set an end time for the video '
                         'conversion; the format should be '
                         'hh:mm:ss; if set to 00:00:00, convert '
                         'the video till the end')
parser.add_argument('--fps',
                    type=int,
                    default=10,
                    help='frames per second to sample from a '
                         ' video to do the conversion')
parser.add_argument('--delete_temp',
                    action='store_false',
                    help='remove temporary images during '
                         'conversion')
parser.add_argument('--num_threads',
                    type=int,
                    help='number of worker threads. the default '
                         ' is the CPU cores.')
parser.add_argument('--batch_size',
                    type=int, default=4,
                    help='batch size to run Mask-RCNN model')
parser.add_argument('--threshold',
                    type=float, default=0.5,
                    help='score threshold for Mask-RCNN model')
parser.add_argument('--mask',
                    type=str, choices=['none', 'mono', 'color', 
                        'instance'], default='mono',
                    help='what segmentation masks to generate '
                         'none: do not generate masks '
                         'mono(HxW): generate binary masks '
                         'color(HxW): pixel values vary on masks by '
                         'object instances '
                         'instance(NxHxW): instance masks in npy '
                         'format with first dimension as instance '
                         'number')
parser.add_argument('--single_process',
                    action='store_true',
                    help='only use a single cpu process '
                         'this option is mainly for debugging')
parser.add_argument('--to_yaml',
                    action='store_true',
                    help='save the options to a yaml file')

FLAGS = parser.parse_args()

NUM_CHUNKS = 100

def _generate_data():
    r"""
    Extract sequences from dataset_dir and store them in save_dir.
    """
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    if FLAGS.to_yaml:
        # Save the options to a YAML configuration in save_dir
        yaml_filename = os.path.join(FLAGS.save_dir, 'config.yaml')
        with open(yaml_filename, 'w') as f:
            yaml.dump(vars(FLAGS), f, default_flow_style=False)

    global dataloader  # pylint: disable=global-variable-undefined
    if FLAGS.dataset_name == 'video':
        dataloader = Video(
            FLAGS.dataset_dir,
            img_height=FLAGS.img_height,
            img_width=FLAGS.img_width,
            seq_length=FLAGS.seq_length,
            data_format=FLAGS.data_format,
            mask=FLAGS.mask,
            batch_size=FLAGS.batch_size,
            threshold=FLAGS.threshold,
            intrinsics=FLAGS.intrinsics,
            trim=FLAGS.trim,
            crop=FLAGS.crop,
            del_static_frames=FLAGS.del_static_frames,
            augment_strategy=FLAGS.augment_strategy,
            augment_shift_h=FLAGS.augment_shift_h,
            fps=FLAGS.fps,
            video_start=FLAGS.video_start,
            video_end=FLAGS.video_end,
            img_ext=FLAGS.save_img_ext
        )
    elif FLAGS.dataset_name == 'kitti_raw_eigen':
        dataloader = KittiRaw(
            FLAGS.dataset_dir,
            split='eigen',
            img_height=FLAGS.img_height,
            img_width=FLAGS.img_width,
            seq_length=FLAGS.seq_length,
            data_format=FLAGS.data_format,
            mask=FLAGS.mask, 
            batch_size=FLAGS.batch_size,
            threshold=FLAGS.threshold
      )
    elif FLAGS.dataset_name == 'kitti_raw_stereo':
        dataloader = KittiRaw(
            FLAGS.dataset_dir,
            split='stereo',
            img_height=FLAGS.img_height,
            img_width=FLAGS.img_width,
            seq_length=FLAGS.seq_length,
            data_format=FLAGS.data_format,
            mask=FLAGS.mask, 
            batch_size=FLAGS.batch_size,
            threshold=FLAGS.threshold
      )
    else:
        raise ValueError('Unknown dataset')

    all_frames = range(dataloader.num_train)
    # Split into training/validation sets. Fixed seed for repeatability.
    np.random.seed(8964)

    num_cores = multiprocessing.cpu_count()
    # number of processes while using multiple processes
    # number of workers for using either a single or multiple processes
    num_threads = num_cores if FLAGS.num_threads is None else FLAGS.num_threads

    if FLAGS.single_process:
        frame_chunks = list(all_frames)
    else:
        frame_chunks = np.array_split(all_frames, NUM_CHUNKS)
        manager = multiprocessing.Manager()
        all_examples = manager.dict()
        pool = multiprocessing.Pool(num_threads)

    with open(os.path.join(FLAGS.save_dir, 'train_files.txt'), 'w') as train_f:
        with open(os.path.join(FLAGS.save_dir, 'val_files.txt'), 'w') as val_f:
            logging.info('Generating data...')

            for index, frame_chunk in enumerate(frame_chunks):
                if FLAGS.single_process:
                    all_examples = _gen_example(frame_chunk, {})
                    if all_examples is None:
                        continue
                else:
                    all_examples.clear()
                    pool.map(
                        _gen_example_star,
                        zip(frame_chunk, itertools.repeat(all_examples))
                    )
                    logging.info(
                        'Chunk %d/%d: saving %s entries...', 
                        index + 1, NUM_CHUNKS, len(all_examples)
                    )
                for _, example in all_examples.items():
                    if example:
                        s = example['folder_name']
                        frame = example['file_name']
                        if np.random.random() < 0.1:
                            val_f.write('%s %s\n' % (s, frame))
                        else:
                            train_f.write('%s %s\n' % (s, frame))

    if not FLAGS.single_process:
        pool.close()
        pool.join()

    if FLAGS.mask != 'none':
        # Collect filenames of all processed images
        img_dataset = ProcessedImageFolder(FLAGS.save_dir,
                                           FLAGS.save_img_ext)
        img_loader = torch.utils.data.DataLoader(
            img_dataset,
            batch_size=FLAGS.batch_size,
            num_workers=num_threads
        )

        # Generate masks by batch
        logging.info('Generating masks...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for imgs, img_filepaths in tqdm(img_loader):
            mrcnn_results = dataloader.run_mrcnn_model(imgs.to(device))
            for i in range(len(imgs)):
                _gen_mask(mrcnn_results[i], img_filepaths[i], FLAGS.save_img_ext)

    if FLAGS.dataset_name=='video' and FLAGS.delete_temp:
        dataloader.delete_temp_images()
  
def _gen_example(i, all_examples=None):
    r"""
    Save one example to file.  Also adds it to all_examples dict.
    """
    add_to_file, example = dataloader.get_example_with_index(i)
    if not example or dataloader.is_bad_sample(i):
        return
    image_seq_stack = _stack_image_seq(example['image_seq'])
    example.pop('image_seq', None)  # Free up memory.
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    save_dir = os.path.join(FLAGS.save_dir, example['folder_name'])
    os.makedirs(save_dir, exist_ok=True)
    img_filepath = os.path.join(save_dir, f'{example["file_name"]}.{FLAGS.save_img_ext}')
    imageio.imsave(img_filepath, image_seq_stack.astype(np.uint8))
    cam_filepath = os.path.join(save_dir, '%s_cam.txt' % example['file_name'])
    example['cam'] = '%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy)
    with open(cam_filepath, 'w') as cam_f:
        cam_f.write(example['cam'])

    if not add_to_file:
        return

    key = example['folder_name'] + '_' + example['file_name']
    all_examples[key] = example
    return all_examples

def _gen_example_star(params):
    return _gen_example(*params)

def _gen_mask(mrcnn_result, img_filepath, save_img_ext):
    f"""
    Generate a mask and save it to file.
    """
    mask_img = dataloader.generate_mask(mrcnn_result)
    if mask_img.ndim == 2:
        mask_filepath = img_filepath[:-(len(save_img_ext)+1)] + f'-fseg.{save_img_ext}'
        imageio.imsave(mask_filepath, mask_img.astype(np.uint8))
    elif mask_img.ndim == 3:
        mask_filepath = img_filepath[:-(len(save_img_ext)+1)] + f'-fseg.npy'
        np.save(mask_filepath, mask_img)

def _gen_mask_star(params):
    return _gen_mask(*params)

def _stack_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res


if __name__ == '__main__':
    _generate_data()
