"""
Copyright (C) University of Science and Technology of China.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', type=str, help='Path to datasets')
parser.add_argument('--dataset', type=str, default='ade20k', help='which dataset to process')
parser.add_argument('--norm', type=str, default='norm', help='normalization mode of dist map')

def cal_connectedComponents(mask, normal_mode='norm'):
    label_idxs = np.unique(mask)
    H, W = mask.shape
    out_h_offset = np.float32(np.zeros_like(mask))
    out_w_offset = np.float32(np.zeros_like(mask))
    for label_idx in label_idxs:
        if label_idx == 0:
            continue
        tmp_mask = np.float32(mask.copy())
        tmp_mask[tmp_mask!=label_idx] = -1
        tmp_mask[tmp_mask==label_idx] = 255
        tmp_mask[tmp_mask==-1] = 0
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(tmp_mask))
        connected_numbers = len(centroids)-1
        for c_idx in range(1,connected_numbers+1):
            tmp_labels = np.float32(labels.copy())
            tmp_labels[tmp_labels!=c_idx] = 0
            tmp_labels[tmp_labels==c_idx] = 1
            h_offset = (np.repeat(np.array(range(H))[...,np.newaxis],W,1) - centroids[c_idx][1])*tmp_labels
            w_offset = (np.repeat(np.array(range(W))[np.newaxis,...],H,0) - centroids[c_idx][0])*tmp_labels
            h_offset = normalize_dist(h_offset, normal_mode)
            w_offset = normalize_dist(w_offset, normal_mode)
            out_h_offset += h_offset
            out_w_offset += w_offset

    return out_h_offset, out_w_offset

def normalize_dist(offset, normal_mode):
    if normal_mode == 'no':
        return offset
    else:
        return offset / np.max(np.abs(offset)+1e-5)

def show_results(ins):
    plt.imshow(ins)
    plt.show()

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def check_mask(mask, check_idx = 255):
    idx = np.unique(mask)
    if check_idx in idx:
        print(idx)

def make_dist_train_val_ade_datasets(dir, norm='norm'):
    out_root = os.path.join(dir, 'distances')
    make_dir(out_root)
    phases = ['training','validation']
    for phase in phases:
        print('process',phase,'dataset')
        label_path = os.path.join(dir, 'annotations', phase)
        label_names = sorted(os.listdir(label_path))
        out_path = os.path.join(out_root, phase)
        make_dir(out_path)
        for label_name in label_names:
            print(label_name)
            mask = np.array(Image.open(os.path.join(label_path, label_name)))
            h_offset, w_offset = cal_connectedComponents(mask, norm)
            dist_cat_np = np.concatenate((h_offset[np.newaxis, ...], w_offset[np.newaxis, ...]), 0)
            np.save(os.path.join(out_path, label_name[:-4]+'.npy'), dist_cat_np)

def make_dist_train_val_coco_datasets(dir = '/home/tzt/HairSynthesis/SPADE/datasets/cocostuff/dataset/',norm='nrom'):
    phases = ['train','val']
    for phase in phases:
        print('process',phase,'dataset')
        label_path = os.path.join(dir,phase+'_label')
        label_names = sorted(os.listdir(label_path))
        out_path = os.path.join(dir,phase+'_dist')
        make_dir(out_path)
        for label_name in label_names:
            print(label_name)
            mask = np.array(Image.open(os.path.join(label_path, label_name)))
            h_offset, w_offset = cal_connectedComponents(mask, norm)
            dist_cat_np = np.concatenate((h_offset[np.newaxis, ...], w_offset[np.newaxis, ...]), 0)
            np.save(os.path.join(out_path, label_name[:-4]+'.npy'), dist_cat_np)

def make_dist_train_val_cityscapes_datasets(dir = '/home/tzt/dataset/cityscapes/',norm='norm'):
    label_dir = os.path.join(dir, 'gtFine')
    phases = ['val','train']
    for phase in phases:
        if 'test' in phase:
            continue
        print('process',phase,'dataset')
        citys = sorted(os.listdir(os.path.join(label_dir,phase)))
        for city in citys:
            label_path = os.path.join(label_dir, phase, city)
            label_names_all = sorted(os.listdir(label_path))
            label_names = [p for p in label_names_all if p.endswith('_labelIds.png')]
            for label_name in label_names:
                print(label_name)
                mask = np.array(Image.open(os.path.join(label_path, label_name)))
                # check_mask(mask)
                h_offset, w_offset = cal_connectedComponents(mask, norm)
                dist_cat_np = np.concatenate((h_offset[np.newaxis, ...], w_offset[np.newaxis, ...]), 0)
                dist_name = label_name[:-12]+'distance.npy'
                np.save(os.path.join(label_path, dist_name), dist_cat_np)

if __name__ == '__main__':
    print('Start ...')

    args = parser.parse_args()
    if args.dataset == 'ade20k':
        make_dist_train_val_ade_datasets(args.path, args.norm)
    elif args.dataset == 'coco':
        make_dist_train_val_coco_datasets(args.path, args.norm)
    elif args.dataset == 'cityscapes':
        make_dist_train_val_cityscapes_datasets(args.path, args.norm)
    else:
        print('Error! dataset must be one of [ade20k|coco|cityscapes]')
