from tqdm import trange
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from logger import Logger
from modules.discriminator import LipDiscriminator, NoiseDiscriminator
from modules.discriminator import Encoder
from modules.util import landmarkdict_to_mesh_tensor, mesh_tensor_to_landmarkdict, LIP_IDX, get_seg, draw_mesh_images, interpolate_zs
from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from torch.utils.data import Dataset
import argparse
import os
import shutil
import numpy as np
import random
import pickle as pkl
import math
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_dir', type=str, default='../datasets/kkj_v2/test/studio_1_34.mp4')
parser.add_argument('--driving_dir', type=str, default='../datasets/kkj_v2/test/studio_1_34.mp4')
parser.add_argument('--reference', type=str, default=None)
parser.add_argument('--driving_reference', type=str, default=None)
parser.add_argument('--result_dir', type=str, default='studio_1_6.mp4')




args = parser.parse_args()

LIP_DIM = 3 * len(LIP_IDX)

lip_dict_dir = os.path.join(args.data_dir, 'lip_dict_normalized')
if os.path.exists(lip_dict_dir):
    shutil.rmtree(lip_dict_dir)
os.makedirs(lip_dict_dir)

def get_mapping_function(lip_pool, lip_drving_pool):
    # lip_ref, lip_driving_ref: lip_dim
    # lip_pool: P x lip_dim
    # lip_driving_pool: P' x lip_dim
    # all normalized by [-1, 1]
    var, mean = torch.var_mean(lip_pool, dim=0, unbiased=False)
    var *= 3
    var_driving, mean_driving = torch.var_mean(lip_driving_pool, dim=0, unbiased=True)
    mean_driving = lip_driving_pool[0]
    # var, mean: lip_dim
    A = (var / var_driving.clamp(min=1e-6)).sqrt() # lip_dim
    B = mean - A * mean_driving # lip_dim
    print('constructing mapping function: A={}, B={}'.format(A, B))
    return lambda x: A * x + B


lip_pool = torch.load(os.path.join(args.data_dir, 'mesh_stack.pt'))[:, LIP_IDX].flatten(1) / 128
lip_driving_pool = torch.load(os.path.join(args.driving_dir, 'mesh_stack.pt'))[:, LIP_IDX].flatten(1) / 128

mapping_func = get_mapping_function(lip_pool, lip_driving_pool)



for i, lip in enumerate(tqdm(lip_driving_pool)):
    # lip: lip_dim
    key = '{:05d}'.format(i + 1)
    mapped_lip = mapping_func(lip)  # lip_dim
    torch.save(mapped_lip.view(-1, 3) * 128, os.path.join(lip_dict_dir, key + '.pt'))




    

