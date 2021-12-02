import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32, io
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import MeshOcclusionAwareGenerator

from scipy.spatial import ConvexHull
import os
import ffmpeg
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def preprocess_mesh(m, frame_idx):
    roi = [0, 267, 13, 14, 269, 270, 17, 146, 402, 405, 409, 415, 37, 39, 40, 178, 181, 310, 311, 312, 185, 314, 317, 61, 191, 318, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
    res = m.copy()
    for key in res.keys():
        # print('{} shape: {}'.format(key, torch.tensor(res[key][frame_idx]).shape))
        res[key] = torch.tensor(res[key][frame_idx])[None].float().cuda()
    # print('raw shape: {}'.format(res['normed_mesh'].shape))
    res['value'] = res['normed_mesh'][:, roi, :2]
    return res

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = MeshOcclusionAwareGenerator(**config['model_params']['generator_params'], **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)

    generator.eval()
    
    return generator

def get_dataset(path):
    roi_list = [0, 267, 13, 14, 269, 270, 17, 146, 402, 405, 409, 415, 37, 39, 40, 178, 181, 310, 311, 312, 185, 314, 317, 61, 191, 318, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
    video_name = os.path.basename(path)
    frames = sorted(os.listdir(os.path.join(path, 'img')))
    num_frames = len(frames)
    frame_idx = range(num_frames)

    normed_mesh_array = [np.array(list(torch.load(os.path.join(path, 'mesh_dict_normalized', frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
    normed_mesh_array = np.array(normed_mesh_array, dtype='float32') / 128 - 1
    normed_mesh_array = torch.from_numpy(normed_mesh_array).float() # B x N x 3
    out = {}
    out['driving_mesh'] = {'normed_mesh': normed_mesh_array}
    return out

def extract_prior(driving_mesh, generator):
    # driving_mesh: L x N x 3
    motion_prior = generator.module.dense_motion_network.motion_prior
    with torch.no_grad():
        key_pool = []
        mesh_pool = []
        for frame_idx in tqdm(range(driving_mesh['normed_mesh'].shape[0])):
            kp_driving = preprocess_mesh(driving_mesh, frame_idx)
            prior = motion_prior(kp_driving['value'].flatten(start_dim=-2)).detach() # 1 x prior_dim
            key_pool.append(prior)
            mesh_pool.append(kp_driving['normed_mesh']) # 1 x N x 3
            # out = generator(source_frame, kp_source=kp_source, kp_driving=kp_driving, driving_mesh_image=driving_mesh_frame, driving_image=driving_frame)
            # predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

    return key_pool, mesh_pool

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--vid_dir", default='../datasets/train_kkj/kkj04.mp4', help="video directory")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
    parser.add_argument("--result_video", default='recon.mp4', help="path to output")
 
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--use_raw", action="store_true", help="use raw dataset")
 

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    generator = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    print('constructing dataset...')
    dataset = get_dataset(opt.vid_dir)
    print('dataset of size {} constructed'.format(dataset['driving_mesh']['normed_mesh'].shape))
    key_pool, mesh_pool = extract_prior(dataset['driving_mesh'], generator)
    key_pool = torch.cat(key_pool, dim=0)   # L x prior_dim
    mesh_pool = torch.cat(mesh_pool, dim=0) # L x N x 3
    torch.save(key_pool, os.path.join(opt.vid_dir, 'key_pool.pt'))
    torch.save(mesh_pool, os.path.join(opt.vid_dir, 'mesh_pool.pt'))
    print('predicted key pool of size: {}'.format(key_pool.shape))
    print('predicted mesh pool of size: {}'.format(mesh_pool.shape))