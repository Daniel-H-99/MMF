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
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm import DataParallelWithCallback

from modules.generator import MeshOcclusionAwareGenerator
from modules.util import mesh_tensor_to_landmarkdict, draw_mesh_images, interpolate_zs, mix_mesh_tensor, LIP_IDX, MASK_IDX, WIDE_MASK_IDX, get_lip_mask

from scipy.spatial import ConvexHull
import os
import ffmpeg
import cv2
import pickle as pkl

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def preprocess_mesh(m, frame_idx):
    roi = [0, 267, 13, 14, 269, 270, 17, 146, 402, 405, 409, 415, 37, 39, 40, 178, 181, 310, 311, 312, 185, 314, 317, 61, 191, 318, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
    res = m.copy()
    for key in res.keys():
        # print('{} shape: {}'.format(key, torch.tensor(res[key][frame_idx]).shape))
        res[key] = torch.tensor(res[key][frame_idx])[None].float().cuda()
    # print('raw shape: {}'.format(res['normed_mesh'].shape))
    if 'normed_lip' in res:
        res['mesh'][:, roi] = res['normed_lip']
        res['normed_mesh'][:, roi] = res['normed_lip']
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
    d = generator.state_dict()
    d.update(checkpoint['generator'])
    generator.load_state_dict(d)
    
    if not cpu:
        generator = DataParallelWithCallback(generator)

    generator.eval()
    
    return generator

def get_dataset(path):
    roi_list = [0, 267, 13, 14, 269, 270, 17, 146, 402, 405, 409, 415, 37, 39, 40, 178, 181, 310, 311, 312, 185, 314, 317, 61, 191, 318, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
    video_name = os.path.basename(path)
    frames = sorted(os.listdir(os.path.join(path, 'img')))
    num_frames = min(len(frames), 500)
    frame_idx = range(num_frames)

    audio_pool = []
    frames = sorted(os.listdir(os.path.join(path, 'img')))
    num_frames = min(len(frames), len(os.listdir(os.path.join(path, 'audio'))))
    frame_idx = range(num_frames)
    for i in range(len(frame_idx)):
        with open(os.path.join(path, 'audio', '{:05d}.pickle'.format(int(frames[frame_idx[i]][:-4]) - 1)), 'rb') as f:
            mspec = pkl.load(f)
            audio_pool.append(mspec)
    audio_pool = np.array(audio_pool).astype(np.float32)

    reference_frame_path = os.path.join(path, 'frame_reference.png')
    reference_mesh_img_path = os.path.join(path, 'mesh_image_reference.png')
    reference_mesh_dict = torch.load(os.path.join(path, 'mesh_dict_reference.pt'))
    reference_normed_mesh_dict = torch.load(os.path.join(path, 'mesh_dict_reference.pt'))
    reference_frame = img_as_float32(io.imread(reference_frame_path))
    reference_mesh_img = img_as_float32(io.imread(reference_mesh_img_path))
    reference_mesh = np.array(list(reference_mesh_dict.values())[:478])
    reference_normed_mesh = np.array(list(reference_normed_mesh_dict.values())[:478])
    reference_R = np.array(reference_mesh_dict['R'])
    reference_t = np.array(reference_mesh_dict['t'])
    reference_c = np.array(reference_mesh_dict['c'])
    reference_normed_z = torch.load(os.path.join(path, 'z_reference_normalized.pt'))
    video_array = [reference_frame for idx in frame_idx]
    mesh_img_array = [reference_mesh_img for idx in frame_idx]
    mesh_array = [reference_mesh for idx in frame_idx]
    normed_mesh_array = [reference_normed_mesh for idx in frame_idx]
    z_array = [reference_normed_z for idx in frame_idx]
    R_array = [reference_R for idx in frame_idx]
    t_array = [reference_t for idx in frame_idx]
    c_array = [reference_c for idx in frame_idx]

    audio_array = []

    for i in range(len(frame_idx)):
        fid = int(frames[frame_idx[i]][:-4]) - 1
        audio_array.append(audio_pool[fid])
    audio_array = np.array(audio_array, dtype='float32')

    mesh_dict = 'mesh_dict'
    normed_mesh_dict = 'mesh_dict_normalized'
    lip_dict = 'lip_dict_normalized'
    driving_lip_array = [torch.load(os.path.join(path, lip_dict, frames[idx].replace('.png', '.pt'))).cpu().numpy() for idx in frame_idx]
    lip_mask_array = [get_lip_mask(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt'))), (256, 256, 3)) for idx in frame_idx]
    
    _mesh_array = [np.array(list(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
    _normed_mesh_array = [np.array(list(torch.load(os.path.join(path, normed_mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
    _R_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['R']) for idx in frame_idx]
    _t_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['t']) for idx in frame_idx]
    _c_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['c']) for idx in frame_idx]
    
    frames = ['frame_reference.png' for _ in range(len(frames))]

    driving_mesh_array = [np.array(list(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
    driving_normed_mesh_array = [np.array(list(torch.load(os.path.join(path, normed_mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
    driving_mesh_img_array = [img_as_float32(io.imread(os.path.join(path, 'mesh_image', frames[idx]))) for idx in frame_idx]
    driving_video_array = [img_as_float32(io.imread(os.path.join(path, 'img', frames[idx]))) for idx in frame_idx]
    driving_z_array = [torch.load(os.path.join(path, 'z', frames[idx].replace('.png', '.pt'))) for idx in frame_idx]
    driving_R_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['R']) for idx in frame_idx]
    driving_t_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['t']) for idx in frame_idx]
    driving_c_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['c']) for idx in frame_idx]


    video_array = np.array(video_array, dtype='float32')
    mesh_img_array = np.array(mesh_img_array, dtype='float32')
    mesh_array = np.array(mesh_array, dtype='float32') / 128 - 1
    normed_mesh_array = np.array(normed_mesh_array, dtype='float32') / 128 - 1
    R_array = np.array(R_array, dtype='float32')
    c_array = np.array(c_array, dtype='float32') * 128
    t_array = np.array(t_array, dtype='float32')
    t_array = t_array + np.matmul(R_array, (c_array[:, None, None] * np.ones_like(t_array)))
    z_array = torch.stack(z_array, dim=0).float() / 128 - 1
    out = {}


    driving_video_array = np.array(driving_video_array, dtype='float32')
    driving_mesh_img_array = np.array(driving_mesh_img_array, dtype='float32') * np.array(lip_mask_array, dtype='float32')
    driving_mesh_array = np.array(driving_mesh_array, dtype='float32') / 128 - 1
    driving_normed_mesh_array = np.array(driving_normed_mesh_array, dtype='float32') / 128 - 1
    driving_z_array = torch.stack(driving_z_array, dim=0).float() / 128 - 1
    driving_R_array = np.array(driving_R_array, dtype='float32')
    driving_c_array = np.array(driving_c_array, dtype='float32') * 128
    driving_t_array = np.array(driving_t_array, dtype='float32')
    driving_t_array = driving_t_array + np.matmul(driving_R_array, (driving_c_array[:, None, None] * np.ones_like(driving_t_array)))
    driving_lip_array = np.array(driving_lip_array, dtype='float32') / 128 - 1


    _mesh_array = np.array(_mesh_array, dtype='float32') / 128 - 1
    _normed_mesh_array = np.array(_normed_mesh_array, dtype='float32') / 128 - 1
    _R_array = np.array(_R_array, dtype='float32')
    _c_array = np.array(_c_array, dtype='float32') * 128
    _t_array = np.array(_t_array, dtype='float32')
    _t_array = _t_array + np.matmul(_R_array, (_c_array[:, None, None] * np.ones_like(_t_array)))

    video = video_array
    out['video'] = video.transpose((3, 0, 1, 2))
    out['mesh'] = {'mesh': mesh_array, 'normed_mesh': normed_mesh_array, 'R': R_array, 't': t_array, 'c': c_array, 'normed_z': z_array, 'normed_lip': driving_lip_array}
    out['driving_video'] = driving_video_array.transpose((3, 0, 1, 2))
    out['driving_mesh_img'] = driving_mesh_img_array.transpose((3, 0, 1, 2))
    out['driving_mesh'] = {'mesh': driving_mesh_array, 'normed_mesh': driving_normed_mesh_array, 'R': driving_R_array, 't': driving_t_array, 'c': driving_c_array, 'z': driving_z_array, 'driving_mesh': _mesh_array, 'driving_normed_mesh': _normed_mesh_array, 'driving_R': _R_array, 'driving_t': _t_array, 'driving_c': _c_array, 'audio': audio_array}
    out['driving_name'] = video_name
    out['source_name'] = video_name
    return out

def make_animation(source_video, driving_video, source_mesh, driving_mesh, driving_mesh_img, generator, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        source = torch.tensor(np.array(source_video)[np.newaxis].astype(np.float32))
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32))
        driving_mesh_img = torch.tensor(np.array(driving_mesh_img)[np.newaxis].astype(np.float32))

        searched_mesh = []
        normed_mesh = []
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_mesh_frame = driving_mesh_img[:, :, frame_idx]
            source_frame = source[:, :, frame_idx]

            kp_driving = preprocess_mesh(driving_mesh, frame_idx)
            kp_source = preprocess_mesh(source_mesh, frame_idx)

            if not cpu:
                driving_frame = driving_frame.cuda()
                source_frame = source_frame.cuda()
                # kp_driving['value'] = kp_driving['value'].cuda()
                # kp_source['value'] = kp_source['value'].cuda()

            out = generator(source_frame, kp_source=kp_source, kp_driving=kp_driving, driving_mesh_image=driving_mesh_frame, driving_image=driving_frame)
            normed_mesh.append(kp_driving['normed_mesh'])

            driving_mesh_pos = kp_driving['mesh'].cuda()[:, MASK_IDX, None, :2] # B x K x 1 x 2
            driving_mesh_normalized_pos = kp_driving['normed_mesh'].cuda()[:, MASK_IDX]
            motion = kp_driving['driving_normed_mesh'].cuda()
            motion[:, MASK_IDX, :2] = F.grid_sample(out['deformation'].permute(0, 3, 1, 2), driving_mesh_pos).squeeze(3).permute(0, 2, 1)   # B x K x 2
            motion[:, MASK_IDX, 2] = driving_mesh_normalized_pos[:, :, 2]
            # motion[LIP_IDX] = kp_driving['normed_lip']
            searched_mesh.append(motion)
            filename = '{:05d}.pt'.format(frame_idx + 1)
            R = kp_driving['driving_R'][0].cuda()
            RT = R.transpose(0, 1)
            t = kp_driving['driving_t'][0].cuda()
            c = kp_driving['driving_c'][0].cuda()

            t -= torch.matmul(R, (c * torch.ones_like(t)))
            c /= 128
            R = R

            # normalized_base = 128 * (kp_driving['normed_mesh'][0] + 1)
            # base = 128 * (kp_driving['mesh'][0] + 1)
            geometry = 128 * (motion.view(-1, 3) + 1)
            normalised_geometry = geometry.clone().detach().cpu()
            # normalised_geometry = mix_mesh_tensor(normalised_geometry, normalized_base.cpu())
            normalised_landmark_dict = mesh_tensor_to_landmarkdict(normalised_geometry)
            
            geometry = (torch.matmul(RT, (geometry.transpose(0, 1) - t)) / c).transpose(0, 1).cpu().detach()
            
            # mix with original geometry
            _geometry = kp_driving['driving_mesh'][0].cpu()
            _geometry[MASK_IDX] = geometry[MASK_IDX]
            geomtery = _geometry
            # geometry = mix_mesh_tensor(geometry, base.cpu())
            landmark_dict = mesh_tensor_to_landmarkdict(geometry)
            landmark_dict.update({'R': R.cpu().numpy(), 't': t.cpu().numpy(), 'c': c.cpu().numpy()})
            torch.save(normalised_landmark_dict, os.path.join(opt.vid_dir,'mesh_dict_searched_normalized',filename))
            torch.save(landmark_dict, os.path.join(opt.vid_dir, 'mesh_dict_searched', filename))
    
    return searched_mesh, normed_mesh

def save_searched_mesh(searched_mesh_batch, save_dir):
    # searched_mesh_batch: L x N * 3
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(len(searched_mesh_batch))):
        mesh = searched_mesh_batch[i].view(-1, 3)   # N x 3
        mesh_dict = mesh_tensor_to_landmarkdict(mesh)
        torch.save(mesh_dict, os.path.join(save_dir, '{:05d}.pt'.format(i + 1)))
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--vid_dir", default='../datasets/test_kkj/kkj04_1.mp4', help="video directory")

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
    parser.add_argument("--T", type=float, default=1.0, help="temperature for searching")
 

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    fps = 25

    generator = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    generator.module.dense_motion_network.prior_from_audio = False
    generator.module.dense_motion_network.T = opt.T
    dataset = get_dataset(opt.vid_dir)
    pca_pool = torch.load(os.path.join(opt.vid_dir, 'mesh_pca.pt'))
    pool = (pca_pool[0].cuda(), torch.load(os.path.join(opt.vid_dir, 'mesh_pool.pt')).cuda(), pca_pool[2].cuda())
    searched_mesh, normed_mesh = make_animation(dataset['video'], dataset['driving_video'], dataset['mesh'], dataset['driving_mesh'], dataset['driving_mesh_img'], generator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    searched_mesh = torch.cat(searched_mesh, dim=0)
    normed_mesh = torch.cat(normed_mesh, dim=0)
    eval_loss = 100 * F.l1_loss(searched_mesh.flatten(start_dim=-2), normed_mesh.flatten(start_dim=-2))
    print('used temperature: {}'.format(generator.module.dense_motion_network.T))
    print('eval loss: {}'.format(eval_loss))

    image_rows = image_cols = 256
    draw_mesh_images(os.path.join(opt.vid_dir, 'mesh_dict_searched_normalized'), os.path.join(opt.vid_dir, 'mesh_image_searched_normalized'), image_rows, image_cols)
    draw_mesh_images(os.path.join(opt.vid_dir, 'mesh_dict_searched'), os.path.join(opt.vid_dir, 'mesh_image_searched'), image_rows, image_cols)
    interpolate_zs(os.path.join(opt.vid_dir, 'mesh_dict_searched'), os.path.join(opt.vid_dir, 'z_searched'), image_rows, image_cols)
    interpolate_zs(os.path.join(opt.vid_dir, 'mesh_dict_searched_normalized'), os.path.join(opt.vid_dir, 'z_searched_normalized'), image_rows, image_cols)

