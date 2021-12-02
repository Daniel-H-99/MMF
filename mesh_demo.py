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
from modules.util import mesh_tensor_to_landmarkdict, draw_mesh_images, interpolate_zs

from scipy.spatial import ConvexHull
import os
import ffmpeg
import cv2
import pickle as pkl

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def preprocess_mesh(m, frame_idx):
    roi = [0, 267, 13, 14, 269, 270, 17, 146, 402, 405, 409, 415, 37, 39, 40, 178, 181, 310, 311, 312, 185, 314, 317, 61, 191, 318, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
    res = m.copy()
    for key in res.keys():
        # print('{} shape: {}'.format(key, torch.tensor(res[key][frame_idx]).shape))
        res[key] = torch.tensor(res[key][frame_idx])[None].float().cuda()
    # print('raw shape: {}'.format(res['normed_mesh'].shape))
    if 'audio' in res:
        res['value'] = res['audio']
    else:
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
    reference_mesh_dict = torch.load(os.path.join(path, 'mesh_dict_reference.pt'))
    reference_normed_mesh_dict = torch.load(os.path.join(path, 'mesh_dict_reference.pt'))
    reference_frame = img_as_float32(io.imread(reference_frame_path))
    reference_mesh = np.array(list(reference_mesh_dict.values())[:478])
    reference_normed_mesh = np.array(list(reference_normed_mesh_dict.values())[:478])
    reference_R = np.array(reference_mesh_dict['R'])
    reference_t = np.array(reference_mesh_dict['t'])
    reference_c = np.array(reference_mesh_dict['c'])
    reference_normed_z = torch.load(os.path.join(path, 'z_reference_normalized.pt'))
    video_array = [reference_frame for idx in frame_idx]
    mesh_array = [reference_mesh for idx in frame_idx]
    normed_mesh_array = [reference_normed_mesh for idx in frame_idx]
    z_array = [reference_normed_z for idx in frame_idx]
    R_array = [reference_R for idx in frame_idx]
    t_array = [reference_t for idx in frame_idx]
    c_array = [reference_c for idx in frame_idx]

    if opt.use_raw:
        audio_array = []

        for i in range(len(frame_idx)):
            fid = int(frames[frame_idx[i]][:-4]) - 1
            audio_array.append(audio_pool[fid])
        audio_array = np.array(audio_array, dtype='float32')

        mesh_dict = 'mesh_dict'
        normed_mesh_dict = 'mesh_dict_normalized'
        driving_mesh_array = [np.array(list(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
        driving_normed_mesh_array = [np.array(list(torch.load(os.path.join(path, normed_mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
        driving_mesh_img_array = [img_as_float32(io.imread(os.path.join(path, 'mesh_image', frames[idx]))) for idx in frame_idx]
        driving_video_array = [img_as_float32(io.imread(os.path.join(path, 'img', frames[idx]))) for idx in frame_idx]
        driving_z_array = [torch.load(os.path.join(path, 'z', frames[idx].replace('.png', '.pt'))) for idx in frame_idx]
        driving_R_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['R']) for idx in frame_idx]
        driving_t_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['t']) for idx in frame_idx]
        driving_c_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['c']) for idx in frame_idx]

    else:
        audio_array = None
        mesh_dict = 'mesh_dict_reenact'
        normed_mesh_dict = 'mesh_dict_reenact_normalized'
        driving_mesh_array = [np.array(list(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
        driving_normed_mesh_array = [np.array(list(torch.load(os.path.join(path, normed_mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
        driving_mesh_img_array = [img_as_float32(io.imread(os.path.join(path, 'mesh_image_reenact', frames[idx]))) for idx in frame_idx]
        driving_video_array = [img_as_float32(io.imread(os.path.join(path, 'img', frames[idx]))) for idx in frame_idx]
        driving_z_array = [torch.load(os.path.join(path, 'z_reenact', frames[idx].replace('.png', '.pt'))) for idx in frame_idx]
        driving_R_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['R']) for idx in frame_idx]
        driving_t_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['t']) for idx in frame_idx]
        driving_c_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['c']) for idx in frame_idx]

    video_array = np.array(video_array, dtype='float32')
    mesh_array = np.array(mesh_array, dtype='float32') / 128 - 1
    normed_mesh_array = np.array(normed_mesh_array, dtype='float32') / 128 - 1
    R_array = np.array(R_array, dtype='float32')
    c_array = np.array(c_array, dtype='float32') * 128
    t_array = np.array(t_array, dtype='float32')
    t_array = t_array + np.matmul(R_array, (c_array[:, None, None] * np.ones_like(t_array)))
    z_array = torch.stack(z_array, dim=0).float() / 128 - 1
    out = {}


    driving_video_array = np.array(driving_video_array, dtype='float32')
    driving_mesh_img_array = np.array(driving_mesh_img_array, dtype='float32')
    driving_mesh_array = np.array(driving_mesh_array, dtype='float32') / 128 - 1
    driving_normed_mesh_array = np.array(driving_normed_mesh_array, dtype='float32') / 128 - 1
    driving_z_array = torch.stack(driving_z_array, dim=0).float() / 128 - 1
    driving_R_array = np.array(driving_R_array, dtype='float32')
    driving_c_array = np.array(driving_c_array, dtype='float32') * 128
    driving_t_array = np.array(driving_t_array, dtype='float32')
    driving_t_array = driving_t_array + np.matmul(driving_R_array, (driving_c_array[:, None, None] * np.ones_like(driving_t_array)))


    video = video_array
    out['video'] = video.transpose((3, 0, 1, 2))
    out['mesh'] = {'mesh': mesh_array, 'normed_mesh': normed_mesh_array, 'R': R_array, 't': t_array, 'c': c_array, 'normed_z': z_array}
    out['driving_video'] = driving_video_array.transpose((3, 0, 1, 2))
    out['driving_mesh_img'] = driving_mesh_img_array.transpose((3, 0, 1, 2))
    out['driving_mesh'] = {'mesh': driving_mesh_array, 'normed_mesh': driving_normed_mesh_array, 'R': driving_R_array, 't': driving_t_array, 'c': driving_c_array, 'z': driving_z_array, 'audio': audio_array}
    out['driving_name'] = video_name
    out['source_name'] = video_name
    return out

def make_animation(source_video, driving_video, source_mesh, driving_mesh, driving_mesh_img, generator, relative=True, adapt_movement_scale=True, cpu=False, pool=None):
    with torch.no_grad():
        predictions = []
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

            out = generator(source_frame, kp_source=kp_source, kp_driving=kp_driving, driving_mesh_image=driving_mesh_frame, driving_image=driving_frame, pool=pool)
            predictions.append(np.transpose(out['deformed'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            searched_mesh.append(out['searched_mesh'])
            normed_mesh.append(kp_driving['normed_mesh'])

            filename = '{:05d}.pt'.format(frame_idx + 1)
            R = kp_driving['R'][0].cuda()
            RT = R.transpose(0, 1)
            t = kp_driving['t'][0].cuda()
            c = kp_driving['c'][0].cuda()

            base = 128 * (kp_driving['normed_mesh'] + 1)
            geometry = 128 * (out['searched_mesh'].view(-1, 3) + 1)
            normlaised_geometry = geometry.clone().detach()
            normalised_landmark_dict = mesh_tensor_to_landmarkdict(normlaised_geometry)
            
            geometry = (torch.matmul(RT, (geometry.transpose(0, 1) - t)) / c).transpose(0, 1).cpu().detach()
            geometry = 128 * (geometry + 1)
            landmark_dict = mesh_tensor_to_landmarkdict(geometry)
            landmark_dict.update({'R': R.cpu().numpy(), 't': t.cpu().numpy(), 'c': c.cpu().numpy()})
            torch.save(normalised_landmark_dict, os.path.join(opt.vid_dir,'mesh_dict_searched_normalized',filename))
            torch.save(landmark_dict, os.path.join(opt.vid_dir, 'mesh_dict_searched', filename))
    
    return predictions, searched_mesh, normed_mesh

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

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
 

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    fps = 25

    os.makedirs(os.path.join(opt.vid_dir, 'mesh_dict_searched'), exist_ok=True)
    os.makedirs(os.path.join(opt.vid_dir, 'mesh_dict_searched_normalized'), exist_ok=True)

    generator = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    dataset = get_dataset(opt.vid_dir)
    pool = (torch.load(os.path.join(opt.vid_dir, 'key_pool.pt')).cuda(), torch.load(os.path.join(opt.vid_dir, 'mesh_pool.pt')).cuda())
    predictions, searched_mesh, normed_mesh = make_animation(dataset['video'], dataset['driving_video'], dataset['mesh'], dataset['driving_mesh'], dataset['driving_mesh_img'], generator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu, pool=pool)
    os.makedirs(os.path.join(opt.vid_dir, 'demo_img'), exist_ok=True)
    for i, pred in tqdm(enumerate(predictions)):
        cv2.imwrite(os.path.join(opt.vid_dir, 'demo_img', '{:05d}.png'.format(i + 1)), img_as_ubyte(pred)[:, :, [2, 1, 0]])
    imageio.mimsave(os.path.join(opt.vid_dir, 'pre_' + opt.result_video), [img_as_ubyte(frame) for frame in predictions], fps=fps)
    if opt.use_raw:
        audio_name = '_audio.wav'
    else:
        audio_name = 'audio.wav'
    ffmpeg.output(ffmpeg.input(os.path.join(opt.vid_dir, 'pre_' + opt.result_video)), ffmpeg.input(os.path.join(opt.vid_dir, audio_name)), os.path.join(opt.vid_dir, opt.result_video)).overwrite_output().run()
    searched_mesh = torch.cat(searched_mesh, dim=0)
    normed_mesh = torch.cat(normed_mesh, dim=0)
    eval_loss = 100 * F.l1_loss(searched_mesh.flatten(start_dim=-2), normed_mesh.flatten(start_dim=-2))
    print('used temperature: {}'.format(generator.module.dense_motion_network.T))
    print('eval loss: {}'.format(eval_loss))
    # searched_mesh = 128 * (searched_mesh + 1)
    # normed_mesh = 128 * (normed_mesh + 1)
    # searched_mesh = mix_mesh_tensor(searched_mesh, normed_mesh)
    # save_searched_mesh(searched_mesh, os.path.join(opt.vid_dir, 'mesh_dict_search'))

    image_rows = image_cols = 256
    draw_mesh_images(os.path.join(opt.vid_dir, 'mesh_dict_searched_normalized'), os.path.join(opt.vid_dir, 'mesh_image_searched_normalized'), image_rows, image_cols)
    draw_mesh_images(os.path.join(opt.vid_dir, 'mesh_dict_searched'), os.path.join(opt.vid_dir, 'mesh_image_searched'), image_rows, image_cols)
    interpolate_zs(os.path.join(opt.vid_dir, 'mesh_dict_searched'), os.path.join(opt.vid_dir, 'z_searched'), image_rows, image_cols)
    interpolate_zs(os.path.join(opt.vid_dir, 'mesh_dict_searched_normalized'), os.path.join(opt.vid_dir, 'z_searched_normalized'), image_rows, image_cols)

