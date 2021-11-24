import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from scipy.spatial import ConvexHull
import numpy as np

from sync_batchnorm import DataParallelWithCallback

import ffmpeg

def preprocess_mesh(m):
    roi = [0, 267, 13, 14, 269, 270, 17, 146, 402, 405, 409, 415, 37, 39, 40, 178, 181, 310, 311, 312, 185, 314, 317, 61, 191, 318, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
    res = dict()
    res['value'] = m[:, roi, :2]
    return res

def animate(config, generator, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'animation')
    png_dir = os.path.join(log_dir, 'png')
    animate_params = config['animate_params']

    # dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)

    generator.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            video = x['video']  # B x C x T x H x W
            mesh = x['mesh']    # B x T X :
            driving_video = x['driving_video']
            driving_mesh = x['driving_mesh']    # B x T x :
            driving_mesh_img = x['driving_mesh_img']

            for frame_idx in tqdm(range(video.shape[2])):
                driving_mesh_image = driving_mesh_img[:, :, frame_idx]
                driving_frame = driving_video[:, :, frame_idx]
                source_frame = video[:, :, frame_idx]
                kp_driving = preprocess_mesh(driving_mesh['mesh'][:, frame_idx])
                kp_source = preprocess_mesh(mesh['mesh'][:, frame_idx])
                out = generator(source_frame, kp_source=kp_source, kp_driving=kp_driving, driving_mesh_image=driving_mesh_image, driving_image=driving_frame)

                del out['sparse_deformed']

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source_frame,
                                                                                    driving=driving_frame, out=out)
                visualization = visualization
                visualizations.append(visualization)

            predictions = np.concatenate(predictions, axis=1)
            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            imageio.imsave(os.path.join(png_dir, result_name + '.png'), (255 * predictions).astype(np.uint8))

            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations, fps=25)
            data_dir = os.path.join(config['dataset_params']['root_dir'], x['driving_name'][0])
            ffmpeg.output(ffmpeg.input(os.path.join(log_dir, image_name)), ffmpeg.input(os.path.join(data_dir, 'audio.wav')), os.path.join(data_dir, "animation.mp4")).overwrite_output().run()
