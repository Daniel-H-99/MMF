import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationWithMeshTransform
import glob
import random

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class MeshFramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_train=True,
                 random_seed=0, id_sampling=False, pairs_list=None, augmentation_params=None, num_dummy_set=0):
        self.root_dir = root_dir
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.num_dummy_set = num_dummy_set
        # if os.path.exists(os.path.join(root_dir, 'train')):
        #     assert os.path.exists(os.path.join(root_dir, 'test'))
        #     print("Use predefined train-test split.")
        #     train_videos = os.listdir(os.path.join(root_dir, 'train'))
        #     test_videos = os.listdir(os.path.join(root_dir, 'test'))
        #     self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        # else:
        #     print("Use random train-test split.")
        #     train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        self.videos = list(filter(lambda x: x.endswith('.mp4'), os.listdir(root_dir)))
        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationWithMeshTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):

        name = self.videos[idx]
        path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)
    
        frames = sorted(os.listdir(os.path.join(path, 'img')))
        num_frames = len(frames)
        frame_idx = np.random.choice(num_frames, replace=True, size=2) if self.is_train else range(min(500, num_frames))

        if self.is_train:
            mesh_dicts = [torch.load(os.path.join(path, 'mesh_dict', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
            mesh_dicts_normed = [torch.load(os.path.join(path, 'mesh_dict_normalized', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
            R_array = [np.array(mesh_dict['R']) for mesh_dict in mesh_dicts]
            t_array = [np.array(mesh_dict['t']) for mesh_dict in mesh_dicts]
            c_array = [np.array(mesh_dict['c']) for mesh_dict in mesh_dicts]
            mesh_array = [np.array(list(mesh_dict.values())[:478]) for mesh_dict in mesh_dicts]
            normed_mesh_array = [np.array(list(mesh_dict_normed.values())[:478]) for mesh_dict_normed in mesh_dicts_normed]
            z_array = [torch.load(os.path.join(path, 'z', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
            normed_z_array = [torch.load(os.path.join(path, 'z_normalized', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
            video_array = [img_as_float32(io.imread(os.path.join(path, 'img', frames[frame_idx[i]]))) for i in range(len(frame_idx))]
            mesh_img_array = [img_as_float32(io.imread(os.path.join(path, 'mesh_image', frames[frame_idx[i]]))) for i in range(len(frame_idx))]
            if self.num_dummy_set > 0:
                dummy_idx = random.randrange(self.num_dummy_set)
                dummy_mesh_dict = f'mesh_dict_dummy_{dummy_idx}'
                dummy_img_dir = f'mesh_image_dummy_{dummy_idx}'
                R_array.append(R_array[0])
                t_array.append(t_array[0])
                c_array.append(c_array[0])
                mesh_array.append(np.array(list(torch.load(os.path.join(path, dummy_mesh_dict, frames[frame_idx[0]].replace('.png', '.pt'))).values())[:478]))
                mesh_img_array.append(img_as_float32(io.imread(os.path.join(path, dummy_img_dir, frames[frame_idx[0]]))))
                video_array.append(mesh_img_array[-1])
            else:
                R_array.append(R_array[1])
                t_array.append(t_array[1])
                c_array.append(c_array[1])
                mesh_array.append(mesh_array[1])
                normed_mesh_array.append(normed_mesh_array[1])
                video_array.append(img_as_float32(io.imread(os.path.join(path, 'img', frames[frame_idx[1] - 1]))))
                mesh_img_array.append(mesh_img_array[1])
                z_array.append(z_array[1])
                normed_z_array.append(normed_z_array[1])

            if self.transform is not None:
                video_array, mesh_array, R_array, t_array, c_array, mesh_img_array = self.transform(video_array, mesh_array, R_array, t_array, c_array, mesh_img_array)
        else:
            # mesh_dict = 'mesh_dict'
            # video_array = [img_as_float32(io.imread(os.path.join(path, 'img', frames[idx]))) for idx in frame_idx]
            # mesh_array = [np.array(list(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
            # R_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['R']) for idx in frame_idx]
            # t_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['t']) for idx in frame_idx]
            # c_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['c']) for idx in frame_idx]
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
        if self.is_train:
            mesh_img_array = np.array(mesh_img_array, dtype='float32')
            normed_z_array = torch.stack(normed_z_array, dim=0).float() / 128 - 1
        mesh_array = np.array(mesh_array, dtype='float32') / 128 - 1
        normed_mesh_array = np.array(normed_mesh_array, dtype='float32') / 128 - 1
        R_array = np.array(R_array, dtype='float32')
        c_array = np.array(c_array, dtype='float32') * 128
        t_array = np.array(t_array, dtype='float32')
        t_array = t_array + np.matmul(R_array, (c_array[:, None, None] * np.ones_like(t_array)))
        z_array = torch.stack(z_array, dim=0).float() / 128 - 1

        out = {}
        if self.is_train:
            source = video_array[0]
            real = video_array[1]
            driving = video_array[2]
            source_mesh = mesh_array[0]
            real_mesh = mesh_array[1]
            driving_mesh = mesh_array[2]
            source_normed_mesh = normed_mesh_array[0]
            real_normed_mesh = normed_mesh_array[1]
            driving_normed_mesh = normed_mesh_array[2]
            source_R = R_array[0]
            real_R = R_array[1]
            driving_R = R_array[2]
            source_t = t_array[0]
            real_t = t_array[1]
            driving_t = t_array[2]
            source_c = c_array[0]
            real_c = c_array[1]
            driving_c = c_array[2]
            source_mesh_image = mesh_img_array[0]
            real_mesh_image = mesh_img_array[1]
            driving_mesh_image = mesh_img_array[2]
            source_z = z_array[0]
            real_z = z_array[1]
            driving_z = z_array[2]
            source_normed_z = normed_z_array[0]
            real_normed_z = normed_z_array[1]
            driving_normed_z = normed_z_array[2]

            out['driving'] = driving.transpose((2, 0, 1))
            out['real'] = real.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            out['driving_mesh'] = {'mesh': driving_mesh, 'normed_mesh': driving_normed_mesh, 'R': driving_R, 't': driving_t, 'c': driving_c, 'z': driving_z, 'normed_z': driving_normed_z}
            out['real_mesh'] = {'mesh': real_mesh, 'normed_mesh': real_normed_mesh, 'R': real_R, 't': real_t, 'c': real_c, 'z': real_z, 'normed_z': real_normed_z}
            out['source_mesh'] = {'mesh': source_mesh, 'normed_mesh': source_normed_mesh, 'R': source_R, 't': source_t, 'c': source_c, 'z': source_z, 'normed_z': source_normed_z}
            out['driving_mesh_image'] = driving_mesh_image.transpose((2, 0, 1))
            out['real_mesh_image'] = real_mesh_image.transpose((2, 0, 1))
            out['source_mesh_image'] = source_mesh_image.transpose((2, 0, 1))
            out['name'] = video_name
        else:
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
            out['driving_mesh'] = {'mesh': driving_mesh_array, 'normed_mesh': driving_normed_mesh_array, 'R': driving_R_array, 't': driving_t_array, 'c': driving_c_array, 'z': driving_z_array}
            out['driving_name'] = video_name
            out['source_name'] = video_name
        return out

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
