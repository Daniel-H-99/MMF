from tqdm import trange
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from logger import Logger
from modules.discriminator import LipDiscriminator
from modules.discriminator import Encoder
from modules.util import landmarkdict_to_mesh_tensor, mesh_tensor_to_landmarkdict, LIP_IDX, get_seg, draw_mesh_images, interpolate_zs
from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from torch.utils.data import Dataset
import argparse
import os
import numpy as np
import random
import pickle as pkl
import math
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_dir', type=str, default='../datasets/test_kkj/kkj04_1.mp4')
parser.add_argument('--pool_dir', type=str, default='../datasets/train_kkj/kkj04.mp4')
parser.add_argument('--embedding_dim', type=int, default='512')
parser.add_argument('--lipdisc_path', type=str, default='expert_v3/00010000.pt')
parser.add_argument('--window_size', type=int, default=5)
parser.add_argument('--N', type=int, default=30)
parser.add_argument('--T', type=float, default=1.0)
parser.add_argument('--mode', type=str, default='A')
parser.add_argument('--device_id', type=str, default='1')



args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.device_id


# prepare dataset
class MeshSyncDataset(Dataset):
    def __init__(self, audio, prior):
        # audio: L x : (tensor)
        # prior: L x prior_dim (tensor)
        super(MeshSyncDataset, self).__init__()
        self.audio = torch.cat([audio[:2], audio, audio[-2:]], dim=0)
        self.prior = torch.cat([torch.zeros_like(prior[0]).unsqueeze(0).repeat(2, 1), prior, torch.zeros_like(prior[0]).unsqueeze(0).repeat(2, 1)], dim=0)
    def __len__(self):
        return len(self.audio) - 4
    def __getitem__(self, index):
        return self.audio[index:index + 5], self.prior[index:index + 5]
            
audio_pool = []
path = os.path.join(args.data_dir, 'audio')
# frames = sorted(os.listdir(os.path.join(args.data_dir, 'img')))
audio_frames = sorted(os.listdir(path))
num_frames = len(audio_frames)
# num_frames = 500
frame_idx = range(num_frames)
for i in range(len(frame_idx)):
    with open(os.path.join(path, '{:05d}.pickle'.format(i)), 'rb') as f:
        mspec = pkl.load(f)
        audio_pool.append(mspec)

audio_pool = torch.from_numpy(np.array(audio_pool).astype(np.float32))
prior_pool = torch.load(os.path.join(args.data_dir, 'mesh_pca.pt'))[0]
prior_pool = prior_pool

audio_pool_size = len(audio_pool)
prior_pool_size = len(prior_pool)
size = min(audio_pool_size, prior_pool_size)

audio_pool = audio_pool[:size]
prior_pool = prior_pool[:size]

audio_pool_size = len(audio_pool)
prior_pool_size = len(prior_pool)

assert audio_pool_size <= prior_pool_size, 'the size of audio / prior pool do not match: {} - {}'.format(audio_pool_size, prior_pool_size)
print('Audio Pool Size: {}, Prior Pool: {}'.format(audio_pool_size, prior_pool_size))


dataset = MeshSyncDataset(audio_pool, prior_pool)

# prepare model
lipdisc = LipDiscriminator(prior_dim=20, embedding_dim=args.embedding_dim).cuda()
lipdisc.load_state_dict(torch.load(args.lipdisc_path))
for p in lipdisc.parameters():
    p.requires_grad = False
lipdisc.eval()
torch.backends.cudnn.enabled=False

# setup training
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
loss_fn = nn.L1Loss(reduction='sum')

normed_searched_mesh_dir = os.path.join(args.data_dir, 'mesh_dict_searched_normalized')
normed_searched_mesh_image_dir = os.path.join(args.data_dir, 'mesh_image_searched_normalized')
searched_mesh_dir = os.path.join(args.data_dir, 'mesh_dict_searched')
searched_mesh_image_dir = os.path.join(args.data_dir, 'mesh_image_searched')
z_dir = os.path.join(args.data_dir, 'z_searched')
driving_mesh_dir = os.path.join(args.data_dir, 'mesh_dict')

os.makedirs(normed_searched_mesh_dir, exist_ok=True)
os.makedirs(normed_searched_mesh_image_dir, exist_ok=True)
os.makedirs(searched_mesh_dir, exist_ok=True)
os.makedirs(searched_mesh_image_dir, exist_ok=True)
reference_mesh = torch.load(os.path.join(args.data_dir, 'mesh_dict_reference.pt'))

pca_pool = torch.load(os.path.join(args.pool_dir, 'mesh_pca.pt'))
prior_pool = pca_pool[0].cuda()
pool_S = torch.diag(pca_pool[1].cuda())
pca_V = pca_pool[2].cuda() # N * 3 x pca_dim
mesh_pool = torch.load(os.path.join(args.pool_dir, 'mesh_stack.pt')).cuda()  # B x N x 3
audio_pool = []
path = os.path.join(args.pool_dir, 'audio')
# frames = sorted(os.listdir(os.path.join(args.pool_dir, 'img')))
audio_frames = sorted(os.listdir(path))
num_frames = len(audio_frames)
frame_idx = range(num_frames)
for i in range(len(frame_idx)):
    with open(os.path.join(path, '{:05d}.pickle'.format(i)), 'rb') as f:
        mspec = pkl.load(f)
        audio_pool.append(mspec)

audio_pool = torch.from_numpy(np.array(audio_pool).astype(np.float32))
# key_pool = prior_pool
# key_pool = mesh_pool[:, LIP_IDX]
# # key_pool -= key_pool.mean(dim=1, keepdim=True)
# key_pool = key_pool.flatten(-2)
# prior_pool = prior_pool
# mesh_pool = mesh_pool
pool_dataset = MeshSyncDataset(audio_pool, prior_pool)
pool_dataloader = DataLoader(pool_dataset, batch_size=1, shuffle=False)

key_pool_path = os.path.join(args.data_dir, f'key_pool_{os.path.basename(args.pool_dir)}_{args.mode}.pt')
if os.path.exists(key_pool_path):
    print(f'loading key pool from {key_pool_path}...')
    key_pool = torch.load(key_pool_path)
else:
    print(f'constructing key pool on {key_pool_path}...')
    key_pool = []
    for audio, prior in tqdm(pool_dataloader):
        if args.mode == 'L':
            prior = prior.cuda()
            key_pool.append(lipdisc.prior_encoder(prior))
        else:
            audio = audio.cuda()
            key_pool.append(lipdisc.audio_encoder(audio[:, 2]))

    key_pool = torch.cat(key_pool, dim=0)
    torch.save(key_pool, key_pool_path)

print(f'Pool Size: {len(key_pool)}')

def search_from_pool(audio_prior, pool, N=30, T=0.5, verbose=False):
    result = {}
    # audio_prior: B x prior_dim
    # print('audio prior shape: {}'.format(audio_prior.shape))
    key_pool, mesh_pool = pool[0], pool[1] # P x prior_dim, P x N x 3
    # print(f'pool shape: {key_pool.shape} - {mesh_pool.shape}')
    ### L2 Search
    # weights = audio_prior.unsqueeze(1) - key_pool.unsqueeze(0)  # B x P x prior_dim
    # weights = weights ** 2  # B x P x prior_dim
    # weights = weights.sum(dim=2)    # B x P
    ### Cosine Search
    weights = nn.CosineEmbeddingLoss(reduction='none')(audio_prior.repeat(len(key_pool), 1), key_pool, torch.ones(len(key_pool)).long().cuda())[None]   # B x P

    # ### best choice
    # print('weights: {}'.format(weights.shape))
    # # print('mesh: {}'.format(mesh_pool.shape))
    # # best_index = weights.argmin(dim=1) # B
    # # result = mesh_pool[best_index] # B x N x 3
    # ### weighted sum
    weights, index = weights.sort(dim=1)
    weights = weights[:, :N]
    mesh_pool = mesh_pool[index[:, :N]]
    weights = nn.Softmax(dim=1)(-weights / T) # B x P
    result['mesh'] = torch.einsum('bp,bpni->bni', weights, mesh_pool) # B x N x 3
    # result[:, LIP_IDX] = audio_prior

    if verbose:
        result['weights'] = weights
        result['index'] = index[:, :N]
    # print('results: {}'.format(result.shape))

    ### Lipdisc Search
    # weights = nn.CosineEmbeddingLoss(reduction='none')(audio_prior.repeat(len(key_pool), 1), key_pool, torch.ones(len(key_pool)).long().cuda())  # P
    # ### best choice
    # # best_index = weights.argmin(dim=0) # 1
    # # result = mesh_pool[[[best_index]]] # 1 x N x 3
    # ### weighted sum
    # weights, index = weights.sort(dim=0)
    # weights = weights[:N]
    # mesh_pool = mesh_pool[index[:N]]
    # weights = nn.Softmax(dim=0)(-weights / T) # P
    # result = torch.einsum('p,pni->ni', weights, mesh_pool)[None] # 1 x N x 3
    # if get_key:
    #     new_key = torch.einsum('p,pi->i', weights, key_pool[:N])[None] # 1 x prior_dim
    #     result = (result, new_key)

    ### Online Lipdisc Seach 
    # key_pool: P x 5 x prior_dim
    # audio_prior: 1 x C x H x W
    # weights = []
    # for i in range(0, len(key_pool), 100):
    #     key_pool_chunk = key_pool[i:i+100]
    #     weight = lipdisc(audio_prior.repeat(len(key_pool_chunk), 1, 1, 1), key_pool_chunk, torch.ones(len(key_pool_chunk)).long().cuda())   # P
    #     weights.append(weight)
    # weights = torch.cat(weights, dim=0)
    # encoded_key_pool = lipdisc.prior_encoder(key_pool)   # P x embedding_dim
    # weights = nn.CosineEmbeddingLoss(reduction='none')(audio_prior.repeat(len(encoded_key_pool), 1), encoded_key_pool, torch.ones(len(encoded_key_pool)).long().cuda())  # P

    ### best choice
    # best_index = weights.argmin(dim=0) # 1
    # result = mesh_pool[[[best_index]]] # 1 x N x 3
    # if get_key:
    #     new_key = key_pool[best_index, 2][None] # 1 x prior_dim
    #     result = (result, new_key)
    ## weighted sum
    # weights, index = weights.sort(dim=0)
    # weights = weights[:N]
    # mesh_pool = mesh_pool[index[:N]]
    # weights = nn.Softmax(dim=0)(-weights / T) # P
    # result = torch.einsum('p,pni->ni', weights, mesh_pool)[None] # N x 3
    # print(f'key pool shape: {key_pool.shape}')
    # if get_key:
    #     new_key = torch.einsum('p,pti->ti', weights, key_pool[index[:N]])[None, 2] # 1 x prior_dim
    #     result = (result, new_key)
    return result

class Window():
    def __init__(self, size=5, cache_size=30, prior_pool=None, mesh_pool=None, encoder=None):
        assert key_pool is not None, f'key pool is required'
        assert mesh_pool is not None, f'mesh pool is required'
        assert encoder is not None, f'encoder is required'
        # key_pool: P x prior_dim, mesh_pool: P x N x 3
        self.size = size
        self.cache_size = cache_size
        self.encoder = encoder
        self.prior_dim = prior_pool.size(1)
        self.prior_pool = prior_pool
        self.mesh_pool = mesh_pool
        self.display = torch.zeros(self.size, self.prior_dim).cuda()
        self.weights = torch.zeros(self.size, self.cache_size).cuda() # weights
        self.index = torch.zeros(self.size, self.cache_size).long().cuda() # index
        self.history = []
    def add(self, display, weights, index):
        # display: prior_dim, idx: cahe_size
        if self.weights[0].sum() > 0:
            self.history.append(torch.einsum('p,pni->ni', self.weights[0], self.mesh_pool[self.index[0]]))
        self.display[:4] = self.display.clone()[1:]
        self.weights[:4] = self.weights.clone()[1:]
        self.index[:4] = self.index.clone()[1:]
        self.display[4] = display
        self.weights[4] = weights
        self.index[4] = index
    def update_display(self, display, pos=2):
        # display: prior_dim
        self.display[pos] = display
    def get_search_pool(self, pos=2):
        cache_size = self.size * self.cache_size
        keys = self.display[None].repeat(self.cache_size, 1, 1) # cache_size x size x prior_dim
        keys[:, pos] = self.prior_pool[self.index[pos]]
        keys = self.encoder(keys) # cache_size x embedding_dim
        values = self.mesh_pool[self.index[pos]] # cache_size x N x 3
        return keys, values
    def flush(self):
        for i in range(5):
            self.history.append(torch.einsum('p,pni->ni', self.weights[i], self.mesh_pool[self.index[i]]))
    def get_history(self):
        return self.history
    


item_size = 0
eval_lipdisc_loss = 0
eval_loss = 0

window = Window(size=args.window_size, cache_size=100, prior_pool=prior_pool, mesh_pool=mesh_pool, encoder=lipdisc.prior_encoder)

with torch.no_grad():
    for step, (audio, prior) in tqdm(enumerate(dataloader)):
        # print('input shape: {}'.format((audio.shape, prior.shape, label.shape)))
        chunked_audio_shape = audio.shape
        chunked_prior_shape = prior.shape
        audio = audio.flatten(0, 1).cuda()
        prior = prior.flatten(0, 1).cuda()

        audio = audio.view(chunked_audio_shape)[:, 2] # B x :
        prior = prior.view(chunked_prior_shape)

        key = '{:05d}'.format(step + 1)

        driving_mesh = torch.load(os.path.join(driving_mesh_dir, key + '.pt'))
        R, t, c = torch.tensor(driving_mesh['R']).float().cuda(), torch.tensor(driving_mesh['t']).float().cuda(), torch.tensor(driving_mesh['c']).float().cuda()

        query = lipdisc.audio_encoder(audio)



        if args.mode != 'O':
            search_result = search_from_pool(query, (key_pool, mesh_pool), N=args.N, T=args.T, verbose=args.mode=='O')
            searched_mesh = search_result['mesh'][0]
            # search_results: B x N x 3
            normed_searched_mesh = searched_mesh
            # print('search result: {}'.format(search_result.shape))
            torch.save(mesh_tensor_to_landmarkdict(normed_searched_mesh), os.path.join(normed_searched_mesh_dir, key + '.pt'))
            searched_mesh = torch.matmul(R.t(), normed_searched_mesh.t() - t).t() / c
            searched_mesh_dict = mesh_tensor_to_landmarkdict(searched_mesh)
            searched_mesh_dict.update({'R': R.cpu().numpy(), 't': t.cpu().numpy(), 'c': c.cpu().numpy()})
            torch.save(searched_mesh_dict, os.path.join(searched_mesh_dir, key + '.pt'))

        else:
            search_result = search_from_pool(query, (key_pool, mesh_pool), N=100, T=1.0, verbose=args.mode=='O')
            searched_mesh = search_result['mesh'][0]
            # search_results: B x N x 3
            weights = search_result['weights'][0]
            index = search_result['index'][0]
            # print(f'weights shape: {weights.shape}, index: {index.shape}')
            prior = torch.einsum('p,pd->d', weights, prior_pool[index])
            # print(f'prior shape: {prior.shape}')
            window.add(prior, weights, index)

            if step >=4:
                conditioned_search_pool = window.get_search_pool()
                search_result = search_from_pool(query, conditioned_search_pool, N=args.N, T=args.T, verbose=True)
                searched_mesh = search_result['mesh']
                # search_results: B x N x 3
                weights = search_result['weights'][0]
                index = search_result['index'][0]
                # print(f'weights shape: {weights.shape}, index: {index.shape}')
                prior = torch.einsum('p,pd->d', weights, prior_pool[index])
                # print(f'prior shape: {prior.shape}')
                window.update_display(prior)

if args.mode == 'O':
    window.flush()
    history = window.get_history()
    for step, searched_mesh in enumerate(history):
        key = '{:05d}'.format(step + 1)
        driving_mesh = torch.load(os.path.join(driving_mesh_dir, key + '.pt'))
        R, t, c = torch.tensor(driving_mesh['R']).float().cuda(), torch.tensor(driving_mesh['t']).float().cuda(), torch.tensor(driving_mesh['c']).float().cuda()
        normed_searched_mesh = searched_mesh
        # print('search result: {}'.format(search_result.shape))
        torch.save(mesh_tensor_to_landmarkdict(normed_searched_mesh), os.path.join(normed_searched_mesh_dir, key + '.pt'))
        searched_mesh = torch.matmul(R.t(), normed_searched_mesh.t() - t).t() / c
        searched_mesh_dict = mesh_tensor_to_landmarkdict(searched_mesh)
        searched_mesh_dict.update({'R': R.cpu().numpy(), 't': t.cpu().numpy(), 'c': c.cpu().numpy()})
        torch.save(searched_mesh_dict, os.path.join(searched_mesh_dir, key + '.pt'))

image_rows = image_cols = 256
draw_mesh_images(os.path.join(args.data_dir, 'mesh_dict_searched_normalized'), os.path.join(args.data_dir, 'mesh_image_searched_normalized'), image_rows, image_cols)
draw_mesh_images(os.path.join(args.data_dir, 'mesh_dict_searched'), os.path.join(args.data_dir, 'mesh_image_searched'), image_rows, image_cols)
interpolate_zs(os.path.join(args.data_dir, 'mesh_dict_searched'), os.path.join(args.data_dir, 'z_searched'), image_rows, image_cols)
interpolate_zs(os.path.join(args.data_dir, 'mesh_dict_searched_normalized'), os.path.join(args.data_dir, 'z_searched_normalized'), image_rows, image_cols)


    

