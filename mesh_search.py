from tqdm import trange
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from logger import Logger
from modules.discriminator import LipDiscriminator, NoiseDiscriminator
from modules.discriminator import Encoder
from modules.util import landmarkdict_to_mesh_tensor, mesh_tensor_to_landmarkdict, LIP_IDX, get_seg, draw_mesh_images, interpolate_zs
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
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
parser.add_argument('--data_dir', type=str, default='../datasets/kkj_v2/test/studio_1_34.mp4')
parser.add_argument('--pool_dir', type=str, default='../datasets/kkj_v2/pool_XL')
parser.add_argument('--embedding_dim', type=int, default='512')
parser.add_argument('--lipdisc_path', type=str, default='expert_v3.1_W5/best.pt')
parser.add_argument('--denoiser_path', type=str, default='denoiser_ckpt/00002000.pt')
parser.add_argument('--window_size', type=int, default=5)
parser.add_argument('--N', type=int, default=50)
parser.add_argument('--k', type=int, default=4)
parser.add_argument('--T', type=float, default=0.5)
parser.add_argument('--l1_weight', type=float, default=1)
parser.add_argument('--l2_weight', type=float, default=0)
parser.add_argument('--opt_steps', type=int, default=1000)
parser.add_argument('--mode', type=str, default='O')
parser.add_argument('--device_id', type=str, default='1')



args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.device_id


# prepare dataset
class MeshSyncDataset(Dataset):
    def __init__(self, audio, prior, aux=None):
        # audio: L x : (tensor)
        # prior: L x prior_dim (tensor)
        super(MeshSyncDataset, self).__init__()
        self.audio = torch.cat([audio[:2], audio, audio[-2:]], dim=0)
        self.prior = torch.cat([prior[:2], prior, prior[-2:]], dim=0)
        self.aux = None if aux is None else torch.cat([aux[:2], aux, aux[-2:]], dim=0)
    def __len__(self):
        return len(self.audio) - 4
    def __getitem__(self, index):
        return (self.audio[index:index + 5], self.prior[index:index + 5]) if self.aux is None else (self.audio[index:index + 5], self.prior[index:index + 5], self.aux[index:index + 5])
            
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
input_audio_pool = audio_pool

# prepare model
lipdisc = LipDiscriminator(prior_dim=20, embedding_dim=args.embedding_dim).cuda()
lipdisc.load_state_dict(torch.load(args.lipdisc_path))
for p in lipdisc.parameters():
    p.requires_grad = False
lipdisc.eval()
torch.backends.cudnn.enabled=False

denoiser = NoiseDiscriminator().cuda()
denoiser.load_state_dict(torch.load(args.denoiser_path))
denoiser.eval()

# setup training
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
loss_fn = nn.L1Loss(reduction='sum')

normed_searched_mesh_dir = os.path.join(args.data_dir, 'mesh_dict_searched_normalized')
normed_searched_mesh_image_dir = os.path.join(args.data_dir, 'mesh_image_searched_normalized')
searched_mesh_dir = os.path.join(args.data_dir, 'mesh_dict_searched')
searched_mesh_image_dir = os.path.join(args.data_dir, 'mesh_image_searched')
lip_dict_dir = os.path.join(args.data_dir, 'lip_dict_normalized')
z_dir = os.path.join(args.data_dir, 'z_searched')
driving_mesh_dir = os.path.join(args.data_dir, 'mesh_dict')

os.makedirs(normed_searched_mesh_dir, exist_ok=True)
os.makedirs(normed_searched_mesh_image_dir, exist_ok=True)
os.makedirs(searched_mesh_dir, exist_ok=True)
os.makedirs(searched_mesh_image_dir, exist_ok=True)
os.makedirs(lip_dict_dir, exist_ok=True)
reference_mesh = torch.load(os.path.join(args.data_dir, 'mesh_dict_reference.pt'))

pca_pool = torch.load(os.path.join(args.pool_dir, 'mesh_pca.pt'))
prior_pool = pca_pool[0].cuda() / 128
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


S = torch.diag(pca_pool[1].cuda())
V = pca_pool[2].cuda()

def recon(coef):
    # coef: B x prior_dim
    # S: prior_dim x prior_dim
    # V: real_dim x prior_dim
    # out: B x recon_dim
    return coef @ S @ V.t()

reference_mesh = torch.load(os.path.join('../datasets/train_kkj/kkj04.mp4', 'mesh_dict_reference.pt'))

def recon_lip(mesh_pca):
    mesh_lip = recon(mesh_pca)  # B x recon_dim
    mesh_recon = landmarkdict_to_mesh_tensor(reference_mesh).cuda()
    bias = mesh_recon[LIP_IDX].flatten(-2)
    return (mesh_lip * 128 + bias[None]).view(len(mesh_pca), -1, 3)

def proj(real):
    # real: B x N * 3
    # S: prior_dim x prior_dim
    # V: N * 3 x prior_dim
    return real @ V @ S.inverse()

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
    logits = weights
    mesh_pool = mesh_pool[index[:, :N]]
    weights = nn.Softmax(dim=1)(-weights / T) # B x P
    result['mesh'] = torch.einsum('bp,bpni->bni', weights, mesh_pool) # B x N x 3
    # result[:, LIP_IDX] = audio_prior

    if verbose:
        result['logits'] = logits
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
        self.index_history = []
        self.weight_history = []
    def add(self, display, weights, idx):
        # display: prior_dim, idx: cahe_size
        if self.weights[0].sum() > 0:
            self.history.append(torch.einsum('p,pni->ni', self.weights[0], self.mesh_pool[self.index[0]]))
            self.index_history.append(torch.tensor(self.index[0]))
            self.weight_history.append(torch.tensor(self.weights[0]))
        self.display[:4] = self.display.clone()[1:]
        self.weights[:4] = self.weights.clone()[1:]
        self.index[:4] = self.index.clone()[1:]
        self.display[4] = display
        self.weights[4] = weights
        self.index[4] = idx
    def update_display(self, display, pos=2):
        # display: prior_dim
        self.display[pos] = display
    def get_search_pool(self, pos=2):
        cache_size = self.size * self.cache_size
        index = self.index.view(-1)
        keys = self.display[None].repeat(cache_size, 1, 1) # cache_size x size x prior_dim
        keys[:, pos] = self.prior_pool[index]
        keys = self.encoder(keys) # cache_size x embedding_dim
        values = self.mesh_pool[index] # cache_size x N x 3
        return keys, values
    def flush(self):
        for i in range(5):
            self.history.append(torch.einsum('p,pni->ni', self.weights[i], self.mesh_pool[self.index[i]]))
            self.index_history.append(self.index[i])
            self.weight_history.append(self.weights[i])
    def get_history(self):
        return self.history, self.index_history, self.weight_history
    


item_size = 0
eval_lipdisc_loss = 0
eval_loss = 0

window = Window(size=args.window_size, cache_size=args.N, prior_pool=prior_pool, mesh_pool=mesh_pool, encoder=lipdisc.prior_encoder)

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
            search_result = search_from_pool(query, (key_pool, mesh_pool), N=args.N, T=args.T, verbose=True)
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
            search_result = search_from_pool(query, (key_pool, mesh_pool), N=args.N, T=args.T, verbose=args.mode=='O')
            searched_mesh = search_result['mesh'][0]
            # search_results: B x N x 3
            weights = search_result['logits'][0]
            index = search_result['index'][0]
            # print(f'weights shape: {weights.shape}, index: {index.shape}')
            prior = torch.einsum('p,pd->d', weights, prior_pool[index])
            # print(f'prior shape: {prior.shape}')
            window.add(prior, weights, index)

            # if step >=4:
            #     conditioned_search_pool = window.get_search_pool()
            #     search_result = search_from_pool(query, conditioned_search_pool, N=args.N, T=args.T, verbose=True)
            #     searched_mesh = search_result['mesh']
            #     # search_results: B x N x 3
            #     weights = search_result['weights'][0]
            #     index = search_result['index'][0]
            #     # print(f'weights shape: {weights.shape}, index: {index.shape}')
            #     prior = torch.einsum('p,pd->d', weights, prior_pool[index])
            #     # print(f'prior shape: {prior.shape}')
            #     window.update_display(prior)

if args.mode == 'O':
    window.flush()
    history, index_history, weight_history = window.get_history()
    history = torch.stack(history, dim=0).cuda()
    weight_history = torch.stack(weight_history, dim=0).cuda()
    # B = 64
    # weights = torch.zeros_like(history)
    # # weights[2:-1, args.N:2*args.N] = 10
    # # weights[0, :args.N] = 10
    # # weights[-1, -args.N:] = 10
    # weights = torch.nn.Parameter(weights.cuda())
    # optimizer = optim.Adam([weights], lr=0.001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 800], gamma=0.1)
    # index_stack = torch.stack(index_history, dim=0)  # B x N
    # indice = torch.arange(1).unsqueeze(0).repeat(len(index_stack), 1) + torch.arange(len(index_stack)).unsqueeze(1) # L - 4 x 5
    # indice = torch.cat([indice], dim=0) # L x 5
    # indice = index_stack[indice].flatten(1) # L x 5 * N
    # mesh_stack = mesh_pool[indice].cuda() # L x 5 * N x ...
    # loss_fn = nn.CosineEmbeddingLoss(reduction='sum')
    # input_audio_pool = input_audio_pool.cuda()
    # print(f'weights_stack: {weights.shape}, index_history: {len(index_history)}, prior_stack: {prior_stack.shape}, mesh_stack: {mesh_stack.shape}')
    # for e in range(args.opt_steps):
    #     coefs = torch.nn.Softmax()(-weights / args.T)
    #     priors = torch.einsum('bm,bmp->bp', coefs, prior_stack) # L x prior_dim
    #     tmp_dataset = MeshSyncDataset(input_audio_pool, priors)
    #     tmp_loader = DataLoader(tmp_dataset, batch_size=B)
    #     loss = 0
    #     optimizer.zero_grad()
    #     for a, p in tmp_loader:
    #         a = a.cuda()
    #         p = p.cuda()
    #         audio_embedding = lipdisc.audio_encoder(a[:, 2])
    #         prior_embedding = lipdisc.prior_encoder(p)
    #         local_loss = loss_fn(audio_embedding, prior_embedding, torch.ones(len(audio_embedding)).long().cuda())
    #         # local_loss = -loss_fn(audio_embedding, prior_embedding).sum()
    #         loss += local_loss
    #         # print(f'local loss: {local_loss}')

    #     loss = loss / 100
    #     print(f'[{e}] loss: {loss}')
    #     loss.backward()
    #     optimizer.step()
    #     scheduler.step()

    # coefs = torch.nn.Softmax()(-weights / args.T)
    # optimized_mesh_stack = torch.einsum('bm,bmni->bni', coefs, mesh_stack) # L x ...

    # for step, searched_mesh in enumerate(optimized_mesh_stack):
    #     key = '{:05d}'.format(step + 1)
    #     driving_mesh = torch.load(os.path.join(driving_mesh_dir, key + '.pt'))
    #     R, t, c = torch.tensor(driving_mesh['R']).float().cuda(), torch.tensor(driving_mesh['t']).float().cuda(), torch.tensor(driving_mesh['c']).float().cuda()
    #     normed_searched_mesh = searched_mesh
    #     # print('search result: {}'.format(search_result.shape))
    #     torch.save(mesh_tensor_to_landmarkdict(normed_searched_mesh), os.path.join(normed_searched_mesh_dir, key + '.pt'))
    #     searched_mesh = torch.matmul(R.t(), normed_searched_mesh.t() - t).t() / c
    #     searched_mesh_dict = mesh_tensor_to_landmarkdict(searched_mesh)
    #     searched_mesh_dict.update({'R': R.cpu().numpy(), 't': t.cpu().numpy(), 'c': c.cpu().numpy()})
    #     torch.save(searched_mesh_dict, os.path.join(searched_mesh_dir, key + '.pt'))

    B = 64
    weights = torch.zeros(len(weight_history), 5 * args.N).cuda()
    bias = torch.zeros(len(weight_history), prior_pool.size(1))
    # bias = torch.zeros(len(weight_history), 3 * len(LIP_IDX))
    # weights[2:-1, args.N:2*args.N] = 10
    # weights[0, :args.N] = 10
    # weights[-1, -args.N:] = 10
    weights = torch.nn.Parameter(weights.cuda())
    bias = torch.nn.Parameter(bias.cuda())
    optimizer = optim.Adam([weights, bias], lr=1e-2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 800], gamma=0.1)

    index_stack = torch.stack(index_history, dim=0)  # B x N
    indice = torch.Tensor([2, 0, 1, 3, 4]).long().unsqueeze(0).repeat(len(index_stack) - 4, 1) + torch.arange(len(index_stack) - 4).unsqueeze(1) # L - 4 x 5
    indice = torch.cat([torch.Tensor([[0, 1, 2, 3, 4], [1, 0, 2, 3, 4]]).long(), indice, torch.Tensor([[3, 0, 1, 2, 4], [4, 0, 1, 2, 3]]).long() + len(index_stack) - 5], dim=0) # L x 5
    indice = index_stack[indice].flatten(1) # L x 5 * N

    prior_stack = prior_pool[indice].cuda() # L x 5 * N x prior_dim
    mesh_stack = mesh_pool[indice].cuda() # L x 5 * N x ...

    ### initialize
    weights_base = weight_history    # L x N
    p_base = nn.Softmax(dim=1)(weights_base) # L x N
    pure_prior_stack = prior_stack[:, :args.N] # L x N x prior_dim
    neighbor_prior_stack = prior_stack[:, args.N:]  # L x 4 * N x prior_dim
    var_x = torch.einsum('bn,bn->b', p_base, torch.norm(pure_prior_stack, dim=2) ** 2) - torch.norm(torch.einsum('bn,bnp->bp', p_base, pure_prior_stack), dim=1) ** 2
    var_x_tilda =  torch.mean(torch.norm(neighbor_prior_stack, dim=2) ** 2, dim=1) - torch.norm(torch.mean(neighbor_prior_stack, dim=1), dim=1) ** 2 
    e =  (var_x_tilda / torch.clamp(var_x + var_x_tilda, min=1e-6)).sqrt()
    w_tilda = - args.T * torch.log((1 / (args.k * args.N)) * (1 / torch.clamp(e, min=1e-6) - 1) * torch.sum(torch.exp(-weights_base / args.T), dim=1)) # B
    weights_base = torch.cat([weights_base, w_tilda.unsqueeze(1).repeat(1, 4 * args.N)], dim=1)
    loss_fn = nn.CosineEmbeddingLoss(reduction='sum')
    input_audio_pool = input_audio_pool.cuda()
    print(f'weights_stack: {weights.shape}, index_history: {len(index_history)}, prior_stack: {prior_stack.shape}, mesh_stack: {mesh_stack.shape}')
    for e in range(args.opt_steps):
        tmp_weights = weights_base + weights
        coefs = torch.nn.Softmax()(-tmp_weights / args.T)
        priors = torch.einsum('bm,bmp->bp', coefs, prior_stack) # L x prior_dim
        tmp_dataset = MeshSyncDataset(input_audio_pool, priors, aux=bias)
        tmp_loader = DataLoader(tmp_dataset, batch_size=B)
        loss = 0
        noise_loss = 0
        optimizer.zero_grad()
        for a, p, b in tmp_loader:
            a = a.cuda()
            p = p.cuda()
            b = b.cuda()

            # lip_recon = recon(p.flatten(0, 1)) + b.flatten(0, 1)    # B * T x N * 3
            # lip_recon_proj = proj(lip_recon).view(p.shape)    # B x T x prior_dim
            lip_recon_proj = p + b 
            audio_embedding = lipdisc.audio_encoder(a[:, 2])
            prior_embedding = lipdisc.prior_encoder(p)
            local_loss = loss_fn(audio_embedding, prior_embedding, torch.ones(len(audio_embedding)).long().cuda())
            # local_loss = -loss_fn(audio_embedding, prior_embedding).sum()
            loss += local_loss
            noise_loss += denoiser.reality_loss(lip_recon_proj) * len(p)
            # noise = lip_recon_proj[1:] - lip_recon_proj[:-1]
            # noise_loss += torch.norm(noise, p=2) * len(lip_recon_proj) / (noise.size(0) * noise.size(1))
            # print(f'local loss: {local_loss}')

        loss = loss / len(priors)
        noise_loss = noise_loss / len(priors)
        # noise_loss = torch.norm(priors[1:] - priors[:-1], p=1) / (priors.size(0) * priors.size(1))
        # tmp_mesh_stack = torch.einsum('bm,bmni->bni', coefs, mesh_stack)[:, LIP_IDX] # L x ...
        loss += args.l2_weight * (torch.norm(bias, p=2) / len(bias))
        # loss += args.l1_weight * torch.norm(tmp_mesh_stack[1:] - tmp_mesh_stack[:-1], p=1) /  ((tmp_mesh_stack.size(0) - 1) * tmp_mesh_stack.size(1) * tmp_mesh_stack.size(2))
        print(f'[{e}] loss: {loss}, noise loss: {noise_loss}')
        loss += args.l1_weight * noise_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    coefs = torch.nn.Softmax(dim=1)(-(weights_base + weights) / args.T)
    # print(f'coefs: {coefs}')
    optimized_priors = torch.einsum('bm,bmp->bp', coefs, prior_stack) # L x prior_dim
    optimized_lip = recon_lip(optimized_priors)
    # optimized_priors = torch.einsum('bm,bmp->bp', coefs, prior_stack) # L x prior_dim
    # optimized_lip = recon_lip(optimized_priors)
    # optimized_lip += bias.view(optimized_lip.shape) * 128
    optimized_mesh_stack = torch.einsum('bm,bmni->bni', coefs, mesh_stack) # L x ...
    optimized_mesh_stack[:, LIP_IDX] = optimized_lip.view(optimized_mesh_stack[:, LIP_IDX].shape)
    
    for step, searched_mesh in enumerate(optimized_mesh_stack):
        key = '{:05d}'.format(step + 1)
        torch.save(searched_mesh[LIP_IDX], os.path.join(lip_dict_dir, key + '.pt'))

    # for step, searched_mesh in enumerate(optimized_mesh_stack):
    #     key = '{:05d}'.format(step + 1)
    #     driving_mesh = torch.load(os.path.join(driving_mesh_dir, key + '.pt'))
    #     R, t, c = torch.tensor(driving_mesh['R']).float().cuda(), torch.tensor(driving_mesh['t']).float().cuda(), torch.tensor(driving_mesh['c']).float().cuda()
    #     normed_searched_mesh = searched_mesh
    #     # print('search result: {}'.format(search_result.shape))
    #     torch.save(mesh_tensor_to_landmarkdict(normed_searched_mesh), os.path.join(normed_searched_mesh_dir, key + '.pt'))
    #     searched_mesh = torch.matmul(R.t(), normed_searched_mesh.t() - t).t() / c
    #     searched_mesh_dict = mesh_tensor_to_landmarkdict(searched_mesh)
    #     searched_mesh_dict.update({'R': R.cpu().numpy(), 't': t.cpu().numpy(), 'c': c.cpu().numpy()})
    #     torch.save(searched_mesh_dict, os.path.join(searched_mesh_dir, key + '.pt'))

    # for step, searched_mesh in enumerate(history):
    #     key = '{:05d}'.format(step + 1)
    #     driving_mesh = torch.load(os.path.join(driving_mesh_dir, key + '.pt'))
    #     R, t, c = torch.tensor(driving_mesh['R']).float().cuda(), torch.tensor(driving_mesh['t']).float().cuda(), torch.tensor(driving_mesh['c']).float().cuda()
    #     normed_searched_mesh = searched_mesh
    #     # print('search result: {}'.format(search_result.shape))
    #     torch.save(mesh_tensor_to_landmarkdict(normed_searched_mesh), os.path.join(normed_searched_mesh_dir, key + '.pt'))
    #     searched_mesh = torch.matmul(R.t(), normed_searched_mesh.t() - t).t() / c
    #     searched_mesh_dict = mesh_tensor_to_landmarkdict(searched_mesh)
    #     searched_mesh_dict.update({'R': R.cpu().numpy(), 't': t.cpu().numpy(), 'c': c.cpu().numpy()})
    #     torch.save(searched_mesh_dict, os.path.join(searched_mesh_dir, key + '.pt'))
"""
class Maker(nn.Model):
    def __init__(self, seq_len, N, prior_pool, key_pool, encoder, weight_l1=0.2):
        super(Maker, self).__init__()
        self.seq_len = seq_len
        self.N = N
        self.prior_pool = prior_pool
        self.key_pool = key_pool
        self.encoder = encoder
        self.weight_l1 = weight_l1
        self.register_parameter('weight', nn.Parameter(torch.tensor(self.seq_len, self.N)))
    def forward(self, index):
        # index: B x N
        input = self.prior_pool[index]  # B x N x prior_dim
        
class OptDataset(Dataset):
    def __init__(self, index, window=5):
        # index: L x N
        super(OptDataset, self).__init__()
        self.index = index
        self.window = window

    def __len__(self):
        return len(index)
    def __getitem__(self, index):
        start = max(0, index - 2)
        end = start + self.window
        overflow = end - self.__len__() + 1
        if overflow >= 1:
            start -= overflow
            end -= overflow
        return torch.arange(start, end), index



opt_data = OptDataset(index_stack, window=args.window_size)
B = 128
opt_dataloader = DataLoader(opt_data, batch_size=B)
optimizer = optim.Adam([weights], lr=1e-3)
loss_fn = nn.CosineEmbeddingLoss(reduction='sum')
for e in range(args.opt_steps):
    optimizer.zero_grad()
    loss = 0
    for w, fid in opt_dataloader:
        # w: B x 5, fid: B
        # prior_pool: P x prior_dim
        index = index_stack[w]
        priors = prior_pool[index]  # B x 5 * N x prior_dim
        priors = torch.einsum('bm,bmp->bp',)
        prior_embedding = lipdisc.prior_encoder(priors)    # B x embedding_dim
        audio_embedding = audio_pool[fid]   # B x embedding_dim
        loss += loss_fn(audio_embedding, prior_embedding, torch.ones(len(audio_embedding)).long().cuda())
    for w, fid in opt_dataloader:
        # w: B x 5, fid: B
        # prior_pool: P x prior_dim
        index = index_stack[w]
        priors = prior_pool[index]  # B x 5 * N x prior_dim
        priors = torch.einsum('bm,bmp->bp',)
        prior_embedding = lipdisc.prior_encoder(priors)    # B x embedding_dim
        audio_embedding = audio_pool[fid]   # B x embedding_dim
        loss += loss_fn(audio_embedding, prior_embedding, torch.ones(len(audio_embedding)).long().cuda())
    loss /= 5
    loss.backward()
    optimizer.step()

    


for step, index in index_history:

    
"""
# image_rows = image_cols = 256
# draw_mesh_images(os.path.join(args.data_dir, 'mesh_dict_searched_normalized'), os.path.join(args.data_dir, 'mesh_image_searched_normalized'), image_rows, image_cols)
# draw_mesh_images(os.path.join(args.data_dir, 'mesh_dict_searched'), os.path.join(args.data_dir, 'mesh_image_searched'), image_rows, image_cols)
# interpolate_zs(os.path.join(args.data_dir, 'mesh_dict_searched'), os.path.join(args.data_dir, 'z_searched'), image_rows, image_cols)
# interpolate_zs(os.path.join(args.data_dir, 'mesh_dict_searched_normalized'), os.path.join(args.data_dir, 'z_searched_normalized'), image_rows, image_cols)


    

