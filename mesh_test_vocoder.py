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
parser.add_argument('--data_dir', type=str, default='../datasets/kkj_v2/studio_1_6.mp4')
parser.add_argument('--result_dir', type=str, default='studio_1_6.mp4')
parser.add_argument('--ckpt_path', type=str, default='vocoder/lw0.8/best.pt')
parser.add_argument('--result_dir', type=str, default='vocoder')
parser.add_argument('--embedding_dim', type=int, default='512')
parser.add_argument('--device_id', type=str, default='1')




args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

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
model = Encoder(output_dim=20).cuda()
assert args.ckpt_path is not None, 'pretrained checkpoint was not given'
model.load_state_dict(torch.load(args.ckpt_path))
model.eval()

# setup training
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
loss_fn = nn.L1Loss(reduction='sum')

ckpt_dir = os.path.join(args.result_dir, args.ckpt_dir)
normed_searched_mesh_dir = os.path.join(args.data_dir, 'mesh_dict_searched_normalized')
normed_searched_mesh_image_dir = os.path.join(args.data_dir, 'mesh_image_searched_normalized')
searched_mesh_dir = os.path.join(args.data_dir, 'mesh_dict_searched')
searched_mesh_image_dir = os.path.join(args.data_dir, 'mesh_image_searched')
z_dir = os.path.join(args.data_dir, 'z_searched')
driving_mesh_dir = os.path.join(args.data_dir, 'mesh_dict')
lip_dict_dir = os.path.join(args.data_dir, 'lip_dict_normalized')
os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(normed_searched_mesh_dir, exist_ok=True)
os.makedirs(normed_searched_mesh_image_dir, exist_ok=True)
os.makedirs(searched_mesh_dir, exist_ok=True)
os.makedirs(searched_mesh_image_dir, exist_ok=True)
os.makedirs(lip_dict_dir, exist_ok=True)
reference_mesh = torch.load(os.path.join(args.data_dir, 'mesh_dict_reference.pt'))

pca_pool = torch.load(os.path.join(args.pool_dir, 'mesh_pca.pt'))
prior_pool = pca_pool[0].cuda()
pool_S = torch.diag(pca_pool[1].cuda())
pca_V = pca_pool[2].cuda() # N * 3 x pca_dim


def recon_lip(mesh_pca):
    mesh_lip = torch.matmul(mesh_pca @ pool_S, pca_V.t())
    mesh_recon = landmarkdict_to_mesh_tensor(reference_mesh).cuda()
    bias = mesh_recon[LIP_IDX].flatten(-2)
    return (mesh_lip * 128 + bias[None]).view(len(mesh_pca), -1, 3)

def save_segmap(mesh_pca, save_name):
    # mesh_pca: B x pca_dim
    meshes_lip = torch.matmul(mesh_pca @ pool_S, pca_V.t()) 
    # print('meshes shape: {}'.format(meshes_lip.shape))
    result_img = []
    for i in range(len(meshes_lip)):
        mesh_lip = meshes_lip[i]
        mesh_recon = landmarkdict_to_mesh_tensor(reference_mesh).cuda()
        bias = mesh_recon[LIP_IDX].flatten(-2)
        print('bias shape: {}'.format(bias.shape))
        print('mesh_lip shape: {}'.format(mesh_lip.shape))
        mesh_recon[LIP_IDX] = (mesh_lip * 128 + bias).view(-1, 3)
        mesh_dict_recon = mesh_tensor_to_landmarkdict(mesh_recon)
        result_img.append(get_seg(mesh_dict_recon, (256, 256, 3)))
    result_img = np.concatenate(result_img, axis=0) * 32
    cv2.imwrite(os.path.join(ckpt_dir, save_name), result_img)

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
model.eval()
window = Window(size=args.window_size, cache_size=args.N, prior_pool=prior_pool, mesh_pool=mesh_pool, encoder=lipdisc.prior_encoder)

with torch.no_grad():
    for step, (audio, prior) in tqdm(enumerate(dataloader)):
        # print('input shape: {}'.format((audio.shape, prior.shape, label.shape)))
        chunked_audio_shape = audio.shape
        chunked_prior_shape = prior.shape
        audio = audio.flatten(0, 1).cuda()
        prior = prior.flatten(0, 1).cuda()
        # print('input shape: {} {}'.format(audio.shape, prior.shape))
        pred = model(audio) # B x pca_dim
        # print('output shape: {}'.format(pred.shape))
        loss = 0.2 * loss_fn(pred, prior) # 1
        eval_loss += loss.item()
        num_items = len(audio) // 5
        audio = audio.view(chunked_audio_shape)[:, 2] # B x :
        pred = pred.view(chunked_prior_shape)   # B x 5 x 
        prior = prior.view(chunked_prior_shape)
        if lipdisc is not None:
            label = torch.ones(len(audio)).long().cuda()
            lipdisc_loss = lipdisc(audio, pred, label).sum()
            eval_lipdisc_loss += lipdisc_loss.item()
            loss += lipdisc_loss
        else:
            eval_lipdisc_loss += 0
        item_size += num_items
        key = '{:05d}'.format(step + 1)
        # save_segmap(pred[:, 2], key + '.png')
        # pred_lip = recon_lip(pred[:, 2])    # B x Lip x 3
        # GT_idx = step
        # GT_prior = prior_pool[GT_idx][None]
        # save_segmap(GT_prior, key + '.png')
        # pred_lip = recon_lip(GT_prior)
        # torch.save(pred_lip[0], os.path.join(args.data_dir, 'lip_dict_normalized', key + '.pt'))
        # continue

        driving_mesh = torch.load(os.path.join(driving_mesh_dir, key + '.pt'))
        R, t, c = torch.tensor(driving_mesh['R']).float().cuda(), torch.tensor(driving_mesh['t']).float().cuda(), torch.tensor(driving_mesh['c']).float().cuda()
        # query = pred[:, 2]
        # query = audio
        query = lipdisc.audio_encoder(audio)
        # query = recon_lip(prior[:, 2]) / 128 - 1
        # query -= query.mean(dim=1, keepdim=True)
        # query = query.flatten(-2)
        # print('key_pool shape: {}'.format(key_pool.shape))
        # print('mesh_pool shape: {}'.format(mesh_pool.shape))
        # print(f'key pool shape: {key_pool.shape}')
        # # key pool: P x 5 x prior_dim
        # if len(prior_chain) > 0:
        #     condition = torch.cat(prior_chain[-2:], dim=0).cuda()   # ~2 x prior_dim
        #     print('condition shape: {}'.format(condition.shape))
        #     conditioned_key = torch.cat([condition.unsqueeze(0).repeat(len(key_pool), 1, 1), key_pool[:, len(condition):]], dim=1)
        # else:
        #     conditioned_key = key_pool
        # prevs = len(prior_chain[-2:])
        # conditioned_key = torch.zeros_like(pred)
        # if prevs > 0:
        #     condition = torch.cat(prior_chain[-2:], dim=0).cuda()   # ~2 x prior_dim
        #     conditioned_key[:, 2 - prevs:2] = condition.unsqueeze(0).repeat(len(conditioned_key), 1, 1)
        # conditioned_key = conditioned_key.repeat(len(key_pool), 1, 1) # P x 5 x prior_dim
        # conditioned_key[:, 2:] = key_pool[:, 2:]
        # conditioned_key = key_pool
        # print('condition pool shape: {}'.format(conditioned_key.shape))
        # search_result, search_prior = search_from_pool(query, (conditioned_key, mesh_pool), get_key=True)

        search_result = search_from_pool(query, (key_pool, mesh_pool), N=args.N, T=args.T, verbose=args.mode=='O')
        searched_mesh = search_result['mesh'][0]
        # search_results: B x N x 3

        if args.mode != 'O':
            # print('searched prior shape: {}'.format(search_prior.shape))

            # prior_chain.append(search_prior)

            # search_result = mesh_pool[GT_idx][None]
            # normed_searched_mesh = 128 * (search_result + 1)[0]  # N x 3
            normed_searched_mesh = searched_mesh
            # print('search result: {}'.format(search_result.shape))
            torch.save(mesh_tensor_to_landmarkdict(normed_searched_mesh), os.path.join(normed_searched_mesh_dir, key + '.pt'))
            searched_mesh = torch.matmul(R.t(), normed_searched_mesh.t() - t).t() / c
            searched_mesh_dict = mesh_tensor_to_landmarkdict(searched_mesh)
            searched_mesh_dict.update({'R': R.cpu().numpy(), 't': t.cpu().numpy(), 'c': c.cpu().numpy()})
            torch.save(searched_mesh_dict, os.path.join(searched_mesh_dir, key + '.pt'))

        else:
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

# eval_loss = eval_loss / item_size
# eval_lipdisc_loss = eval_lipdisc_loss / item_size
# print('(Test) test_loss: {}, lipdisc_loss: {}'.format(eval_loss, eval_lipdisc_loss))

# train_dataloader.dataset.update_p(positive_p)
# eval_dataloader.dataset.update_p(positive_p)
# draw_mesh_images(os.path.join(normed_searched_mesh_dir), os.path.join(normed_searched_mesh_image_dir), 256, 256)
# draw_mesh_images(os.path.join(searched_mesh_dir), os.path.join(searched_mesh_image_dir), 256, 256)
# interpolate_zs(searched_mesh_dir, z_dir, 256, 256)

image_rows = image_cols = 256
draw_mesh_images(os.path.join(args.data_dir, 'mesh_dict_searched_normalized'), os.path.join(args.data_dir, 'mesh_image_searched_normalized'), image_rows, image_cols)
draw_mesh_images(os.path.join(args.data_dir, 'mesh_dict_searched'), os.path.join(args.data_dir, 'mesh_image_searched'), image_rows, image_cols)
interpolate_zs(os.path.join(args.data_dir, 'mesh_dict_searched'), os.path.join(args.data_dir, 'z_searched'), image_rows, image_cols)
interpolate_zs(os.path.join(args.data_dir, 'mesh_dict_searched_normalized'), os.path.join(args.data_dir, 'z_searched_normalized'), image_rows, image_cols)


    

