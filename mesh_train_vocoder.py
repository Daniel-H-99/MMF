from tqdm import trange
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from logger import Logger
from modules.discriminator import LipDiscriminator
from modules.discriminator import Encoder
from modules.util import landmarkdict_to_mesh_tensor, mesh_tensor_to_landmarkdict, LIP_IDX, get_seg
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
parser.add_argument('--data_dir', type=str, default='../datasets/kkj_v2/train')
parser.add_argument('--ckpt_dir', type=str, default='checkpoint')
parser.add_argument('--result_dir', type=str, default='vocoder')
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--split_ratio', type=float, default=0.9)
parser.add_argument('--steps', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--milestone', type=str, default='5,10,15')
parser.add_argument('--embedding_dim', type=int, default=512)
parser.add_argument('--log_pth', type=str, default='log.txt')
parser.add_argument('--lipdisc_path', type=str, default='expert_v3.1_W5/best.pt')
parser.add_argument('--lipdisc_weight', type=float, default=0.2)
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


prior_pool = torch.load(os.path.join(args.data_dir, 'mesh_pca.pt'))[0]
audio_pool = []
path = os.path.join(args.data_dir, 'audio')
# frames = sorted(os.listdir(os.path.join(args.data_dir, 'img')))
audio_frames = sorted(os.listdir(path))
num_frames = min(len(prior_pool), len(audio_frames))
frame_idx = range(num_frames)
for i in range(len(frame_idx)):
    with open(os.path.join(path, '{:05d}.pickle'.format(i)), 'rb') as f:
        mspec = pkl.load(f)
        audio_pool.append(mspec)

audio_pool = torch.from_numpy(np.array(audio_pool).astype(np.float32))
prior_pool = prior_pool[:num_frames] / 128

audio_pool_size = len(audio_pool)
prior_pool_size = len(prior_pool)
assert audio_pool_size == prior_pool_size, 'the size of audio / prior pool do not match: {} - {}'.format(audio_pool_size, prior_pool_size)
print('Audio Pool Size: {}, Prior Pool: {}'.format(audio_pool_size, prior_pool_size))

training_size = int(args.split_ratio * audio_pool_size)
train_audio_pool, eval_audio_pool = audio_pool[:training_size], audio_pool[training_size:]
train_prior_pool, eval_prior_pool = prior_pool[:training_size], prior_pool[training_size:]

train_dataset = MeshSyncDataset(train_audio_pool, train_prior_pool)
eval_dataset = MeshSyncDataset(eval_audio_pool, eval_prior_pool)


# prepare model
lipdisc = None
if args.lipdisc_weight > 0:
    lipdisc = LipDiscriminator(prior_dim=20, embedding_dim=args.embedding_dim).cuda()
    lipdisc.load_state_dict(torch.load(args.lipdisc_path))
    for p in lipdisc.parameters():
        p.requires_grad = False
    lipdisc.eval()
    torch.backends.cudnn.enabled=False

model = Encoder(output_dim=20).cuda()

# setup training
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)
scheduler = MultiStepLR(optimizer, [int(m) for m in args.milestone.split(',')], gamma=0.1,
                                      last_epoch=- 1)
loss_fn = nn.L1Loss(reduction='sum')

total_steps = args.steps
save_freq = args.save_freq
log_freq = args.log_freq
ckpt_dir = os.path.join(args.result_dir, args.ckpt_dir)
log_path = os.path.join(ckpt_dir, args.log_pth)
os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

with open(log_path, 'w') as f:
    f.write('Training Started\n')

def write_log(text):
    with open(log_path, 'a') as f:
        f.write(text + '\r\n')

reference_mesh = torch.load(os.path.join(args.data_dir, 'mesh_dict_reference.pt'))

pool_S = torch.diag(torch.load(os.path.join(args.data_dir, 'mesh_pca.pt'))[1].cuda())
pca_V = torch.load(os.path.join(args.data_dir, 'mesh_pca.pt'))[2].cuda() # N * 3 x pca_dim

def save_segmap(mesh_pca, save_name):
    print(f'mesh shape : {mesh_pca.shape}')
    # mesh_pca: B x pca_dim
    meshes_lip = torch.matmul(mesh_pca @ pool_S, pca_V.t()) 
    print('meshes shape: {}'.format(meshes_lip.shape))
    result_img = []
    for i in range(len(meshes_lip)):
        mesh_lip = meshes_lip[i]
        mesh_recon = landmarkdict_to_mesh_tensor(reference_mesh).cuda()
        bias = mesh_recon[LIP_IDX].flatten(-2)
        # print('bias shape: {}'.format(bias.shape))
        # print('mesh_lip shape: {}'.format(mesh_lip.shape))
        mesh_recon[LIP_IDX] = (mesh_lip * 128 + bias).view(-1, 3)
        mesh_dict_recon = mesh_tensor_to_landmarkdict(mesh_recon)
        result_img.append(get_seg(mesh_dict_recon, (256, 256, 3)))
    result_img = np.concatenate(result_img, axis=0) * 32
    cv2.imwrite(os.path.join(ckpt_dir, save_name), result_img)

# train
train_iter = iter(train_dataloader)
item_size = 0
train_loss = 0
train_lipdisc_loss = 0
best_eval_loss = 100000
for step in range(1, total_steps + 1):
    model.train()
    optimizer.zero_grad()
    try:
        audio, prior = train_iter.next()
    except StopIteration:
        train_iter = iter(train_dataloader)
        audio, prior = train_iter.next()
    # audio: B x :
    # prior: B x T x prior_dim
    chunked_audio_shape = audio.shape
    chunked_prior_shape = prior.shape
    audio = audio.flatten(0, 1).cuda()
    # print('input shape: {} {}'.format(audio.shape, prior.shape))
    pred = model(audio) # num_chunk x pca_dim
    # print('output shape: {}'.format(pred.shape))
    audio = audio.view(chunked_audio_shape)[:, 2] # B x :
    pred = pred.view(chunked_prior_shape)   # B x 5 x :
    loss = loss_fn(pred[:, 2], prior[:, 2].cuda()) # 1
    train_loss += loss.item()
    num_items = len(audio)
    if lipdisc is not None:
        label = torch.ones(len(audio)).long().cuda()
        lipdisc_loss = lipdisc(audio, pred, label).sum()
        train_lipdisc_loss += lipdisc_loss.item()
        loss += args.lipdisc_weight * lipdisc_loss
    else: 
        train_lipdisc_loss += 0
    item_size += num_items
    loss /= num_items
    loss.backward()
    optimizer.step()
    if step % log_freq == 0:
        print('step [{:05d}] train_loss: {}, train_lipdisc_loss {}'.format(step, train_loss / item_size, train_lipdisc_loss / item_size))
        write_log('step [{:05d}] train_loss: {}, train_lipdisc_loss {}'.format(step, train_loss / item_size, train_lipdisc_loss / item_size))
        train_loss = 0
        train_lipdisc_loss = 0
        item_size = 0

    if step % save_freq == 0:
        eval_item_size = 0
        eval_lipdisc_loss = 0
        eval_loss = 0
        model.eval()
        with torch.no_grad():
            for audio, prior, in eval_dataloader:
                # print('input shape: {}'.format((audio.shape, prior.shape, label.shape)))
                chunked_audio_shape = audio.shape
                chunked_prior_shape = prior.shape
                audio = audio.flatten(0, 1).cuda()
                # print('input shape: {} {}'.format(audio.shape, prior.shape))
                pred = model(audio) # num_chunk x pca_dim
                # print('output shape: {}'.format(pred.shape))
                audio = audio.view(chunked_audio_shape)[:, 2] # B x :
                pred = pred.view(chunked_prior_shape)   # B x 5 x :
                loss = loss_fn(pred[:, 2], prior[:, 2].cuda()) # 1
                eval_loss += loss.item()
                num_items = len(audio)
                if lipdisc is not None:
                    label = torch.ones(len(audio)).long().cuda()
                    lipdisc_loss = lipdisc(audio, pred, label).sum()
                    eval_lipdisc_loss += lipdisc_loss.item()
                    loss += lipdisc_loss
                else:
                    eval_lipdisc_loss += 0
                eval_item_size += num_items
        eval_loss = eval_loss / eval_item_size
        eval_lipdisc_loss = eval_lipdisc_loss / eval_item_size
        print('(Eval) step [{:08d}] eval_loss: {}, lipdisc_loss: {}'.format(step, eval_loss, eval_lipdisc_loss))
        write_log('(Eval) step [{:08d}] eval_loss: {}, lipdisc_loss: {}'.format(step, eval_loss, eval_lipdisc_loss))
        # pred = pred.view(chunked_prior_shape)   # B x 5 x :
        save_segmap(pred[:, 2], '{:05d}.png'.format(step))
        torch.save(model.state_dict(), os.path.join(ckpt_dir, '{:08d}.pt'.format(step)))
        if eval_loss + eval_lipdisc_loss <= best_eval_loss:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pt'))
            best_eval_loss = eval_loss + eval_lipdisc_loss
            print('best loss of step {} saved'.format(step))
            write_log('best loss of step {} saved\n'.format(step))
        scheduler.step()
        # train_dataloader.dataset.update_p(positive_p)
        # eval_dataloader.dataset.update_p(positive_p)

print('Training Finished with Best Eval Loss: {}'.format(best_eval_loss))
write_log('Training Finished with Best Eval Loss: {}\n'.format(best_eval_loss))



    

