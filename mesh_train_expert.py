from tqdm import trange
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.discriminator import LipDiscriminator

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from torch.utils.data import Dataset
import argparse
import os
import numpy as np
import random
import pickle as pkl
import math

os.environ['CUDA_VISIBLE_DEVICES']='1'
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_dir', type=str, default='../datasets/train_kkj/kkj04.mp4')
parser.add_argument('--ckpt_dir', type=str, default='expert_ckpt')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--split_ratio', type=float, default=0.9)
parser.add_argument('--steps', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--milestone', type=str, default='50,80,100')
parser.add_argument('--embedding_dim', type=int, default='256')
parser.add_argument('--log_pth', type=str, default='log.txt')



args = parser.parse_args()

# prepare dataset
class MeshSyncDataset(Dataset):
    def __init__(self, audio, prior):
        # audio: L x : (tensor)
        # prior: L x prior_dim (tensor)
        super(MeshSyncDataset, self).__init__()
        self.audio = audio
        self.prior = torch.cat([torch.zeros_like(prior[0]).unsqueeze(0).repeat(2, 1), prior, torch.zeros_like(prior[0]).unsqueeze(0).repeat(2, 1)], dim=0)
        self.positive_p = 0.5
    def __len__(self):
        return len(self.audio)
    def __getitem__(self, index):
        negative = random.random() >= self.positive_p
        if negative:
            r = random.random()
            if r <= 0.3:
                negative_index = (index + random.choice(list(range(1, self.__len__())))) % self.__len__()
                return self.audio[index], self.prior[negative_index:negative_index + 5], -1 # :, T x prior_dim
            elif r <= 0.6:
                negative_index = (index + random.choice(list(range(1, self.__len__())))) % self.__len__()
                return self.audio[index], torch.cat([self.prior[negative_index:negative_index+2], self.prior[index+2:index+5]], dim=0), -1
            else:
                negative_index = (index + random.choice(list(range(1, self.__len__())))) % self.__len__()
                return self.audio[index], torch.cat([self.prior[index:index+2], self.prior[negative_index+2:negative_index+5]], dim=0), -1
        else:
            return self.audio[index], self.prior[index:index + 5], 1
    def update_p(self, p):
        self.positive_p = p
    def reset_p(self):
        self.positive_p = 0.5
            
audio_pool = []
path = os.path.join(args.data_dir, 'audio')
frames = sorted(os.listdir(os.path.join(args.data_dir, 'img')))
audio_frames = sorted(os.listdir(path))
num_frames = min(len(frames), len(audio_frames))
frame_idx = range(num_frames)
for i in range(len(frame_idx)):
    with open(os.path.join(path, '{:05d}.pickle'.format(int(frames[frame_idx[i]][:-4]) - 1)), 'rb') as f:
        mspec = pkl.load(f)
        audio_pool.append(mspec)

audio_pool = torch.from_numpy(np.array(audio_pool).astype(np.float32))
prior_pool = torch.load(os.path.join(args.data_dir, 'mesh_pca.pt'))[0]

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
model = LipDiscriminator(prior_dim=20, embedding_dim=args.embedding_dim).cuda()

# setup training
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
scheduler = MultiStepLR(optimizer, [int(m) for m in args.milestone.split(',')], gamma=0.1,
                                      last_epoch=- 1)

total_steps = args.steps
save_freq = args.save_freq
log_freq = args.log_freq
ckpt_dir = os.path.join(args.ckpt_dir)
log_path = os.path.join(args.ckpt_dir, args.log_pth)
os.makedirs(ckpt_dir, exist_ok=True)

with open(log_path, 'w') as f:
    f.write('Training Started\n')

def write_log(text):
    with open(log_path, 'a') as f:
        f.write(text + '\r\n')

# train
train_iter = iter(train_dataloader)
item_size = 0
train_loss = 0
best_eval_loss = 100000
for step in range(1, total_steps + 1):
    model.train()
    optimizer.zero_grad()
    try:
        audio, prior, label = train_iter.next()
    except StopIteration:
        train_iter = iter(train_dataloader)
        audio, prior, label = train_iter.next()
    # audio: B x :
    # prior: B x T x prior_dim
    # label: B (1, -1)
    loss = model(audio.cuda(), prior.cuda(), label.cuda()) # B
    loss = loss.sum() # 1
    train_loss += loss.item()
    item_size += len(label)
    loss = loss / len(label)
    loss.backward()
    optimizer.step()
    if step % log_freq == 0:
        print('step [{:05d}] train_loss: {}'.format(step, train_loss / item_size))
        write_log('step [{:05d}] train_loss: {}'.format(step, train_loss / item_size))
        train_loss = 0
        item_size = 0
    if step % save_freq == 0:
        item_size = 0
        positive_size = 0
        negative_size = 0
        eval_loss = 0
        positive_loss = 0
        negative_loss = 0
        model.eval()
        with torch.no_grad():
            for audio, prior, label in eval_dataloader:
                # print('input shape: {}'.format((audio.shape, prior.shape, label.shape)))
                loss = model(audio.cuda(), prior.cuda(), label.cuda()) # B
                positive_loss += loss[label == 1].sum()
                negative_loss += loss[label == -1].sum()
                positive_size += len(loss[label == 1])
                negative_size += len(loss[label == -1])
                loss = loss.sum() # 1
                eval_loss += loss.item()
                item_size += len(label)
        eval_loss = eval_loss / item_size
        positive_loss = positive_loss / positive_size
        negative_loss = negative_loss / negative_size
        print('(Eval) step [{:08d}] eval_loss: {} ({}), positive_loss: {} ({}), negative_loss: {} ({})'.format(step, eval_loss, item_size, positive_loss, positive_size, negative_loss, negative_size))
        write_log('(Eval) step [{:08d}] eval_loss: {} ({}), positive_loss: {} ({}), negative_loss: {} ({})\n'.format(step, eval_loss, item_size, positive_loss, positive_size, negative_loss, negative_size))

        torch.save(model.state_dict(), os.path.join(ckpt_dir, '{:08d}.pt'.format(step)))
        if eval_loss <= best_eval_loss:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pt'))
            best_eval_loss = eval_loss
            print('best loss of step {} saved'.format(step))
            write_log('best loss of step {} saved\n'.format(step))
        scheduler.step()
        positive_p = math.exp(positive_loss) / (math.exp(positive_loss) + math.exp(negative_loss))
        # train_dataloader.dataset.update_p(positive_p)
        # eval_dataloader.dataset.update_p(positive_p)

print('Training Finished with Best Eval Loss: {}'.format(best_eval_loss))
write_log('Training Finished with Best Eval Loss: {}\n'.format(best_eval_loss))



    

