from tqdm import trange
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import MeshGeneratorFullModel, MeshDiscriminatorFullModel
from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater


def train(config, generator, discriminator, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.dense_motion_network.audio_prior.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator)
        start_epoch = 0

        # start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, optimizer_generator=optimizer_generator)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch= start_epoch - 1)


    if 'num_repeats' in train_params and train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = MeshGeneratorFullModel(generator, discriminator, train_params)
    discriminator_full = MeshDiscriminatorFullModel(generator, discriminator, train_params)

    # freeze models
    for name, p in generator_full.named_parameters():
        if 'audio_prior' not in name:
            p.requires_grad = False
    for name, p in generator.dense_motion_network.audio_prior.named_parameters():
        p.requires_grad = True

        
    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        epoch = start_epoch
        for _ in trange(0, 200):
            for step, x in tqdm(enumerate(dataloader)):
                pool = dataloader.dataset.get_pool(train_params['pool_size'])
                pool = (pool[0].unsqueeze(0).repeat(3, 1, 1), pool[1].unsqueeze(0).repeat(3, 1, 1, 1), pool[2].unsqueeze(0).repeat(3, 1, 1))
                x['pool'] = pool
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()

                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

                if (step + 1) % train_params['log_freq'] == 0:
                    logger.log_epoch(epoch, {'generator': generator,
                                        'discriminator': discriminator,
                                        'optimizer_generator': optimizer_generator}, inp=x, out=generated)
                    epoch += 1
                    scheduler_generator.step()

                    
                   
