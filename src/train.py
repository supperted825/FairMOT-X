from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch

import json
import torch.utils.data
from torchvision.transforms import transforms as T
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory


def run(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.task, opt.multi_scale)  # if opt.task==mot -> JointDataset

    with open(opt.data_cfg) as f:  # choose which dataset to train '../src/lib/cfg/mot15.json',
        data_config = json.load(f)
        train_path = list(data_config['train'].values())[0]
        dataset_root = data_config['root']
    print("Dataset Root: %s" % dataset_root)

    # Dataset
    dataset = Dataset(train_path, opt=opt)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    opt.nID_dict = dataset.nID_dict
    print("opt:\n", opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Creating model...', flush=True)
    model = create_model(opt.arch, opt=opt)

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model,
                                                   opt.load_model,
                                                   optimizer,
                                                   opt.resume,
                                                   opt.lr,
                                                   opt.lr_step)

    # Get dataloader
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                drop_last=True,
                                                collate_fn=dataset.collate_fn)


    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    
    if opt.l1_loss:
        print("Training with L1 Loss from Start", flush=True)
        model.head.use_l1 = True
    
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        
        if epoch >= opt.num_epochs - 5:
            print('No Mosaic from Epoch 15 - Now Using L1 Loss', flush=True)
            dataset.mosaic = False
            model.head.use_l1 = True

        # Train an epoch
        log_dict_train, _ = trainer.train(epoch, train_loader)

        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f}'.format(k, v))
            if 'time' not in k:
                logger.write(' | ')

        if not os.path.isdir(f'/hpctmp/e0425991/modelrepo/FairMOT-X/{opt.exp_id}/'):
            os.mkdir(f'/hpctmp/e0425991/modelrepo/FairMOT-X/{opt.exp_id}/')
            
        save_model(os.path.join(f'/hpctmp/e0425991/modelrepo/FairMOT-X/{opt.exp_id}/', f'model_{epoch}.pth'), epoch, model, optimizer)
        save_model(os.path.join(f'/hpctmp/e0425991/modelrepo/FairMOT-X/{opt.exp_id}/', 'model_last.pth'), epoch, model, optimizer)

        logger.write('\n')

        if epoch in opt.lr_step:
            
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr, flush=True)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    logger.close()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # '0, 1'
    opt = opts().parse()
    run(opt)
