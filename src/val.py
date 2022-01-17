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
        val_path = list(data_config['test'].values())[0]

    # Image data transformations
    transforms = T.Compose([T.ToTensor()])

    # Dataset
    dataset = Dataset(val_path, opt=opt)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    opt.nID_dict = dataset.nID_dict
    
    opt.load_model = os.path.join("/hpctmp/e0425991/modelrepo/FairMOT-X/", opt.exp_id)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda:0' if opt.gpus[0] >= 0 else 'cpu')

    for epoch in range(opt.start_epoch, 60 + 1):

        print('Setting up validation data...')

        dataset = Dataset(val_path, opt=opt)
        opt = opts().update_dataset_info_and_set_heads(opt, dataset)

        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset.collate_fn
        )
        
        print('Creating model for Epoch {}...'.format(epoch))

        pthpath = os.path.join(opt.load_model, "model_{}.pth".format(epoch))
        if not os.path.exists(pthpath):
            continue

        model = create_model(opt.arch, opt=opt)
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        model, optimizer, start_epoch = load_model(model, pthpath, optimizer, opt.resume, opt.lr, opt.lr_step)
        
        Trainer = train_factory[opt.task]
        trainer = Trainer(opt, model, optimizer)
        trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

        print("Running forward passes on validation data...")

        with torch.no_grad():
            log_dict_val, preds = trainer.val(epoch, val_loader)
        logger.write('Val Epoch: {} |'.format(epoch))
        for k, v in log_dict_val.items():
            logger.scalar_summary('val_{}'.format(k), v, epoch)
            logger.write('{} {:8f}'.format(k, v))
            if 'time' not in k:
                logger.write(' | ')
        logger.write('\n')

    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    run(opt)
