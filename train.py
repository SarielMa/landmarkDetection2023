# !/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os
import tqdm
import torch
import torch.nn.functional as nnF
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')
import logging

from utils.tranforms import Rescale, RandomHorizontalFlip, ToTensor
from utils.dataset import CephaDataset2023
from utils.model import ShapeDeformer

from utils.cldetection_utils import check_and_make_dir
# %%
def setup_logging(log_name, log_file):
    """Setup logging configuration
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # FileHandler to log messages to the specific file
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("[%(asctime)s:%(msecs)03d] - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # StreamHandler to sys.stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger
# %%
def cal_loss_train(obj_pred, obj_true, stage):
    if stage == 0:
        loss = nnF.mse_loss(input=obj_pred.mean(dim=1), target=obj_true.mean(dim=1))
    else: # stage == 1 or 2
        loss = nnF.mse_loss(input=obj_pred, target=obj_true)
    return loss
# %%
@torch.no_grad()
def cal_mrse_eval(obj_pred, obj_true, stage):
    if stage == 0:
        mrse = ((obj_pred.mean(dim=1) - obj_true.mean(dim=1))**2).sum(dim=-1).sqrt().mean()
    else: # stage == 1 or 2
        mrse = ((obj_pred-obj_true)**2).sum(dim=-1).sqrt().mean()
    mrse = float(mrse)
    return mrse
# %%
def main(config):
    # GPU device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    gpu_id = config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # train and valid dataset
    train_transform = transforms.Compose([Rescale(output_size=(config.image_height, config.image_width)),
                                          RandomHorizontalFlip(p=config.flip_augmentation_prob),
                                          ToTensor()])
    train_dataset = CephaDataset2023(csv_file_path=config.train_csv_path, transform=train_transform)
    valid_transform = transforms.Compose([Rescale(output_size=(config.image_height, config.image_width)),
                                          ToTensor()])
    valid_dataset = CephaDataset2023(csv_file_path=config.valid_csv_path, transform=valid_transform)

    # train and valid dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.batch_size_valid,
                              shuffle=False,
                              num_workers=config.num_workers)

    # load model
    model_args = {'patch_size': [4, 4],
                  'n_layers'  : 2,
                  'n_heads'   : 16,
                  'embed_dim' : 512,
                  'alpha'     : 100,
                  'attn_drop' : 0,
                  'C_in'      : 128,
                  'H_in'      : 128,
                  'W_in'      : 128,
                  'beta'      : 1
                  }
    model = ShapeDeformer(model_args)
    model = model.to(device)

    # optimizer and StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 betas=(config.beta1, config.beta2))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=config.scheduler_step_size,
    #                                             gamma=config.scheduler_gamma)

    # model training preparation
    train_losses = []
    valid_losses = []
    best_loss = 1e10
    num_epoch_no_improvement = 0
    check_and_make_dir(config.save_model_dir)
    
    # logging
    logger = setup_logging(log_name=config.setup_log_name, log_file=config.save_model_dir + "log.txt")
    logger.debug(model_args)
    
    # extension
    template = torch.tensor(data=torch.load('./dataset/template.pt', map_location='cpu'), dtype=torch.float32)
    template = template.view(1, -1, 2)
    
    # start to train and valid
    for epoch in range(config.train_max_epoch):
        # scheduler.step(epoch)
        model.train()
        for (img, obj) in tqdm.tqdm(train_loader):
            img, obj = img.float().to(device), obj.float().to(device)
            batch_template = template.expand(obj.shape[0], -1, 2).to(device)
            
            # Extension 2-phase training 
            stage = np.random.choice([0, 1])
            if stage == 0:
                obj_init = batch_template
                noise_large = 2 * torch.rand((obj.shape[0], 1, 2), device=device) - 1
                noise_large[..., 0] *= 20
                noise_large[..., 1] *= 20
                obj_init = obj_init + noise_large
                del noise_large
            else:
                obj_init = batch_template + obj.mean(dim=1, keepdim=True) - batch_template.mean(dim=1, keepdim=True)
                noise_small = 2 * torch.rand((obj.shape[0], 1, 2), device=device) - 1
                noise_small[..., 0] *= 10
                noise_small[..., 1] *= 10
                obj_init = obj_init + noise_small
                del noise_small
            
            obj_pred = model(obj_init, batch_template, img)
            loss = cal_loss_train(obj_pred, obj, stage)
            optimizer.zero_grad()
            loss.backward()
            # extension
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            train_losses.append(round(loss.item(), 3))
        print('Train epoch [{:<4d}/{:<4d}], Loss: {:.6f}'.format(epoch + 1, config.train_max_epoch, np.mean(train_losses)))
        logger.info('Train epoch [{:<4d}/{:<4d}], Loss: {:.6f}'.format(epoch + 1, config.train_max_epoch, np.mean(train_losses)))

        # save model checkpoint
        if epoch % config.save_model_step == 0:
            torch.save(model.state_dict(), os.path.join(config.save_model_dir, 'checkpoint_epoch_%s.pt' % epoch))
            print("Saving checkpoint model ", os.path.join(config.save_model_dir, 'checkpoint_epoch_%s.pt' % epoch))
            logger.info("Saving checkpoint model ", os.path.join(config.save_model_dir, 'checkpoint_epoch_%s.pt' % epoch))

        # valid model, save best_checkpoint.pkl
        with torch.no_grad():
            model.eval()
            print("Validating....")
            logger.info("Validating....")
            for stage in [0, 1]:
                for (img, obj) in tqdm.tqdm(valid_loader):
                    img, obj = img.float().to(device), obj.float().to(device)
                    batch_template = template.expand(obj.shape[0], -1, 2).to(device)
                    if stage == 0:
                        obj_init = batch_template
                        noise_large = 2 * torch.rand((obj.shape[0], 1, 2), device=device) - 1
                        noise_large[..., 0] *= 20
                        noise_large[..., 1] *= 20
                        obj_init = obj_init + noise_large
                        del noise_large
                    else:
                        obj_init = batch_template + obj.mean(dim=1, keepdim=True) - batch_template.mean(dim=1, keepdim=True)
                        noise_small = 2 * torch.rand((obj.shape[0], 1, 2), device=device) - 1
                        noise_small[..., 0] *= 10
                        noise_small[..., 1] *= 10
                        obj_init = obj_init + noise_small
                        del noise_small
                    
                    obj_pred = model(obj_init, batch_template, img)
                    loss = cal_mrse_eval(obj_pred, obj, stage)
                    valid_losses.append(loss)
            valid_loss = np.mean(valid_losses)
            print('Validation loss: {:.6f}'.format(valid_loss))
            logger.info('Validation loss: {:.6f}'.format(valid_loss))

        # early stop mechanism
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.6f} to {:.6f}".format(best_loss, valid_loss))
            logger.info("Validation loss decreases from {:.6f} to {:.6f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            torch.save(model.state_dict(), os.path.join(config.save_model_dir, "best_model.pt"))
            print("Saving best model ", os.path.join(config.save_model_dir, "best_model.pt"))
            logger.info("Saving best model ", os.path.join(config.save_model_dir, "best_model.pt"))
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss, num_epoch_no_improvement))
            logger.info("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss, num_epoch_no_improvement))
            num_epoch_no_improvement += 1
        if num_epoch_no_improvement == config.epoch_patience:
            print("Early Stopping!")
            logger.debug("Early Stopping!")
            break

        # reset parameters
        train_losses = []
        valid_losses = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv_path', type=str, default="./dataset/images/train.csv")
    parser.add_argument('--valid_csv_path', type=str, default="./dataset/images/valid.csv")

    # model hyper-parameters: image_width and image_height
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)

    # model training hyper-parameters
    parser.add_argument('--cuda_id', type=int, default=1)
    parser.add_argument('--train_max_epoch', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_valid', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_model_step', type=int, default=2)

    # data augmentation
    parser.add_argument('--flip_augmentation_prob', type=float, default=0.5)

    # early stop mechanism
    parser.add_argument('--epoch_patience', type=int, default=5)

    # learning rate
    parser.add_argument('--lr', type=float, default=1e-4)

    # Adam optimizer parameters
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Step scheduler parameters
    parser.add_argument('--scheduler_step_size', type=int, default=10)
    parser.add_argument('--scheduler_gamma', type=float, default=0.9)

    # result & save
    parser.add_argument('--save_model_dir', type=str, default='./saves/')
    parser.add_argument('--setup_log_name', type=str, default='ShapeDeformer')
    
    experiment_config = parser.parse_args()
    main(experiment_config)

