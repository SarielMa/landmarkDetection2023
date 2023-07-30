import argparse
import datetime
import os
from pathlib import Path
import time
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.nn import functional as F
from torch.nn import BCELoss

from network import UNet, UNet_Pretrained
from data_loader import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from myTest import Tester
import matplotlib.pyplot as plt
import numpy as np
from metric import total_loss
# max = 1
# min = -1

if __name__ == "__main__":
    #CUDA_VISIBLE_DEVICES=0

    #device = torch.device('cuda:1')
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='base', help="name of the run")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--iterations", default = "100")
    parser.add_argument("--cuda_id", default=1, type = int, help="cuda id")
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_id)
 
    # Load yaml config file
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    
    # Create Logger
    logger = get_mylogger()
    logger.info(config)

    # Create runs dir
    tag = str(datetime.datetime.now()).replace(' ', '_') if args.tag == '' else args.tag
    runs_dir = "./runs/" + tag
    runs_path = Path(runs_dir)
    config['runs_dir'] = runs_dir
    if not runs_path.exists():
        runs_path.mkdir()
    #set_logger_dir(logger, runs_dir)

    dataset = Cephalometric(config['dataset_pth'], 'Train')
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                shuffle=True, num_workers=config['num_workers'])
    
    # net = UNet(3, config['num_landmarks'])
    net = UNet_Pretrained(3, config['num_landmarks'])
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    logger.info(net)

    optimizer = optim.Adam(params=net.parameters(), \
        lr=config['learning_rate'], betas=(0.9,0.999), eps=1e-08, weight_decay=1e-4)
    
    scheduler = StepLR(optimizer, config['decay_step'], gamma=config['decay_gamma'])

    # Tester
    tester = Tester(logger, config, tag=args.tag)
    
    loss_train_list = list()
    loss_val_list = list()
    MRE_list = list()
    iterations = int(args.iterations)
    for epoch in range(iterations):
        logic_loss_list = list()
        regression_loss_list = list()
        net.train()
        for img, mask, guassian_mask, offset_y, offset_x, landmark_list in tqdm(dataloader):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
            heatmap, regression_y, regression_x = net(img)
            loss, loss_regression = total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
            """
            logic_loss = loss_logic_fn(heatmap, guassian_mask) # find min size          
            regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
            regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)
            loss =  regression_loss_x + regression_loss_y + logic_loss * config['lambda']
            loss_regression = regression_loss_y + regression_loss_x
            """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logic_loss_list.append(loss.cpu().item())
            regression_loss_list.append(loss_regression.cpu().item())     
        loss_train = sum(logic_loss_list) / dataset.__len__()
        
        logger.info("Epoch {} Training total loss {} regression loss {}".\
            format(epoch, loss_train, \
                sum(regression_loss_list) / dataset.__len__()))
        loss_train_list.append(loss_train)
        MRE, loss_val,_, loss_reg = tester.validate(net)
        logger.info("Epoch {} Testing MRE {} logic loss {} regression loss {}".format(epoch, MRE, loss_val, loss_reg))
        loss_val_list.append(loss_val)
        MRE_list.append(MRE)
        
        
        # save model and plot the trend
        if (epoch + 1) % config['save_seq'] == 0:
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
            # plot the trend
            cols = ['b','g','r','y','k','m','c']
            
            fig,axs = plt.subplots(1,3, figsize=(15,5))

            #ax = fig.add_subplot(111)
            X = list(range(epoch+1))
            axs[0].plot(X, loss_train_list, color=cols[0], label="Training Loss")
            axs[1].plot(X, loss_val_list, color=cols[1], label="Validation Loss")
            axs[2].plot(X, MRE_list, color=cols[2], label="MRE")
            axs[0].set_xlabel("epoch")   
            axs[1].set_xlabel("epoch")
            axs[2].set_xlabel("epoch")
            axs[0].legend()
            axs[1].legend()
            axs[2].legend()           
            fig.savefig(runs_dir +"training.png")
            #save the last epoch
            config['last_epoch'] = epoch

        # dump yaml
        with open(runs_dir + "/config.yaml", "w") as f:
            yaml.dump(config, f)

    # # Test
    tester.test(net)
