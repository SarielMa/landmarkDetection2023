# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
from skimage import transform
from skimage import io as sk_io

import warnings
warnings.filterwarnings('ignore')

from utils.model import ShapeDeformer
from utils.cldetection_utils import check_and_make_dir, calculate_prediction_metrics, visualize_prediction_landmarks


def main(config):
    # GPU device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    gpu_id = config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

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
    model.load_state_dict(torch.load(config.load_weight_path, map_location=device))
    model = model.to(device)

    # load test.csv
    df = pd.read_csv(config.test_csv_path)

    # test result dict
    test_result_dict = {}
    
    # extension
    template = torch.tensor(data=torch.load('./dataset/template.pt', map_location='cpu'), dtype=torch.float32)
    template = template.view(1, -1, 2)
    template = template.to(device)

    # test mode
    with torch.no_grad():
        model.eval()
        # test all images
        for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            image_file_path, scale = str(df.iloc[index, 0]), float(df.iloc[index, 1])
            landmarks = df.iloc[index, 2:].values.astype('float')
            landmarks = landmarks.reshape(-1, 2)

            # load image array
            image = sk_io.imread(image_file_path)
            h, w = image.shape[:2]
            new_h, new_w = config.image_height, config.image_width

            # preprocessing image for model input
            image = transform.resize(image, (new_h, new_w), mode='constant', preserve_range=False)
            transpose_image = np.transpose(image, (2, 0, 1))
            torch_image = torch.from_numpy(transpose_image[np.newaxis, :, :, :]).float().to(device)
            predict_landmarks = []
            
            # model predict
            # stage 0
            obj_pred_0 = template     
            for n_iter in range(0, 1):
                obj_deformed = model(obj_init=obj_pred_0, template=template, img=torch_image)
                u_0 = (obj_deformed - obj_pred_0).mean(dim=1, keepdim=True)
                obj_pred_0 = obj_pred_0 + u_0            
            # stage 1
            obj_pred_1 = obj_pred_0
            for n_iter in range(0, 1):
                obj_pred_1 = model(obj_init=obj_pred_1, template=template, img=torch_image)
            obj_pred = obj_pred_1 # (num_nodes, 2)
            
            for obj_iter in range(obj_pred[0].shape[0]):
                predict_landmarks.append([obj_pred[0][obj_iter][0].item(), obj_pred[0][obj_iter][1].item()])

            test_result_dict[image_file_path] = {'scale': scale,
                                                 'gt': np.asarray(landmarks),
                                                 'predict': np.asarray(predict_landmarks)}

    # calculate prediction metrics
    calculate_prediction_metrics(test_result_dict)

    # visualize prediction landmarks
    if config.save_image:
        check_and_make_dir(config.save_image_dir)
        visualize_prediction_landmarks(test_result_dict, config.save_image_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--test_csv_path', type=str, default='./dataset/images/test.csv')

    # model load dir path
    parser.add_argument('--load_weight_path', type=str, default='./saves/best_model.pt')

    # model hyper-parameters: image_width and image_height
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)

    # model test hyper-parameters
    parser.add_argument('--cuda_id', type=int, default=0)

    # result & save
    parser.add_argument('--save_image', type=bool, default=True)
    parser.add_argument('--save_image_dir', type=str, default='./visualize/')

    experiment_config = parser.parse_args()
    main(experiment_config)

