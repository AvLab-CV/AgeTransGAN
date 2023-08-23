import os 
import time 
import json 
import argparse
import torch 
import torchvision
import random
import numpy as np 
from data import FaceDataset
from tqdm import tqdm 
from torch import nn
from torch import optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from model.resnet50_ft_dims_2048 import resnet50_ft
import cv2
import torch.nn.functional as F
import csv
LAMBDA_1 = 0.2
LAMBDA_2 = 0.05
START_AGE = 0
END_AGE = 100
VALIDATION_RATE= 0.1

def predict(model, image):

    model.eval()
    with torch.no_grad():
        #image = image.astype(np.float32)
      
        image = np.transpose(image, (2,0,1))
        img = torch.from_numpy(image).cuda()
        img = img.type('torch.FloatTensor').cuda()
       
        output = model(img[None])
        m = nn.Softmax(dim=1)
        
        output = m(output)
        
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
        mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
        pred = np.around(mean)[0][0]
    return pred


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_directory', type=str)
    parser.add_argument('-pm', '--pred_model', type=str, default='./weights/mean_variance_ffhq/model_best_loss')
    parser.add_argument('-path','--pred_path',type=str,default='../result/10')
    parser.add_argument('-out','--outcsv',type=str,default='./test_result/test')
    return parser.parse_args()


def main():
    
    args = get_args()
    
    if args.pred_path and args.pred_model:
        all_num = 0
        all_MAE = 0
        all_age = 0
        model = resnet50_ft()
        model.load_state_dict(torch.load(args.pred_model))
        model.eval()
        model.cuda()
        with open(args.outcsv+'.csv', 'a+', newline='') as csvfile:
            for filename in os.listdir(args.pred_path):
                img = cv2.imread(args.pred_path+"/"+filename)
                resized_img = cv2.resize(img, (224, 224))
                pred = predict(model, resized_img) 
                writer =  csv.writer(csvfile)
                writer.writerow([filename,pred])
                all_age+= pred
                all_num+=1

if __name__ == "__main__":
    main()
