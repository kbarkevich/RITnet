# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:36:02 2020

@author: Kevin Barkevich
"""
import torch
from dataset import IrisDataset
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
from dataset import transform
import os
import cv2
from opt import parse_args
from models import model_dict
from tqdm import tqdm
from utils import get_predictions
from PIL import Image

if __name__ == '__main__':
    
    args = parse_args()
   
    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)

    if args.useGPU:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
        
    model = model_dict[args.model]
    model  = model.to(device)
    filename = args.load
    if not os.path.exists(filename):
        print("model path not found !!!")
        exit(1)
        
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()
    
    if not os.path.exists(args.video):
        print("input video not found!")
        exit(1)
    
    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    videowriter = cv2.VideoWriter("out.mp4", fourcc, fps, (int(width),int(height)))
    while not video.isOpened():
        video = cv2.VideoCapture(args.video)
        cv2.waitKey(1000)
        print("Wait for the header")
    
    pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    table = 255.0*(np.linspace(0, 1, 256)**0.8)
    count = 0
    
    def process_frame(frame, do_corrections=True):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert("L")
        if do_corrections:
            img = cv2.LUT(np.array(img), table)
            img = clahe.apply(np.array(np.uint8(img)))
            img = Image.fromarray(img)
        img = transform(img)
        return img
    
    while True:
        flag, frame = video.read()
        if flag:
            count += 1
            img = process_frame(frame)
            img = img.unsqueeze(1)
            data = img.to(device)
            output = model(data)
            predict = get_predictions(output)
            
            # cv2.imshow('video', frame)
            # cv2.imshow('output', output[0][0].cpu().detach().numpy()/3.0)
            # cv2.imshow('mask', predict[0].cpu().numpy()/3.0)
            pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
            
            pred_img = predict[0].cpu().numpy()/3.0
            inp = process_frame(frame, False).squeeze() * 0.5 + 0.5
            img_orig = np.clip(inp,0,1)
            img_orig = np.array(img_orig)
            combine = np.hstack([img_orig,pred_img])
            cv2.imshow('RITnet', combine)
            videowriter.write((pred_img * 255).astype('uint8'))  # write to video output
            print(str(pos_frame)+" frames")
        else:
            # Wait for next frame
            video.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            cv2.waitKey(1000)
        
        if cv2.waitKey(10) == 27:
            video.release()
            videowriter.release()
            cv2.destroyAllWindows()
            break
        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
            video.release()
            videowriter.release()
            cv2.destroyAllWindows()
            break
            

    # os.rename('test',args.save)