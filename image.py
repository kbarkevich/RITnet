# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 19:09:16 2020

@author: Kevin Barkevich
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from models import model_dict
from dataset import transform
import cv2
from utils import get_predictions
from PIL import Image
from helperfunctions import get_pupil_parameters

def init_model(devicestr="cuda"):
    device = torch.device(devicestr)
    model = model_dict["densenet"]
    model  = model.to(device)
    filename = os.path.dirname(os.path.abspath(__file__)) + "/ritnet_pupil.pkl";
        
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()
    return model

def process_PIL_image(frame, do_corrections=True, clahe=None, table=None):
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    if table is None:
        table = 255.0*(np.linspace(0, 1, 256)**0.8)
    img = Image.fromarray(frame).convert("L")
    if do_corrections:
        img = cv2.LUT(np.array(img), table)
        img = clahe.apply(np.array(np.uint8(img)))
        img = Image.fromarray(img)
    img = transform(img)
    return img

def get_mask_from_path(path: str, model, useGpu=True):
    if useGpu:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    
    pilimg = Image.open(path).convert("L")
    table = 255.0*(np.linspace(0, 1, 256)**0.8)
    pilimg = cv2.LUT(np.array(pilimg), table)
    img = clahe.apply(np.array(np.uint8(pilimg)))    
    img = Image.fromarray(img)
    img = img.unsqueeze(1)
    data = img.to(device)   
    output = model(data)
    predict = get_predictions(output)
    return predict
    
def get_mask_from_cv2_image(image, model, useGpu=True, pupilOnly=False):
    if useGpu:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
        
    img = image.unsqueeze(1)
    data = img.to(device)   
    output = model(data)
    predict = get_predictions(output)
    pred_img = 1 - predict[0].cpu().numpy()/3.0
    if pupilOnly:
        pred_img = np.ceil(pred_img) * 0.5
    return pred_img

def get_mask_from_PIL_image(pilimage, model, useGpu=True, pupilOnly=False):
    img = process_PIL_image(pilimage)
    return get_mask_from_cv2_image(img, model, useGpu, pupilOnly)
    
def get_pupil_ellipse_from_cv2_image(image, model, useGpu=True):
    if useGpu:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
        
    img = image.unsqueeze(1)
    data = img.to(device)   
    output = model(data)
    predict = get_predictions(output)
    return get_pupil_parameters(predict[0].numpy())
    
def get_pupil_ellipse_from_PIL_image(pilimage, model, useGpu=True):
    img = process_PIL_image(pilimage)
    res = get_pupil_ellipse_from_cv2_image(img, model, useGpu)
    if res is not None:
        res[4] = res[4] * 180 / np.pi
    return res