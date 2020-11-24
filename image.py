# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 19:09:16 2020

@author: Kevin Barkevich
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision
import numpy as np
from models import model_dict
from dataset import transform
import cv2
from utils import get_predictions
from PIL import Image
from helperfunctions import get_pupil_parameters
from skimage import io, measure

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
    
def get_mask_from_cv2_image(image, model, useGpu=True, pupilOnly=False, includeRawPredict=False, channels=3, trim_pupil=False):
    if useGpu:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    
    img = image.unsqueeze(1)
    data = img.to(device)   
    output = model(data)
    rawpredict = get_predictions(output)
    predict = rawpredict + 1
    # print(np.unique(predict[0].cpu().numpy()))
    pred_img = 1 - predict[0].cpu().numpy()/channels
    
    # trim pupil if asked to
    if trim_pupil:
        newimg = np.invert(pred_img>0)
        labeled_img = measure.label(newimg)
        labels = np.unique(labeled_img)
        newimg = np.zeros((newimg.shape[0],newimg.shape[1]))
        old_sum = 0
        old_label = None
        for label in (y for y in labels if y != 0):
            if np.sum(labeled_img==label) > old_sum:
                old_sum = np.sum(labeled_img==label)
                old_label = label
        if old_label is not None:
            newimg = newimg + (labeled_img == old_label)
        newimg[newimg == 0] = 2
        newimg[newimg == 1] = 0
        newimg[newimg == 2] = 1
        pred_img[pred_img == 0] = 1-(1/channels)
        pred_img[newimg == 0] = 0
        
    #print(np.unique(pred_img))
    if pupilOnly:
        pred_img = np.ceil(pred_img) * 0.5
    if includeRawPredict:
        return pred_img, rawpredict
    return pred_img

def get_mask_from_PIL_image(pilimage, model, useGpu=True, pupilOnly=False, includeRawPredict=False, channels=3, trim_pupil=False):
    img = process_PIL_image(pilimage)
    return get_mask_from_cv2_image(img, model, useGpu, pupilOnly, includeRawPredict, channels, trim_pupil)
    
def get_pupil_ellipse_from_cv2_image(image, model, useGpu=True, predict=None):
    if useGpu:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    
    if predict is None:
        img = image.unsqueeze(1)
        data = img.to(device)   
        output = model(data)
        predict = get_predictions(output)
        pred_img = predict[0].numpy()
        
    return get_pupil_parameters(pred_img)
    
def get_pupil_ellipse_from_PIL_image(pilimage, model, useGpu=True, predict=None):
    img = process_PIL_image(pilimage)
    res = get_pupil_ellipse_from_cv2_image(img, model, useGpu, predict)
    if res is not None:
        res[4] = res[4] * 180 / np.pi
    return res

def get_area_perimiters_from_mask(image, iris_thresh=0.9, pupil_thresh=0.1):    

    #get threshold image
    ret,thresh_img = cv2.threshold(image, iris_thresh, 255, cv2.THRESH_BINARY_INV)
    thresh_img = thresh_img.astype(np.uint8)
    iris_area = np.sum(thresh_img != 0)
    #find iris contours
    iris_image, iris_contours, iris_hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #get threshold image
    ret,thresh_img = cv2.threshold(image, pupil_thresh, 255, cv2.THRESH_BINARY_INV)
    thresh_img = thresh_img.astype(np.uint8)
    pupil_area = np.sum(thresh_img != 0)
    #find pupil contours
    pupil_image, pupil_contours, pupil_hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    iris_perimeter = 0
    pupil_perimeter = 0
    
    for c in iris_contours:
        peri = cv2.arcLength(c, True)
        iris_perimeter = iris_perimeter + peri
    for c in pupil_contours:
        peri = cv2.arcLength(c, True)
        pupil_perimeter = pupil_perimeter + peri
    
    #print(f'Perimeter = {int(round(perimeter,0))} pixels')
    
    # print("IRIS PERIMETER: " + str(iris_perimeter))
    # print("PUPIL PERIMETER: " + str(pupil_perimeter))

    return iris_perimeter, pupil_perimeter, iris_area, pupil_area

def get_polsby_popper_score(perimeter, area):
    try:
        return (4 * np.pi * area) / (np.square(perimeter))
    except:
        return 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    