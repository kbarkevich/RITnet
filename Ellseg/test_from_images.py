#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:12:40 2020

@author: aaa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tqdm
import torch
import pickle
#import resource
import numpy as np
import matplotlib.pyplot as plt

from args import parse_args
from modelSummary import model_dict
from pytorchtools import load_from_file
from torch.utils.data import DataLoader
from utils import get_nparams, get_predictions
from helperfunctions import mypause, stackall_Dict
from utils import getSeg_metrics, getPoint_metric, generateImageGrid, unnormPts,getPoint_metric_norm
from utils import getAng_metric,Logger
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (2048*10, rlimit[1]))
import time
#%%
if __name__ == '__main__':

    args = parse_args()
#%%
    device=torch.device("cuda")
    torch.cuda.manual_seed(12)
    if torch.cuda.device_count() > 1:
        print('Moving to a multiGPU setup.')
        args.useMultiGPU = True
    else:
        args.useMultiGPU = False
    torch.backends.cudnn.deterministic=False

    if args.model not in model_dict:
        print("Model not found.")
        print("valid models are: {}".format(list(model_dict.keys())))
        exit(1)

    if args.seg2elactivated:
        path_intermediate='_0_0'#'with_seg2el'
    else:
        path_intermediate='_1_0'#'without_seg2el'       
#RC_e2e_allvsone_ritnet_v2_allvsone_0_0
#    if args.expname=='':
#    args.expname='RC_e2e_baseline_'+args.model+'_'+args.curObj+path_intermediate#'_0_0'
    args.expname='RC_e2e_allvsone_'+args.model+'_allvsone'+path_intermediate#'_0_0'

#    args.expname='RC_e2e_leaveoneout_'+args.model+'_'+args.curObj+path_intermediate#'_0_0'
    
    LOGDIR = os.path.join(os.getcwd(), 'ExpData', 'logs',\
                          args.model, args.expname)
#    LOGDIR = os.path.join(os.getcwd(), 'logs', args.model, args.expname)
#    path2model = os.path.join(LOGDIR, 'weights')
    path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
    path2op = os.path.join(os.getcwd(), 'op5', str(args.curObj),args.model)
    path2op_mask = os.path.join(os.getcwd(), 'op5', str(args.curObj), args.model,'mask')

#%%
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(path2checkpoint, exist_ok=True)
    os.makedirs(path2op, exist_ok=True)
    os.makedirs(path2op_mask, exist_ok=True)
    os.makedirs('./images/images/',exist_ok=True)
    os.makedirs('./images/output/',exist_ok=True)

    model = model_dict[args.model]

    checkpointfile = os.path.join(path2checkpoint, 'checkpoint.pt')
    netDict = load_from_file([checkpointfile, args.loadfile])
    model.load_state_dict(netDict['state_dict'])
    
#    netDict = load_from_file([args.loadfile, path2checkpoint])
#    startEp = netDict['epoch'] if 'epoch' in netDict.keys() else 0
#    if 'state_dict' in netDict.keys():
#        model.load_state_dict(netDict['state_dict'])

    print('Parameters: {}'.format(get_nparams(model)))
    model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
    model = model.to(device).to(args.prec)

    filepath='./images/'
    list_all=[]
    for file in os.listdir(os.path.join(filepath,'images')):   
        if file.endswith(".png"):
           list_all.append(file.strip(".png"))

    
    model.eval()

    with torch.no_grad():
        for i in range (len(list_all)):
            print (i)
            img=np.array(Image.open(filepath+'images/'+list_all[i]+'.png').convert("L"))
            img = (img - img.mean())/img.std()
            img = torch.from_numpy(img).unsqueeze(0).to(args.prec)  # Adds a singleton for channels
            img = img.unsqueeze(0)
            
            x4, x3, x2, x1, x =model.enc(img.to(device).to(args.prec))
            op = model.dec(x4, x3, x2, x1, x)
            output=get_predictions(op)
            
            pred_img = output[0].cpu().numpy()/3.0
            plt.imsave(filepath+'output/{}.jpg'.format(i),pred_img)

            