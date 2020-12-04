#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:50:11 2019

@author: manoj
"""


from densenet import DenseNet2D
model_dict = {}

model_dict['densenet'] = DenseNet2D(dropout=True,prob=0.2)
model_dict['densenet_4ch'] = DenseNet2D(dropout=True,prob=0.2,out_channels=4)
model_dict['densenet_3ch'] = DenseNet2D(dropout=True,prob=0.2,out_channels=3)

model_channel_dict = {}
model_channel_dict['best_model.pkl'] = ('densenet_4ch', 4)
model_channel_dict['ritnet_pupil.pkl'] = ('densenet_4ch', 4)
model_channel_dict['ritnet_400400.pkl'] = ('densenet_3ch', 3)
