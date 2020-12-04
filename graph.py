# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:47:52 2020

@author: Kevin Barkevich
"""

import matplotlib.pyplot as plt
import json
from os import path


FILE_NAME = "pp_data.txt"
SPACING = 1
RANGE = None  # None for entire viceo
IGNORE_ZEROS = True  # Frames where no pupil at all could be detected, such as during blinks and/or before the eye comes into frame, are treated as 0.0.

def print_stats(file_name=FILE_NAME, spacing=SPACING, frame_range=RANGE, ignore_zeros=IGNORE_ZEROS):
    x = []
    y_pp = []
    y_pp_diff = []
    y_shape_conf = []
    
    with open(file_name) as json_file:
        data = json.load(json_file)
        for p in data:
            if int(p) % spacing == 0 and (frame_range is None or (int(p) >= frame_range[0] and int(p) <= frame_range[1])) and (ignore_zeros == False or float(data[p]["pp"]) != 0.0):
                x.append(int(p))
                if "pp" in data[p].keys():
                    y_pp.append(float(data[p]["pp"]))
                if "pp_diff" in data[p].keys():
                    y_pp_diff.append(float(data[p]["pp_diff"]))
                if "IOU" in data[p].keys():
                    y_shape_conf.append(float(data[p]["IOU"]))
                elif "shape_conf" in data[p].keys():
                    y_shape_conf.append(float(data[p]["shape_conf"]))
                
        plt.title("Image Pupil Scoring (every " + str(spacing) + " pixel[s])")
        plt.xlabel("frame")
        plt.ylabel("score")
        plt.plot(x, y_pp, color='olive', label="Image Score")
        plt.plot(x, y_pp_diff, color='blue', label="Difference Image Score, Ellipse Score")
        plt.plot(x, y_shape_conf, color='red', label="Pupil Shape Confidence")
        plt.ylim(bottom=0, top=1)
        plt.legend()
        plt.show()
        
    y_pp_copy = y_pp.copy()
    y_pp_diff_copy = y_pp_diff.copy()
    y_shape_conf_copy = y_shape_conf.copy()
    y_pp_copy.sort()
    y_pp_diff_copy.sort(reverse=True)
    y_shape_conf_copy.sort()
    
    onePercent = int(len(x)/100)
    ptOnePercent = int(len(x)/1000)
    
    print("-------------PP: Higher Is Better-------------")
    if (len(y_pp_copy) > 0):
        val = sum(y_pp_copy)/len(y_pp_copy)
        print("            PP mean: ", val)
    if (len(y_pp_copy) > 0):
        val = y_pp_copy[int(len(y_pp_copy)/2)]
        print("          PP median: ", val)
    if len(y_pp_copy) >= onePercent and onePercent != 0:
        val = sum(y_pp_copy[0:onePercent])/onePercent
        print("  PP 1% low average: ", val)
    if len(y_pp_copy) >= ptOnePercent and ptOnePercent != 0:
        val = sum(y_pp_copy[0:ptOnePercent])/ptOnePercent
        print("PP 0.1% low average: ", val)
    print('\n-------------PP Difference: Lower Is Better-------------')
    if (len(y_pp_diff_copy) > 0):
        val = sum(y_pp_diff_copy)/len(y_pp_diff_copy)
        print("            PP Diff mean: ", val)
    if (len(y_pp_diff_copy) > 0):
        val = y_pp_diff_copy[int(len(y_pp_diff_copy)/2)]
        print("          PP Diff median: ", val)
    if len(y_pp_diff_copy) >= onePercent and onePercent != 0:
        val = sum(y_pp_diff_copy[0:onePercent])/onePercent
        print("  PP Diff 1% low average: ", val)
    if len(y_pp_diff_copy) >= ptOnePercent and ptOnePercent != 0:
        val = sum(y_pp_diff_copy[0:ptOnePercent])/ptOnePercent
        print("PP Diff 0.1% low average: ", val)
    print('\n-------------Pupil Shape Confidence: Higher Is Better-------------')
    if (len(y_shape_conf_copy) > 0):
        val = sum(y_shape_conf_copy)/len(y_shape_conf_copy)
        print("            Pupil Shape Conf mean: ", val)
    if (len(y_shape_conf_copy) > 0):
        val = y_shape_conf_copy[int(len(y_shape_conf_copy)/2)]
        print("          Pupil Shape Conf median: ", val)
    if len(y_shape_conf_copy) >= onePercent and onePercent != 0:
        val = sum(y_shape_conf_copy[0:onePercent])/onePercent
        print("  Pupil Shape Conf 1% low average: ", val)
    if len(y_shape_conf_copy) >= ptOnePercent and ptOnePercent != 0:
        val = sum(y_shape_conf_copy[0:ptOnePercent])/ptOnePercent
        print("Pupil Shape Conf 0.1% low average: ", val) 
    

if __name__ == '__main__':
    print_stats()
