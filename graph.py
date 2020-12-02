# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:47:52 2020

@author: Kevin Barkevich
"""

import matplotlib.pyplot as plt
import json
from os import path


FILE_NAME = "pp_data.txt"
SPACING = 50

def main():
    x = []
    y_pp = []
    y_pp_diff = []
    y_shape_conf = []
    
    with open(FILE_NAME) as json_file:
        data = json.load(json_file)
        for p in data:
            if int(p) % SPACING == 0:
                x.append(int(p))
                if "pp" in data[p].keys():
                    y_pp.append(float(data[p]["pp"]))
                if "pp_diff" in data[p].keys():
                    y_pp_diff.append(float(data[p]["pp_diff"]))
                if "IOU" in data[p].keys():
                    y_shape_conf.append(float(data[p]["IOU"]))
                elif "shape_conf" in data[p].keys():
                    y_shape_conf.append(float(data[p]["shape_conf"]))
                
        plt.title("Image Pupil Scoring (every " + str(SPACING) + " pixel[s])")
        plt.xlabel("frame")
        plt.ylabel("score")
        plt.plot(x, y_pp, color='olive', label="Image Score")
        plt.plot(x, y_pp_diff, color='blue', label="Difference Image Score, Ellipse Score")
        plt.plot(x, y_shape_conf, color='red', label="Pupil Shape Confidence")
        plt.ylim(bottom=0, top=1)
        plt.legend()
        plt.show()
    

if __name__ == '__main__':
    main()
