# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:36:02 2020

@author: Kevin Barkevich
"""
import torch
import numpy as np
import os
import cv2
from opt import parse_args
from models import model_dict, model_channel_dict
import matplotlib.pyplot as plt
from image import get_mask_from_PIL_image, process_PIL_image, get_area_perimiters_from_mask, get_polsby_popper_score, get_pupil_ellipse_from_PIL_image
import asyncio
import math
import datetime
import json
from graph import print_stats

from helperfunctions import get_pupil_parameters, ellipse_area, ellipse_circumference
from Ellseg.pytorchtools import load_from_file
from Ellseg.utils import get_nparams, get_predictions
from Ellseg.args import parse_precision

# INITIAL LOADING OF ARGS
args = parse_args()
filename = args.load
if not os.path.exists(filename):
    print("model path not found !!!")
    exit(1)
MODEL_DICT_STR, CHANNELS, IS_ELLSEG, ELLSEG_MODEL = model_channel_dict[filename]
ELLSEG_FOLDER = 'Ellseg'
ELLSEG_FILEPATH = './'+ELLSEG_FOLDER
ELLSEG_PRECISION = 32  # precision. 16, 32, 64

ELLSEG_PRECISION = parse_precision(ELLSEG_PRECISION)

# SETTINGS
ROTATION = 0
PAD = False
THREADED = False
SEPARATE_ORIGINAL_VIDEO = False
SAVE_SEPARATED_PP_FRAMES = True  # Setting enables Polsby-Popper scoring, which slows down processing
SHOW_PP_OVERLAY = True  # Setting enables Polsby-Popper scoring, which slows down processing
SHOW_PP_GRAPH = False  # Setting enables Polsby-Popper scoring, which slows down processing
OUTPUT_PP_DATA_TO_JSON = True  # Setting enables Polsby-Popper scoring, which slows down processing
OVERLAP_MASK = True
KEEP_BIGGEST_PUPIL_BLOB_ONLY = True
START_FRAME = 0
ISOLATE_FRAMES = [606,808,14152,17751,23714]  # Set to save independent frames of the output into a dedicated folder, for easy mass-data-gathering.

def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=3, lineType=cv2.LINE_AA, shift=10):
    center = (
        int(round(center[0] * 2**shift)),
        int(round(center[1] * 2**shift))
    )
    axes = (
        int(round(axes[0] * 2**shift)),
        int(round(axes[1] * 2**shift))
    )
    try:
        return cv2.ellipse(
            img, center, axes, angle,
            startAngle, endAngle, color,
            thickness, lineType, shift)
    except:
        return None


def main():
    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)

    if args.useGPU:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    
    if torch.cuda.device_count() > 1:
        print('Moving to a multiGPU setup for Ellseg.')
        args.useMultiGPU = True
    else:
        args.useMultiGPU = False
        
    model = model_dict[MODEL_DICT_STR]
    
    if not IS_ELLSEG:
        model  = model.to(device)
        model.load_state_dict(torch.load(filename))
        model = model.to(device)
        model.eval()
    else:
        LOGDIR = os.path.join(os.getcwd(), ELLSEG_FOLDER, 'ExpData', 'logs',\
                          'ritnet_v2', ELLSEG_MODEL)
        path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
        checkpointfile = os.path.join(path2checkpoint, 'checkpoint.pt')
        netDict = load_from_file([checkpointfile, ''])
        #print(checkpointfile)
        #print(netDict)
        model.load_state_dict(netDict['state_dict'])
        #print('Parameters: {}'.format(get_nparams(model)))
        model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
        model = model.to(device)
        model = model.to(ELLSEG_PRECISION)
        model.eval()
    
    if not os.path.exists(args.video):
        print("input video not found!")
        exit(1)
    
    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    os.makedirs('video/images/',exist_ok=True)
    os.makedirs('video/outputs/',exist_ok=True)
    os.makedirs('video/pp-separation/0.'+str(0)+"-0."+str(1),exist_ok=True)
    os.makedirs('video/pp-diff-separation/0.'+str(0)+"-0."+str(1),exist_ok=True)
    for i in range(1, 9):
        os.makedirs('video/pp-separation/0.'+str(i)+"-0."+str((i+1)),exist_ok=True)
        os.makedirs('video/pp-diff-separation/0.'+str(i)+"-0."+str((i+1)),exist_ok=True)
    os.makedirs('video/pp-separation/0.'+str(9)+"-"+str(1)+".0",exist_ok=True)
    os.makedirs('video/pp-diff-separation/0.'+str(9)+"-"+str(1)+".0",exist_ok=True)
    os.makedirs('video/outputs/',exist_ok=True)
    os.makedirs('video/isolated/', exist_ok=True)
    
    mult = 2
    if OVERLAP_MASK:
        mult = 3
    if PAD and width == 192 and height == 192:
        videowriter = cv2.VideoWriter("video/outputs/out.mp4", fourcc, fps, (int(width*mult+(64*mult)),int(height*2)))
    elif PAD and width == 400 and height == 400:
        videowriter = cv2.VideoWriter("video/outputs/out.mp4", fourcc, fps, (int(width*mult+(133*mult)),int(height*2)))
    else:
        videowriter = cv2.VideoWriter("video/outputs/out.mp4", fourcc, fps, (int(width*mult),int(height)))
    # maskvideowriter = cv2.VideoWriter("video/mask.mp4", fourcc, fps, (int(width),int(height)))
    while not video.isOpened():
        video = cv2.VideoCapture(args.video)
        cv2.waitKey(1000)
        print("Wait for the header")
    
    pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    
    # GAMMA CORRECTION STEP
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))  # EDIT NUMBERS HERE FOR POSSIBLE BETTTER LOW-LIGHT PERFORMANCE
    table = 255.0*(np.linspace(0, 1, 256)**0.6)  # CHANGE 0.8 TO 0.6 FOR THE DARKER VIDEO
    
    count = 0

    def get_stretched_combine(frame, pad):
        frame1 = cv2.copyMakeBorder(
                    frame,
                    top=0,
                    bottom=0,
                    left=int(pad),
                    right=int(pad),
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0,0,0)
            )
        # Perform the rotation
        (h, w) = frame1.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, -25, 1.0)
        frame1 = cv2.warpAffine(frame1, M, (w, h))
        pred_img = get_mask_from_PIL_image(frame1, model, True, False)

        inp = process_PIL_image(frame1, False, clahe, table).squeeze() * 0.5 + 0.5

        img_orig = np.clip(inp,0,1)
        img_orig = np.array(img_orig)
        stretchedcombine = np.hstack([img_orig,get_mask_from_PIL_image(frame1, model, True, False),pred_img])
        return stretchedcombine
    
    max_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    pp_x = []
    pp_iris_y = []
    pp_pupil_y = []
    pp_pupil_diff_y = []
    iou_y = []
    pp_data = {}
    seconds_arr = []
    while True:
        seconds_start = datetime.datetime.now()  # Start timer
        flag, frame = video.read()
        if flag:
            count += 1
            if count < START_FRAME:
                continue
            pp_x.append(count)
            # cv2.imshow('video', frame)
            # cv2.imshow('output', output[0][0].cpu().detach().numpy()/CHANNELS)
            # cv2.imshow('mask', predict[0].cpu().numpy()/CHANNELS)
            pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
            
            pad = False
            
            # If the video is 192x192, pad the sides 32 pixels each
            # STEP - VIDEO TO 4:3 RATIO VIA PADDING - ADD 32 BLACK PIXELS TO EACH SIDE FOR A 192x192 IMAGE
            if PAD and (tuple(frame.shape[1::-1]) == (192, 192) or tuple(frame.shape[1::-1]) == (400, 400)):
                #pass
                pad = True
                comb = get_stretched_combine(frame.copy(), tuple(frame.shape[1::-1])[0]/6)
            
            # ---------------------------------------------------
            
            # Perform the rotation
            (h, w) = frame.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, ROTATION, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
                
            pred_img, predict = get_mask_from_PIL_image(frame, model, True, False, True, CHANNELS, KEEP_BIGGEST_PUPIL_BLOB_ONLY, isEllseg=IS_ELLSEG, ellsegPrecision=ELLSEG_PRECISION)
            
            IOU = 0
            
            # Calculate PP score data only if PP score is used
            if SAVE_SEPARATED_PP_FRAMES or SHOW_PP_OVERLAY or SHOW_PP_GRAPH or OUTPUT_PP_DATA_TO_JSON:
                # Scoring step 1: Get area/perimeter directly from mask
                if CHANNELS == 4:
                    iris_perimeter, pupil_perimeter, iris_area, pupil_area = get_area_perimiters_from_mask(pred_img, iris_thresh=0.6, pupil_thresh=0.1)
                else:
                    iris_perimeter, pupil_perimeter, iris_area, pupil_area = get_area_perimiters_from_mask(pred_img)
                # Scoring step 2: Get pupil & iris scores from area/perimeter
                pp_iris = get_polsby_popper_score(iris_perimeter, iris_area)
                pp_pupil = get_polsby_popper_score(pupil_perimeter, pupil_area)
                pp_pupil_diff = 0
                # Scoring step 3: Get ellipse from mask
                params_get = predict[0].numpy()/CHANNELS
                params_get[params_get < (CHANNELS-1)/CHANNELS] = 0
                params_get[params_get >= (CHANNELS-1)/CHANNELS] = 0.75
                pupil_ellipse = get_pupil_parameters(1-params_get)
                
              
                
                if pupil_ellipse is not None and pupil_ellipse[0] >= -0:
                    center_coordinates = (int(pupil_ellipse[0]), int(pupil_ellipse[1]))
                    #axesLength = (int(pupil_ellipse[2])+2, int(pupil_ellipse[3])+2)
                    axesLength = (int(pupil_ellipse[2]), int(pupil_ellipse[3]))
                    angle = pupil_ellipse[4]*180/(np.pi)
                    print("angle: ",angle)
                    startAngle = 0
                    endAngle = 360
                    color = (0, 0, 255)
                    
                    ellimage = np.zeros((int(height), int(width), 3), dtype="uint8")
                    ellimage = draw_ellipse(ellimage, center_coordinates, axesLength,
                                            angle, startAngle, endAngle, color, -1)
                    
                    image_copy = np.zeros((int(height), int(width), 3), dtype = "uint8")
                    
                    pupilimage = np.where(pred_img == 0)

                    image_copy[pupilimage] =  [0, 255, 0]
                    ellimage = (ellimage + image_copy) if ellimage is not None else ellimage
                    # cv2.imshow("ELLIPSE", ellimage)
                    
                    intersection = np.sum(np.all(ellimage == [0, 255, 255], axis=2)) if ellimage is not None else 0
                    union = np.sum(~np.all(ellimage == [0, 0, 0], axis=2)) if ellimage is not None else 0
                    if union != 0:
                        IOU = intersection/union
                    
                    major_axis = pupil_ellipse[2]
                    minor_axis = pupil_ellipse[3]
                    pupil_ellipse_area = ellipse_area(major_axis, minor_axis)
                    # print(pupil_ellipse_area)
                    pupil_ellipse_perimeter = ellipse_circumference(major_axis, minor_axis)
                    # Scoring step 4: Get pupil ellipse area/perimeter
                    pp_pupil_ellipse = get_polsby_popper_score(pupil_ellipse_perimeter, pupil_ellipse_area)
                    if math.isnan(pp_pupil) or pp_pupil >= 1 or pp_pupil <= 0:
                        pp_pupil_diff = 0
                    else:
                        pp_pupil_diff = abs(pp_pupil - pp_pupil_ellipse)
                else:
                    pp_pupil = 0
                if math.isnan(pp_pupil) or pp_pupil >= 1 or pp_pupil <= 0:
                    pp_pupil = 0
                    
                if math.isnan(pp_iris) or pp_iris >= 1 or pp_iris <= 0:
                    if len(pp_iris_y) > 0:
                        pp_iris = pp_iris_y[len(pp_iris_y)-1]
                    else:
                        pp_iris = 0
    
                pp_iris_y.append(pp_iris)
                pp_pupil_y.append(pp_pupil)
                pp_pupil_diff_y.append(pp_pupil_diff)
                iou_y.append(IOU)
                if OUTPUT_PP_DATA_TO_JSON:
                    pp_data[count] = {
                            'pp': pp_pupil,
                            'pp_diff': pp_pupil_diff,
                            'shape_conf': IOU
                    }
            
            if SHOW_PP_GRAPH:
                plt.title("Pupil Polsby-Popper Score")
                plt.xlabel("frame")
                plt.ylabel("score")
                plt.plot(pp_x, pp_pupil_y, color='olive', label="Image Score")
                plt.plot(pp_x, pp_pupil_diff_y, color='blue', label="Difference Image Score, Ellipse Score")
                plt.plot(pp_x, iou_y, color='red', label="IOU Pupil Ellipse Fit & Pupil Mask")
                plt.ylim(bottom=0, top=1)
                plt.legend()
                plt.show()
            
            # Add score overlay to image
            if SHOW_PP_OVERLAY:
                font = cv2.FONT_HERSHEY_SIMPLEX
                orgPP = (10, 15)
                orgPPDiff = (10, 35)
                orgIouDiff = (10, 55)
                fontScale = 0.5
                colorWhite = (255, 255, 255)
                colorBlack= (0, 0, 0)
                thickness = 2
                frame = cv2.putText(frame, "PP:           "+"{:.4f}".format(pp_pupil), orgPP, font, fontScale,
                                    colorBlack, thickness*2, cv2.LINE_AA)
                frame = cv2.putText(frame, "PP:           "+"{:.4f}".format(pp_pupil), orgPP, font, fontScale,
                                    colorWhite, thickness, cv2.LINE_AA)
                frame = cv2.putText(frame, "PP Diff:       "+"{:.4f}".format(pp_pupil_diff), orgPPDiff, font, fontScale,
                                    colorBlack, thickness*2, cv2.LINE_AA)
                frame = cv2.putText(frame, "PP Diff:       "+"{:.4f}".format(pp_pupil_diff), orgPPDiff, font, fontScale,
                                    colorWhite, thickness, cv2.LINE_AA)
                frame = cv2.putText(frame, "Shape Conf.:  "+"{:.4f}".format(IOU), orgIouDiff, font, fontScale,
                                    colorBlack, thickness*2, cv2.LINE_AA)
                frame = cv2.putText(frame, "Shape Conf.:  "+"{:.4f}".format(IOU), orgIouDiff, font, fontScale,
                                    colorWhite, thickness, cv2.LINE_AA)
            
            
            inp = process_PIL_image(frame, False, clahe, table).squeeze() * 0.5 + 0.5
            img_orig = np.clip(inp,0,1)
            img_orig = np.array(img_orig)
            if OVERLAP_MASK:
                combine = np.hstack([img_orig, (img_orig + pred_img) / 2, pred_img])
            else:
                combine = np.hstack([img_orig,pred_img])
            #combine = get_stretched_combine(frame.copy(), tuple(frame.shape[1::-1])[0]/6)
            if pad:
                stretchedcombine = comb
                height = len(combine[0])
                r = []
                e = []
                for j in range(len(combine)):
                    e.append(0)
                for i in range(len(stretchedcombine[0]) - len(combine[0])):
                    r.append(e)
                
                combine = np.append(combine, r, axis=1)
                combine = np.vstack([combine, stretchedcombine])
            cv2.imshow('RITnet', combine)
            if SEPARATE_ORIGINAL_VIDEO:
                cv2.imshow('Original', img_orig)
            if SAVE_SEPARATED_PP_FRAMES:
                pp_folder = "{}-{}".format(str(round(int(math.floor(pp_pupil * 10.0)) / 10, 1)), str(round(int(math.floor(pp_pupil * 10.0)) / 10 + .10, 1)))
                pp_diff_folder = "{}-{}".format(str(round(int(math.floor(pp_pupil_diff * 10.0)) / 10, 1)), str(round(int(math.floor(pp_pupil_diff * 10.0)) / 10 + .10, 1)))
                plt.imsave('video/pp-separation/{}/{}.png'.format(pp_folder, str(count)), combine)
                plt.imsave('video/pp-diff-separation/{}/{}.png'.format(pp_diff_folder, str(count)), combine)
                if count in ISOLATE_FRAMES:
                    plt.imsave('video/isolated/{}.png'.format(str(count)), combine)
            pred_img_3=np.zeros((pred_img.shape[0],pred_img.shape[1],3))
            pred_img_3[:,:,0]=pred_img
            pred_img_3[:,:,1]=pred_img
            pred_img_3[:,:,2]=pred_img

            plt.imsave('video/images/{}.png'.format(count),np.uint8(pred_img_3 * 255))
            # maskvideowriter.write((pred_img * 255).astype('uint8'))  # write to mask video output
            videowriter.write((combine * 255).astype('uint8')) # write to video output
            
            # Calculate time remaining using last (up to) 10 frames
            seconds_end = datetime.datetime.now()
            seconds_diff = seconds_end - seconds_start
            if len(seconds_arr) == 25:
                seconds_arr.pop(0)
            seconds_arr.append(seconds_diff.total_seconds())
            avg = 0
            for i in seconds_arr:
                avg += i
            avg = avg / len(seconds_arr)
            seconds_remaining = avg * (max_frames - pos_frame)
            time_remaining = "{}:{}:{}".format("{0:0=2d}".format(int(seconds_remaining/60/60)),"{0:0=2d}".format(int(seconds_remaining/60%60)),"{0:0=2d}".format(int(seconds_remaining%60)))
            print(str(pos_frame)+"/"+str(max_frames)+" frames  (ETA: " + time_remaining+")")
        else:
            # Wait for next frame
            video.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            cv2.waitKey(1000)
        
        if cv2.waitKey(10) == 27:
            video.release()
            # maskvideowriter.release()
            videowriter.release()
            cv2.destroyAllWindows()
            break
        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
            video.release()
            # maskvideowriter.release()
            videowriter.release()
            cv2.destroyAllWindows()
            break
    if OUTPUT_PP_DATA_TO_JSON:
        with open('pp_data.txt', 'w') as outfile:
            json.dump(pp_data, outfile, indent=4)
        print_stats(file_name='pp_data.txt', spacing=1, frame_range=None, ignore_zeros=True)
    
    os.system('cd "'+os.path.dirname(os.path.realpath(__file__))+'" & ffmpeg -r '+str(fps)+' -i ".\\video\\images\\%d.png" -c:v mpeg4 -vcodec libx264 -r '+str(fps)+' ".\\video\\outputs\\mask.mp4"')
    

    # os.rename('test',args.save)


if __name__ == '__main__':
    if THREADED:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    else:
        main()