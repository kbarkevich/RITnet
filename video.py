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
from models import model_dict
import matplotlib.pyplot as plt
from image import get_mask_from_PIL_image, process_PIL_image, get_area_perimiters_from_mask, get_polsby_popper_score
import asyncio
import math

ROTATION = -25
PAD = False

def main():
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
    os.makedirs('video/images/',exist_ok=True)
    os.makedirs('video/outputs/',exist_ok=True)
    if PAD and width == 192 and height == 192:
        videowriter = cv2.VideoWriter("video/outputs/out.mp4", fourcc, fps, (int(width*3+192),int(height*2)))
    elif PAD and width == 400 and height == 400:
        videowriter = cv2.VideoWriter("video/outputs/out.mp4", fourcc, fps, (int(width*3+400),int(height*2)))
    else:
        videowriter = cv2.VideoWriter("video/outputs/out.mp4", fourcc, fps, (int(width*3),int(height)))
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
    
    pp_x = []
    pp_iris_y = []
    pp_pupil_y = []
    while True:
        flag, frame = video.read()
        if flag:
            count += 1
            pp_x.append(count)
            # cv2.imshow('video', frame)
            # cv2.imshow('output', output[0][0].cpu().detach().numpy()/3.0)
            # cv2.imshow('mask', predict[0].cpu().numpy()/3.0)
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
            M = cv2.getRotationMatrix2D(center, -25, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
                
            pred_img = get_mask_from_PIL_image(frame, model, True, False)
            iris_perimeter, pupil_perimeter, iris_area, pupil_area = get_area_perimiters_from_mask(pred_img)
            
            pp_iris = get_polsby_popper_score(iris_perimeter, iris_area)
            pp_pupil = get_polsby_popper_score(pupil_perimeter, pupil_area)
            if math.isnan(pp_iris) or pp_iris >= 1 or pp_iris <= 0:
                if len(pp_iris_y) > 0:
                    pp_iris = pp_iris_y[len(pp_iris_y)-1]
                else:
                    pp_iris = 0
            if math.isnan(pp_pupil) or pp_pupil >= 1 or pp_pupil <= 0:
                pp_pupil = 0
            pp_iris_y.append(pp_iris)
            pp_pupil_y.append(pp_pupil)
            
            plt.title("Pupil Polsby-Popper Score")
            plt.xlabel("frame")
            plt.ylabel("score (higher is better)")
            plt.plot(pp_x, pp_pupil_y)
            plt.ylim(bottom=0, top=1)
            plt.show()
            
            inp = process_PIL_image(frame, False, clahe, table).squeeze() * 0.5 + 0.5
            img_orig = np.clip(inp,0,1)
            img_orig = np.array(img_orig)
            combine = np.hstack([img_orig,pred_img,pred_img])
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
            pred_img_3=np.zeros((pred_img.shape[0],pred_img.shape[1],3))
            pred_img_3[:,:,0]=pred_img
            pred_img_3[:,:,1]=pred_img
            pred_img_3[:,:,2]=pred_img
            plt.imsave('video/images/{}.png'.format(count),np.uint8(pred_img_3 * 255))
            # maskvideowriter.write((pred_img * 255).astype('uint8'))  # write to mask video output
            videowriter.write((combine * 255).astype('uint8')) # write to video output
            print(str(pos_frame)+" frames")
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
    os.system('cd "'+os.path.dirname(os.path.realpath(__file__))+'" & ffmpeg -r '+str(fps)+' -i ".\\video\\images\\%d.png" -c:v mpeg4 -vcodec libx264 -r '+str(fps)+' ".\\video\\outputs\\mask.mp4"')
            

    # os.rename('test',args.save)

THREADED = False

if __name__ == '__main__':
    if THREADED:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    else:
        main()