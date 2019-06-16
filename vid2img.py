"""
Date:3/6/19
Authors: Phat Truong
Simple script extract a video to several images 
"""

import cv2
import os
import argparse
def main(input_vid,output_dir,rate,bias):
    vid2img(input_vid,output_dir,rate,bias)
def vid2img(input_vid,output_dir,rate,bias):     
    cap = cv2.VideoCapture(input_vid)
    frame_count = 0
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
            file_name = 'img%010d.jpg' % (frame_count)
            if frame_count % rate - bias == 0:
                cv2.imwrite(os.path.join(output_dir,file_name),frame)
            frame_count += 1
        else:
            break          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Must give to script')
    parser.add_argument('-v','--video',help = 'Input Video to extract frame')
    parser.add_argument('-d','--dir',help = 'directory of output image')
    parser.add_argument('-r','--rate',help = 'rate take photo in total frames,num_image = total/rate')
    parser.add_argument('-b','--bias',help = 'bias take photo in total frames,for example: bias = 1 will take those frames 1,101,201,...')
    args = parser.parse_args()
    main(args.video,args.dir,int(args.rate),int(args.bias))