"""
TEST OF EYEQ TECH
Author: Truong Loc Phat
Date: 5/6/2019
"""
import numpy as np
import cv2

def iou(box1,box2):
    """
    iou: intersec/union
    x1,y1: left-top corner of intersection box
    x2,y2: right-bot corner of intersection box
    """
    x1 = max(box1[1],box2[1])
    y1 = max(box1[0],box2[0])
    x2 = min(box1[3],box2[3])
    y2 = min(box1[2],box2[2])
    #area of intersection box
    w,h = (x2-x1),(y2-y1)
    if w < 0 or h < 0:
        return 0
    inter_area = w*h
    #area of both 2 box
    area1 = (box1[3]-box1[1])*(box1[2]-box1[0])
    area2 = (box2[3]-box2[1])*(box2[2]-box2[0])
    #IOU
    iou_value = inter_area/(area1+area2-inter_area+1e-5)
    return iou_value

def inside_door(box,threshold1 = 0.9,threshold2 = 0.8):
    """
    Find if box is inside the door area
    rate : percentage of box's part inside door area
    """
    door_box = [400,550,1080,1350]
    #overlaps points
    x1 = max(box[1],door_box[1])
    y1 = max(box[0],door_box[0])
    x2 = min(box[3],door_box[3])
    y2 = min(box[2],door_box[2])
    #area of intersection box
    w,h = (x2-x1),(y2-y1)
    if w < 0 or h < 0:
        return 0
    inter_area = w*h
    area_box = (box[3]-box[1])*(box[2]-box[0])
    rate = inter_area/area_box
    if rate > threshold1:
        return True 
    elif rate < threshold2:
        return False
    else:
        return None
def visualize(image,boxes,classes,scores,ID_tracking,person_in,frame):#boxes : (y1,x1,y2,x2)
    """
    Show information on image and terminal
    
    """
    print('frame',frame)
    for i,box in enumerate(boxes):
        print("box",i)
        print("ID:",ID_tracking[i])
        p1 , p2 = (box[1],box[0]),(box[3],box[2])
        print(p1)
        print(p2)
        cv2.rectangle(image,p1,p2,(0,244,249),6)
        cv2.rectangle(image,p1,(int(p1[0]+0.8*(p2[0]-p1[0])),int(p1[1]+0.1*(p2[1]-p1[1]))),(0,244,249),-1)
        cv2.putText(image,'person: %3d' % (scores[i]*100),(p1[0],int(p1[1]+0.1*(p2[1]-p1[1]))),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),thickness = 3)
    cv2.rectangle(image,(550,400),(1350,1080),(100,0,249),6)
    cv2.putText(image,'number person go in: '+str(person_in),(550,490),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,244,249),thickness = 3)
    print('person in',person_in)