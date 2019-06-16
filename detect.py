"""
TEST OF EYEQ TECH
Author: Truong Loc Phat
Date: 5/6/2019
"""
import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import argparse
sys.path.append("models/build/lib/object_detection")
sys.path.append("models/build/lib/")

from utils import label_map_util
from my_utils import iou,inside_door,visualize

def main(input_vid_name,output_vid_name,model,label_map):
    # Define the video stream
    cap = cv2.VideoCapture(os.path.join("demo/input",input_vid_name))
    output = os.path.join('demo/output',output_vid_name)  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output, fourcc, 20.0, (1920, 1080))
    #cap = cv2.VideoCapture("train_video/trainvideo.mp4")
    # folder of model.
    MODEL_NAME = model

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(label_map)

    # Number of classes to detect
    NUM_CLASSES = 1

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # Detection
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            frame = 0
            person_in = 0
            next_id = 0
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            boxes = tf.reshape(boxes,(300,4))
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            scores = tf.reshape(scores,(300,))
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            classes = tf.reshape(classes,(300,))
            # Extract number of detections
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            #Non_max_suppresion to clean many box for 1 object
            mask = tf.image.non_max_suppression(boxes,scores,10,iou_threshold = 0.3,score_threshold = 0.9)
            boxes = tf.gather(boxes,mask)
            scores = tf.gather(scores,mask)
            classes = tf.gather(classes,mask)
            while(cap.isOpened()):
                # Read frame from camera
                ret, image_np = cap.read()
                if ret == True:
                    if frame % 2 == 0: #skip frame to reduce noise, fail-boxs
                        img_shape = image_np.shape
                        #Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        # Run session to take suitable variables
                        (true_boxes, true_scores, true_classes) = sess.run(
                            [boxes, scores, classes],
                            feed_dict={image_tensor: image_np_expanded})
                        true_boxes = true_boxes*np.array([img_shape[0],img_shape[1],img_shape[0],img_shape[1]])
                        true_boxes = true_boxes.astype(int)
                        num_box = true_boxes.shape[0]
                        #Tracking using IOU between box in previous frame, and counting
                        frame_box_id = np.zeros((num_box),dtype = int)
                        track_stt = np.zeros((num_box),dtype = int)
                        in_out = np.zeros((num_box),dtype = int)
                        if frame == 0:#first frame
                            for i,box in enumerate(true_boxes):
                                frame_box_id[i] = next_id
                                if inside_door(box) is True:
                                    in_out[i] = 1
                                next_id += 1

                        else:#other frames
                            for i,box in enumerate(true_boxes):#box in current frame
                                if len(prev_boxes) != 0:
                                    for k,prev_box in enumerate(prev_boxes):#box in previous frame
                                        if iou(box,prev_box) > 0.3:#threshold = 0.3 for tracking
                                            frame_box_id[i] = prev_box_ID[k]
                                            if prev_in_out[k] == 1:
                                                if inside_door(box) is False:
                                                    person_in += 1
                                                else:
                                                    in_out[i] = 1
                                            track_stt[i] = 1
                                    if track_stt[i] == 0:    
                                        frame_box_id[i] = next_id
                                        if inside_door(box) is True:
                                            in_out[i] = 1
                                        next_id += 1
                                else:
                                    frame_box_id[i] = next_id
                                    if inside_door(box) is True:
                                        in_out[i] = 1
                                    next_id += 1
                        prev_in_out = in_out
                        prev_boxes = true_boxes
                        prev_box_ID = frame_box_id            

                    # Visualization of the results of a detection.
                    visualize(image_np,true_boxes,true_classes,true_scores,frame_box_id,person_in,frame)
                    out.write(image_np)
        
                #if run on Google Colab, command the below line to avoid corrupted
                #or if run on local laptop, this line will show the result every frame in video
                    #cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
                    
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    frame += 1
                else:
                    break   
            cap.release()
            out.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Must give to script')
    parser.add_argument('-i','--input',help = "Input Video, please make sure that this video's in demo/input/... ")
    parser.add_argument('-o','--output',help = 'name of output video, the result will be placed at demo/output/...')
    parser.add_argument('-m','--model',help = 'name of AI model')
    parser.add_argument('-l','--label',help = 'label map')
    args = parser.parse_args()
    main(args.input,args.output,args.model,args.label)     