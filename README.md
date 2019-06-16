# Counting people go into a shop

## *Brief summary*
*Last update: 16/6/2019 with Tensorflow 1.13*

This is my code for EyeQ Tech company's test. Please feel free to check it and give me comments
## Step to predict video
###   1.Input video:
  Put your input video at demo/input (Required).
###   2.Pre-trained CNN model:
  I used Tensorflow Object Detection API, chose pre-trained model "faster-rcnn-resnet101" and trained it for my task.
  Here are link of my trained model: https://drive.google.com/open?id=1XgAz09LRWgLkT_zYjiNrB_aQq4A4sFUZ.
  Just download its content and run code

  Your folder tree will like that:

    counting-people-shop

    └─ output_inference_graph_step31713

        ├─ saved_model

        ├─ pipeline.config

        ├─ model.ckpt.meta

        ├─ model.ckpt.index

        ├─ model.ckpt.data-00000-of-00001

        ├─ frozen_inferrence_graph.pb

        └─ checkpoint

###   3.Run code to predict video:
  Change direction to counting-people-shop folder by following command:
  
    cd counting-people-shop
  
  Run script detect.py by following command:
  
    python detect.py -i [Input-vid-name] -o [Output-vid-name] -m [Model-name] -l [Label-map]
  
  Example: 
  
    python detect.py -i input.mp4 -o output.mp4 -m output_inference_graph_step31713 -l label_map.pbtxt
  
