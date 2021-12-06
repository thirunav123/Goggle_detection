import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2 
import numpy as np
from matplotlib import pyplot as plt
import threading
import time

CUSTOM_MODEL_NAME = 'my_ssd_mobnet-cropped_more_img2' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations-cropped_more_img2'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    #print(shapes)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
    
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
cap = cv2.VideoCapture(r'D:\videos\1.mp4')
x=200
y=1100
w=640
h=640
x1,x2,y1,y2,Info=0,0,0,0,""



#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
P_count=0
count=0
def thread_task(img):
    global x1,x2,y1,y2,Info,P_count
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    coordinates = viz_utils.return_coordinates(
                        image_np_with_detections,
                        np.squeeze(detections['detection_boxes']),
                        np.squeeze(detections['detection_classes']+label_id_offset),
                        np.squeeze(detections['detection_scores']),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.80)
    #print(coordinates)
    if not coordinates==[]:
        y1=int(coordinates[0][0])
        y2=int(coordinates[0][1])
        x1=int(coordinates[0][2])
        x2=int(coordinates[0][3])
        Info=coordinates[0][5]
    else:
        x1,x2,y1,y2,Info=0,0,0,0,""
        pass
    P_count=P_count+1
    #x1=x1+1
    #time.sleep(3)

t1 = threading.Thread(target=thread_task,args=(1,))
#t1.start()
while cap.isOpened(): 
    ret, or_read_img = cap.read()
    cv2.rectangle(or_read_img, (y,x), (y+h,x+w), (255,102,255), 5)

    frame = or_read_img[x:x+w, y:y+h]
    if not t1.is_alive():
        t1 = threading.Thread(target=thread_task,args=(frame,))
        #print('Not alive, started')
        t1.start()
        if Info=="Goggle":
            frame=cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 5)
        elif Info=="NoGoggle":
            frame=cv2.rectangle(frame, (x1,y1), (x2, y2), (0,0,255), 5)
        else:
            pass
    else:
        if Info=="Goggle":
            frame=cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 5)
        elif Info=="NoGoggle":
            frame=cv2.rectangle(frame, (x1,y1), (x2, y2), (0,0,255), 5)
        else:
            pass
        #cv2.rectangle(or_read_img, (x,y), (x+w, y+h), (0,255,0), 3)
    cv2.imshow('object detection', cv2.resize(or_read_img,(800,600)) )
    #except:
        #pass
    count=count+1
    print(count,P_count)
    #image_np = np.array(frame)
    #print("NEW:",image_np)q
    if cv2.waitKey(30) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    