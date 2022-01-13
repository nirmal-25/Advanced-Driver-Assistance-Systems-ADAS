import numpy as np
import os
import sys
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from glob import glob

from utils import label_map_util
from utils import visualization_utils as vis_util

CKPT = 'IG/combined_1024_6L/frozen_inference_graph.pb'
PATH_TO_LABELS = 'data/label.pbtxt'
NUM_CLASSES = 49

detection_graph = tf.Graph()

with detection_graph.as_default():
    
  od_graph_def = tf.GraphDef()

  with tf.gfile.GFile(CKPT, 'rb') as fid:
        
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(category_index)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
      

PATH_TO_TEST_IMAGES_DIR = '/media/nirmal/DATA2/Final-Year-Project/Dataset/Traffic Lights/Traffic Light-alter/dataset-alter/dataset_test_rgb/bosch_jpg_test/rgb/test/'

#print(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.png'))
#TEST_IMAGE_PATHS = glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.png'))  # PNG OR JPG
TEST_IMAGE_PATHS = glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))
print("Length of test images:", len(TEST_IMAGE_PATHS))

print(TEST_IMAGE_PATHS[45][137:-4])
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

import time
import json
counter = 0
with detection_graph.as_default():
    time0 = time.time()
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            counter = counter + 1
            print('Image {}'.format(counter))
            filename_string = image_path[137:-4]
            image = Image.open(image_path)
            # print(image.size)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            #time0 = time.time()

            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

            

            # Visualization of the results of a detection.
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #  image_np,
            #  np.squeeze(boxes),
            #  np.squeeze(classes).astype(np.int32),
            #  np.squeeze(scores),
            #  category_index,
            #  use_normalized_coordinates=True,
            #  line_thickness=8)
            #plt.figure(figsize=IMAGE_SIZE)
            #plt.imshow(image_np)
            #plt.show()
            coordinates = vis_util.return_coordinates(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.50)
            #textfile = open("json/"+filename_string+".json", "a")
            #textfile.write(json.dumps(coordinates))
            #textfile.write("\n")
            #print(coordinates)
            if coordinates == []:
                with open('test-detections-bosch/combined-1024-6L/' + filename_string + '.txt','a') as text_file:
                    text_file.close()
                
            for coordinate in coordinates:
                #print(coordinate)
                (xmax, xmin, ymax, ymin, accuracy, classification_num) = coordinate
                label = classification_num
                detection_box = (label, accuracy, xmin, ymin, xmax, ymax)
                
                if label == 46:
                    light = 'Red'
                    
                if label == 47:
                    light = 'Yellow'
                    
                if label == 48:
                    light = 'Green'
                    
                if label == 49:
                    light = 'off'
                    
                #print('{} {} {} {} {} {}'.format(label, accuracy, xmin, ymin, xmax, ymax))
                
                if label == 46 or label == 47 or label == 48 or label == 49: 
                    with open('test-detections-bosch/combined-1024-6L/' + filename_string + '.txt','a') as text_file:
                        text_file.write(light + ' ' + str(accuracy) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')
                        text_file.close()    
                
                
                
                #print('{} {} {} {} {} {}'.format(label, accuracy, xmin, ymin, xmax, ymax))
                print(detection_box)
                #if str(label) != '46' and str(label) != '47' and str(label) != '48' and str(label) != '49':
                    
                        
            time1 = time.time()

            print("Time in milliseconds", (time1 - time0) * 1000) 
                
             

