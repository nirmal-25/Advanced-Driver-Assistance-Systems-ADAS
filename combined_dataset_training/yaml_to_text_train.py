import os
from anno_conversion import test_annotation

#train_instance = {'boxes': [{'label': 'Red', 'occluded': True, 'x_max': 615.5, 'x_min': 612.0, 'y_max': 360.375, 'y_min': 354.75}, {'label': 'Red', 'occluded': False, 'x_max': 636.625, 'x_min': 632.25, 'y_max': 355.375, 'y_min': 346.375}, {'label': 'Red', 'occluded': False, 'x_max': 653.875, 'x_min': 649.25, 'y_max': 364.875, 'y_min': 353.5}], 'path': './rgb/train/2017-02-03-11-44-56_los_altos_mountain_view_traffic_lights_bag/207390.png'}

image_h = 720
image_w = 1280

train_annotation_folder = '/media/gokul/DOCUMENTS/Amrita University/Final Year Project/Dataset/Traffic Lights/Bosch Small Traffic Lights/Extracted files/dataset_train_rgb'

class_dict = {'off': 0, 'Red': 1, 'Yellow':2, 'Green':3}

#os.chdir(os.getcwd() + '/Bosch_annotation_train_text')

#To be place inside the loop
for train_instance in test_annotation:
	boxes = train_instance['boxes']
	path = train_instance['path']
	
	path_separate = path.split('/')
	image_name = path_separate[-1]
	image_name = image_name[::-1][4::][::-1]

	os.chdir('/media/gokul/PROJECTS & BACKUPS/Final Year Project/Code/My_TLDR/Darknet_PjReddie/Other_files/Bosch_annotation_test_text') 	
	
	f = open(image_name + '.txt', 'w+')
	for single_box in boxes:
	
		object_class = class_dict[single_box['label']]
		x_center = (0.5 * (single_box['x_min'] + single_box['x_max']))/image_w
		y_center = (0.5 * (single_box['y_min'] + single_box['y_max']))/image_h
		width = (single_box['x_max'] - single_box['x_min'])/image_w
		height = (single_box['y_max'] - single_box['y_min'])/image_h
		print(object_class, x_center, y_center, width, height)
		
		
		f.write(str(object_class) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')
	f.close()


