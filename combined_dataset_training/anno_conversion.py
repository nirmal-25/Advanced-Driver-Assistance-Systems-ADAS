import yaml
import cv2
import numpy as np
import copy

#train_image_folder = "/media/gokul/DOCUMENTS/Amrita University/Final Year Project/Dataset/Traffic Lights/Bosch Small Traffic Lights/Extracted files/dataset_train_rgb/"
#train_image_folder = "/media/gokul/PROJECTS & BACKUPS/Final Year Project/Code/My_TLDR/Darknet/Extracted files/dataset_train_rgb/"
test_image_folder  = "/media/gokul/PROJECTS & BACKUPS/Final Year Project/Dataset/Traffic Lights/Bosch Small Traffic Lights/Extracted files/dataset_test_rgb/"
Labels = ['Red', 'Yellow', 'Green', 'off']


#with open(train_image_folder + "train.yaml", 'r') as stream:
#    try:
#        train_annotation = yaml.safe_load(stream)
#    except yaml.YAMLError as exc:
#        print(exc)

with open(test_image_folder +"test.yaml", 'r') as stream:
    try:
        test_annotation = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#for image in range(len(train_annotation)):
#	image_path = train_annotation[image]['path']
#	print(image)
#	image_open = cv2.imread(train_image_folder+image_path)
#	if type(image_open) == type(None):
#		train_annotation[image] = 0

#print(len(train_annotation))		
#for i in range(train_annotation.count(0)):
#	train_annotation.remove(0)

#TRAINING DATA
#print(len(train_annotation))

#def label_alter(labels, annotations):
#	true_label = annotations['label']
#	for label_name in labels:
#		if label_name in true_label:
#			assign_label = label_name
#			annotations['label'] = assign_label
#			break

#for train_instance in train_annotation:
#	boxes = train_instance['boxes']
#	for single_box in boxes:
#		label_alter(Labels, single_box)
#
#for index in range(len(train_annotation)):
#	if train_annotation[index]['boxes'] == []:
#		train_annotation[index] = 0
#for i in range(train_annotation.count(0)):
#	train_annotation.remove(0)

#print(len(train_annotation))
#print(train_annotation[3])


#TESTING DATA
print(len(test_annotation))

def label_alter(labels, annotations):
	true_label = annotations['label']
	for label_name in labels:
		if label_name in true_label:
			assign_label = label_name
			annotations['label'] = assign_label
			break

for test_instance in test_annotation:
	boxes = test_instance['boxes']
	for single_box in boxes:
		label_alter(Labels, single_box)

#for index in range(len(test_annotation)):
#	if test_annotation[index]['boxes'] == []:
#		test_annotation[index] = 0
#for i in range(test_annotation.count(0)):
#	test_annotation.remove(0)

print(len(test_annotation))


