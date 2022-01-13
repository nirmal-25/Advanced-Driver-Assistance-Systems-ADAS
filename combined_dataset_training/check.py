import os
from glob import glob
path1 = 'test-detections-bosch/combined-1024-6L/'
test1 = glob(os.path.join(path1, '*.txt'))

list1 = []
for image in test1:
	list1.append(image[39:-4])
	
path2 = 'groundtruths/bosch/'
test2 = glob(os.path.join(path2, '*.txt'))

list2 = []
for image in test2:
	#print(image[20:-4])
	list2.append(image[19:-4])
	
#list1.sort()
#list2.sort()

#print(list1)

for i in list2:
	if i not in list1:
		print('{} is missing'.format(i))
	#print(i)

