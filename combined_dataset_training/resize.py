from PIL import Image
from glob import glob
import os

paths = glob(os.path.join('test/', '*.jpg'))
print(len(paths))
counter = 1

for pic in paths:
	image = Image.open(pic)
	new_image = image.resize((1024, 1024))	
	new_image.save('test/resized/' + pic[5:-4] + '.jpg')
	print(counter)
	counter = counter + 1
	
#print(len(paths))


#print (paths[56][6:-4])

