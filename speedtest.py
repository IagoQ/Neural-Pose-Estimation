import numpy as np
from random import randint
from datautils import sumHeats,loadDatapoint,sumPafs
from datagenutils import blurImage,increaseBrightness,generateHeat,generatePafs
from pprint import pprint
import cv2
from model import *
import time
from keras.models import load_model
dataDir = 'data'
dataType = 'processed'
imagesDir = '{}/{}/'.format(dataDir,dataType)

names = np.load('./data/testNames.npy')
keypoints = np.load('./data/testLabels.npy')
paths = [imagesDir + name for name in names]


img,_ = loadDatapoint(paths[0],keypoints[0])
# model = load_model('./models/depthtestv1.h5')
model = mobtest()
print('start')
start = time.time()
for i in range(200):
    model.predict(np.expand_dims(img,axis=0))

end = time.time() - 1


print('')
print('total time:  ' + str(end-start))
print('single inference time:  ' + str((end-start)/200))
print('fps aprox: ' + str(1/((end-start)/200)))