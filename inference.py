import numpy as np
import cv2
from datautils import loadDatapoint,sumHeats,parseHeat
from random import randint
from keras.models import load_model
from pprint import pprint
from model import *


dataDir = 'data'
dataType = 'processed'
imagesDir = '{}/{}/'.format(dataDir,dataType)

names = np.load('./data/testNames.npy')
keypoints = np.load('./data/testLabels.npy')


paths = [imagesDir + name for name in names]


model = load_model('./models/mobtest.h5')
# model = mobiletest()

ind = randint(0,1000)


img, heat = loadDatapoint(paths[ind],keypoints[ind])

img = img.astype('float32') / 255

result = model.predict(np.expand_dims(img, axis=0))
result = np.squeeze(result)


flat = sumHeats(result)
heatimg = sumHeats(heat)
norm = 0.95 / np.amax(flat)
normheat = flat*norm
normheat[normheat < 0] = 0
print(np.amax(normheat))
print(np.amin(normheat))


img = cv2.resize(img,(448,448))
heatimg = cv2.resize(heatimg,(448,448))
flat = cv2.resize(flat,(448,448))
normheat = cv2.resize(normheat,(448,448))


cv2.imshow('img',img)
cv2.imshow('y',heatimg)
cv2.imshow('yb',flat)
cv2.imshow('norm',normheat)
cv2.waitKey(0)


