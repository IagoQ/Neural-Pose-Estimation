import numpy as np
from datautils import sumHeats,parseHeat,getSkeleton
from model import *
from keras.models import load_model
import cv2
from skimage.transform import resize
from math import floor,ceil
import time


#TODO overlay ironman suit when body detection is more stable
#TODO refactor everything eww


camera = cv2.VideoCapture(0)

# model = depthposev1()
# model = resposev4()
model = load_model('./models/resposev3b.h5')

def prepare(img,target=224):
    height, width, channels = img.shape
    aspect = floor(target/width * height)
    scale = target/width
    img = resize(img,(aspect,target),preserve_range=True)
    height, width, channels = img.shape
    padding_t = ceil((target - height)/2)
    padding_b = floor((target - height)/2)
    img = np.pad(img, ((padding_t, padding_b),(0, 0),(0,0)), 'constant').astype('float32') / 255
    return img

while True:
    start = time.time()
    _, img = camera.read()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = prepare(img)

    heat = model.predict(np.expand_dims(img,axis=0))
    heat = np.squeeze(heat)


    flat = sumHeats(heat)
    coords = parseHeat(heat)
    skeli = getSkeleton(coords)
    for (x,y) in coords:
        cv2.circle(img,(x,y), 2, (0,255,0), -1)


    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    img = cv2.resize(img,(448,448))
    skeli = cv2.resize(skeli,(448,448))
    flat = cv2.resize(flat,(448,448))


    cv2.imshow('img',img)
    cv2.imshow('ske',skeli)
    cv2.imshow('flat',flat)
    end = time.time()
    fps = 1/(end-start)
    
    print("fps:{0:.2f}".format(fps),end='\r')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
