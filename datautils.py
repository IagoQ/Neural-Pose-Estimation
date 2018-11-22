import numpy as np
from math import sqrt,floor,ceil
from PIL import Image
from skimage.transform import resize
from random import randint
import cv2

def distance(x1,y1,x2,y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def generateHeat(coords,imgsize = 224,spotsize = 8):
    heats = np.zeros(shape=(imgsize,imgsize,17))
    for person in coords:    
        for part, (x, y) in enumerate(person):
            if x == 0 and y == 0:
                continue
            delta = spotsize//2
            xmin = max(x-delta, 0)
            ymin = max(y-delta, 0)
            xmax = min(x+delta,imgsize-1)
            ymax = min(y+delta,imgsize-1)
            heats[y][x][part] = 4
            for xx in range(xmin,xmax):
                for yy in range(ymin,ymax):
                    dist = distance(x,y,xx,yy)
                    if dist>delta:
                        continue
                    if (xx != x or yy != y):
                        value = 1 / dist
                        heats[yy][xx][part] = value
                    else:
                        heats[yy][xx][part] = 1 

    return heats

def loadDatapoint(path,keypoints, target = 224):
    #TODO delete this and replace references
    imgopen = Image.open(path)
    img = np.array(imgopen)
    imgopen.close()

    keypoints = np.array(keypoints)
    heat = generateHeat(keypoints)
    return img, heat

def parseHeat(heat):
    #TODO detect multiple people
    coords = np.zeros(shape=(17,2))
    for i in range(heat.shape[2]):
        maximum = np.amax(heat[...,i])
        if maximum > 0.160:
            x,y = np.where(heat[...,i] >= maximum)
            coords[i][0] = y[0]
            coords[i][1] = x[0]
    return coords.astype(int)

def sumHeats(heat,size = 224):
    result = np.zeros(shape=(224,224))
    for i in range(17):
        temp = heat[...,i]
        temp = np.squeeze(temp)
        result += temp
    return result


def sumPafs(pafs):
    base = np.zeros(shape=(224,224,3))
    for i in range(19):
        base[:,:,0] += pafs[:,:,i*2]
        base[:,:,2] += pafs[:,:,i*2+1]
    return base


def getSkeleton(coords,target=224):
    img = np.zeros(shape=(target,target,3))
    conections = [[0,1],[0,2],[1,3],[2,4],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,13],[13,15],[12,14],[14,16],[5,6],[11,12]]
    for con in conections:
        x1,y1 = coords[con[0]]
        x2,y2 = coords[con[1]]
        if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
            continue
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)

    for (x,y) in coords:
        cv2.circle(img,(x,y), 2, (0,255,0), -1)
    return img
