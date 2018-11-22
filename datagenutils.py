import cv2
import numpy as np
import random
from math import sqrt,floor,ceil
import cv2

#functions for loading and generating data for training/testing


def getcoords(ann):
    ncoords = []
    for person in ann:
        coords = person['keypoints']
        tempcoords = []
        for x in range(17):
            x = x*3
            tempcoords.append((coords[x],coords[x+1]))
        ncoords.append(tempcoords)
    return ncoords

def getName(imgid):
    numchar = 12
    sid = str(imgid)
    while(len(sid) is not numchar):
        sid = '0' + sid   
    name = sid + '.jpg'
    return name

def getImgIds(anns):#HACK very feo
    imgd = {}
    imgs = []
    for ann in anns:
        imgd[ann['image_id']] = 0
        if ann['num_keypoints'] is 0:
            imgd[ann['image_id']] = 2
    for ann in anns:
        imgd[ann['image_id']] += 1
    for k in imgd:
        if imgd[k] < 16:
            imgs.append(k)
    return imgs

def checkCoords(coords,numkeys):
    for p in coords:
        current = 0
        for (x,y) in p:
            if (x != 0 or y != 0) and (x != 223 or y != 223):
                current += 1
        if current >= numkeys:
            return True
    return False

def normcoords(coords,scale, xoffset = 0, yoffset = 0,img_size = 224):
    ncoords = []
    for person in coords:
        tempcoords = []
        for x,y in person:
            nx, ny = int(x*scale), int(y*scale)
            nx, ny = nx + xoffset, ny + yoffset
            if nx <= 0 or ny <= 0:
                nx, ny = 0,0
            if nx == img_size:
                nx -= 1
            if ny == img_size:
                ny -= 1
            tempcoords.append((nx,ny))
        ncoords.append(tempcoords)
    return ncoords

def getSample(path,coords,target=224):
    #returns RGB image resized and cropped to 224x224
    #TODO make number vars
    img = cv2.imread(path)
    
    height, width, channels = img.shape

    if height > width:
        aspect = floor(target/height * width)
        scale = target/height
        img = cv2.resize(img,(aspect,target))
        height, width, channels = img.shape

        padding_l = ceil((target - width)/2)
        padding_r = floor((target - width)/2)
        img = np.pad(img, ((0, 0), (padding_l, padding_r),(0,0)), 'constant').astype('uint8')
        coords = normcoords(coords,scale,padding_l,0)
    else:
        aspect = floor(target/width * height)
        scale = target/width
        img = cv2.resize(img,(target,aspect))
        height, width, channels = img.shape

        padding_t = ceil((target - height)/2)
        padding_b = floor((target - height)/2)

        img = np.pad(img, ((padding_t, padding_b),(0, 0),(0,0)), 'constant').astype('uint8')
        coords = normcoords(coords,scale,0,padding_t)

    return img,coords

def increaseBrightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def decreaseBrightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 0 + value
    v[v < lim] = 0
    v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def translateImage(img,x,y,imgsize = 224):
    translation_matrix = np.float32([ [1,0,x], [0,1,y] ])
    img_translation = cv2.warpAffine(img, translation_matrix, (imgsize, imgsize))
    return img_translation

def translateLabel(label,dx,dy,imgsize = 224):
    translatedPoints = []
    for person in label:
        translatedPerson = []
        for (x,y) in person:
            if x == 0 or y == 0:
                translatedPerson.append((0,0))
                continue

            x += dx
            y += dy
            
            if x < 0 or y < 0 or x >= imgsize or y >= imgsize:
                x,y = 0,0

            translatedPerson.append((x,y))
        translatedPoints.append(translatedPerson)
    return translatedPoints

def resizeSample(img,label,scale):
    #resize image
    side = img.shape[0]
    nside = floor(side * scale)
    nimg = cv2.resize(img,(nside,nside))


    base = np.zeros([side,side,3])

    center = int(side / 2)
    offset = int(nside / 2)
    origin = center - offset


    base[origin:origin+nside,origin:origin+nside] = nimg / 255
        
    #adjust points
    pointOffset = center - offset 

    normPoints = []
    for person in label:
        normPerson = []
        for (x,y) in person:
            if x == 0 or y == 0:
                normPerson.append((0,0))
                continue
                
            x = floor(x*scale + pointOffset) 
            y = floor(y*scale + pointOffset) 

            normPerson.append((x,y))
        normPoints.append(normPerson)
        
    return base, normPoints

def blurImage(img,direction):
    size = 7
    if direction == 0:
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
    if direction == 1:
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[:,int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
    if direction == 2:
        size = 5
        kernel_motion_blur = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i == j:
                    kernel_motion_blur[i][j] = 1
        kernel_motion_blur = kernel_motion_blur / size
    if direction == 3:
        size = 5
        kernel_motion_blur = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i + j == size-1:
                    kernel_motion_blur[i][j] = 1
        kernel_motion_blur = kernel_motion_blur / size

    output = cv2.filter2D(img, -1, kernel_motion_blur)
    return output

def flipSample(img,coords,imgsize=224):
    img = cv2.flip( img, 1 )



    flipped = []
    for person in coords:
        pflip = []
        for (x,y) in person:
            if x == 0 or y == 0:
                pflip.append((0,0))
                continue

            nx = imgsize - x

            if nx < 0 or nx >= imgsize:
                pflip.append((0,0))
                continue

            pflip.append((nx,y))
        flipped.append(pflip)
    
    return img,flipped

def augmentSample(img,label,maxTranslate = 45,maxBrightness = 70,minresize = 85):

    #brightness -90 - 90 range
    #translate -65 - 65 range

    deltaBrightness = random.randint(5,maxBrightness)
    blurOrNot = random.randint(0,3)

    if blurOrNot == 0:
        blurdir = random.randint(0,3)
        img = blurImage(img,blurdir)

    #at inference time is much more likely for the image to be dark than bright
    darkOrlight = random.randint(0,3)
    if darkOrlight:
        img = decreaseBrightness(img,deltaBrightness)
    else:
        img = increaseBrightness(img,deltaBrightness)
    
    #flip image 
    flipOrNot = random.randint(0,1)

    if(flipOrNot):
        img,label = flipSample(img,label)


    resizeFactor = random.randint(minresize,100) / 100

    img, label = resizeSample(img,label,resizeFactor)


    deltaX = random.randint(-maxTranslate,maxTranslate)
    deltaY = random.randint(-maxTranslate,maxTranslate)

    img = translateImage(img,deltaX,deltaY)
    label = translateLabel(label,deltaX,deltaY)

    img = (img * 255).astype('uint8')

    return img,label

def distance(x1,y1,x2,y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def genSpot(size):
    spot = np.zeros(shape=(size,size))
    center = size//2
    for x in range(size):
        for y in range(size):
            dist = distance(center,center,x,y)
            if dist > center:
                continue
            if x != center or y != center:
                spot[y][x] = 1 / dist
            else:
                spot[y][x] = 1
    return spot

def generateHeat(labels,imgsize = 224,spotsize = 8):
    heats = np.zeros(shape=(imgsize,imgsize,17))
    spot = genSpot(spotsize)
    for person in labels:
        for part, (x, y) in enumerate(person):
            if x == 0 and y == 0:
                continue

            delta = spotsize//2
            xmin = max(x-delta, 0)
            ymin = max(y-delta, 0)
            xmax = min(x+delta,imgsize-1)
            ymax = min(y+delta,imgsize-1)
            xsize = xmax - xmin
            ysize = ymax - ymin
            
            heats[ymin:ymax,xmin:xmax,part] = spot[:ysize,:xsize]

    return heats

def validCoords(pair,imgsize=224):
    if pair[0] == 0 or pair[1] == 0 or pair[0] == 223 or pair[1] == 223:
        return False
    return True

def generatePafs(labels,imgsize=224,width=3):
    vectors = np.zeros(shape=(imgsize,imgsize,19*2))
    counts = np.zeros(shape=(imgsize,imgsize,19))

    connections = [(15, 13),
               (13, 11),
               (16, 14),
               (14, 12),
               (11, 12),
               (5, 11),
               (6, 12),
               (5, 6),
               (5, 7),
               (6, 8),
               (7, 9),
               (8, 10),
               (1, 2),
               (0, 1),
               (0, 2),
               (1, 3),
               (2, 4),
               (3, 5),
               (4, 6)]

    for person in labels:
        for planeid,link in enumerate(connections):
            centerfrom = person[link[0]]
            centerto = person[link[1]]

            if (not validCoords(centerfrom)) or (not validCoords(centerto)):
                continue

            vecx = centerfrom[0] - centerto[0]
            vecy = centerfrom[1] - centerto[1]
        
            min_x = max(0, int(min(centerfrom[0], centerto[0]) - width))
            min_y = max(0, int(min(centerfrom[1], centerto[1]) - width))

            max_x = min(imgsize, int(max(centerfrom[0], centerto[0]) + width))
            max_y = min(imgsize, int(max(centerfrom[1], centerto[1]) + width))

            norm = sqrt(vecx ** 2 + vecy ** 2)
            if norm < 1e-8:
                continue

            vecx /= norm
            vecy /= norm

            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    bec_x = x - centerfrom[0]
                    bec_y = y - centerfrom[1]
                    
                    dist = abs(bec_x * vecy - bec_y * vecx)

                    if dist > width:
                        continue
                    
                    counts[y][x][planeid] += 1

                    vectors[y][x][planeid*2] = vecx
                    vectors[y][x][planeid*2+1] = vecy
    
    nonzeros = np.nonzero(counts)
    for y, x, p in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
        if counts[y][x][p] <= 0:
            continue
        vectors[y][x][p*2+0] /= counts[y][x][p]
        vectors[y][x][p*2+1] /= counts[y][x][p]
    
    return vectors.astype(np.float16)


