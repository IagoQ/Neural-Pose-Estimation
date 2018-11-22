from pycocotools.coco import COCO
from pprint import pprint
import numpy as np
from math import floor,ceil
from datagenutils import *
import cv2



numAugmentations = 2


dataDir = 'data'
dataType = 'train2017'
imagesDir = '{}/{}/'.format(dataDir,dataType)
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
saveDir = 'data/processed/'

#Get all annotations
coco = COCO(annFile)
personid = coco.getCatIds(catNms=['person'])
imgids = coco.getImgIds(catIds=personid)
annids = coco.getAnnIds(catIds=personid)
anns = coco.loadAnns(ids=annids)
cats = coco.loadCats(coco.getCatIds())

target = 224

imgids = getImgIds(anns)
annids = coco.getAnnIds(imgIds=imgids)

print(len(imgids))


#get single person annotations annotations
annlist = coco.loadAnns(ids=annids)
anns = {}
for ann in annlist:
    anns[ann['image_id']]  = []
for ann in annlist:
    anns[ann['image_id']].append(ann)

names = []
labels = []

for imid in imgids:
    name = getName(imid)
    coords = getcoords(anns[imid])
    names.append(name)
    labels.append(coords)

dataset = []





for i in range(len(names)):
    name = names[i]
    coords = labels[i]

    path = imagesDir + name
    
    img,coords = getSample(path,coords)


    if not checkCoords(coords,5):
        continue

    for j in range(numAugmentations):
        aug,augCoord = augmentSample(img,coords)

        augname = name[:12] + '_' + str((j+1)) + '.jpg'

        dataset.append((augname,augCoord))

        cv2.imwrite(saveDir + augname, aug)        


    
    nname = name[:12] + '_0.jpg'

    dataset.append((nname,coords))

    cv2.imwrite(saveDir + nname, img)

    print(' ' +  str(i) + '/' +str(len(names)) + '     ',end='\r')
        

testnames= []
testlabels= []
trainnames = []
trainlabels = []
random.shuffle(dataset)

for i, sample in enumerate(dataset):
    if i%10 ==0:
        testnames.append(sample[0])
        testlabels.append(sample[1])
        continue
    
    trainnames.append(sample[0])
    trainlabels.append(sample[1])

testnames = np.array(testnames)
testlabels = np.array(testlabels)

trainnames = np.array(trainnames)
trainlabels = np.array(trainlabels)

print('Train: {}'.format(trainlabels.shape))
print('test: {}'.format(testlabels.shape))

savePath =  './data/'

np.save(savePath + 'testNames',testnames)
np.save(savePath + 'testLabels',testlabels)
np.save(savePath + 'trainNames',trainnames)
np.save(savePath + 'trainLabels',trainlabels)


    

