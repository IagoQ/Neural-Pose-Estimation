import numpy as np
from model import *
from keras.models import load_model
from generator import DataGenerator


dataDir = 'data'
dataType = 'processed'
imagesDir = '{}/{}/'.format(dataDir,dataType)

names = np.load('./data/testNames.npy')
keypoints = np.load('./data/testLabels.npy')

paths = [imagesDir + name for name in names]
testing_generator = DataGenerator(paths,keypoints,shuffle=True,batch_size=10)



model = load_model('./models/depthtestv1.h5')

loss = model.evaluate_generator(generator=testing_generator,use_multiprocessing=True,workers=6,verbose=1)
print(loss)


