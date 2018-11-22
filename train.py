import numpy as np
from model import *
from oldmodels import *
from keras.models import load_model
from generator import DataGenerator
from keras.callbacks import TensorBoard,ModelCheckpoint
from time import time



# model = resposev3()
model = load_model('./models/depthtestv1.h5')

savename = './models/depthtestv1.h5'

# savename = './models/depthtest.h5'
# saveweight = './models/depthtestw.h5'

dataDir = 'data'
dataType = 'processed'
imagesDir = '{}/{}/'.format(dataDir,dataType)

tensorboard = TensorBoard(log_dir="logs/{}".format('depthtest'),write_graph=True,write_images=True)
checkpoint = ModelCheckpoint(savename,verbose=1,save_best_only=True,monitor='loss')


names = np.load('./data/trainNames.npy')
keypoints = np.load('./data/trainLabels.npy')

# names = names[:100]
# keypoints = keypoints[:100]


paths = [imagesDir + name for name in names]
training_generator = DataGenerator(paths,keypoints,shuffle=True,batch_size=4)



model.fit_generator(generator=training_generator,
                use_multiprocessing=True,
                workers=4,
                epochs=100,
                callbacks=[tensorboard,checkpoint])


