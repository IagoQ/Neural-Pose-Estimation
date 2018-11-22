import keras
import numpy as np
from PIL import Image
from datagenutils import generateHeat,generatePafs

class DataGenerator(keras.utils.Sequence):
    def __init__(self,paths,keypoints,batch_size=32,dim=(224,224,3),shuffle = True):
        self.paths = paths
        self.keypoints = keypoints
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self,paths,keypoints):
        'Generates data containing batch_size samples' # x : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size,self.dim[0],self.dim[1],self.dim[2]))
        y = np.empty((self.batch_size,self.dim[0],self.dim[1],17)) #+38

        # Generate data
        for i,(name,keys) in enumerate(zip(paths,keypoints)):
            # Store sample
            imgopen = Image.open(name)
            img = np.array(imgopen).astype('float32') / 255
            imgopen.close()


            x[i,] = img
            heat = generateHeat(keys)
            
            y[i] = heat


            # heat = generateHeat(nkeys)
            # pafs = generatePafs(keys)
            # out = np.concatenate((heat,pafs),axis=-1)
            # x[i,] = nimg
            # y[i] = out
            
        return x, y
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        paths_temp = [self.paths[k] for k in indexes]
        keys_temp = [self.keypoints[k] for k in indexes]
        # Generate data
        x, y = self.__data_generation(paths_temp,keys_temp)
        return x,y