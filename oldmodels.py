from keras.layers import *
from keras.models import Model
from keras.optimizers import adadelta,adam

#TODO test model with l2 reg, longer pooling shorter upsampling model, get depthwise working


adamopt = adam(lr=0.000050)

def resblock(first,depth,squeeze,small,big,smallsize=1,bigsize=3):
    x = Conv2D(squeeze,1,padding='same')(first)
    x = LeakyReLU()(x)

    x1 = Conv2D(small,smallsize,padding='same')(x)
    x2 = Conv2D(big,bigsize,padding='same')(x)

    x = Concatenate()([x1,x2])
    x = LeakyReLU()(x)

    x = Conv2D(depth,1,padding='same')(x)
    x = LeakyReLU()(x)

    x = Add()([first,x])
    return x

def poolblock(first,depth,filters):
    x = Conv2D(depth,3,padding='same')(first)
    x = LeakyReLU()(x)
    x = Conv2D(filters,5,strides=2,padding='same')(x)
    x = LeakyReLU()(x)
    return x

def upsampblock(first,filters):
    x = UpSampling2D()(first)
    x = Conv2D(filters,5,padding='same')(x)
    x = LeakyReLU()(x)
    return x

def condenseblock(first,filters):
    x = Conv2D(filters,3,padding='same')(first)
    x = Conv2D(filters,5,padding='same')(x)

    return x

def resposev1():
    img = Input(shape=(224,224,3))
    x = Conv2D(32,3,padding='same')(img)
    x = LeakyReLU()(x)
    x = Conv2D(64,3,padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)

    x = poolblock(x,64,64)
    
    x = resblock(x,64,16,32,32)
    x = resblock(x,64,16,32,32,3,5)
    x = Dropout(0.2)(x)

    x = poolblock(x,64,128)

    x = resblock(x,128,32,128,64)
    x = resblock(x,128,32,64,64,3,5)
    x = Dropout(0.2)(x)

    x = poolblock(x,128,256)
    
    x = resblock(x,256,64,128,128)
    x = resblock(x,256,64,128,128,3,5)
    x = Dropout(0.2)(x)

    x = upsampblock(x,128)

    x = resblock(x,128,32,128,64)
    x = resblock(x,128,32,64,64,3,5)
    x = Dropout(0.2)(x)
    
    x = upsampblock(x,64)

    x = resblock(x,64,16,32,32)
    x = resblock(x,64,16,32,32,3,5)
    x = Dropout(0.2)(x)
    
    x = upsampblock(x,32)

    x = resblock(x,32,16,32,32)
    x = resblock(x,32,16,32,32,3,5)
    x = Dropout(0.2)(x)

    out = Conv2D(17,7,padding='same')(x)
    
    model = Model(img,out)
    model.summary()
    model.compile(optimizer=adamopt,loss='mse')
    return model

def resposev2():
    img = Input(shape=(224,224,3))
    x = Conv2D(32,3,padding='same')(img)
    x = LeakyReLU()(x)

    x = resblock(x,32,16,32,32)
    x = Dropout(0.25)(x)

    x = poolblock(x,32,64)#112

    x = resblock(x,64,32,64,64)
    x = Dropout(0.25)(x)
    x = resblock(x,64,32,64,64,3,5)                
    x = Dropout(0.3)(x)

    x = poolblock(x,64,128)#56

    x = resblock(x,128,64,128,128)
    x = Dropout(0.25)(x)
    x = resblock(x,128,64,128,128,3,5)          
    x = Dropout(0.25)(x)

    x = poolblock(x,128,256)#28

    x = Dropout(0.3)(x)
    x = resblock(x,256,128,128,128)
    x = resblock(x,256,128,128,128,3,7)
    x = resblock(x,256,128,128,128)

    x = upsampblock(x,128)#56

    x = Dropout(0.25)(x)
    x = resblock(x,128,64,128,128)              
    x = resblock(x,128,64,128,128,3,5)
    
    x = upsampblock(x,64)#112

    x = Dropout(0.25)(x)
    x = resblock(x,64,32,64,64)                     
    x = resblock(x,64,32,64,64,3,5)

    x = upsampblock(x,32)#224

    x = Dropout(0.25)(x)
    x = resblock(x,32,16,32,32)
    x = resblock(x,32,16,32,32,3,5)

    out = Conv2D(17,7,padding='same')(x)
    
    model = Model(img,out)
    model.summary()
    model.compile(optimizer=adamopt,loss='mse')
    return model


def poseblock(first,depth,squeeze,small,big):
    x = Conv2D(squeeze,1,padding='same')(first)
    x1 = LeakyReLU()(x)
    x = Conv2D(squeeze,1,padding='same')(first)
    x2 = LeakyReLU()(x)

    x1 = Conv2D(small,3,padding='same')(x1)
    x2 = Conv2D(big,3,padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(big,3,padding='same')(x2)

    x = Concatenate()([x1,x2])
    x = LeakyReLU()(x)

    x = Conv2D(depth,1,padding='same')(x)
    x = LeakyReLU()(x)

    x = Add()([first,x])
    return x

def poseblocksmall(first,depth,squeeze,small,big):
    x = Conv2D(squeeze,1,padding='same')(first)
    x1 = LeakyReLU()(x)
    x = Conv2D(squeeze,1,padding='same')(first)
    x2 = LeakyReLU()(x)

    x1 = Conv2D(small,1,padding='same')(x1)
    x2 = Conv2D(big,3,padding='same')(x2)
    x = Concatenate()([x1,x2])
    x = LeakyReLU()(x)

    x = Conv2D(depth,1,padding='same')(x)
    x = LeakyReLU()(x)

    x = Add()([first,x])
    return x

def poseblockbig(first,depth,squeeze,small,big):
    x = Conv2D(squeeze,1,padding='same')(first)
    x1 = LeakyReLU()(x)
    x = Conv2D(squeeze,1,padding='same')(first)
    x2 = LeakyReLU()(x)

    x1 = Conv2D(small,3,padding='same')(x1)

    x2 = Conv2D(big,3,padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(big,3,padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(big,3,padding='same')(x2)

    x = Concatenate()([x1,x2])
    x = LeakyReLU()(x)

    x = Conv2D(depth,1,padding='same')(x)
    x = LeakyReLU()(x)

    x = Add()([first,x])
    return x


def resposev3():
    img = Input(shape=(224,224,3))
    x = Conv2D(32,3,padding='same')(img)
    x = LeakyReLU()(x)
    x = poseblocksmall(x,32,16,32,32)
    x = Dropout(0.25)(x)

    x = poolblock(x,32,64) #112
    x = Dropout(0.25)(x)

    x = Dropout(0.25)(x)
    x = poseblocksmall(x,64,32,64,32)
    x = poseblock(x,64,32,64,32)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,64,32,64,32)
    
    x = poolblock(x,64,128) #56

    x = Dropout(0.25)(x)
    x = poseblocksmall(x,128,32,128,64)
    x = poseblock(x,128,32,128,64)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,128,32,128,64)

    x = poolblock(x,128,256) #28

    x = Dropout(0.25)(x)
    x = poseblocksmall(x,256,64,256,128)
    x = poseblock(x,256,64,256,128)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,256,64,256,128)

    x = upsampblock(x,128) #56

    x = Dropout(0.25)(x)
    x = poseblocksmall(x,128,32,128,64)
    x = poseblock(x,128,32,128,64)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,128,32,128,64)

    x = upsampblock(x,64) #112

    x = Dropout(0.25)(x)
    x = poseblocksmall(x,64,32,64,32)
    x = poseblock(x,64,32,64,32)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,64,32,64,32)

    x = upsampblock(x,32) #224

    x = Dropout(0.25)(x)
    x = poseblock(x,32,32,64,32)
    x = Dropout(0.25)(x)
    x = poseblock(x,32,32,64,32)


    out = Conv2D(17,7,padding='same')(x)
    
    model = Model(img,out)
    model.summary()
    model.compile(optimizer=adamopt,loss='mse')
    return model