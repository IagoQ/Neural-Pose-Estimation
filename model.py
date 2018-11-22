from keras.layers import *
from keras.models import Model
from keras.optimizers import Adagrad,Adam,RMSprop

#TODO test model with l2 reg, longer pooling shorter upsampling model, get depthwise working


adamopt = Adam(lr=0.000050)



def poseblock(first,depth,squeeze,small,big):
    x1 = Conv2D(squeeze,1,padding='same')(first)
    x1 = LeakyReLU()(x1)

    x2 = Conv2D(squeeze,1,padding='same')(first)
    x2 = LeakyReLU()(x2)

    x1 = Conv2D(small,(3,3),padding='same')(x1)
    x1 = LeakyReLU()(x1)

    x2 = Conv2D(small,(1,3),padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(small,(3,1),padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(small,(1,3),padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(small,(3,1),padding='same')(x2)
    x2 = LeakyReLU()(x2)

    x = Concatenate()([x1,x2])

    x = Conv2D(depth,1,padding='same')(x)
    # x = LeakyReLU()(x) #test

    x = Add()([first,x])

    return x

def poseblockbig(first,depth,squeeze,small,big):
    x1 = Conv2D(squeeze,1,padding='same')(first)
    x1 = LeakyReLU()(x1)

    x2 = Conv2D(squeeze,1,padding='same')(first)
    x2 = LeakyReLU()(x2)

    x1 = Conv2D(small,(3,3),padding='same')(x1)
    x1 = LeakyReLU()(x1)


    x2 = Conv2D(small,(1,3),padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(small,(3,1),padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(small,(1,3),padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(small,(3,1),padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(small,(1,3),padding='same')(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(small,(3,1),padding='same')(x2)
    x2 = LeakyReLU()(x2)
    
    x = Concatenate()([x1,x2])

    x = Conv2D(depth,1,padding='same')(x)
    # x = LeakyReLU()(x) #test

    x = Add()([first,x])

    return x
    
def poolblock(first,filters):
    x = Conv2D(filters,3,strides=2,padding='same')(first)
    x = LeakyReLU()(x)
    return x

def upsampblock(first,filters):
    x = UpSampling2D()(first)
    x = Conv2D(filters,3,padding='same')(first)
    x = LeakyReLU()(x)
    return x

def resposev4():
    img = Input(shape=(224,224,3))
    x = Conv2D(32,3,padding='same',strides=1)(img)
    x = LeakyReLU()(x)
    x = poseblock(x,32,16,32,32)
    x = Dropout(0.25)(x)
    
    x = poolblock(x,64) #112

    x = Dropout(0.25)(x)
    x = poseblock(x,64,32,64,32)
    x = poseblock(x,64,32,64,32)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,64,32,64,32)
    
    x = poolblock(x,128) #56

    x = Dropout(0.25)(x)
    x = poseblock(x,128,32,128,64)
    x = poseblock(x,128,32,128,64)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,128,32,128,64)

    x = poolblock(x,128) #28
    
    x = Dropout(0.25)(x)
    x = poseblock(x,128,32,128,64)
    x = poseblock(x,128,32,128,64)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,128,32,128,64)

    x = poolblock(x,128) #14

    x = Dropout(0.25)(x)
    x = poseblock(x,128,32,128,64)
    x = poseblock(x,128,32,128,64)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,128,32,128,64)

    x = upsampblock(x,128) #28

    x = Dropout(0.25)(x)
    x = poseblock(x,128,32,128,64)
    x = poseblock(x,128,32,128,64)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,128,32,128,64)

    x = upsampblock(x,128) #56

    x = Dropout(0.25)(x)
    x = poseblock(x,128,32,128,64)
    x = poseblock(x,128,32,128,64)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,128,32,128,64)

    x = upsampblock(x,64) #112

    x = Dropout(0.25)(x)
    x = poseblock(x,64,32,64,32)
    x = poseblock(x,64,32,64,32)
    x = Dropout(0.25)(x)
    x = poseblockbig(x,64,32,64,32)

    x = upsampblock(x,64)#224

    x = Dropout(0.25)(x)
    x = poseblock(x,64,32,64,32)


    out = Conv2D(17,7,padding='same')(x)
    
    model = Model(img,out)
    model.summary()
    model.compile(optimizer=adamopt,loss='mse')
    return model












    











    




depthadam = Adam(lr=0.000050)

def DepthBlock(first,depth,small,big):

    x1 = DepthwiseConv2D(3,padding='same')(first)
    x1 = LeakyReLU()(x1)
    x1 = Conv2D(small,1,padding='same')(x1)
    

    x2 = DepthwiseConv2D(5,padding='same')(first)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(big,1,padding='same')(x2)


    x = Concatenate()([x1,x2])
    x = LeakyReLU()(x)

    x = Conv2D(depth,1,padding='same')(x)
    x = LeakyReLU()(x)

    x = Add()([first,x])
    return x

def DepthPool(first,filters):
    x = Conv2D(filters,3,strides=2,padding='same')(first)
    x = LeakyReLU()(x)
    return x

def DepthUpsamp(first,filters):
    x = UpSampling2D()(first)
    x = Conv2D(filters,3,padding='same')(x)
    x = LeakyReLU()(x)
    return x

def depthposev1base():

    img = Input(shape=(224,224,3))
    
    x = Conv2D(32,3,padding='same')(img)
    x = LeakyReLU()(x)
    x = DepthBlock(x,32,32,16)

    x = DepthPool(x,32) #112

    x = Dropout(0.2)(x)
    x = DepthBlock(x,32,32,16)
    x = DepthBlock(x,32,32,16)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,32,32,16)

    x = DepthPool(x,64) #56

    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)
    x = DepthBlock(x,64,64,32)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)

    x = DepthPool(x,128) #28

    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)
    x = DepthBlock(x,128,128,64)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)

    x = DepthPool(x,128) #14
    
    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,128)
    x = DepthBlock(x,128,128,128)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,128)

    x = DepthPool(x,256) #7
    
    x = Dropout(0.2)(x)
    x = DepthBlock(x,256,256,128)
    x = DepthBlock(x,256,256,128)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,256,256,128)

    x = DepthUpsamp(x,128) #14

    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)
    x = DepthBlock(x,128,128,64)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)

    x = DepthUpsamp(x,128) #28

    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)
    x = DepthBlock(x,128,128,64)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)
    
    x = DepthUpsamp(x,64) #56

    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)
    x = DepthBlock(x,64,64,32)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)

    x = DepthUpsamp(x,64) #112

    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)
    x = DepthBlock(x,64,64,32)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)

    x = DepthUpsamp(x,32) #224

    x = DepthBlock(x,32,32,32)

    out = Conv2D(17,7,padding='same')(x)
    
    model = Model(img,out)
    model.summary()
    model.compile(optimizer=depthadam,loss='mse')
    return model

def depthposev1shorter():

    img = Input(shape=(224,224,3))
    
    x = Conv2D(32,3,padding='same')(img)
    x = LeakyReLU()(x)
    x = DepthBlock(x,32,32,16)

    x = DepthPool(x,32) #112

    x = Dropout(0.2)(x)
    x = DepthBlock(x,32,32,16)
    x = DepthBlock(x,32,32,16)

    x = DepthPool(x,64) #56

    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)
    x = DepthBlock(x,64,64,32)

    x = DepthPool(x,128) #28

    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)
    x = DepthBlock(x,128,128,64)

    x = DepthPool(x,256) #14
    
    x = Dropout(0.2)(x)
    x = DepthBlock(x,256,256,128)
    x = DepthBlock(x,256,256,128)

    x = DepthPool(x,512) #7
    
    x = Dropout(0.2)(x)
    x = DepthBlock(x,512,512,256)
    x = DepthBlock(x,512,512,256)

    x = DepthUpsamp(x,256) #14

    x = Dropout(0.2)(x)
    x = DepthBlock(x,256,256,128)
    x = DepthBlock(x,256,256,128)

    x = DepthUpsamp(x,128) #28

    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)
    x = DepthBlock(x,128,128,64)
    
    x = DepthUpsamp(x,64) #56

    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)
    x = DepthBlock(x,64,64,32)

    x = DepthUpsamp(x,64) #112

    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)
    x = DepthBlock(x,64,64,32)

    x = DepthUpsamp(x,32) #224

    x = DepthBlock(x,32,32,32)

    out = Conv2D(17,7,padding='same')(x)
    
    model = Model(img,out)
    model.summary()
    model.compile(optimizer=depthadam,loss='mse')
    return model

def depthposev1shallow():

    img = Input(shape=(224,224,3))
    
    x = Conv2D(32,3,padding='same')(img)
    x = LeakyReLU()(x)
    x = DepthBlock(x,32,32,16)

    x = DepthPool(x,32) #112

    x = Dropout(0.2)(x)
    x = DepthBlock(x,32,32,16)
    x = DepthBlock(x,32,32,16)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,32,32,16)

    x = DepthPool(x,64) #56

    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)
    x = DepthBlock(x,64,64,32)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)

    x = DepthPool(x,128) #28

    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)
    x = DepthBlock(x,128,128,64)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)

    x = DepthPool(x,128) #14
    
    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,128)
    x = DepthBlock(x,128,128,128)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,128)

    x = DepthUpsamp(x,128) #28

    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)
    x = DepthBlock(x,128,128,64)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,128,128,64)
    
    x = DepthUpsamp(x,64) #56

    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)
    x = DepthBlock(x,64,64,32)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)

    x = DepthUpsamp(x,64) #112

    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)
    x = DepthBlock(x,64,64,32)
    x = Dropout(0.2)(x)
    x = DepthBlock(x,64,64,32)

    x = DepthUpsamp(x,32) #224

    x = DepthBlock(x,32,32,32)

    out = Conv2D(17,7,padding='same')(x)
    
    model = Model(img,out)
    model.summary()
    model.compile(optimizer=depthadam,loss='mse')
    return model















mobrms = RMSprop(lr=0.00005)
mobadagrad = Adagrad(lr=0.00005)
mobadam = Adam(lr=0.000050)


def mobblock(first,filters):
    x = Conv2D(filters//2,1,padding='valid',activation='relu')(first)
    x = DepthwiseConv2D(3,padding='same',activation='relu')(x)
    x = Conv2D(filters,1,padding='same')(x)
    x = Add()([first,x])
    return x

def mobpool(first,filters):
    # x = Conv2D(filters*2,1,padding='valid',activation='relu')(first)
    # x = DepthwiseConv2D(3,padding='same',strides=2,activation='relu')(x)
    # x = Conv2D(filters,1,padding='same')(x)
    x = Conv2D(filters,3,strides=2,padding='same',activation='relu')(first)
    return x

def mobupsamp(first,filters):
    x = UpSampling2D()(first)
    x = Conv2D(filters,3,padding='same',activation='relu')(x)
    # x = Conv2D(filters*2,1,padding='valid',activation='relu')(x)
    # x = DepthwiseConv2D(3,padding='same',activation='relu')(x)
    # x = Conv2D(filters,1,padding='same')(x)
    return x


def mobtest():
    img = Input(shape=(224,224,3))

    x = Conv2D(32,3,strides=1,padding='same',activation='relu')(img)

    x = mobblock(x,32)

    x = mobpool(x,64) #112

    x = mobblock(x,64)
    x = mobblock(x,64)
    x = mobblock(x,64)

    x = mobpool(x,128) #56

    x = mobblock(x,128)
    x = mobblock(x,128)
    x = mobblock(x,128)

    x = mobpool(x,256) #28

    x = mobblock(x,256)
    x = mobblock(x,256)
    x = mobblock(x,256)

    x = mobpool(x,512) #14

    x = mobblock(x,512)
    x = mobblock(x,512)
    x = mobblock(x,512)

    x = mobupsamp(x,256) #28
    
    x = mobblock(x,256)
    x = mobblock(x,256)

    x = mobupsamp(x,128) #56
    
    x = mobblock(x,128)
    x = mobblock(x,128)

    x = mobupsamp(x,64) #112
    
    x = mobblock(x,64)
    x = mobblock(x,64)

    x = mobupsamp(x,32) #224
    
    x = mobblock(x,32)
    x = mobblock(x,32)

    out = Conv2D(17,7,padding='same')(x)
    
    model = Model(img,out)
    model.summary()
    model.compile(optimizer=mobadagrad,loss='mse')
    return model