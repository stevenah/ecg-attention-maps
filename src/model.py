import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Reshape, Add, Lambda
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling2D, AveragePooling1D, ZeroPadding1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate, Activation
from tensorflow.keras.layers import SeparableConv1D
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from math import floor, ceil

class BaseModel():

    def __init__(self, input_shape, output_size):
        self.input_shape = input_shape
        self.output_size = output_size

        self.build_model()

        print(self.model.summary())

    def fit(self, x_train, y_train, **kwargs):
        return self.model.fit(x_train, y_train, **kwargs)

    def compile(self, *argv, **kwargs):
        self.model.compile(*argv, **kwargs)

    def evaluate(self, *argv, **kwargs):
        return self.model.evaluate(*argv, **kwargs)

    def save(self, path):
        self.model.save(path)
        
def kanres_init(INP_TENSOR, Filterno_1, Filterno_2, Filtersize_1, Filtersize_2, STRIDE):
    X = Conv1D(Filterno_1, Filtersize_1, activation= None, strides=STRIDE, data_format = 'channels_last')(INP_TENSOR)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu') (X)
    X = Conv1D(Filterno_2, Filtersize_2, activation= None, data_format = 'channels_last')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu') (X)
    return X
    
def kanres_module(INPUT, Filterno_1, Filterno_2, Filtersize_1, Filtersize_2, STRIDE):
    X = Conv1D(Filterno_1, Filtersize_1, activation= None, strides=STRIDE, padding='same', data_format = 'channels_last')(INPUT)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu') (X)
    X = Conv1D(Filterno_2, Filtersize_2, activation= None, padding='same', data_format = 'channels_last')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu') (X)
    X = Add()([X,INPUT])
    return X

class KanResWide_X(BaseModel):

    def __init__(self, *args, **kwargs):
        super(KanResWide_X, self).__init__(*args, **kwargs)
        
    def build_model(self):    
        ECG = Input(shape=self.input_shape, name = 'ECG')
       
        x = kanres_init(ECG, 64, 32, 8, 3, 1) 
        x = AveragePooling1D(pool_size=2, data_format = 'channels_last')(x)
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = GlobalAveragePooling1D()(x)           

        output = Dense(self.output_size)(x)
        
        model = Model(inputs=ECG, outputs=output)
        
        self.model = model  

def attia_module(INPUT, k_length, Ch_out):
    X = BatchNormalization(axis = -1)(INPUT)
    X = Activation('relu')(X)
    X = Conv2D(filters = Ch_out, kernel_size = (k_length, 1), activation= None, strides = 1, padding='same')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = Ch_out, kernel_size = (k_length, 1), activation= None, strides = (2,1), padding='same')(X)
    Y = Conv2D(filters = Ch_out, kernel_size = (1, 1), activation = None, padding = 'same')(INPUT)
    Y = MaxPooling2D(pool_size=(2,1))(Y)
    X = Add()([X,Y])
    return X