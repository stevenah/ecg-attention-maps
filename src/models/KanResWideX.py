
from models import BaseModel

from tensorflow.keras import layers
from tensorflow.keras.models import Model

def kanres_init(INP_TENSOR, Filterno_1, Filterno_2, Filtersize_1, Filtersize_2, STRIDE):
    X = layers.Conv1D(Filterno_1, Filtersize_1, activation= None, strides=STRIDE, data_format = 'channels_last')(INP_TENSOR)
    X = layers.BatchNormalization(axis = -1)(X)
    X = layers.Activation('relu') (X)
    X = layers.Conv1D(Filterno_2, Filtersize_2, activation= None, data_format = 'channels_last')(X)
    X = layers.BatchNormalization(axis = -1)(X)
    X = layers.Activation('relu') (X)
    return X
    
def kanres_module(INPUT, Filterno_1, Filterno_2, Filtersize_1, Filtersize_2, STRIDE):
    X = layers.Conv1D(Filterno_1, Filtersize_1, activation= None, strides=STRIDE, padding='same', data_format = 'channels_last')(INPUT)
    X = layers.BatchNormalization(axis = -1)(X)
    X = layers.Activation('relu') (X)
    X = layers.Conv1D(Filterno_2, Filtersize_2, activation= None, padding='same', data_format = 'channels_last')(X)
    X = layers.BatchNormalization(axis = -1)(X)
    X = layers.Activation('relu') (X)
    X = layers.Add()( [ X, INPUT ] )
    return X

class KanResWide_X(BaseModel):

    def __init__(self, *args, **kwargs):
        super(KanResWide_X, self).__init__(*args, **kwargs)
        
    def build_model(self):    
        ECG = layers.Input(shape=self.input_shape, name = 'ECG')
       
        x = kanres_init(ECG, 64, 32, 8, 3, 1) 
        x = layers.AveragePooling1D(pool_size=2, data_format = 'channels_last')(x)
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = kanres_module(x, 64, 32, 50, 50, 1) 
        x = layers.GlobalAveragePooling1D()(x)           

        output = layers.Dense(self.output_size)(x)
        
        model = Model(inputs=ECG, outputs=output)
        
        self.model = model