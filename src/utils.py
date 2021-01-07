
import csv
import numpy as np
import os

from config import TIMESTAMP, DATA_LABELS

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except:
    plt = None

def read_csv(csv_file, delimiter=';', transpose=False, skip_header=True, dtype=None):

    data = []
    header = []

    with open(csv_file, 'r') as f:
        csv_data = csv.reader(f, delimiter=delimiter)
        
        if skip_header:
            temp = next(csv_data)
            header = {k: v for v, k in enumerate(temp)}
        
        for row in csv_data:
            data.append(row)

    data = np.array(data, dtype=dtype)

    if transpose:
        data = np.transpose(data)

    return data, header
    
def prepare_csv_data(data, prediction_labels, training_path, x_shape=None, data_labels = DATA_LABELS):

    prediction_indicies = [ data_labels[label] for label in prediction_labels ]
    
    counter = 0
    for row in data:
        if ([ row[index] for index in prediction_indicies ].count('NA')  == 0) & (os.path.isfile(f'{ training_path }/{ row[0] }.asc')) :
            counter += 1
    
    tempcount = 0
    while not 'median_data' in locals():
        tempcount += 1
        try:
            median_data, _ = read_csv(f'{ training_path }/{ row[0] }.asc', ' ', x_shape, False)
        except FileNotFoundError:
            pass
    
    x_data = np.ones( ((counter,) + median_data.shape), dtype = np.int16 )
    y_data = []
    
    counter = 0
    for row in data:
        try:
            if [ row[index] for index in prediction_indicies ].count('NA') == 0:
                median_data, _ = read_csv(f'{ training_path }/{ row[0] }.asc', ' ', x_shape, False)
                y_data.append([ row[index] for index in prediction_indicies ])
                x_data[counter,:,:] = np.array(median_data).astype(np.int16) 
                counter += 1
        except FileNotFoundError:
            pass
    
    y_data = np.array(y_data)
    
    if not x_data.shape[0] == y_data.shape[0]:
        raise ValueError('Shape of x_data and y_data do not match')
    
    return x_data, y_data

def k_fold(x_data, y_data, k=3):

    assert len(x_data) == len(y_data)
    
    x_split_length = len(x_data) // k
    y_split_length = len(y_data) // k

    x_folds = []
    y_folds = []

    for k_index in range(k - 1):
        x_folds += [ x_data[ k_index * x_split_length : (k_index + 1) * x_split_length ] ]
        y_folds += [ y_data[ k_index * y_split_length : (k_index + 1) * y_split_length ] ]

    x_folds += [ x_data[ (k - 1) * x_split_length : len(x_data) ] ] 
    y_folds += [ y_data[ (k - 1) * y_split_length : len(y_data) ] ]

    for fold_index in range(k):
        
        x_train = []
        y_train = []

        for train_index in range(k):
            if train_index != fold_index:
                x_train.extend(x_folds[train_index])
                y_train.extend(y_folds[train_index])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_test = x_folds[fold_index]
        y_test = y_folds[fold_index]
    
        yield x_train, x_test, y_train, y_test

def plot_loss(history, file_name):

    if plt is None:
        return
    
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])

    plt.title(str(TIMESTAMP))

    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    plt.legend(['validation','training'], loc='upper left')

    plt.savefig(file_name)
    plt.gcf().clear()