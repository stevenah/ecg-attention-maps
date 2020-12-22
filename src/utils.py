
import csv
import numpy as np
import os

from tensorflow.keras.preprocessing import image as kimage
from settings import TIMESTAMP, DATA_LABELS 
import cv2

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

def read_image(image_file, image_shape):

    image = cv2.imread(image_file)

    if image is None:
        raise FileNotFoundError()

    image = cv2.resize(image, (image_shape[1], image_shape[0]))

    image_tensor = kimage.img_to_array(image)
    image_tensor /= 255. 

    return image_tensor

def normalize(x):
    return (2 * ((x - x.min()) / (x.max() - x.min()))) - 1

def shorten(x):
    return [y[1::2] for y in x]

def prepare_csv_data_old(data, prediction_labels, training_path, x_shape=None, data_labels = DATA_LABELS):

    x_data = []
    y_data = []
    
    prediction_indicies = [ data_labels[label] for label in prediction_labels ]
    
    for row in data:
        try:
            median_data, _ = read_csv(f'{ training_path }/{ row[0] }.asc', ' ', x_shape, False) 
            if [ row[index] for index in prediction_indicies ].count('NA') == 0:
                y_data.append([ row[index] for index in prediction_indicies ])
                x_data.append(median_data)
        except FileNotFoundError:
            pass
    
    x_data = np.array(x_data).astype(np.int16)
    y_data = np.array(y_data)
    
    return x_data, y_data
    
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

   

def prepare_csv_data_cov(data, prediction_labels, cov_labels, training_path, x_shape=None):

    x_data = []
    y_data = []
    z_data = []
    
    prediction_indicies = [ DATA_LABELS[label] for label in prediction_labels ]
    cov_indicies = [ DATA_LABELS[label] for label in cov_labels ]

    for row in data:
        try:
            median_data, _ = read_csv(f'{ training_path }/{ row[0] }.asc', ' ', x_shape, False)[0]
            if [ row[index] for index in prediction_indicies + cov_indicies ].count('NA') == 0:
                y_data.append([ row[index] for index in prediction_indicies ])
                x_data.append(median_data)
                z_data.append([ row[index] for index in cov_indicies ])
        except FileNotFoundError:
            pass
    
    x_data = np.array(x_data).astype(np.int16)
    y_data = np.array(y_data)
    z_data = np.array(z_data)
    
    return x_data, y_data, z_data
    
def prepare_image_data(data, prediction_labels, training_path, x_shape=None):

    x_data = []
    y_data = []

    prediction_indicies = [ DATA_LABELS[label] for label in prediction_labels ]

    for row in data:
        try:
            median_data = read_image(f'{ training_path }/{ row[0].astype(np.float).astype(np.int) }.png', x_shape)
            if [ row[index] for index in prediction_indicies ].count('NA') == 0:
                y_data.append([ row[index] for index in prediction_indicies ])
                x_data.append(median_data)
        except FileNotFoundError:
            pass
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

def prepare_image_data_cov(data, prediction_labels, cov_labels, training_path, x_shape=None):

    x_data = []
    y_data = []
    z_data = []

    prediction_indicies = [ DATA_LABELS[label] for label in prediction_labels ]
    cov_indicies = [ DATA_LABELS[label] for label in cov_labels ]

    for row in data:
        try:
            median_data = read_image(f'{ training_path }/{ row[0].astype(np.float).astype(np.int) }.png', x_shape)
            if [ row[index] for index in prediction_indicies + cov_indicies ].count('NA') == 0:
                y_data.append([ row[index] for index in prediction_indicies ])
                x_data.append(median_data)
                z_data.append([ row[index] for index in cov_indicies ])
        except FileNotFoundError:
            pass
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    z_data = np.array(z_data)

    return x_data, y_data, z_data

def split_data(data, ratio=.3):
    split_index = int(len(data) - (len(data) * ratio))
    return data[:split_index], data[split_index:]

def k_fold_cov(x_data, y_data, z_data, k=3):

    assert len(x_data) == len(y_data)
    assert len(z_data) == len(y_data)
    
    split_length = len(x_data) // k

    x_folds = []
    y_folds = []
    z_folds = []

    for k_index in range(k - 1):
        x_folds += [ x_data[ k_index * split_length : (k_index + 1) * split_length ] ]
        y_folds += [ y_data[ k_index * split_length : (k_index + 1) * split_length ] ]
        z_folds += [ z_data[ k_index * split_length : (k_index + 1) * split_length ] ]

    x_folds += [ x_data[ (k - 1) * split_length : len(x_data) ] ] 
    y_folds += [ y_data[ (k - 1) * split_length : len(y_data) ] ]
    z_folds += [ z_data[ (k - 1) * split_length : len(z_data) ] ]

    for fold_index in range(k):
        
        x_train = []
        y_train = []
        z_train = []

        for train_index in range(k):
            if train_index != fold_index:
                x_train.extend(x_folds[train_index])
                y_train.extend(y_folds[train_index])
                z_train.extend(z_folds[train_index])

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        z_train = np.array(z_train)

        x_test = x_folds[fold_index]
        y_test = y_folds[fold_index]
        z_test = z_folds[fold_index]
    
        yield x_train, x_test, y_train, y_test, z_train, z_test

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

def k_fold_2(x_data, y_data, k=3, fold_index=0):

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
    
    return x_train, x_test, y_train, y_test

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

def plot_saliency_background_2d(ecg, grad, path_to_print):
    x = np.arange(0,len(ecg))
    fig, axs = plt.subplots(4,2, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs = axs.ravel()
    for idx, ax in enumerate(axs):
        ax.imshow(np.stack([grad[:,idx].T for _ in range(10)]), cmap="autumn_r", origin='lower',extent=[x.min(),x.max(),-1000,2000])
        ax.plot(ecg[:,idx], color='blue',lw=1)
        ax.set_aspect('auto')
    fig.suptitle(title)
    fig.savefig(path_to_print)
    plt.close()


def plot_cam_background(ecg, cam, title, path_to_print):
    x = np.arange(0,len(ecg))
    fig, axs = plt.subplots(4,2, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs = axs.ravel()
    for idx, ax in enumerate(axs):
        ax.imshow(np.stack([cam[:].T for _ in range(10)]), cmap="autumn_r", origin='lower',extent=[x.min(),x.max(),-1000,2000])
        ax.plot(ecg[:,idx], color='blue',lw=1)
        ax.set_aspect('auto')
    fig.suptitle(title)
    fig.savefig(path_to_print)
    plt.close()