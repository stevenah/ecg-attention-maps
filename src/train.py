print('\n\n\nBeginning train_1d.py')

import numpy as np

import datetime 
import sys
import os
import config

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from models import KanResWide_X

from utils import plot_loss, read_csv
from utils import prepare_csv_data, k_fold

try:
    from sacred import Experiment
    from sacred.utils import apply_backspaces_and_linefeeds
    from sacred.observers import FileStorageObserver
except:
    Experiment = None    

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except:
    plt = None

print('Running...')
print(str(config.MODEL_PATH)) 

default_stdout = sys.stdout 
sys.stdout = open(config.LOG_PATH, 'w') 

tf.compat.v1.set_random_seed(config.SEED)
np.random.seed(config.SEED)

default_stdout.write(str(config.TIMESTAMP)+'\n')
default_stdout.write(str(config.PREDICTION_LABELS)+'\n')
default_stdout.write('Data set: ' + str(config.DATASET) + '\n')
default_stdout.write('Input type: Raw ' + config.INPUT_TYPE + '\n\n')

def train():
    print('Prediction label(s): ' + str(config.PREDICTION_LABELS)) 
    print('Covariates: None!\n')
    print('K folds: ' + str(config.K_FOLDS))
    print('Loss function: ' + str(config.LOSS_FUNCTION))
    print('Model path: ' + str(config.MODEL_PATH))
    print('Model name: ' + str(config.TIMESTAMP))
    print('Optimizer: ' + str(config.OPTIMIZER.__class__.__name__) + ', parameters: ' + str(config.OPTIMIZER.get_config()))
    print('Batch size: ' + str(config.BATCH_SIZE))
    print('Epochs: ' + str(config.EPOCHS)) 
    print('Data set: ' + str(config.DATASET))
    print('Data type: Raw ' + config.INPUT_TYPE + '\n')
    
    data, header = read_csv(
        csv_file=config.GROUND_TRUTH_PATH,
        delimiter=';',
        transpose=False,
        skip_header=True
    )

    x_data, y_data = prepare_csv_data(
        data=data,
        prediction_labels=config.PREDICTION_LABELS,
        training_path=config.ECG_PATH, 
        x_shape=config.X_TRANSPOSE,
        data_labels = header
    )
    
    try:
        y_data = np.array(y_data).astype(np.int16)
    except:
        y_data = np.round(np.array(y_data).astype(np.float32)).astype(np.int16)
    
    for fold_index, (x_train, x_test, y_train, y_test) in enumerate(k_fold(x_data, y_data, config.K_FOLDS)):
        default_stdout.write('\n\n' + datetime.datetime.now().strftime("%H%M") + ': Running fold ' + str(fold_index+1) + ' of ' + str(config.K_FOLDS) + '\n') 
        print('\n\n' + datetime.datetime.now().strftime("%H%M") + ': Running fold ' + str(fold_index+1) + ' of ' + str(config.K_FOLDS) + '\n') 
        
        xshape = x_data[0].shape
        
        MODEL_CALLBACKS = [
                            ModelCheckpoint(save_best_only=True,
                                            filepath = f'{os.path.splitext(config.MODEL_PATH)[0] }_{ fold_index }_best.h5')
                          ]
        
        
        with config.STRATEGY.scope():

            model = KanResWide_X(input_shape=xshape,output_size=len(config.PREDICTION_LABELS))
            
            model.compile(
            optimizer=config.OPTIMIZER,
            loss=config.LOSS_FUNCTION,
            metrics=config.METRICS)
        
        
        print('Model type: ' + model.__class__.__name__ + '\n')
        
        history = model.fit(x_train, y_train, 
            epochs=config.EPOCHS,
            batch_size= config.BATCH_SIZE,
            verbose=2,
            callbacks=MODEL_CALLBACKS,
            validation_data=(x_test, y_test),
            shuffle=True)
        
        print('\nModel name: ' + model.__class__.__name__ + '\n')
        print('Training mean absolute error (last 20 epochs). Mean: ' + str(np.mean(history.history['mae'][-20:])) + ', SD: ' + str(np.std(history.history['mae'][-20:])) + ', median: ' + str(np.median(history.history['mae'][-20:])) + '.\n')
        print('Training MSE (last 20 epochs). Mean: ' + str(np.mean(history.history['mse'][-20:])) + ', SD: ' + str(np.std(history.history['mse'][-20:])) + ', median: ' + str(np.median(history.history['mse'][-20:])) + '\n')
        print('\nValidation mean absolute error (last 20 epochs). Mean: ' + str(np.mean(history.history['val_mae'][-20:])) + ', SD: ' + str(np.std(history.history['val_mae'][-20:])) + ', median: ' + str(np.median(history.history['val_mae'][-20:])) + '.\n')
        print('Validation MSE (last 20 epochs). Mean: ' + str(np.mean(history.history['val_mse'][-20:])) + ', SD: ' + str(np.std(history.history['val_mse'][-20:])) + ', median: ' + str(np.median(history.history['val_mse'][-20:])) + '\n')
        bestEpoch = np.argmin(history.history['val_loss'])
        print('\nBest validation loss at epoch (1-based): ' + str(bestEpoch+1))
        print('Stats at best epoch: Training MAE: ' + str(history.history['mae'][bestEpoch]) + ', Validation MAE: ' 
        + str(history.history['val_mae'][bestEpoch]) + ', Validation MSE: ' + str(history.history['val_mse'][bestEpoch]) + '.\n')
        
        default_stdout.write('\nModel name: ' + model.__class__.__name__ + '\n')
        default_stdout.write('\nValidation mean absolute error (last 20 epochs). Mean: ' + str(np.mean(history.history['val_mae'][-20:])) + ', SD: ' + str(np.std(history.history['val_mae'][-20:])) + ', median: ' + str(np.median(history.history['val_mae'][-20:])) + '.\n')
        default_stdout.write('\nValidation MSE (last 20 epochs). Mean: ' + str(np.mean(history.history['val_mse'][-20:])) + ', SD: ' + str(np.std(history.history['val_mse'][-20:])) + ', median: ' + str(np.median(history.history['val_mse'][-20:])) + '\n')
        default_stdout.write('Stats at best epoch (' + str(bestEpoch+1) + '): Training MAE: ' + str(history.history['mae'][bestEpoch]) + ', Validation MAE: ' 
        + str(history.history['val_mae'][bestEpoch]) + ', Validation MSE: ' + str(history.history['val_mse'][bestEpoch]) + '.\n') 
        
        
        try:
            if plt is not None:
                plt.plot(history.history['val_mae'][20:])
                plt.plot(history.history['mae'][20:])
                plt.legend(['Validation','Training'])
                plt.title(str(config.TIMESTAMP) + '_' + str(fold_index) + ' - learning ' + config.PREDICTION_LABELS[0])
                plt.xlabel('Epoch')
                plt.ylabel('Mean absolute error')
                plt.savefig(os.path.join(config.ROOT_DIRECTORY,'models', config.TIMESTAMP + f'_mean_abs_error_{ fold_index }.png'))
                plt.gcf().clear()
        except Exception as e:
            default_stdout.write('Error plot could not be made: ' + str(e) + '\n')
            print('Error plot could not be made: ' + str(e) + '\n')
        
        default_stdout.flush()
        
        plot_path = f'{ os.path.splitext(config.PLOT_FILE)[0] }_{ fold_index }.png'
        model_path = f'{ os.path.splitext(config.MODEL_PATH)[0] }_{ fold_index }.h5'
        history_path = f'{ os.path.splitext(config.HISTORY_PATH)[0] }_{ fold_index }.npy'

        if config.EPOCHS > 5:
            try:
                plot_loss(history, plot_path)
            except Exception as e:
                default_stdout.write('Loss plot could not be made: ' + str(e) + '\n')
                print('Loss plot could not be made: ' + str(e) + '\n')
            model.save(model_path)
            np.save(history_path, history.history)

        if Experiment is not None:
            experiment.add_artifact(config.PLOT_FILE)
            experiment.add_artifact(model_path)
        
                

if __name__ == '__main__':

    if Experiment is not None:

        experiment_path = f'{config.EXPERIMENT_ROOT}/{ config.EXPERIMENT_NAME }'

        experiment = Experiment(config.EXPERIMENT_NAME)
        experiment.captured_out_filter = apply_backspaces_and_linefeeds
        experiment.observers.append(FileStorageObserver.create(experiment_path))
        
        experiment.automain( train )

    else:
        train()

    print('\n\nDone!')
    default_stdout.write('Done!\n\n')
    sys.stdout = default_stdout

