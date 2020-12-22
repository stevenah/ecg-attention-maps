print('\n\n\nBeginning train_1d.py')
import numpy as np
import os, sys, GPUtil

DEVICE_ID_LIST = GPUtil.getFirstAvailable(order = 'memory', maxLoad=0.9, maxMemory=0.8, attempts=3, interval=15, verbose=True)
DEVICE_ID = DEVICE_ID_LIST[0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

import tensorflow as tf
from tensorflow.keras.optimizers import Nadam, SGD, Adagrad, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from customcallbacks import SaveSpecificEpochs
import datetime 

from model import KanRes8, JRes8, KanRes8815, testmodel, KanResWide, Attia, testmodel, Hannun, KanResWide_X

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



from settings import PLOT_FILE, TIMESTAMP, LOG_PATH, HISTORY_PATH
from settings import X_TRANSPOSE, SEED, EXPERIMENT_NAME, EXPERIMENT_ROOT
from settings import MODEL_PATH, ROOT_DIRECTORY

PREDICTION_LABELS = [
    'Asymmetry'
]

DATASET = 'gesustimeshift'
INPUT_TYPE = 'medians'

EPOCHS = 100 
SAVE_AFTER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30, 40, 50, 62, 75, 87, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
SAVE_AFTER = []
BATCH_SIZE = 32 
K_FOLDS = 5 
METRICS = [ 'mse', 'mae' ]
LOSS_FUNCTION = 'mean_squared_error' 
OPTIMIZER = Nadam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
    pass
STRATEGY = tf.distribute.OneDeviceStrategy(device="/gpu:0")


GROUND_TRUTH_PATH = os.path.join(ROOT_DIRECTORY, '..', DATASET, 'asc', 'ground_truth.csv')
MEDIANS_PATH = os.path.join(ROOT_DIRECTORY, '..',  DATASET, 'asc', 'medians')
RHYTHM_PATH = os.path.join(ROOT_DIRECTORY, '..', DATASET, 'asc', 'rhythm')
if INPUT_TYPE == 'rhythm':
    ECG_PATH = RHYTHM_PATH
else:
    ECG_PATH = MEDIANS_PATH

print('Running...')
print(str(MODEL_PATH)) 

default_stdout = sys.stdout 
sys.stdout = open(LOG_PATH, 'w') 

tf.compat.v1.set_random_seed(SEED)
np.random.seed(SEED)

default_stdout.write(str(TIMESTAMP)+'\n')
default_stdout.write(str(PREDICTION_LABELS)+'\n')
default_stdout.write('Data set: ' + str(DATASET) + '\n')
default_stdout.write('Input type: Raw ' + INPUT_TYPE + '\n\n')

def train():
    print('Prediction label(s): ' + str(PREDICTION_LABELS)) 
    print('Covariates: None!\n')
    print('K folds: ' + str(K_FOLDS))
    print('Loss function: ' + str(LOSS_FUNCTION))
    print('Model path: ' + str(MODEL_PATH))
    print('Model name: ' + str(TIMESTAMP))
    print('Optimizer: ' + str(OPTIMIZER.__class__.__name__) + ', parameters: ' + str(OPTIMIZER.get_config()))
    print('Batch size: ' + str(BATCH_SIZE))
    print('Epochs: ' + str(EPOCHS)) 
    print('Data set: ' + str(DATASET))
    print('Data type: Raw ' + INPUT_TYPE + '\n')
    
    data, header = read_csv(
        csv_file=GROUND_TRUTH_PATH,
        delimiter=';',
        transpose=False,
        skip_header=True)

    x_data, y_data = prepare_csv_data(
        data=data,
        prediction_labels=PREDICTION_LABELS,
        training_path=ECG_PATH, 
        x_shape=X_TRANSPOSE,
        data_labels = header)
    
    try:
        y_data = np.array(y_data).astype(np.int16)
    except:
        y_data = np.round(np.array(y_data).astype(np.float32)).astype(np.int16)
    
    for fold_index, (x_train, x_test, y_train, y_test) in enumerate(k_fold(x_data, y_data, K_FOLDS)):
        default_stdout.write('\n\n' + datetime.datetime.now().strftime("%H%M") + ': Running fold ' + str(fold_index+1) + ' of ' + str(K_FOLDS) + '\n') 
        print('\n\n' + datetime.datetime.now().strftime("%H%M") + ': Running fold ' + str(fold_index+1) + ' of ' + str(K_FOLDS) + '\n') 
        
        xshape = x_data[0].shape
        
        MODEL_CALLBACKS = [SaveSpecificEpochs(save_epochs = SAVE_AFTER,
                                     filepath = f'{os.path.splitext(MODEL_PATH)[0] }_{ fold_index }_' + 'epoch_{epoch:d}.h5' ),
                            ModelCheckpoint(save_best_only=True,
                                            filepath = f'{os.path.splitext(MODEL_PATH)[0] }_{ fold_index }_best.h5')
                          ]
        
        
        with STRATEGY.scope():

            model = KanResWide_X(input_shape=xshape,output_size=len(PREDICTION_LABELS))
            
            model.compile(
            optimizer=OPTIMIZER,
            loss=LOSS_FUNCTION,
            metrics=METRICS)
        
        
        print('Model type: ' + model.__class__.__name__ + '\n')
        
        history = model.fit(x_train, y_train, 
            epochs=EPOCHS,
            batch_size= BATCH_SIZE,
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
                plt.title(str(TIMESTAMP) + '_' + str(fold_index) + ' - learning ' + PREDICTION_LABELS[0])
                plt.xlabel('Epoch')
                plt.ylabel('Mean absolute error')
                plt.savefig(os.path.join(ROOT_DIRECTORY,'models', TIMESTAMP + f'_mean_abs_error_{ fold_index }.png'))
                plt.gcf().clear()
        except Exception as e:
            default_stdout.write('Error plot could not be made: ' + str(e) + '\n')
            print('Error plot could not be made: ' + str(e) + '\n')
        
        default_stdout.flush()
        
        plot_path = f'{ os.path.splitext(PLOT_FILE)[0] }_{ fold_index }.png'
        model_path = f'{ os.path.splitext(MODEL_PATH)[0] }_{ fold_index }.h5'
        history_path = f'{ os.path.splitext(HISTORY_PATH)[0] }_{ fold_index }.npy'

        if EPOCHS > 5:
            try:
                plot_loss(history, plot_path)
            except Exception as e:
                default_stdout.write('Loss plot could not be made: ' + str(e) + '\n')
                print('Loss plot could not be made: ' + str(e) + '\n')
            model.save(model_path)
            np.save(history_path, history.history)

        if Experiment is not None:
            experiment.add_artifact(PLOT_FILE)
            experiment.add_artifact(model_path)
        
                

if __name__ == '__main__':

    if Experiment is not None:

        experiment_path = f'{EXPERIMENT_ROOT}/{ EXPERIMENT_NAME }'

        experiment = Experiment(EXPERIMENT_NAME)
        experiment.captured_out_filter = apply_backspaces_and_linefeeds
        experiment.observers.append(FileStorageObserver.create(experiment_path))
        
        experiment.automain( train )

    else:
        train()

print('\n\nDone!')
default_stdout.write('Done!\n\n')
sys.stdout = default_stdout

