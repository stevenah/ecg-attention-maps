import datetime
import os

DATA_LABELS = {
    'TestID': 0, 'VentricularRate': 1, 'P_RInterval': 2, 'QRSDuration': 3,
    'Q_TInterval': 4, 'QTCFridericia': 5, 'PAxis': 6, 'RAxis': 7, 'TAxis': 8,
    'QRSCount': 9, 'QOnset': 10, 'QOffset': 11, 'POnset': 12, 'POffset': 13,
    'TOffset': 14, 'MCS': 15, 'Flatness': 16, 'Asymmetry': 17, 'Notch': 18,
    'NotchBinom': 19, 'Potassium': 20, 'Sodium': 21, 'Sex': 22, 'Age': 23,
    'BMI': 24, 'ECGCategory': 25, 'TPeakAmp_I': 26, 'TPeakAmp_II': 27, 
    'TPeakAmp_V1': 28, 'TPeakAmp_V2': 29, 'TPeakAmp_V3': 30, 'TPeakAmp_V4': 31,
    'TPeakAmp_V5': 32, 'TPeakAmp_V6': 33,
    'RPeakAmp_I': 34, 'RPeakAmp_II': 35,'RPeakAmp_V1': 36, 'RPeakAmp_V2': 37,
    'RPeakAmp_V3': 38, 'RPeakAmp_V4': 39,'RPeakAmp_V5': 40, 'RPeakAmp_V6': 41,
    'STM_I': 42, 'STM_II': 43, 'STM_V1': 44, 'STM_V2': 45,
    'STM_V3': 46, 'STM_V4': 47,'STM_V5': 48, 'STM_V6': 49,
    'fysiskaktiv_1': 50, 'styrketraening': 51, 'fysisk_fritid': 52,
    'age10': 53, 'fysisk_score': 54, 'fysisk_score_w': 55,
    'STJ_I': 56, 'STJ_II': 57, 'STJ_V1': 58, 'STJ_V2': 59,
    'STJ_V3': 60, 'STJ_V4': 61,'STJ_V5': 62, 'STJ_V6': 63,
    'SPeakAmp_I': 64, 'SPeakAmp_II': 65, 'SPeakAmp_V1': 66, 'SPeakAmp_V2': 67,
    'SPeakAmp_V3': 68, 'SPeakAmp_V4': 69, 'SPeakAmp_V5': 70, 'SPeakAmp_V6': 71,
    'RPeakAmp_aVL': 72, 'SPeakAmp_aVL': 73, 'TPeakAmp_aVL': 74, 'NormalECG': 75,
    'CornellVoltage': 76, 'SokolowLyonVoltage': 77, 'CornellVoltageCriterion': 78,
    'SokolowLyonCriterion': 79
}


ROOT_DIRECTORY = ''

SEED = 2

X_TRANSPOSE = False
IMAGE_SHAPE = (8, 600, 3)

TIMESTAMP = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
PLOT_FILE = os.path.join(ROOT_DIRECTORY, 'models', str(TIMESTAMP) + '_loss_plot.png')
MODEL_FILE = os.path.join('models', str(TIMESTAMP) + '_model.h5')
LOG_FILE = os.path.join('models', str(TIMESTAMP) + '_log.txt')
HISTORY_FILE = os.path.join('models', str(TIMESTAMP) + '_history.npy')

EXPERIMENT_ROOT = os.path.join(ROOT_DIRECTORY, '..', 'experiments')
MODEL_PATH = os.path.join(ROOT_DIRECTORY, MODEL_FILE)
LOG_PATH = os.path.join(ROOT_DIRECTORY, LOG_FILE)
HISTORY_PATH = os.path.join(ROOT_DIRECTORY, HISTORY_FILE)

EXPERIMENT_NAME = 'ECG_CNN_MODEL'




