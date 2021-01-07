import cv2
import os
import math
import glob
import csv
import logging

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mline

from matplotlib.collections import LineCollection
from tensorflow import keras

__MODEL_PATH__ = ""
__ECG_PATHS__ = [ '' + str(sub) + '.asc' for sub in list(range(600,610)) ]
__ECG_PATHS__ = [ '' for sub in list(range(139271,139281))]

__OUTPUT_PATH__ = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s"
)

def load_ecg_data_from_file(filepath, delimiter=" "):
    
    data_item_list = []

    with open(filepath) as f:
        for row in csv.reader(f, delimiter=delimiter):
            data_item_list.append(row)

    return np.array(data_item_list)

def explain(image, model, class_index, layer_name, weighted=None):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    guided_grads = (
        tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    )

    output = conv_outputs[0]
    guided_grad = guided_grads[0]
    
    if weighted is not None:
        guided_grad *= weighted.T

    weights = tf.reduce_mean(guided_grad, axis=(0, 1))
    heatmap = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
    
    image = np.squeeze(image)

    heatmap = cv2.resize(heatmap.numpy(), tuple([image.shape[1], image.shape[0]]))
    heatmap = (heatmap - np.min(heatmap)) / (heatmap.max() - heatmap.min())

    return cv2.applyColorMap(
        cv2.cvtColor((heatmap * 255).astype("uint8"), cv2.COLOR_GRAY2BGR), cv2.COLORMAP_JET
    )

def plot_ecg_image(ax, sensor_data, heatmap, name):

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False) 
    ax.spines["left"].set_visible(False) 
    ax.spines["bottom"].set_visible(False) 

    ax.tick_params(axis=u'both', which=u'both',length=0)

    heatmap = heatmap / 255

    data_points = np.zeros((len(sensor_data), 1, 2))

    for row_index, point in enumerate(sensor_data):
        data_points[ row_index, 0, 0 ] = row_index
        data_points[ row_index, 0, 1 ] = point

    segments = np.hstack([data_points[:-1], data_points[1:]])
    coll = LineCollection(segments, colors=[ [ 0.5, 0.5, 0.5 ] ] * len(segments), linewidths=(1.3)) 
    
    ax.add_collection(coll)
    ax.autoscale_view()

    colors = np.mean(heatmap, axis=0)
    for c_index, color in enumerate(colors):
        ax.axvspan(c_index, c_index+1, facecolor=color)

def visualize_ecg_prediction_tf_explain(model_path, ecg_paths, output_path, class_indicies=[ 0 ]):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        logging.error("Could not load %s!" % model_path)
        return
    
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    layer_names = [ layer.name for layer in model.layers if type(layer) == keras.layers.Add ][ -1 : ]
    layer_names.extend ([ layer.name for layer in model.layers if type(layer) == keras.layers.Conv1D ][ -1 : ])

    logging.info("Starting to visualize the predictions of %i ECGs" % len(ecg_paths))

    for ecg_index, path in enumerate(ecg_paths):
        
        logging.info("Visualizing %s..." % path)
        try: 
            ecg_values = load_ecg_data_from_file(path)
        except:
            continue
        ecg_values = np.array(ecg_values, dtype=np.float32)
        ecg_values = np.reshape(ecg_values, (1, *ecg_values.shape))
        
    
        cols = [ "Last Add", "Last Conv"]
        rows = [ 
            os.path.splitext(os.path.basename(path))[0]
        ]

        ecg_name = os.path.splitext(os.path.basename(path))[0]
        
        plt.figure()
        plt.axis("off")

        fig, axes = plt.subplots(len(rows), len(cols), figsize=(9, 3))
        
        for ax, col in zip(axes, cols):
            ax.set_title(col, fontsize= 9)
        
        for index, (ax, row) in enumerate(zip(axes, rows)):
            ax.set_ylabel(row, fontsize= 9)
        
        for layer_index, layer_name in enumerate(layer_names):

            logging.info("Visualizing layer %s..." % layer_name)

            for class_index in class_indicies:

                output = explain(
                    ecg_values,
                    model=model,
                    layer_name=layer_name,
                    class_index=class_index
                )

                sensor_values = np.transpose(ecg_values.squeeze())[6]
                output = cv2.cvtColor(np.transpose(output, (1, 0, 2)), cv2.COLOR_BGR2RGB)
                plot_ecg_image(axes[ layer_index ], sensor_values, output, "%s_%s" % (ecg_name, layer_name))

        file_format = "png"
        image_output_name = "%s_%s.%s" % (model_name, ecg_name, file_format)
    
        logging.info("Saving %s..." % image_output_name)

        plt.savefig(
            fname=os.path.join(output_path, image_output_name),
            format=file_format,
            dpi=1000,
        )

        plt.cla()
        plt.clf()
        plt.close("all")
    
if __name__ == "__main__":

    visualize_ecg_prediction_tf_explain(__MODEL_PATH__, __ECG_PATHS__, __OUTPUT_PATH__)