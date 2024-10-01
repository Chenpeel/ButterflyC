import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
from main.utils.config import load_config
configs = load_config()
from main.model import *

def plot_loss_acc(path,save_name):
    data = pd.read_csv(path)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(data['epoch'], data['loss'], label='Training Loss')
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(data['epoch'], data['accuracy'], label='Training Accuracy')
    plt.plot(data['epoch'], data['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.savefig(save_name)
    plt.close()

def show_cam(model,img_path,class_idx):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    heatmap = model.generate_cam(img_array, class_idx)
    model.visualize_cam(img_path, heatmap)

def get_random_img():
    from utils.encode_label import encode_label
    train_csv = pd.read_csv(configs['train_csv'])
    random_row = train_csv.sample(n=1).iloc[0]
    img_path = os.path.join(configs['train_data'], str(random_row['filename']))
    class_idx = encode_label(random_row['label'])
    return img_path, class_idx

if __name__ == '__main__':
    # ,'VGG16','ResNet50','DenseNet121'
    for model_name in ['br']:
        # path = os.path.join(configs['log_dir'],model_name+'.csv')
        # save_name = os.path.join(configs['log_dir'],model_name+'.png')
        # plot_loss_acc(path,save_name)

        # 'VGG16': VGG16M, 'ResNet50': ResNet50M, 'DenseNet121': DenseNet121M
        model_class = {'br': br, }[model_name]
        model = model_class(input_shape=(configs['image_size'], configs['image_size'], 3), num_classes=configs['num_classes'])
        # model.model.load_weights(os.path.join(configs['model_path'], model_name + configs['model_suffix']))
        model.model.load_weights(os.path.join(configs['model_path'], 'br.keras'))
        img_path , class_idx = get_random_img()
        show_cam(model, img_path, class_idx)
