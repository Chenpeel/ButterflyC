import matplotlib.pyplot as plt
from main.utils.process import process_img
import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import main.utils.config as config

def plot_accloss(csv_path):
    data = pd.read_csv(csv_path)
    epochs = data['epoch']
    accuracy = data['accuracy']
    val_accuracy = data['val_accuracy']
    loss = data['loss']
    val_loss = data['val_loss']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

def display_cam(model, img_paths, layer_name=None):
    if layer_name is None:
        layer_name = get_last_conv_layer_name(model)
        if layer_name is None:
            raise ValueError("No convolutional layer found in the model.")

    plt.figure(figsize=(20, 10))

    for i, img_path in enumerate(img_paths):
        img_array = process_img(img_path, (configs['image_size'], configs['image_size']))

        # 确保img_array的形状是正确的
        print("Image array shape before processing:", img_array.shape)
        img_array = np.squeeze(img_array)  # 移除任何单一维度
        print("Image array shape after processing:", img_array.shape)

        # 添加批次维度
        if img_array.ndim == 3:  # 如果img_array是3维的，添加一个批次维度
            img_array = np.expand_dims(img_array, axis=0)
        else:
            raise ValueError("Image array does not have the expected number of dimensions after squeeze.")

        print("Image array shape before prediction:", img_array.shape)

        preds = model.predict(img_array)
        class_idx = np.argmax(preds[0])

        last_conv_layer = model.get_layer(layer_name)
        last_conv_layer_output = last_conv_layer.output
        output_model = tf.keras.Model([model.inputs], [last_conv_layer_output, model.output])

        with tf.GradientTape() as tape:
            last_conv_layer_output, model_output = output_model(img_array)
            tape.watch(last_conv_layer_output)
            grads = tape.gradient(model_output[:, class_idx], last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # 检查 np.max(heatmap) 是否为零
        max_heatmap = np.max(heatmap)
        if max_heatmap != 0:
            heatmap = np.maximum(heatmap, 0) / max_heatmap
        else:
            print(f"Warning: max_heatmap is zero for image {img_path}")
            heatmap = np.maximum(heatmap, 0)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (configs['image_size'], configs['image_size']))
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        plt.subplot(2, len(img_paths), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Image {i+1}')
        plt.axis('off')

        plt.subplot(2, len(img_paths), i + 1 + len(img_paths))
        plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        plt.title(f'CAM {i+1}')
        plt.axis('off')

    plt.show()

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            return layer.name
    return None

if __name__ == "__main__":
    configs = config.load_config()
    for model_name in ['ButterflyC','ResNet50M','DenseNet121M']:
        model = tf.keras.models.load_model(configs['model_path'] + '/'+model_name+'.keras')
        img_paths = [
            "data/train/Image_1001.jpg",
            "data/train/Image_1002.jpg",
            "data/train/Image_1003.jpg",
            "data/train/Image_1004.jpg",
            "data/train/Image_1005.jpg",
            "data/train/Image_1006.jpg",
            "data/train/Image_1007.jpg",
            "data/train/Image_1008.jpg",
        ]

        display_cam(model, img_paths)
