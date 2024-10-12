import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from main.utils.config import load_config
import tensorflow as tf

def process_img(img_path, target_size):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def apply_smote(train_images, train_labels):
    try:
        smote = SMOTE()
        print("Applying SMOTE...")
        train_images_flat = train_images.reshape(len(train_images), -1)
        train_images_resampled, train_labels_resampled = smote.fit_resample(train_images_flat, train_labels)
        return train_images_resampled.reshape(-1, configs['image_size'], configs['image_size'], 3), train_labels_resampled
    except Exception as e:
        print(f"Error applying SMOTE: {e}")
        return train_images, train_labels

def load_data():
    configs = load_config()
    train_df = pd.read_csv(configs['train_csv'])
    label_encoder = encode_labels(configs['train_csv'])
    train_df['label'] = label_encoder.transform(train_df['label'])
    train_images = []
    train_labels = []

    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0], desc="Processing images"):
        img_path = os.path.join(configs['train_data'], str(row['filename']))
        img = process_img(img_path, (configs['image_size'], configs['image_size']))
        if img is not None:
            train_images.append(img)
            train_labels.append(row['label'])

    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels)
    # train_images_resampled, train_labels_resampled = apply_smote(train_images, train_labels)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

    # One-hot encode the labels
    train_labels = to_categorical(train_labels, num_classes=configs['num_classes'])
    val_labels = to_categorical(val_labels, num_classes=configs['num_classes'])

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])

    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_data = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_data = train_data.batch(configs['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.batch(configs['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)

    return train_data, val_data

def encode_labels(csv_path):
    df = pd.read_csv(csv_path)
    labels = df['label'].unique()
    label_encoder = LabelEncoder()
    global encoded_labels
    encoded_labels = label_encoder.fit_transform(labels)
    label_mapping = {
        label: int(encoded_label) for label, encoded_label in zip(labels, encoded_labels)}
    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)
    return label_encoder
