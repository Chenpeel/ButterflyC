import os
import sys
import shutil
import psutil
import tempfile
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array
)
import tensorflow as tf
from tqdm import tqdm
from ctypes import Array
from imblearn.over_sampling import SMOTE
from main.utils.config import load_config
from numpy.core.multiarray import ndarray
from main.utils.encode_label import label_encoder
from pandas.core.dtypes.cast import NumpyArrayT
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


configs = load_config()


def transform(labels):
    return label_encoder.transform(labels)

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")

def genDatas():
    train_csv = pd.read_csv(configs['train_csv'])
    test_csv = pd.read_csv(configs['test_csv'])  # 加载测试集

    # 清理列名
    train_csv.columns = train_csv.columns.str.strip()
    test_csv.columns = test_csv.columns.str.strip()

    # 打印列名以进行调试
    print("Train CSV Columns:\n", train_csv.columns)
    print("Test CSV Columns:\n", test_csv.columns)

    # 打印前几行以进行调试
    print("Train CSV Head:\n", train_csv.head())

    # 拟合 LabelEncoder
    label_encoder.fit(train_csv['label'])

    # 处理训练集
    train_images = []
    train_labels = []
    for index, row in tqdm(train_csv.iterrows(), total=train_csv.shape[0], desc="Processing train images"):
        img_path = os.path.join(configs['train_data'], str(row['filename']))
        img = load_img(img_path, target_size=(configs['image_size'], configs['image_size']))
        img = img_to_array(img)
        train_images.append(img)
        train_labels.append(transform([row['label']])[0])

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # 检查原始数据集的类别分布
    unique_classes, class_counts = np.unique(train_labels, return_counts=True)
    print("Class distribution in original dataset:\n", dict(zip(unique_classes, class_counts)))

    if len(unique_classes) <= 1:
        raise ValueError("The target 'y' in the original dataset needs to have more than 1 class. Got 1 class instead")

    # 处理测试集
    test_images = []
    test_labels = None
    for index, row in tqdm(test_csv.iterrows(), total=test_csv.shape[0], desc="Processing test images"):
        img_path = os.path.join(configs['test_data'], str(row['filename']))
        img = load_img(img_path, target_size=(configs['image_size'], configs['image_size']))
        img = img_to_array(img)
        test_images.append(img)

    test_images = np.array(test_images)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    # 数据增强
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()

    augmented_images = []
    augmented_labels = []
    for img, label in tqdm(zip(X_train, y_train), total=len(X_train), desc="Augmenting images"):
        aug_imgs, aug_labels = augment_images(img, label, datagen, num_augmented=2, temp_dir=temp_dir)
        augmented_images.extend(aug_imgs)
        augmented_labels.extend(aug_labels)

    augmented_images = np.array([img_to_array(load_img(img_path)) for img_path in augmented_images])
    augmented_labels = np.array(augmented_labels)

    # 检查类别分布
    unique_classes, class_counts = np.unique(augmented_labels, return_counts=True)
    print("Class distribution after augmentation:\n", dict(zip(unique_classes, class_counts)))

    if len(unique_classes) <= 1:
        raise ValueError("The target 'y' needs to have more than 1 class. Got 1 class instead")

    # 过采样
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(augmented_images.reshape(len(augmented_images), -1), augmented_labels)
    X_res = X_res.reshape(-1, configs['image_size'], configs['image_size'], 3)

    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_res), y=y_res)
    class_weights_dict = dict(enumerate(class_weights))

    # 将目标标签转换为 one-hot 编码格式
    y_res = tf.keras.utils.to_categorical(y_res, num_classes=len(unique_classes))
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(unique_classes))

    print_memory_usage()

    return X_res, y_res, X_val, y_val, test_images, test_labels, class_weights_dict, temp_dir

def augment_images(image, label, datagen, num_augmented, temp_dir):
    image = np.expand_dims(image, 0)
    augmented_images = []
    for i in range(num_augmented):
        aug_img = datagen.flow(image, batch_size=1)[0][0]
        aug_img_path = os.path.join(temp_dir, f"aug_{label}_{i}.png")
        tf.keras.preprocessing.image.save_img(aug_img_path, aug_img)
        augmented_images.append(aug_img_path)
    return augmented_images, [label] * num_augmented



def process(path,size=(224,224))->ndarray:
    img = load_img(path, target_size=size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array
