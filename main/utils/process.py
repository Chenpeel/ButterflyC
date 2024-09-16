import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)


def create_data_generators(configs, label_encoder):
    # 读取 CSV 文件
    train_df = pd.read_csv(configs["train_csv_path"])
    test_df = pd.read_csv(configs["test_csv_path"])

    # 配置数据生成器
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # 将像素值缩放到0-1之间
        rotation_range=30,  # 随机旋转
        width_shift_range=0.1,  # 水平位移
        height_shift_range=0.1,  # 垂直位移
        shear_range=0.2,  # 剪切变换
        zoom_range=0.2,  # 缩放
        horizontal_flip=True,  # 随机水平翻转
        fill_mode="nearest",  # 填充像素
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # 创建训练数据生成器
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=configs["train_path"],
        x_col="filename",
        y_col="label",
        target_size=(configs["image_size"], configs["image_size"]),
        batch_size=configs["batch_size"],
        class_mode="categorical",
        classes=list(label_encoder.classes_),  # Ensure labels are used as classes
    )

    # 创建测试数据生成器
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=configs["test_path"],
        x_col="filename",
        y_col=None,
        target_size=(configs["image_size"], configs["image_size"]),
        batch_size=configs["batch_size"],
        class_mode=None,  # No labels in the test set
    )

    return train_generator, test_generator



def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


if __name__ == "__main__":
    # Ensure prepare function is called before this script
    from utils.prepare import prepare

    configs, train_data, test_data, model_path, label_encoder = prepare("configs.yaml")
    create_data_generators(configs, label_encoder)
