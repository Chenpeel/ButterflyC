import os
import gc
import numpy as np
import tensorflow as tf
import sys
import shutil
import psutil
from tensorflow.keras.callbacks import CSVLogger
from main.model import br, VGG16M, ResNet50M, DenseNet121M
from main.utils.process_img import genDatas, process
from main.utils.config import load_config

configs = load_config()

def main(model_name):
    # 设置参数
    image_size = configs['image_size']
    input_shape = (image_size, image_size, 3)  # 输入图像的形状
    num_classes = configs['num_classes']
    epochs = configs['epochs']
    batch_size = configs['batch_size']
    X_train, y_train, X_val, y_val, X_test, y_test, class_weights, temp_dir = genDatas()

    if model_name == 'br':
        model = br(input_shape, num_classes)
    elif model_name == 'VGG16':
        model = VGG16M(input_shape, num_classes)
    elif model_name == 'ResNet50':
        model = ResNet50M(input_shape, num_classes)
    elif model_name == 'DenseNet121':
        model = DenseNet121M(input_shape, num_classes)
    else:
        raise ValueError("Invalid model choice")

    # 设置日志目录和CSVLogger回调
    log_dir = configs['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    csv_logger = CSVLogger(os.path.join(log_dir, model_name + '.csv'))

    train_dataset = tf.data.Dataset.from_generator(
        lambda: process(X_train, y_train, batch_size=batch_size, size=(image_size, image_size)),
        output_signature=(
            tf.TensorSpec(shape=(None, image_size, image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).batch(batch_size)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: process(X_val, y_val, batch_size=batch_size, size=(image_size, image_size)),
        output_signature=(
            tf.TensorSpec(shape=(None, image_size, image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).batch(batch_size)

    # 训练模型
    model.train(
        train_dataset,
        val_dataset,
        class_weights,
        epochs=epochs,
        callbacks=[csv_logger]
    )

    # 评估模型
    if y_test is not None:
        test_dataset = tf.data.Dataset.from_generator(
            lambda: process(X_test, y_test, batch_size=batch_size, size=(image_size, image_size)),
            output_signature=(
                tf.TensorSpec(shape=(None, image_size, image_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        ).batch(batch_size)
        test_loss, test_accuracy = model.evaluate(test_dataset)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    else:
        print("No test labels provided, skipping evaluation on test set.")

    # 保存模型
    model.model.save(os.path.join(configs['model_path'], model_name + configs['model_suffix']))

    # 清理临时目录
    clean_temp_dir(temp_dir)
    # 清理不必要的变量
    del model
    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()

def clean_temp_dir(temp_dir):
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    for model_name in ['br', 'VGG16', 'ResNet50', 'DenseNet121']:
        main(model_name)
