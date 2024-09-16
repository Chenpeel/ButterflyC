import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from main.utils.prepare import prepare, configs
from main.utils.process import create_data_generators
from tensorflow.keras.models import load_model


def evaluate_model():
    # test 无 label 卒😭
    return
    # 加载配置
    configs, _, test_data, model_path, label_encoder = prepare("configs.yaml")

    # 创建测试数据生成器
    _, test_generator = create_data_generators(configs, label_encoder)

    # 加载模型
    model = load_model(os.path.join(model_path, configs["model_name"]))

    # 评估模型
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")

    # 获取预测值
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)


def plot_training_loss_acc(loss_acc_path):
    # 检查文件是否存在
    if not os.path.exists(loss_acc_path):
        print("Loss and accuracy file not found.")
        return

    # 读取 CSV 文件
    try:
        loss_acc = pd.read_csv(loss_acc_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # 检查数据
    print("loss and accuracy:", loss_acc.head())

    plt.figure(figsize=(5, 5))
    plt.plot(loss_acc["Loss"], label="Loss")
    # plt.plot(loss_acc["Accuracy"], label="Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # evaluate_model()
    plot_training_loss_acc(
        os.path.join(configs["result_model_path"], configs["loss_acc"])
    )
