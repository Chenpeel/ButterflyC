import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from main.utils.prepare import prepare, configs
from main.utils.process import create_data_generators
from tensorflow.keras.models import load_model
from main.model import ButterflyR, VGG16Model, ResNet50Model, DenseNet121Model

def evaluate_model(model_class, layer_name):
    try:
        # 加载配置
        configs, _, test_data, model_path, label_encoder = prepare("configs.yaml")

        # 创建测试数据生成器
        _, _, test_generator = create_data_generators(configs, label_encoder)

        # 加载模型
        model = model_class((configs["image_size"], configs["image_size"], 3), configs["num_classes"])
        model.model = load_model(os.path.join(model_path, configs["model_name"]))

        # 评估模型
        loss, accuracy = model.model.evaluate(test_generator, verbose=1)
        print(f"Test loss: {loss}")
        print(f"Test accuracy: {accuracy}")

        # 获取预测值
        predictions = model.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)

        # 生成分类报告和混淆矩阵
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())
        report = classification_report(true_classes, predicted_classes, target_names=class_labels)
        matrix = confusion_matrix(true_classes, predicted_classes)

        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", matrix)

        # 生成并可视化 CAM
        for i in range(5):  # 只展示前5个样本的CAM
            img, label = test_generator[i]
            img_array = np.expand_dims(img[0], axis=0)
            heatmap = model.generate_cam(img_array, layer_name)
            visualize_cam(img[0], heatmap, label_encoder.classes_[label[0]])

    except Exception as e:
        print(f"An error occurred: {e}")

def visualize_cam(img, heatmap, label):
    # 归一化热图
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = np.repeat(heatmap, 3, axis=-1)

    # 将热图应用于原始图像
    superimposed_img = heatmap * 0.4 + img

    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image - {label}")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"CAM - {label}")
    plt.imshow(superimposed_img)
    plt.axis('off')

    plt.show()

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

    plt.figure(figsize=(10, 5))

    # 绘制训练损失和准确率
    plt.subplot(1, 2, 1)
    plt.plot(loss_acc["Loss"], label="Loss")
    plt.plot(loss_acc["Accuracy"], label="Accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()

    # 绘制验证损失和准确率
    plt.subplot(1, 2, 2)
    plt.plot(loss_acc["Val_Loss"], label="Val Loss")
    plt.plot(loss_acc["Val_Accuracy"], label="Val Accuracy")
    plt.title("Validation Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    for model_class, layer_name in [
        (ButterflyR, "top_conv"),  # EfficientNetB0 的最后一个卷积层名称
        (VGG16Model, "block5_conv3"),  # VGG16 的最后一个卷积层名称
        (ResNet50Model, "conv5_block3_out"),  # ResNet50 的最后一个卷积层名称
        (DenseNet121Model, "conv5_block16_concat")  # DenseNet121 的最后一个卷积层名称
    ]:
        evaluate_model(model_class, layer_name)
        plot_training_loss_acc(
            os.path.join(configs["result_model_path"], "loss_acc.csv")
        )
