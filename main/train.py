import os
import csv
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 确保正确导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main.utils.prepare import prepare, clear_temp_data, configs
from main.utils.process import create_data_generators
from main.model import ButterflyR, VGG16Model, ResNet50Model, DenseNet121Model

def train_model(model_class) -> list:
    try:
        # 加载配置
        configs, train_data, test_data, model_path, label_encoder = prepare("configs.yaml")

        # 设置设备
        if configs["device"] == "GPU":
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("Using GPU")
            else:
                print("No GPU found, using CPU")
        else:
            print("Using CPU")

        # 检查配置
        print(f"Configs: {configs}")
        print(f"Label Encoder Classes: {label_encoder.classes_}")
        print(f"Model path: {model_path}")

        # 创建数据生成器
        train_generator, val_generator, test_generator = create_data_generators(configs, label_encoder)

        # 检查数据生成器
        print(f"Train generator: {train_generator}")
        print(f"Validation generator: {val_generator}")
        print(f"Test generator: {test_generator}")

        # 初始化模型
        input_shape = (configs["image_size"], configs["image_size"], 3)  # 假设RGB图像输入
        num_classes = configs["num_classes"]

        model = model_class(input_shape, num_classes, learning_rate=configs["lr"])
        model.compile()

        # 设置回调函数
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(configs["result_model_path"], "best_model.h5"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            EarlyStopping(monitor="val_accuracy", patience=4, verbose=1),
        ]

        # 训练模型
        history = model.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=configs["epochs"],
            callbacks=callbacks,
            verbose=1,
        )

        # 保存最终模型
        final_model_path = os.path.join(configs["result_model_path"], model_class.__name__ + configs["model_name"])
        model.model.save(final_model_path)
        print(f"Model saved at {final_model_path}")

        # 保存训练和验证的损失和准确率
        losses = history.history["loss"]
        accuracies = history.history["accuracy"]
        val_losses = history.history["val_loss"]
        val_accuracies = history.history["val_accuracy"]
        loss_acc = list(zip(losses, accuracies, val_losses, val_accuracies))
        with open(os.path.join(configs["result_model_path"], "loss_acc.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Loss", "Accuracy", "Val_Loss", "Val_Accuracy"])
            writer.writerows(loss_acc)

        return loss_acc

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    for model_class in [ButterflyR, VGG16Model, ResNet50Model, DenseNet121Model]:
        loss_acc = train_model(model_class)
        print(loss_acc)
        clear_temp_data(configs["work_dir"])
