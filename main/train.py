import os
import csv
from main.utils.prepare import prepare,clear_temp_data,configs
from main.utils.process import create_data_generators
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from main.model import ButterflyR


def train_model() -> list:
    # 加载配置
    configs, train_data, test_data, model_path, label_encoder = prepare("configs.yaml")

    # 检查配置
    print(f"Configs: {configs}")
    print(f"Label Encoder Classes: {label_encoder.classes_}")
    print(f"Model path: {model_path}")

    # 创建数据生成器
    train_generator, test_generator = create_data_generators(configs, label_encoder)

    # 检查数据生成器
    print(f"Train generator: {train_generator}")
    print(f"Test generator: {test_generator}")

    # 初始化模型
    input_shape = (configs["image_size"], configs["image_size"], 3)  # 假设RGB图像输入
    num_classes = configs["num_classes"]

    model = ButterflyR(input_shape, num_classes, learning_rate=configs["lr"])
    model.compile()

    # 设置回调函数
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(configs["result_model_path"], "best_model.h5"),
            monitor="accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="accuracy", patience=4, verbose=1),
    ]

    # 训练模型
    history = model.model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=configs["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # 保存最终模型
    final_model_path = os.path.join(configs["result_model_path"], configs["model_name"])
    model.model.save(final_model_path)
    print(f"Model saved at {final_model_path}")

    losses = history.history["loss"]
    accuracies = history.history["accuracy"]
    loss_acc = list(zip(losses, accuracies))
    with open(os.path.join(configs["result_model_path"], "loss_acc.csv"), "w",newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Loss", "Accuracy"])
        writer.writerows(loss_acc)


if __name__ == "__main__":
    loss_acc = train_model()
    clear_temp_data(configs["work_dir"])
