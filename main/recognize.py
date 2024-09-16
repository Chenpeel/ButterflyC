import os
import train
import main.model
import main.utils.prepare
import main.utils.process
import main.utils.eval
import numpy as np
import tensorflow as tf

configs = main.utils.prepare.configs


def get_uploaded_pic():
    uploaded_path = configs["uploaded_picture"]
    uploaded_pics = []
    for pic in os.listdir(uploaded_path):
        full_path = os.path.join(uploaded_path, pic)
        uploaded_pics.append(full_path)
    return uploaded_pics


def load_model():
    model_path = os.path.join(configs["result_model_path"], configs["model_name"])
    loss_acc_path = os.path.join(configs["result_model_path"], configs["loss_acc"])
    if not os.path.exists(model_path):
        train.train_model()
        main.utils.eval.plot_training_loss_acc(loss_acc_path)
    else:
        main.utils.eval.plot_training_loss_acc(loss_acc_path)
    model = tf.keras.models.load_model(model_path)

    return model


def recognize_list():
    uploaded_pics = get_uploaded_pic()
    model = load_model()
    for pic in uploaded_pics:
        pc_pic = main.utils.process.preprocess_image(
            pic,
            (configs["image_size"], configs["image_size"]),
        )
        predictions = model.predict(pc_pic)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_label = main.utils.prepare.label_encoder.inverse_transform(
            predicted_classes
        )
        print(f"{pic} ---- {predicted_label}")


def recognize(pic_path):
    uploaded_pic = get_uploaded_pic()  # 获取上传的图片路径
    model = load_model()  # 加载模型

    if pic_path:
        result = []
        result.append(uploaded_pic[0])  # 第一个元素是上传的图片路径

        pc_pic = main.utils.process.preprocess_image(
            pic_path,
            (configs["image_size"], configs["image_size"]),
        )

        predictions = main.model.predict(pc_pic)
        predicted_classes = np.argmax(predictions, axis=1)

        # 转换为 Python 列表，确保返回结果可以被 JSON 序列化
        recognized_label = main.utils.prepare.label_encoder.inverse_transform(
            predicted_classes
        ).tolist()

        # 将标签追加到结果中
        result.append(recognized_label)

        print(f"{result[0]} ---- {result[1]}")
        return result
    else:
        print("No selected file")
        return [None, "default"]


if __name__ == "__main__":
    recognize_list()
