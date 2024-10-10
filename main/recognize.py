import os
import sys
import main.train as train
import main.model as model
import main.utils.config as config
import main.utils.process as process
import main.utils.eval as eval
import numpy as np
import tensorflow as tf

configs = config.load_config()


def get_uploaded_pic():
    uploaded_path = configs["upload_dir"]
    uploaded_pics = []
    for pic in os.listdir(uploaded_path):
        full_path = os.path.join(uploaded_path, pic)
        uploaded_pics.append(full_path)
    return uploaded_pics


def load_model():
    model_path= configs['model_path']
    bc = os.path.join(model_path,'ButterflyC.keras')
    if not os.path.exists(bc):
        print('Train model first!')
    model = tf.keras.models.load_model(bc)
    return model


def recognize(pic_path):
    uploaded_pic = get_uploaded_pic()  # 获取上传的图片路径
    model = load_model()  # 加载模型
    label_encoder  = process.encode_labels(configs['train_csv'])

    if pic_path:
        result = []
        result.append(uploaded_pic[0])

        pic_array = process.process_img(
            pic_path,
            (configs["image_size"], configs["image_size"]),
        )

        predictions = model.predict(pic_array)
        predicted_classes = np.argmax(predictions, axis=1)

        # 转换为 Python 列表，确保返回结果可以被 JSON 序列化
        recognized_label = label_encoder.inverse_transform(
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
    path = "data/test/Image_10.jpg"
    recognize(path)
