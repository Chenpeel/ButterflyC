import os
import shutil
import yaml
import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# 加载配置文件函数
def load_configs(config_file):
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)
    return configs


# 验证目录存在性，若不存在则创建
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 复制数据集到工作目录
def copy_dataset_to_work_dir(src_dir, dest_dir):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)


# 加载CSV并创建图片路径-标签映射
def load_dataset(csv_path, data_dir, is_test=False):
    df = pd.read_csv(csv_path)
    print(f"CSV Columns for {csv_path}:", df.columns)

    image_paths = [os.path.join(data_dir, filename) for filename in df["filename"]]

    if not is_test:
        # 训练集需要标签
        if "label" not in df.columns:
            raise KeyError("找不到标签列，请检查 CSV 文件的列名。")
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df["label"])
        return list(zip(image_paths, labels)), label_encoder
    else:
        # 测试集不需要标签
        return image_paths


# 准备函数
def prepare(config_file):
    # 加载配置
    configs = load_configs(config_file)

    # 处理路径中的变量
    dataset = configs["dataset"]
    train_csv = configs["train_csv"]
    test_csv = configs["test_csv"]

    configs["train_path"] = configs["train_path"].replace("{{dataset}}", dataset)
    configs["test_path"] = configs["test_path"].replace("{{dataset}}", dataset)
    configs["train_csv_path"] = (
        configs["train_csv_path"]
        .replace("{{dataset}}", dataset)
        .replace("{{train_csv}}", train_csv)
    )
    configs["test_csv_path"] = (
        configs["test_csv_path"]
        .replace("{{dataset}}", dataset)
        .replace("{{test_csv}}", test_csv)
    )

    # 验证所需目录是否存在，否则创建
    ensure_dir(configs["work_dir"])
    ensure_dir(configs["uploaded_picture"])
    ensure_dir(configs["result_model_path"])
    ensure_dir(configs["log_path"])

    # 将数据集复制到工作目录
    copy_dataset_to_work_dir(
        os.path.join("main", "data", dataset), configs["temp_data"]
    )

    # 读取CSV，生成图片路径-标签映射
    train, label_encoder = load_dataset(
        configs["train_csv_path"], configs["train_path"]
    )
    test = load_dataset(configs["test_csv_path"], configs["test_path"], is_test=True)

    # 返回配置、训练集、测试集、模型路径、以及标签编码器
    return configs, train, test, configs["result_model_path"], label_encoder

def clear_temp_data(work_dir):
    if os.path.exists(work_dir):
        if os.path.exists(configs["temp_data"]):
            shutil.rmtree(configs["temp_data"])


configs, train, test, model_path, label_encoder = prepare("configs.yaml")
if __name__ == "__main__":
    # 打印检查
    print("Configs:", configs)
    print("Train samples:", len(train))
    print("Test samples:", len(test))
    print("Model Path:", model_path)
    print("Label classes:", label_encoder.classes_)
