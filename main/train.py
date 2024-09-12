import tensorflow as tf
import os
import prepare
from utils.process import get_batch  # 确保正确导入 get_batch
from model import create_model  # 确保正确导入 create_model

def create_directories(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def copy_train_data(src, dest):
    os.system(f'cp -r {src}/* {dest}/')
    if os.name == 'posix':
        os.system(f'rm -rf {dest}/.DS_Store')

# --------------------------------------------
# 加载配置文件内容
configs = prepare.load_configs()
trainData_dir = configs.get("train_path")
work_dir = configs.get("temp_path")
json_path = configs.get("classes_path")
log_path = configs.get("log_path")
model_path = configs.get("res_model_path")
create_directories([log_path, work_dir, model_path])
copy_train_data(trainData_dir, work_dir)

batch_size = configs.get("batch_size")
num_classes = configs.get("num_classes")
image_size = configs.get("image_size")
lr = configs.get("lr")
epochs = configs.get("epochs")
precision = configs.get("precision")
capacity = configs.get("capacity")

if image_size is None:
    raise ValueError("image_size cannot be None")

# --------------------------------------------
# 准备数据
train_images, train_labels = prepare.label_files(work_dir, json_path, trainData_dir)
train_dataset = get_batch(train_images, train_labels, image_size, batch_size, capacity, num_augments=3)

# --------------------------------------------
# 定义模型
model = create_model(num_classes=num_classes, input_shape=(image_size, image_size, 3))

# --------------------------------------------
# 配置训练过程
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# 创建检查点对象
checkpoint = tf.train.Checkpoint(model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, model_path, max_to_keep=3)

# 创建回调函数
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_path),
    # Keras ModelCheckpoint does not support `.ckpt` format
]

# 训练模型
history = model.fit(train_dataset, epochs=epochs, callbacks=callbacks, steps_per_epoch=len(train_images) // batch_size)

# 保存检查点
checkpoint_manager.save()
