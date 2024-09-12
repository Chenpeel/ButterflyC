import model
import prepare
import utils.process
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
import random

configs = prepare.load_configs()
json_path = configs.get("classes_path")
work_dir = configs.get("temp_path")
image_size = configs.get("image_size")


def random_test():
    test_time = 30
    valid_path = configs.get("valid_path")
    os.system(f'rm -rf {work_dir}/*')
    os.system(f'cp -r {valid_path}/* {work_dir}/')
    
    rand_n, images, corr_name = [], [], []
    
    # 确保 work_dir 中的每个目录都是合法的
    directories = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    
    if not directories:
        raise ValueError("No directories found in the work directory.")
    
    for i in range(test_time):
        rand_dir = random.choice(directories)
        rand_dir_path = os.path.join(work_dir, rand_dir)
        image_files = [f for f in os.listdir(rand_dir_path) if os.path.isfile(os.path.join(rand_dir_path, f))]
        
        if not image_files:
            raise ValueError(f"No image files found in directory {rand_dir_path}.")
        
        rand_image = random.choice(image_files)
        image_path = os.path.join(rand_dir_path, rand_image)
        
        rand_n.append(rand_dir)
        images.append(image_path)
        corr_name.append(prepare.json2dict(configs.get("classes_path"))[rand_dir][1])
    
    print(rand_n, corr_name, images)
    return rand_n, corr_name, images

def upload():
    return image, path


def evaluation_1_image(path, label):
    global work_dir
    if work_dir != configs.get("temp_path"):
        pass
    with tf.Session() as sess:
        num_classes = configs.get("num_classes")

        # 调用 parse_function 处理图像
        image, label = utils.process.parse_function(path, label, image_size)
        
        # 为模型实例化并进行前向传播
        model_instance = model.RecognizeNet(num_classes=num_classes)
        logits = model_instance(tf.expand_dims(image, 0))  # 增加一个 batch 维度
        logits = tf.nn.softmax(logits)
        
        trained_model_path = configs.get("res_model_path")
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(trained_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        else:
            print("No checkpoint file found")
            return

        # 运行 session 获取预测结果
        prediction = sess.run(logits)
        max_index = np.argmax(prediction)

        result = prepare.json2dict(configs.get("classes_path"))[str(max_index)][1]

    return result



def __main__():
    rand_n, corr_name, images = random_test()
    res = []
    for i in range(len(rand_n)):
        res.append([corr_name[i], evaluation_1_image(images[i], rand_n[i])])
    print("real / predict (category name):")
    for i in res:
        print(i[0], i[1])
    os.system(f"rm -rf {work_dir}/*")

if __name__ == '__main__':
    __main__()