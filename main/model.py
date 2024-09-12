import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_model(num_classes, input_shape=(224, 224, 3), pretrained_weights='imagenet'):
    """
    创建一个基于 ResNet-50 的模型，并添加自定义分类层。
    
    参数:
    - num_classes: 类别数量。
    - input_shape: 输入图像的形状。
    - pretrained_weights: 预训练权重来源，'imagenet' 或 None。
    
    返回:
    - model: 构建的模型。
    """
    # 加载 ResNet-50 基础模型
    base_model = ResNet50(weights=pretrained_weights, include_top=False, input_shape=input_shape)
    
    # 冻结预训练的层
    for layer in base_model.layers:
        layer.trainable = False
    
    # 添加自定义分类层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # 全局平均池化
    x = Dense(1024, activation='relu')(x)  # 自定义全连接层
    x = Dense(num_classes, activation='softmax')(x)  # 分类层
    
    # 定义模型
    model = Model(inputs=base_model.input, outputs=x)
    
    return model
