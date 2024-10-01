import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau
from main.utils.process_img import genDatas
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, EfficientNetB4
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

class br:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        base_model = EfficientNetB4(include_top=False, weights='imagenet', input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_generator, val_generator, class_weights, epochs, callbacks=[]):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        self.model.fit(train_generator, validation_data=val_generator, epochs=epochs, class_weight=class_weights, callbacks=callbacks)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def generate_cam(self, img_array, class_idx):
        # 获取模型的最后一个卷积层
        last_conv_layer = self.model.get_layer('top_conv')

        # 创建一个模型，输入为模型的输入，输出为最后一个卷积层的输出和模型的预测
        grad_model = Model([self.model.inputs], [last_conv_layer.output, self.model.output])

        # 计算梯度
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        # 计算权重
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 生成类激活图
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # 归一化
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

    def visualize_cam(self, img_path, heatmap, alpha=0.4):
        # 加载原始图像
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)

        # 调整热图大小
        heatmap = np.uint8(255 * heatmap)
        heatmap = tf.keras.preprocessing.image.array_to_img(heatmap)
        heatmap = heatmap.resize((img.shape[1], img.shape[0]))
        heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)

        # 叠加热图到原始图像
        superimposed_img = heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        # 显示图像
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.show()

# 对比
class VGG16M:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # 冻结预训练模型的层
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_generator, val_generator, class_weights, epochs, callbacks=[]):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        self.model.fit(train_generator, validation_data=val_generator, epochs=epochs, class_weight=class_weights, callbacks=callbacks)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def generate_cam(self, img_array, class_idx):
        last_conv_layer = self.model.get_layer('block5_conv3')
        grad_model = Model([self.model.inputs], [last_conv_layer.output, self.model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

    def visualize_cam(self, img_path, heatmap, alpha=0.4):
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        heatmap = np.uint8(255 * heatmap)
        heatmap = tf.keras.preprocessing.image.array_to_img(heatmap)
        heatmap = heatmap.resize((img.shape[1], img.shape[0]))
        heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)
        superimposed_img = heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.show()

class ResNet50M:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # 冻结预训练模型的层
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_generator, val_generator, class_weights, epochs, callbacks=[]):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        self.model.fit(train_generator, validation_data=val_generator, epochs=epochs, class_weight=class_weights, callbacks=callbacks)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def generate_cam(self, img_array, class_idx):
        last_conv_layer = self.model.get_layer('conv5_block3_out')
        grad_model = tf.keras.models.Model([self.model.inputs], [last_conv_layer.output, self.model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

    def visualize_cam(self, img_path, heatmap, alpha=0.4):
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        heatmap = np.uint8(255 * heatmap)
        heatmap = tf.keras.preprocessing.image.array_to_img(heatmap)
        heatmap = heatmap.resize((img.shape[1], img.shape[0]))
        heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)
        superimposed_img = heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.show()

class DenseNet121M:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_generator, val_generator, class_weights, epochs, callbacks=[]):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        self.model.fit(train_generator, validation_data=val_generator, epochs=epochs, class_weight=class_weights, callbacks=callbacks)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def generate_cam(self, img_array, class_idx):
        last_conv_layer = self.model.get_layer('conv5_block16_concat')
        grad_model = tf.keras.models.Model([self.model.inputs], [last_conv_layer.output, self.model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

    def visualize_cam(self, img_path, heatmap, alpha=0.4):
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        heatmap = np.uint8(255 * heatmap)
        heatmap = tf.keras.preprocessing.image.array_to_img(heatmap)
        heatmap = heatmap.resize((img.shape[1], img.shape[0]))
        heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)
        superimposed_img = heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.show()
