import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, VGG16, ResNet50, DenseNet121
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

class ButterflyR:
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        base_model = EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs=base_model.input, outputs=output)
        return model

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )


    def generate_cam(self, img_array, layer_name):
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(layer_name).output, self.model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, tf.argmax(predictions[0])]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.linalg.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

# 对比模型
class VGG16Model:
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        base_model = VGG16(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs=base_model.input, outputs=output)
        return model

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def generate_cam(self, img_array, layer_name):
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(layer_name).output, self.model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, tf.argmax(predictions[0])]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.linalg.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

class ResNet50Model:
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        base_model = ResNet50(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs=base_model.input, outputs=output)
        return model

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def generate_cam(self, img_array, layer_name):
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(layer_name).output, self.model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, tf.argmax(predictions[0])]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.linalg.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

class DenseNet121Model:
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        base_model = DenseNet121(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation="softmax")(x)

        model = models.Model(inputs=base_model.input, outputs=output)
        return model

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def generate_cam(self, img_array, layer_name):
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(layer_name).output, self.model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, tf.argmax(predictions[0])]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.linalg.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
