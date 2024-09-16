import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model


class ButterflyR:
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        """
        构建DenseNet121模型
        """
        base_model = DenseNet121(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )

        # 冻结前几层
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output

        # CAM层
        x = layers.GlobalAveragePooling2D()(x)
        output = layers.Dense(self.num_classes, activation="softmax")(x)

        # 创建模型
        model = models.Model(inputs=base_model.input, outputs=output)
        plot_model(
            model,
            to_file="model_structure.png",
            show_shapes=True,
            show_layer_names=True,
        )
        return model

    def compile(self):
        """
        编译模型
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
