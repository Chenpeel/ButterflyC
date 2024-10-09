import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import main.utils.config as config

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.show()

def display_cam(model, img_path, layer_name='top_conv'):
    img_array = process_img(img_path, (configs['image_size'], configs['image_size']))
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer(layer_name)
    grads = tf.keras.backend.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))
    iterate = tf.keras.backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img_array])
    for i in range(pooled_grads_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()

if __name__ == "__main__":
    configs = config.load_config()
    model = tf.keras.models.load_model(configs['model_path'] + '/br.keras')
    plot_history(train_model())
    display_cam(model, 'app/templates/static/butterfly.jpg')
