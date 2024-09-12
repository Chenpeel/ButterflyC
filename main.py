import matplotlib.pyplot as plt
import train
def show_plot(history):
    plt.plot(history.history['loss'], 'r', label='loss')
    plt.plot(history.history['sparse_categorical_accuracy'], 'g', label='accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.show()

history = main.train.history
show_plot(history)