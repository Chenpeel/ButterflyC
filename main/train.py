import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from main.model import ButterflyR,VGG16M,ResNet50M,DenseNet121M
from main.utils.process import load_data, encode_labels
from main.utils.config import load_config

def train(model_name='ButterflyR'):
    configs = load_config()
    n = configs['num_classes']
    size = configs['image_size']
    train_data, val_data = load_data()
    label_encoder = encode_labels(configs['train_csv'])

    if model_name=='ButterflyR':
        butterfly_model = ButterflyR((size, size, 3), num_classes=n)
    elif model_name=='VGG16M':
        butterfly_model = VGG16M((size, size, 3), num_classes=n)
    elif model_name=='ResNet50M':
        butterfly_model = ResNet50M((size, size, 3), num_classes=n)
    elif model_name=='DenseNet121M':
        butterfly_model = DenseNet121M((size, size, 3), num_classes=n)
    else :
        butterfly_model = ButterflyR((size, size, 3), num_classes=n)

    model = butterfly_model.build_model()


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.8,
        patience=5,
        min_lr=0.00005
    )
    checkpoint = ModelCheckpoint(
        os.path.join(configs['model_path'], 'br.keras'),
        monitor='val_loss',
        save_best_only=True
    )
    csv_logger = CSVLogger(os.path.join(configs['log_dir'], 'training_log.csv'))

    history = model.fit(
        train_data,
        epochs=configs['initial_epochs'],
        validation_data=val_data,
        validation_freq=3,
        callbacks=[reduce_lr, checkpoint, csv_logger]
    )

    model = butterfly_model.unfreeze_base_model(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs['fine_tuning_learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_data,
        epochs=configs['fine_tuning_epochs'],
        validation_data=val_data,
        validation_freq=3,
        callbacks=[reduce_lr, checkpoint, csv_logger]
    )

    return history, history_fine

if __name__ == "__main__":
    train()
