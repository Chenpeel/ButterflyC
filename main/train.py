import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from main.model import ButterflyC,VGG16M,ResNet50M,DenseNet121M
from main.utils.process import load_data, encode_labels
from main.utils.config import load_config
def check_device(device="CPU"):
    if device=='GPU':
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
        else:
            return
def train(model_name='ButterflyC'):
    configs = load_config()
    n = configs['num_classes']
    size = configs['image_size']
    train_data, val_data = load_data()
    label_encoder = encode_labels(configs['train_csv'])

    if model_name=='ButterflyC':
        butterfly_model = ButterflyC((size, size, 3), num_classes=n)
    elif model_name=='VGG16M':
        butterfly_model = VGG16M((size, size, 3), num_classes=n)
    elif model_name=='ResNet50M':
        butterfly_model = ResNet50M((size, size, 3), num_classes=n)
    elif model_name=='DenseNet121M':
        butterfly_model = DenseNet121M((size, size, 3), num_classes=n)
    else :
        model_name = 'ButterflyC'
        butterfly_model = ButterflyC((size, size, 3), num_classes=n)


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
        os.path.join(configs['model_path'], 'checkpoint.keras'),
        monitor='val_loss',
        save_best_only=True
    )
    csv_logger = CSVLogger(os.path.join(configs['log_dir'], 'csv_logger.csv'))

    history = model.fit(
        train_data,
        epochs=configs['initial_epochs'],
        validation_data=val_data,
        validation_freq=3,
        callbacks=[reduce_lr, checkpoint, csv_logger]
    )
    model.save(os.path.join(configs['model_path'],f'{model_name}-init.keras'))
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
    model.save(os.path.join(configs['model_path'],f'{model_name}.keras'))
    return history, history_fine

if __name__ == "__main__":
    train()
