import os
import argparse
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

try:
    import keras_tuner as kt
    HAS_TUNER = True
except Exception:
    HAS_TUNER = False

from .config import DEFAULT_IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, MODEL_PATH, LABEL_MAP_PATH, MODELS_DIR
from .utils import set_seed, save_label_map, plot_training


def build_model(num_classes: int, img_size=(224, 224), learning_rate: float = LEARNING_RATE, dropout: float = 0.3):
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = layers.RandomFlip('horizontal')(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model, base_model


def prepare_datasets(train_dir, val_dir, img_size, batch_size):
    train_ds = image_dataset_from_directory(
        train_dir,
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )
    val_ds = image_dataset_from_directory(
        val_dir,
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def _run_tuner(num_classes, img_size, train_ds, val_ds, output_dir, trials):
    if not HAS_TUNER:
        raise RuntimeError("keras-tuner is not installed. Install it or run without --tune")

    def model_builder(hp):
        hp_lr = hp.Choice('learning_rate', values=[1e-4, 3e-4, 1e-3])
        hp_dropout = hp.Choice('dropout', values=[0.2, 0.3, 0.5])
        model, base_model = build_model(num_classes, img_size, learning_rate=hp_lr, dropout=hp_dropout)
        return model

    tuner = kt.RandomSearch(
        hypermodel=model_builder,
        objective='val_accuracy',
        max_trials=trials,
        executions_per_trial=1,
        directory=os.path.join(output_dir, 'tuner'),
        project_name='densenet121_tuning'
    )

    tuner.search(train_ds, validation_data=val_ds, epochs=8)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    return {
        'learning_rate': best_hp.get('learning_rate'),
        'dropout': best_hp.get('dropout')
    }


def train(train_dir, val_dir, output_dir=MODELS_DIR, img_size=DEFAULT_IMG_SIZE, batch_size=BATCH_SIZE,
          epochs=EPOCHS, learning_rate=LEARNING_RATE, tune=False, tune_trials=10):
    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)

    train_ds, val_ds, class_names = prepare_datasets(train_dir, val_dir, img_size, batch_size)
    num_classes = len(class_names)

    tuned = None
    if tune:
        tuned = _run_tuner(num_classes, img_size, train_ds, val_ds, output_dir, tune_trials)
        learning_rate = float(tuned['learning_rate'])
        print(f"[TUNER] Best hyperparameters -> lr={learning_rate}, dropout={tuned['dropout']}")

    model, base_model = build_model(num_classes, img_size, learning_rate=learning_rate, dropout=float(tuned['dropout']) if tuned else 0.3)

    ckpt_path = os.path.join(output_dir, 'densenet121_best.h5')
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # Fine-tune: unfreeze top layers of base model; jumlah layer yang di-unfreeze bisa dipengaruhi hasil tuning di masa depan
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate * 0.1),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history_ft = model.fit(train_ds, validation_data=val_ds, epochs=max(5, epochs//2), callbacks=callbacks)

    # Save label map and plots
    save_label_map(class_names, os.path.join(output_dir, 'label_map.json'))
    plot_training(history, output_dir)
    plot_training(history_ft, output_dir)

    print(f"Training complete. Best model saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=MODELS_DIR)
    parser.add_argument('--img_size', type=int, nargs=2, default=DEFAULT_IMG_SIZE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--tune_trials', type=int, default=10)

    args = parser.parse_args()

    train(args.train_dir, args.val_dir, args.output_dir, tuple(args.img_size), args.batch_size, args.epochs, args.learning_rate, args.tune, args.tune_trials)
