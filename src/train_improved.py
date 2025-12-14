"""
Versi improved dari train.py dengan implementasi rekomendasi perbaikan:
- Class weights untuk handle imbalance
- Augmentation lebih agresif
- Callback lebih optimal
- Fine-tuning bertahap
"""
import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.utils.class_weight import compute_class_weight

from .config import DEFAULT_IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, MODELS_DIR
from .utils import set_seed, save_label_map, plot_training


def build_model_improved(num_classes: int, img_size=(192, 192), learning_rate: float = LEARNING_RATE, dropout: float = 0.3):
    """Build model dengan augmentation lebih agresif."""
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    
    # SUPER AUGMENTATION PIPELINE (UPDATED)
    # Tujuannya: Membuat model "Tangguh" terhadap variasi foto user di lapangan
    x = layers.RandomFlip('horizontal_and_vertical')(inputs)
    x = layers.RandomRotation(0.25)(x)     # 0.25 = 90 derajat (Sangat aman untuk daun)
    x = layers.RandomZoom(0.2)(x)          # 20% zoom in/out (Simulasi jarak kamera)
    x = layers.RandomTranslation(0.1, 0.1)(x) # 10% geser (Simulasi posisi tidak tengah)
    x = layers.RandomBrightness(0.2)(x)    # 20% kecerahan (Simulasi siang/sore/mendung)
    x = layers.RandomContrast(0.4)(x)      # 40% kontras (Pertegas tekstur/bercak)
    
    # Note: Hue/Saturation augmentations bisa ditambahkan jika menggunakan tf.image
    # tapi untuk stabilitas training di Keras Layer, brightness & contrast biasanya cukup.
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model, base_model


def compute_class_weights_from_dataset(train_ds, class_names):
    """Hitung class weights dengan penalty SPESIFIK untuk penyakit yang sulit dideteksi."""
    print("Menghitung class weights (Targeted approach)...")
    all_labels = []
    for _, labels in train_ds.unbatch():
        all_labels.append(labels.numpy())
    
    unique_classes = np.unique(all_labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=all_labels
    )
    
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights)}
    
    # --- Custom Bias Correction ---
    print("  [INFO] Adjusting weights for critical diseases...")
    
    for idx, name in enumerate(class_names):
        name_lower = name.lower()
        
        # 1. Healthy: Biarkan normal (1.0x multiplier) atau sedikit dikurangi relativenya
        if "healthy" in name_lower:
            pass 
            
        # 2. All Diseases (Bercak Target, Hawar Daun, Bakteri, Virus)
        # Samaratakan hukuman 8x lipat agar AI SANGAT agresif pada SEMUA penyakit
        else:
            print(f"    -> Boosting PRIORITY for {name} (8.0x)")
            class_weight_dict[idx] *= 8.0
            
    # ----------------------------------

    print("Class weights (Final Adjusted):")
    for cls, weight in sorted(class_weight_dict.items()):
        name = class_names[cls] if cls < len(class_names) else str(cls)
        print(f"  Class {cls} ({name}): {weight:.3f}")
    
    return class_weight_dict


def train_improved(train_dir, val_dir, output_dir=MODELS_DIR, img_size=DEFAULT_IMG_SIZE, 
                   batch_size=BATCH_SIZE, epochs=20, learning_rate=LEARNING_RATE, use_class_weights=True):
    """Training dengan perbaikan."""
    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare datasets
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
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    # Compute class weights jika diperlukan
    class_weight_dict = None
    if use_class_weights:
        class_weight_dict = compute_class_weights_from_dataset(train_ds, class_names)

    # Build model
    model, base_model = build_model_improved(num_classes, img_size, learning_rate, dropout=0.3)

    ckpt_path = os.path.join(output_dir, 'densenet121_best.keras')
    
    # Callbacks yang lebih optimal (Updated Patience)
    callbacks = [
        ModelCheckpoint(
            ckpt_path, 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy', 
            patience=10,  # Naikkan dari 7 ke 10
            mode='max', 
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5,  # Naikkan dari 4 ke 5
            verbose=1,
            min_lr=1e-6
        ),
        CSVLogger(os.path.join(output_dir, 'training_log.csv'))
    ]

    print(f"\n{'='*60}")
    print("TRAINING PHASE 1: Transfer Learning (Base Model Frozen)")
    print(f"{'='*60}\n")

    # Phase 1: Training dengan base model frozen
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=epochs, 
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    print(f"\n{'='*60}")
    print("TRAINING PHASE 2: Fine-tuning (Unfreeze 30 layers terakhir)")
    print(f"{'='*60}\n")

    # Phase 2: Fine-tuning bertahap - Unfreeze 30 layer terakhir
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate * 0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_ft1 = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=5, 
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    print(f"\n{'='*60}")
    print("TRAINING PHASE 3: Fine-tuning (Unfreeze 20 layers terakhir)")
    print(f"{'='*60}\n")

    # Phase 3: Fine-tuning lebih dalam - Unfreeze 20 layer terakhir
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate * 0.005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_ft2 = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=5, 
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Save label map and plots
    save_label_map(class_names, os.path.join(output_dir, 'label_map.json'))
    plot_training(history, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete. Best model saved to {ckpt_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Improved training dengan class weights dan fine-tuning bertahap')
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=MODELS_DIR)
    parser.add_argument('--img_size', type=int, nargs=2, default=DEFAULT_IMG_SIZE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--no_class_weights', action='store_true', help='Disable class weights')

    args = parser.parse_args()

    train_improved(
        args.train_dir, 
        args.val_dir, 
        args.output_dir, 
        tuple(args.img_size), 
        args.batch_size, 
        args.epochs, 
        args.learning_rate,
        use_class_weights=not args.no_class_weights
    )

