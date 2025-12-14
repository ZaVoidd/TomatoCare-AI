import os
import json
import numpy as np
import tensorflow as tf
from typing import Tuple, List
from .preprocess import load_and_preprocess


def load_model_and_labels(model_path: str, label_map_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Label map not found at {label_map_path}")

    model = tf.keras.models.load_model(model_path)
    with open(label_map_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    # Convert to ordered class names list
    class_names = [mapping[str(i)] if str(i) in mapping else mapping[i] for i in range(len(mapping))]
    return model, class_names


def predict_image(model, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Tuple[str, float, np.ndarray]:
    img = load_and_preprocess(image_path, target_size=target_size)
    x = np.expand_dims(img, axis=0)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    return str(idx), conf, preds  # idx string will be mapped by caller if needed
