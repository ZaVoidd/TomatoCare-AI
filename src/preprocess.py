import cv2
import numpy as np
from typing import Tuple


def preprocess_image_bgr(img_bgr: np.ndarray, target_size: Tuple[int, int] = (224, 224),
                          use_clahe: bool = False) -> np.ndarray:
    # Resize
    img = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    # Optional CLAHE on L-channel in LAB
    if use_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Normalize 0-1 and convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


def load_and_preprocess(path: str, target_size=(224, 224), use_clahe=False) -> np.ndarray:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Unable to read image at {path}")
    img = preprocess_image_bgr(img_bgr, target_size=target_size, use_clahe=use_clahe)
    return img
