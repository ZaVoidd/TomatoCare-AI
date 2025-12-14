import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_label_map(class_names: List[str], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, ensure_ascii=False, indent=2)


def load_label_map(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    # Ensure ordered by index
    return [mapping[str(i)] if isinstance(list(mapping.keys())[0], str) else mapping[i] for i in range(len(mapping))]


def plot_training(history, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    plt.close()

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()
