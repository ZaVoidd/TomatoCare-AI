import os
import argparse
import json
import csv
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from .config import DEFAULT_IMG_SIZE, BATCH_SIZE
from .inference import load_model_and_labels


def _save_misclassified_grid(misclassified: List[dict], out_path: str, max_images: int = 9):
    if not misclassified:
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    subset = misclassified[:max_images]
    cols = min(3, len(subset))
    rows = int(np.ceil(len(subset) / cols))

    plt.figure(figsize=(4 * cols, 4 * rows))
    for idx, sample in enumerate(subset):
        ax = plt.subplot(rows, cols, idx + 1)
        try:
            img = Image.open(sample["image_path"])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"T: {sample['true_class']}\nP: {sample['predicted_class']}", fontsize=10)
        except Exception as exc:  # pragma: no cover - display only
            ax.text(0.5, 0.5, f"Failed to load\n{exc}", ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def evaluate(test_dir: str, model_path: str, label_map: str, img_size=(224, 224), batch_size=32,
             report_path: str = None, cm_path: str = None, plots_dir: str = None,
             report_json: str = None, misclassified_csv: str = None, misclassified_grid: str = None):
    model, class_names = load_model_and_labels(model_path, label_map)

    ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=False
    )

    y_true = []
    y_pred = []
    file_paths = getattr(ds, 'file_paths', None)
    misclassified = []
    sample_idx = 0

    for images, labels in ds:
        batch_paths = None
        if file_paths:
            batch_paths = file_paths[sample_idx: sample_idx + len(labels)]
            sample_idx += len(labels)

        preds = model.predict(images, verbose=0)
        batch_true = labels.numpy().tolist()
        batch_pred = np.argmax(preds, axis=1).tolist()

        y_true.extend(batch_true)
        y_pred.extend(batch_pred)

        if batch_paths:
            for true_cls, pred_cls, path in zip(batch_true, batch_pred, batch_paths):
                if true_cls != pred_cls:
                    misclassified.append({
                        "image_path": path,
                        "true_class": class_names[true_cls],
                        "predicted_class": class_names[pred_cls]
                    })

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    if report_path:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
    if report_json:
        os.makedirs(os.path.dirname(report_json) or ".", exist_ok=True)
        report_dict = classification_report(
            y_true, y_pred, target_names=class_names, digits=4, output_dict=True
        )
        with open(report_json, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if cm_path:
        plt.savefig(cm_path, bbox_inches='tight')
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    if misclassified_csv and misclassified:
        os.makedirs(os.path.dirname(misclassified_csv) or ".", exist_ok=True)
        with open(misclassified_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'true_class', 'predicted_class'])
            writer.writeheader()
            writer.writerows(misclassified)

    if misclassified_grid and misclassified:
        _save_misclassified_grid(misclassified, misclassified_grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--label_map', type=str, required=True)
    parser.add_argument('--img_size', type=int, nargs=2, default=DEFAULT_IMG_SIZE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--report_path', type=str, default=None)
    parser.add_argument('--cm_path', type=str, default=None)
    parser.add_argument('--plots_dir', type=str, default=None)
    parser.add_argument('--report_json', type=str, default=None)
    parser.add_argument('--misclassified_csv', type=str, default=None)
    parser.add_argument('--misclassified_grid', type=str, default=None)

    args = parser.parse_args()

    evaluate(
        args.test_dir,
        args.model_path,
        args.label_map,
        tuple(args.img_size),
        args.batch_size,
        args.report_path,
        args.cm_path,
        args.plots_dir,
        args.report_json,
        args.misclassified_csv,
        args.misclassified_grid
    )
