"""
Script evaluasi model untuk melihat performa per kelas penyakit.
Jalankan: python evaluate_model.py
"""
import os
import sys
from src.evaluate import evaluate
from src.config import MODELS_DIR, DATA_DIR

def main():
    # Path dataset valid (bisa diganti ke test jika ada)
    valid_dir = os.path.join(
        DATA_DIR, 
        "New Plant Diseases Dataset(Filtered)", 
        "valid"
    )
    
    model_path = os.path.join(MODELS_DIR, "densenet121_best.keras")
    label_map = os.path.join(MODELS_DIR, "label_map.json")
    
    # Pastikan semua file ada
    if not os.path.exists(valid_dir):
        print(f"❌ Folder valid tidak ditemukan: {valid_dir}")
        print("Pastikan path dataset benar!")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan: {model_path}")
        return
    
    if not os.path.exists(label_map):
        print(f"❌ Label map tidak ditemukan: {label_map}")
        return
    
    print("=" * 60)
    print("EVALUASI MODEL - TOMATO DISEASE IDENTIFICATION")
    print("=" * 60)
    print(f"Dataset: {valid_dir}")
    print(f"Model: {model_path}")
    print(f"Label Map: {label_map}")
    print("=" * 60)
    print("\nMemulai evaluasi... (ini mungkin memakan waktu beberapa menit)\n")
    
    # Jalankan evaluasi dengan output lengkap
    evaluate(
        test_dir=valid_dir,
        model_path=model_path,
        label_map=label_map,
        img_size=(192, 192),  # Sesuai dengan training
        batch_size=32,
        report_path=os.path.join(MODELS_DIR, "classification_report.txt"),
        cm_path=os.path.join(MODELS_DIR, "confusion_matrix.png"),
        plots_dir=MODELS_DIR,
        report_json=os.path.join(MODELS_DIR, "classification_report.json"),
        misclassified_csv=os.path.join(MODELS_DIR, "misclassified.csv"),
        misclassified_grid=os.path.join(MODELS_DIR, "misclassified_samples.png"),
    )
    
    print("\n" + "=" * 60)
    print("EVALUASI SELESAI!")
    print("=" * 60)
    print("\nFile hasil evaluasi:")
    print(f"  - Classification Report: {os.path.join(MODELS_DIR, 'classification_report.txt')}")
    print(f"  - Classification Report (JSON): {os.path.join(MODELS_DIR, 'classification_report.json')}")
    print(f"  - Confusion Matrix: {os.path.join(MODELS_DIR, 'confusion_matrix.png')}")
    print(f"  - Misclassified Samples: {os.path.join(MODELS_DIR, 'misclassified.csv')}")
    print(f"  - Misclassified Grid: {os.path.join(MODELS_DIR, 'misclassified_samples.png')}")
    print("\nBuka file-file di atas untuk melihat detail performa model per kelas!")

if __name__ == "__main__":
    main()

