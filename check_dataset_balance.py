"""
Script untuk mengecek keseimbangan dataset per kelas.
Jalankan: python check_dataset_balance.py
"""
import os
import sys
from collections import Counter
from src.config import DATA_DIR

# Force output to utf-8 if possible, or just avoid unicode
sys.stdout.reconfigure(encoding='utf-8')

def check_dataset_balance():
    """Cek jumlah gambar per kelas di train dan valid."""
    base_dir = os.path.join(DATA_DIR, "New Plant Diseases Dataset(Augmented)")
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")
    
    if not os.path.exists(train_dir):
        print(f"Error: Folder train tidak ditemukan: {train_dir}")
        return
    
    print("=" * 70)
    print("ANALISIS KESEIMBANGAN DATASET")
    print("=" * 70)
    
    # Cek train
    print("\n[TRAIN SET]:")
    print("-" * 70)
    train_counts = {}
    total_train = 0
    
    for class_name in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            train_counts[class_name] = count
            total_train += count
            print(f"  {class_name:50} {count:6} gambar")
    
    print("-" * 70)
    print(f"  {'TOTAL':50} {total_train:6} gambar")
    
    # Cek valid
    print("\n[VALID SET]:")
    print("-" * 70)
    valid_counts = {}
    total_valid = 0
    
    if os.path.exists(valid_dir):
        for class_name in sorted(os.listdir(valid_dir)):
            class_path = os.path.join(valid_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                valid_counts[class_name] = count
                total_valid += count
                print(f"  {class_name:50} {count:6} gambar")
    
    print("-" * 70)
    print(f"  {'TOTAL':50} {total_valid:6} gambar")
    
    # Analisis keseimbangan
    print("\n" + "=" * 70)
    print("ANALISIS KESEIMBANGAN")
    print("=" * 70)
    
    if train_counts:
        min_count = min(train_counts.values())
        max_count = max(train_counts.values())
        avg_count = sum(train_counts.values()) / len(train_counts)
        
        print(f"\n[STATISTIK TRAIN SET]:")
        print(f"  Minimum: {min_count} gambar")
        print(f"  Maximum: {max_count} gambar")
        print(f"  Rata-rata: {avg_count:.1f} gambar")
        print(f"  Rasio min/max: {min_count/max_count:.2%}")
        
        # Identifikasi kelas yang tidak seimbang
        print(f"\n[!] Kelas yang perlu perhatian (kurang dari 80% rata-rata):")
        imbalanced = []
        for class_name, count in train_counts.items():
            if count < avg_count * 0.8:
                imbalanced.append((class_name, count, avg_count))
                print(f"  - {class_name}: {count} gambar (rata-rata: {avg_count:.0f})")
        
        if not imbalanced:
            print("  [OK] Semua kelas cukup seimbang!")
        
        # Rekomendasi
        print(f"\n[REKOMENDASI]:")
        if min_count / max_count < 0.7:
            print("  [WARNING] Dataset TIDAK SEIMBANG!")
            print("  -> Gunakan class weights saat training")
            print("  -> Atau tambahkan data augmentation lebih agresif untuk kelas minoritas")
        else:
            print("  [OK] Dataset cukup seimbang")
            print("  -> Bisa menggunakan class weights opsional")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_dataset_balance()

