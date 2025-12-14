"""
Script untuk memfilter dataset menjadi 5-6 kelas saja.
Jalankan: python filter_dataset.py
"""
import os
import shutil
from pathlib import Path
from src.config import DATA_DIR

# REKOMENDASI: 5 Kelas Paling Penting
SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
]

# OPSI: Jika mau 6 kelas, uncomment salah satu di bawah:
# SELECTED_CLASSES.append("Tomato___Early_blight")
# SELECTED_CLASSES.append("Tomato___Leaf_Mold")


def filter_dataset():
    """Filter dataset menjadi hanya kelas yang dipilih."""
    base_dir = Path(DATA_DIR) / "New Plant Diseases Dataset(Augmented)"
    train_dir = base_dir / "train"
    valid_dir = base_dir / "valid"
    
    # Buat folder baru untuk dataset yang sudah difilter
    filtered_base = Path(DATA_DIR) / "New Plant Diseases Dataset(Filtered)"
    filtered_train = filtered_base / "train"
    filtered_valid = filtered_base / "valid"
    
    filtered_train.mkdir(parents=True, exist_ok=True)
    filtered_valid.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FILTER DATASET - Hanya 5-6 Kelas Terpilih")
    print("=" * 70)
    print(f"\n📋 Kelas yang dipilih ({len(SELECTED_CLASSES)} kelas):")
    for i, cls in enumerate(SELECTED_CLASSES, 1):
        print(f"  {i}. {cls}")
    
    print(f"\n📁 Folder output: {filtered_base}")
    print("\n⏳ Memulai filtering...\n")
    
    # Filter train set
    print("📂 Filtering TRAIN set...")
    train_count = 0
    for class_name in SELECTED_CLASSES:
        src_path = train_dir / class_name
        dst_path = filtered_train / class_name
        
        if src_path.exists():
            # Copy seluruh folder
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            
            count = len(list(dst_path.glob("*.jpg"))) + len(list(dst_path.glob("*.JPG")))
            train_count += count
            print(f"  ✅ {class_name}: {count} gambar")
        else:
            print(f"  ⚠️  {class_name}: Folder tidak ditemukan!")
    
    # Filter valid set
    print("\n📂 Filtering VALID set...")
    valid_count = 0
    for class_name in SELECTED_CLASSES:
        src_path = valid_dir / class_name
        dst_path = filtered_valid / class_name
        
        if src_path.exists():
            # Copy seluruh folder
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            
            count = len(list(dst_path.glob("*.jpg"))) + len(list(dst_path.glob("*.JPG")))
            valid_count += count
            print(f"  ✅ {class_name}: {count} gambar")
        else:
            print(f"  ⚠️  {class_name}: Folder tidak ditemukan!")
    
    print("\n" + "=" * 70)
    print("✅ FILTERING SELESAI!")
    print("=" * 70)
    print(f"\n📊 Statistik Dataset Baru:")
    print(f"  Train: {train_count:,} gambar")
    print(f"  Valid: {valid_count:,} gambar")
    print(f"  Total: {train_count + valid_count:,} gambar")
    print(f"\n📁 Lokasi dataset baru: {filtered_base}")
    print("\n💡 Langkah selanjutnya:")
    print(f"  1. Gunakan path ini untuk training:")
    print(f"     --train_dir \"{filtered_train}\"")
    print(f"     --val_dir \"{filtered_valid}\"")
    print(f"  2. Model akan otomatis mengenali {len(SELECTED_CLASSES)} kelas")
    print(f"  3. Label map akan dibuat otomatis dengan urutan alfabetis")


if __name__ == "__main__":
    # Konfirmasi sebelum filtering
    print("⚠️  PERINGATAN:")
    print("  Script ini akan membuat COPY dataset baru dengan hanya kelas terpilih.")
    print("  Dataset asli TIDAK akan dihapus.")
    print(f"\n  Kelas yang akan di-copy: {len(SELECTED_CLASSES)} kelas")
    print(f"  Output folder: {Path(DATA_DIR) / 'New Plant Diseases Dataset(Filtered)'}")
    
    response = input("\nLanjutkan? (y/n): ").strip().lower()
    if response == 'y':
        filter_dataset()
    else:
        print("❌ Dibatalkan.")

