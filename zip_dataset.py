
import shutil
import os
from src.config import DATA_DIR, BASE_DIR

def zip_files():
    # 1. Zip Dataset
    source_dir = os.path.join(DATA_DIR, "New Plant Diseases Dataset(Filtered)")
    output_filename = os.path.join(DATA_DIR, "dataset_5_classes")
    
    print(f"📦 Zipping dataset...")
    print(f"   Source: {source_dir}")
    print(f"   Target: {output_filename}.zip")
    shutil.make_archive(output_filename, 'zip', source_dir)
    print(f"   ✅ Done: dataset_5_classes.zip")

    # 2. Zip Source Code (src folder)
    src_dir = os.path.join(BASE_DIR, "src")
    src_output = os.path.join(BASE_DIR, "src_code")
    
    print(f"\n📦 Zipping source code...")
    print(f"   Source: {src_dir}")
    print(f"   Target: {src_output}.zip")
    shutil.make_archive(src_output, 'zip', src_dir)
    print(f"   ✅ Done: src_code.zip")

if __name__ == "__main__":
    zip_files()
