# 🍅 Panduan Lengkap Retraining Model (5 Kelas) via Google Drive

Panduan ini dirancang **sangat detail** agar proses training berjalan lancar tanpa error `BadZipFile`. Kita akan menggunakan **Google Drive** sebagai penyimpanan agar file tidak hilang dan upload lebih stabil.

---

## ✅ Tahap 0: Persiapan Data di Laptop

Pastikan Anda memiliki 2 file ini di laptop Anda sebelum mulai:

1.  **`New Plant Diseases Dataset(Filtered).zip`**
    *   **Lokasi**: Cek di folder `data/` dalam project ini.
    *   **Isi**: Dataset gambar yang sudah difilter menjadi 5 kelas (hasil dari script `filter_dataset.py`).
    *   *Jika belum ada, jalankan dulu `python filter_dataset.py` lalu zip folder hasilnya.*

2.  **`src_code.zip`**
    *   **Lokasi**: Cek di folder root project.
    *   **Isi**: Kode program python (`src/`) yang dizip.
    *   *Cara buat: Klik kanan folder `src`, pilih 'Send to > Compressed (zipped) folder', lalu beri nama `src_code.zip`.*

---

## ☁️ Tahap 1: Upload ke Google Drive

Lakukan ini agar upload stabil dan file aman.

1.  Buka browser, masuk ke **[drive.google.com](https://drive.google.com)**.
2.  Login dengan akun Google yang sama dengan yang akan dipakai di Colab.
3.  Klik tombol **+ New** (Baru) > **Folder**.
4.  Beri nama folder: `BelajarTomat` (supaya mudah ditemukan).
5.  Masuk ke dalam folder `BelajarTomat` tersebut.
6.  **Upload 2 file** tadi (`New Plant Diseases Dataset(Filtered).zip` dan `src_code.zip`) ke sini.
    *   *Caranya: Drag & drop file dari laptop ke browser, atau klik New > File Upload.*
7.  **Tunggu** hingga kedua file selesai terupload 100% (pastikan ada tanda centang hijau).

---

## 🚀 Tahap 2: Buka & Setting Google Colab

1.  Buka **[colab.research.google.com](https://colab.research.google.com)**.
2.  Klik **New Notebook** (Notebook Baru).
3.  **Aktifkan GPU** (Wajib agar training tidak lemot):
    *   Klik menu **Runtime** > **Change runtime type**.
    *   Pada 'Hardware accelerator', pilih **T4 GPU**.
    *   Klik **Save**.

---

## 🔗 Tahap 3: Koneksikan Colab ke Google Drive

Sekarang kita hubungkan Colab dengan Google Drive tempat file tadi.

Ketik kode berikut di **Cell Pertama**, lalu jalankan (klik tombol Play ▶️):

```python
from google.colab import drive
import os

print("🔄 Menghubungkan ke Google Drive...")
drive.mount('/content/drive')
print("✅ Sukses! Google Drive terhubung.")
```

> **PENTING**: Nanti akan muncul pop-up "Permit this notebook to access your Google Drive files?". Klik **Connect to Google Drive**, pilih akun Google Anda, dan klik **Allow/Izinkan**.

---

## 📦 Tahap 4: Ekstrak File di Colab

Kita copy file dari Drive ke penyimpanan sementara Colab dan ekstrak di sana supaya proses baca data cepat.

Buat **Cell Baru** (klik `+ Code`), copy paste kode ini, lalu jalankan:

```python
import zipfile
import shutil

# --- KONFIGURASI NAMA FILE DI SINI ---
# Sesuaikan jika Anda menaruh di folder lain di Drive
drive_folder = "/content/drive/MyDrive/BelajarTomat"
dataset_name = "New Plant Diseases Dataset(Filtered).zip"
source_code_name = "src_code.zip"

dataset_path = f"{drive_folder}/{dataset_name}"
src_path = f"{drive_folder}/{source_code_name}"

# --- PROSES EKSTRAK ---
print(f"📦 Sedang mengekstrak Dataset: {dataset_name}...")
try:
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(".") # Ekstrak ke folder saat ini (/content)
    print("✅ Ekstrak Dataset SELESAI.")
except FileNotFoundError:
    print(f"❌ ERROR: File {dataset_name} tidak ditemukan di {drive_folder}!")
    print("👉 Pastikan nama folder dan file di Drive sudah sesuai (huruf besar/kecil berpengaruh).")

print(f"\n📦 Sedang mengekstrak Source Code: {source_code_name}...")
try:
    # Buat folder src jika belum ada
    os.makedirs("src", exist_ok=True)
    with zipfile.ZipFile(src_path, 'r') as zip_ref:
        zip_ref.extractall("src")
    print("✅ Ekstrak Source Code SELESAI.")
except FileNotFoundError:
    print(f"❌ ERROR: File source code tidak ditemukan!")
```

---

## 🔍 Tahap 5: Verifikasi Struktur Folder

Pastikan folder terekstrak dengan benar agar script training bisa menemukannya.

Buat **Cell Baru**, jalankan ini:

```bash
print("📂 Cek isi folder dataset:")
# Sesuaikan nama folder hasil ekstraksi. Biasanya sama dengan nama zip tanpa .zip
# Jika zip bernama "New Plant Diseases Dataset(Filtered).zip", folder hasil ekstraknya mungkin bernama sama.
!ls "New Plant Diseases Dataset(Filtered)" | head -n 5

print("\n📂 Cek isi folder src:")
!ls src
```

**Hasil yang BENAR harusnya:**
1.  Isi folder dataset ada folder: `train` dan `valid`.
2.  Isi folder src ada file: `config.py`, `train_improved.py`, dll.

---

## 🛠️ Tahap 6: Install & Mulai Training

### 6.1 Install Library
Sebenarnya Google Colab sudah menyediakan TensorFlow, OpenCV, dan Scikit-Learn secara default. Namun untuk memastikan versi yang kita pakai kompatibel dan lengkap, jalankan perintah ini:

```bash
!pip install tensorflow opencv-python matplotlib scikit-learn
```
*Matplotlib wajib diinstall karena kadang versi default Colab minimalis.*

### 6.2 Jalankan Training
Ini adalah perintah utama untuk melatih model. Sesuaikan path folder dataset dengan hasil cek di Tahap 5.

```bash
# Perintah Training
!python -m src.train_improved \
  --train_dir "New Plant Diseases Dataset(Filtered)/train" \
  --val_dir "New Plant Diseases Dataset(Filtered)/valid" \
  --output_dir "models_5_classes" \
  --img_size 192 192 \
  --batch_size 16 \
  --epochs 20 \
  --learning_rate 1e-4
```
*Tunggu proses ini sampai selesai (bisa 1-2 jam).*

---

## 💾 Tahap 7: Simpan Model ke Drive (PENTING!)

Setelah capek-capek training, jangan lupa simpan hasilnya ke Drive agar bisa didownload.

```python
import shutil

source = "models_5_classes"
destination = "/content/drive/MyDrive/BelajarTomat/Hasil_Model_5_Kelas"

print(f"💾 Menyalin hasil training ke: {destination} ...")
try:
    shutil.copytree(source, destination)
    print("✅ SUKSES! Model aman tersimpan di Google Drive.")
except FileExistsError:
    print("⚠️ Folder tujuan sudah ada. Ganti nama folder tujuan atau hapus folder lama di Drive.")
except Exception as e:
    print(f"❌ Gagal menyalin: {e}")
```

---

## 💻 Tahap 8: Download & Pasang di Aplikasi

1.  Buka **Google Drive**, masuk folder `BelajarTomat/Hasil_Model_5_Kelas`.
2.  Download file:
    *   `densenet121_best.keras`
    *   `label_map.json`
3.  Copy kedua file itu ke folder `models/` di project VS Code Anda (timpa file yang lama).
4.  Jalankan aplikasi Flask (`python app.py`) dan coba tes dengan gambar tomat!
