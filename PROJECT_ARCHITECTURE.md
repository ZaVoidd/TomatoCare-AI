# Dokumentasi Lengkap Sistem TomatoCare AI

Dokumen ini menjelaskan seluruh arsitektur proyek TomatoCare dari hulu ke hilir, mulai dari persiapan data mentah hingga menjadi aplikasi web yang siap pakai.

## 1. Tech Stack (Teknologi yang Digunakan)

### Backend & Core AI
*   **Python 3.8+**: Bahasa pemrograman utama.
*   **TensorFlow & Keras**: Framework Deep Learning untuk membuat dan melatih model AI.
*   **OpenCV (cv2)**: Library Computer Vision untuk manipulasi gambar (resize, crop, contrast).
*   **Scikit-Learn**: Untuk perhitungan metrik evaluasi (Confusion Matrix, F1-Score) dan Class Weights.
*   **Flask**: Web framework ringan untuk menjalankan server aplikasi.

### Frontend (Tampilan)
*   **HTML5 Templates**: Struktur halaman web.
*   **TailwindCSS**: Framework CSS modern untuk desain UI yang responsif dan cantik (via CDN).
*   **Jinja2**: Templating engine untuk menyisipkan data Python ke HTML dinamis.
*   **FontAwesome**: Ikon-ikon visual.

---

## 2. Alur Sistem (Workflow)

Sistem ini dibangun dalam 4 TAHAPAN BESAR:

### TAHAP 1: Persiapan Data (Data Preparation)

Sebelum AI bisa belajar, data harus bersih dan fokus.
1.  **Sumber Data**: *New Plant Diseases Dataset (Augmented)* yang berisi puluhan ribu gambar daun.
2.  **Filtering (`filter_dataset.py`)**:
    *   Dataset asli punya 10 kelas.
    *   Script ini membuang 5 kelas yang tidak relevan (seperti Leaf Mold, Septoria, dll) dan hanya menyisakan **5 Kelas Utama**: Healthy, Bacterial Spot, Early Blight (dianggap Target Spot), Late Blight, dan Mosaic Virus.
    *   Tujuannya: Agar AI fokus pada penyakit yang paling umum di lapangan user.
3.  **Zipping (`zip_dataset.py`)**:
    *   Mengemas folder dataset yang sudah difilter menjadi `.zip` agar mudah diupload ke Google Colab.

---

### TAHAP 2: Pelatihan Model (AI Training)

Ini adalah "Otak" dari sistem, dilakukan di Google Colab menggunakan GPU.
File utama: `src/train_improved.py`

#### A. Preprocessing & Augmentasi (Super Augmentation)
Setiap gambar yang masuk dilatih dengan variasi ekstrem agar AI "tahan banting":
*   **Resize**: Diubah ke 192x192 pixel (Standar input DenseNet).
*   **Super Augmentation**:
    *   **Rotasi 90 Derajat (0.25)**: Agar AI mengenali daun dari berbagai sudut.
    *   **Random Flip**: Horizontal & Vertical.
    *   **Contrast (0.4) & Brightness (0.2)**: Simulasi foto siang/mendung/kamera HP berbeda.
    *   **Zoom (0.2) & Translation (0.1)**: Simulasi jarak dan posisi pengambilan gambar.

#### B. Strategi Hukuman (Class Weights 8x - Updated)
Kita menerapkan aturan **SANGAT KETAT** untuk memaksa AI belajar:
*   **Penyakit (4 Kelas)**: Hukuman **8x Lipat (8.0)**.
    *   *Filosofi:* Salah menebak penyakit (False Negative) dianggap kesalahan fatal. Model dipaksa "agresif" dan "paranoid" terhadap ciri penyakit.
*   **Sehat**: Hukuman **1x (Normal)**.

#### C. Arsitektur Model (DenseNet121)
Kita menggunakan **Transfer Learning**:
*   Menggunakan model **DenseNet121** yang sudah pintar (pre-trained di ImageNet).
*   Kita "memotong" kepalanya, dan menggantinya dengan lapisan baru yang khusus mengenali 5 penyakit tomat.
*   **Fine-Tuning**: Di akhir training, kita "mencairkan" (unfreeze) sebagian otak DenseNet agar dia lebih spesifik lagi belajar tekstur daun tomat.

**Output:** File `densenet121_best.keras` (File otak AI) dan `label_map.json` (Kamus nama penyakit).

---

### TAHAP 3: Integrasi Backend (Flask)

Setelah otak AI jadi, kita pasang ke badan aplikasi (Flask).
File utama: `app.py`

1.  **Loading Model**:
    *   Saat aplikasi nyala, `src/inference.py` memuat model ke memori sekali saja (Lazy Loading).
2.  **Routing**:
    *   User membuka `/` -> Muncul halaman Home (`index.html`).
    *   User upload gambar -> Dikirim ke `/predict`.
3.  **Prediction Pipeline**:
    *   Gambar user diterima -> Disimpan sementara di `static/uploads`.
    *   Di-Preprocess (Resize 192x192, Convert RGB) -> `src/preprocess.py`.
    *   Masuk ke Model -> Keluar berupa angka probabilitas (misal: `[0.1, 0.8, 0.05, ...]`).
    *   **Sorting**: Hasil diurutkan dari persentase tertinggi ke terendah.
    *   **Logic UI**: Backend mengecek, jika kelas tertinggi adalah "Healthy", set flag `is_healthy = True`.

#### Fitur Penjelas (Explainability)
File: `src/explain.py`
*   Kita menggunakan teknik **Grad-CAM (Gradient-weighted Class Activation Mapping)**.
*   Sistem melacak "pixel mana yang membuat AI yakin".
*   Hasilnya adalah **Heatmap Merah** yang ditempel di atas gambar asli, menunjukkan lokasi bercak penyakit.

---

### TAHAP 4: Tampilan Frontend (UI/UX)

Bagian yang berinteraksi dengan user.

1.  **Homepage (`index.html`)**:
    *   Menampilkan kartu informasi 5 penyakit.
    *   Form upload gambar yang simpel.
2.  **Result Page (`result.html`)**:
    *   **Badge Dinamis**:
        *   Hijau ("KONDISI PRIMA") jika Sehat.
        *   Merah ("PENYAKIT TERDETEKSI") jika Sakit.
    *   **Visualisasi**: Menampilkan Gambar Asli bersandingan dengan Heatmap AI.
    *   **Progress Bar**: Statistik probabilitas semua kelas (Hijau/Merah dinamis).
    *   **Rekomendasi**: Menarik data penanganan dan pencegahan dari `src/disease_data.py` (database teks lokal) dan menampilkannya dengan rapi.

---

### Ringkasan Ekosistem

1.  **Dataset** (Data Mentah)
    ⬇️ *Filtered by `filter_dataset.py`*
2.  **Dataset 5 Kelas**
    ⬇️ *Training di Colab (`train_improved.py`)*
3.  **Model File** (`.keras`)
    ⬇️ *Download ke Laptop*
4.  **Aplikasi Flask** (`app.py`)
    ⬇️ *User Upload Foto*
5.  **Hasil Diagnosa & Solusi** (Web UI)
