# 🍅 TomatoCare AI - Smart Disease Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![TailwindCSS](https://img.shields.io/badge/Tailwind-CSS-cyan)

**TomatoCare** adalah sistem kecerdasan buatan (AI) berbasis web untuk mendeteksi penyakit pada tanaman tomat secara otomatis melalui foto daun. Dibangun menggunakan arsitektur **DenseNet121** dengan akurasi tinggi dan dilengkapi fitur validasi cerdas ("Satpam Digital") untuk mencegah input yang tidak valid.

---

## ✨ Fitur Unggulan

### 1. 🧠 AI Diagnosis (DenseNet121)
Mampu mendeteksi 5 kondisi tanaman dengan presisi tinggi:
- **Tanaman Sehat (Healthy)**
- **Bercak Bakteri (Bacterial Spot)**
- **Hawar Daun (Late Blight)**
- **Bercak Target (Target Spot)**
- **Virus Mosaik (Mosaic Virus)**

### 2. 🛡️ Smart Image Validation ("Satpam Digital")
Sistem dilengkapi 6 lapis proteksi untuk menolak gambar "sampah":
- **Anti-Blur:** Menolak gambar yang buram/goyang.
- **Anti-Dark/Bright:** Menolak gambar terlalu gelap atau *overexposed*.
- **Anti-Foreign Object:** Menolak gambar coretan spidol, mobil, tembok, dll.
- **Resolution Check:** Memastikan minimal resolusi 200x200 px.
- **Leaf Dominance:** Memastikan objek utama adalah daun (bukan buah tomat utuh).

### 3. 🔍 Explainability (Grad-CAM)
Transparansi AI! Sistem menampilkan **Heatmap** di atas gambar untuk menunjukkan bagian daun mana yang "dilihat" oleh AI sebagai indikasi penyakit.

### 4. 📱 Modern & Responsive UI
Dibangun dengan **TailwindCSS**, tampilan web responsif di HP maupun Laptop, dengan desain yang bersih dan informatif.

---

## 🛠️ Instalasi & Cara Menjalankan

Ikuti langkah ini untuk menjalankan proyek di komputer Anda secara lokal.

### Prasyarat
- Python 3.8 - 3.10
- Git

### 1. Clone Repository
Buka terminal dan jalankan perintah ini:
```bash
git clone https://github.com/USERNAME_ANDA/TomatoCare.git
cd TomatoCare
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install pustaka yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
python app.py
```
Aplikasi akan berjalan di local server: http://127.0.0.1:5000/
Buka alamat tersebut di browser Anda.

---

## 📂 Struktur Project

```
├── data/               # [Di-ignore] Dataset gambar (Download terpisah)
├── models/
│   ├── densenet121_best.keras  # Model AI Utama (Pre-trained)
│   └── label_map.json          # Label kelas
├── src/
│   ├── train_improved.py       # Script Training (Colab)
│   ├── inference.py            # Script Prediksi
│   └── explain.py              # Script Grad-CAM
├── static/             # CSS, Images, Uploads
├── templates/          # Halaman HTML (Jinja2)
├── app.py              # Main Entry Point (Flask)
└── requirements.txt    # Daftar Pustaka
```

---

## 🤖 Algoritma & Teknologi

*   **Model:** DenseNet121 (Transfer Learning from ImageNet)
*   **Optimizer:** Adam
*   **Loss Function:** Sparse Categorical Crossentropy
*   **Metric:** Accuracy, Precision, Recall, F1-Score
*   **Activation:** Softmax (Output Layer)

---

## 👨‍💻 Author

Project ini dikembangkan untuk tujuan edukasi dan penelitian.
**[Nama Anda]** - Universitas Bina Sarana Informatika

---
*Created with ❤️ using Python & TensorFlow.*
