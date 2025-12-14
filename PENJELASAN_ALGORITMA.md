# 🧠 Penjelasan Mendalam: Algoritma & Arsitektur Model

Dokumen ini disusun untuk membantu Anda menjawab pertanyaan teknis (sidang/presentasi) mengenai "Otak" di balik TomatoCare.

---

## 1. CNN (Convolutional Neural Network)
**"Mata Digital yang Belajar Melihat Pola"**

CNN adalah jenis Deep Learning yang dirancang khusus untuk mengolah data gambar. Berbeda dengan program biasa yang melihat gambar sebagai sekumpulan angka acak, CNN melihat gambar seperti manusia melihat objek: **Bertahap dari detail kecil ke bentuk utuh**.

### Cara Kerjanya (Analogi Sederhana):
Bayangkan Anda melihat foto wajah orang. Otak Anda tidak langsung tahu "Itu Budi". Tapi otak Anda memprosesnya bertahap:
1.  **Layer Awal:** Mengenali garis lengkung, garis lurus, dan sudut.
2.  **Layer Tengah:** Menggabungkan garis tadi menjadi bentuk: mata, hidung, telinga.
3.  **Layer Akhir:** Menggabungkan bentuk tadi menjadi wajah utuh "Budi".

Di komputer, proses "Mengenali" ini dilakukan oleh **Filter (Kernel)** yang "menggeser" (convolve) di atas gambar untuk mencari fitur-fitur penting itu.

---

## 2. DenseNet121 (Densely Connected Convolutional Networks)
**"Arsitektur dengan Ingatan Super Kuat"**

DenseNet121 adalah salah satu varian CNN yang sangat canggih. Angka **121** berarti model ini memiliki **121 lapisan (layers)**. 

### Masalah pada CNN Biasa (Zaman Dulu):
Pada CNN biasa yang sangat dalam (misal 100 layer), informasi dari gambar asli sering "hilang" atau "luntur" sebelum sampai ke layer terakhir. Seperti permainan "Bisik Berantai", pesan di awal sering berubah saat sampai di orang terakhir. Ini disebut **Vanishing Gradient Problem**.

### Solusi Jenius DenseNet: "Jalur Pintas (Shortcut Connections)"
DenseNet memperbaiki ini dengan ide radikal:
**"Setiap layer ngobrol dengan SEMUA layer di depannya."**

*   **Dense Block:** Dalam satu blok, Layer 1 mengirim hasil kerjanya ke Layer 2, Layer 3, Layer 4, dst.
*   **Reuse Features:** Layer terakhir tidak hanya menerima info dari layer sebelumnya, tapi juga "melihat kembali" fitur asli dari layer-layer awal.

**Analogi:**
Bayangkan grup diskusi.
*   **CNN Biasa:** Orang A bisik ke B, B ke C, C ke D. Kalau D ditanya apa kata A, mungkin dia lupa.
*   **DenseNet:** Orang A bicara di mikrofon, jadi B, C, dan D semua mendengarnya langsung. Informasi A tidak akan pernah hilang sampai diskusi selesai.

**Kelebihan DenseNet121 untuk Penyakit Tomat:**
1.  **Akurasi Tinggi:** Sangat jeli melihat detail kecil (bercak jamur kecil) karena tidak ada info yang hilang.
2.  **Efisien:** Membutuhkan jumlah parameter yang lebih sedikit dibanding model besar lain (seperti VGG16) tapi performanya setara atau lebih bagus.

---

## 3. Softmax Activation Function
**"Hakim Pemberi Keputusan (Probabilitas)"**

Ini adalah "pintu terakhir" dari model AI. Setelah DenseNet memproses gambar melalui 121 layernya, dia akan mengeluarkan 5 angka mentah (Logits) yang seringkali aneh (misal: 2.5, -0.1, 5.0). Manusia susah membacanya.

**Softmax** bertugas mengubah angka-angka aneh itu menjadi **Persentase Total 100%**.

### Rumus Matematis:
$$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} $$

### Cara Kerjanya:
1.  **Eksponensial ($e^z$):** Membuat angka positif menjadi LEBIH BESAR LAGI, dan angka negatif menjadi angka positif kecil. Ini mempertegas perbedaan antara "Yakin" dan "Ragu".
2.  **Normalisasi ($\sum$):** Memastikan total semua angka jika dijumlahkan adalah 1 (atau 100%).

### Contoh Kasus TomatoCare:
Model mengeluarkan Logits:
*   Sehat: 2.0
*   Bercak: 4.5  <-- Paling tinggi
*   Hawar: 0.1

**Setelah masuk Softmax:**
*   Sehat: 0.08 (8%)
*   **Bercak: 0.91 (91%)**  ✅ *Pemenang!*
*   Hawar: 0.01 (1%)

Jadi, "Confidence Score" yang Anda lihat di web (91%) itu adalah hasil kerja keras rumus Softmax ini.
