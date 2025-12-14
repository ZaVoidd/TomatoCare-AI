import os
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

import cv2
import numpy as np

from src.inference import load_model_and_labels, predict_image
from src.explain import make_gradcam_heatmap, save_and_display_gradcam, find_target_layer
from src.config import (
    DEFAULT_IMG_SIZE,
    ALLOWED_EXTENSIONS,
    MODEL_PATH,
    LABEL_MAP_PATH,
    UPLOAD_FOLDER,
)
from src.disease_data import DISEASE_INFO

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

_MODEL = None
_CLASS_NAMES = None



# --- VALIDATION HELPER ---
def validate_image(filepath):
    """
    Validasi Advanced (Level Max):
    1. Cek File Corrupt
    2. Cek Resolusi (Min 200px)
    3. Cek Cahaya (Terlalu Gelap/Terang)
    4. Cek Blur (Laplacian)
    5. Cek Objek Asing (Warna Buatan/Scribbles)
    6. Cek Dominasi Daun (Wajib Hijau/Kuning/Coklat Alami)
    """
    try:
        img = cv2.imread(filepath)
        if img is None:
            return False, "File gambar rusak atau tidak terbaca."
        
        h, w, _ = img.shape
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. CEK RESOLUSI
        if h < 200 or w < 200:
            return False, f"Resolusi citra terlalu rendah ({w}x{h}). Minimal 200x200 px agar AI bekerja optimal."

        # 2. CEK PENCAHAYAAN (Brightness)
        avg_brightness = np.mean(img_gray)
        if avg_brightness < 30:
            return False, "Citra terlalu GELAP. Harap ambil foto di tempat yang lebih terang."
        if avg_brightness > 220:
            return False, "Citra terlalu TERANG (Overexposed). Detail daun hilang karena cahaya berlebih."

        # 3. CEK KETAJAMAN (Blur Detection)
        # Laplacian Variance: Angka kecil = Flat/Blur, Angka besar = Tajam/Texture
        blur_score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        if blur_score < 50: 
            return False, "Citra terlalu BURAM/HANCUR. Pastikan kamera fokus ke daun saat memotret."

        # 4. CEK OBJEK ASING (Synthetic Colors: Blue, Neon Pink, Cyan)
        # Warna-warna ini JARANG ada di alam (kecuali bunga tertentu, tapi tomat tidak).
        # Biasa muncul di baju, mobil, tembok cat, atau coretan spidol digital.
        # Range Biru/Cyan
        lower_blue = np.array([85, 50, 50])
        upper_blue = np.array([135, 255, 255])
        mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
        
        # Range Pink/Magenta (Sering dipakai spidol)
        lower_pink = np.array([140, 50, 50])
        upper_pink = np.array([170, 255, 255])
        mask_pink = cv2.inRange(img_hsv, lower_pink, upper_pink)

        foreign_pixels = cv2.countNonZero(mask_blue) + cv2.countNonZero(mask_pink)
        total_pixels = h * w
        
        if (foreign_pixels / total_pixels) > 0.05: # Jika > 5% gambar isinya biru/pink
            return False, "Terdeteksi OBJEK ASING atau CORETAN (Warna tidak alami). Harap upload foto daun tomat asli."

        # 5. CEK DOMINASI DAUN (Leaf Segmentation Logic)
        # Kita cari pixel Hijau (Daun Sehat) + Kuning/Coklat (Daun Sakit)
        
        # Hijau (Range Luas)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([95, 255, 255]) 
        mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
        
        # Kuning/Coklat/Oranye (Penyakit)
        lower_disease = np.array([10, 40, 40]) 
        upper_disease = np.array([25, 255, 255])
        mask_disease = cv2.inRange(img_hsv, lower_disease, upper_disease)

        plant_pixels = cv2.countNonZero(mask_green) + cv2.countNonZero(mask_disease)
        plant_ratio = plant_pixels / total_pixels

        # Ambang batas kita naikkan ke 20% agar lebih ketat terhadap background kosong
        if plant_ratio < 0.20:
            return False, "Objek DAUN TOMAT tidak ditemukan atau terlalu kecil. Pastikan foto zoom ke arah daun, bukan background."

        return True, "Valid"

    except Exception as e:
        print(f"[ValidationError] {e}")
        return False, "Gagal memvalidasi gambar. Format mungkin tidak didukung."


def _get_model_and_labels():
    """Lazy-load model + label map sekali lalu cache di memori."""
    global _MODEL, _CLASS_NAMES
    if _MODEL is None or _CLASS_NAMES is None:
        _MODEL, _CLASS_NAMES = load_model_and_labels(MODEL_PATH, LABEL_MAP_PATH)
    return _MODEL, _CLASS_NAMES


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def translate_class_name(class_name: str) -> str:
    """Terjemahkan nama kelas penyakit ke bahasa Indonesia yang mudah dipahami."""
    info = DISEASE_INFO.get(class_name)
    return info["name_id"] if info else class_name


def get_disease_info(class_name_en: str) -> dict:
    """Kembalikan deskripsi dan penanganan untuk setiap penyakit."""
    return DISEASE_INFO.get(class_name_en, {
        "name_id": class_name_en,
        "description": "Informasi penyakit tidak tersedia.",
        "treatment": ["Konsultasikan dengan ahli pertanian untuk penanganan yang tepat."],
        "prevention": [],
        "journals": []
    })


@app.route("/", methods=["GET"])
def index():
    # Filter hanya 5 kelas aktif
    active_classes = [
        "Tomato___healthy",
        "Tomato___Late_blight", 
        "Tomato___Bacterial_spot",
        "Tomato___Target_Spot",
        "Tomato___Tomato_mosaic_virus"
    ]
    filtered_diseases = {k: v for k, v in DISEASE_INFO.items() if k in active_classes}
    
    return render_template("index.html", diseases=filtered_diseases)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("Tidak ada file yang diunggah.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("Tidak ada file yang dipilih.")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        # --- VALIDASI GAMBAR ---
        is_valid, error_msg = validate_image(filepath)
        if not is_valid:
            # Hapus file jika tidak valid (hemat storage)
            try:
                os.remove(filepath)
            except OSError:
                pass
            flash(error_msg)
            return redirect(url_for("index"))
        # -----------------------

        try:
            model, class_names = _get_model_and_labels()
        except FileNotFoundError as exc:
            flash(str(exc))
            return redirect(url_for("index"))

        pred_idx, conf, probs = predict_image(
            model, filepath, target_size=DEFAULT_IMG_SIZE
        )
        predicted_class_en = class_names[int(pred_idx)]
        predicted_class = translate_class_name(predicted_class_en)
        disease_info = get_disease_info(predicted_class_en)

        # Zip names and probs, then sort by probability (descending)
        raw_results = sorted(
            zip(class_names, probs.tolist()), 
            key=lambda x: x[1], 
            reverse=True
        )

        formatted_probs = [
            (translate_class_name(name), f"{p * 100:.2f}%") 
            for name, p in raw_results
        ]



        # --- GRAD-CAM GENERATION ---
        heatmap_filename = f"heatmap_{filename}"
        heatmap_path = os.path.join(app.config["UPLOAD_FOLDER"], heatmap_filename)
        
        try:
            # 1. Preprocess image for Grad-CAM
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=DEFAULT_IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            # 2. Get target layer (otomatis cari layer conv terakhir)
            target_layer = find_target_layer(model)
            
            if target_layer:
                # 3. Generate heatmap
                heatmap = make_gradcam_heatmap(img_array, model, target_layer, pred_index=int(pred_idx))
                
                # 4. Save heatmap image
                save_and_display_gradcam(filepath, heatmap, heatmap_path)
                heatmap_url = url_for("static", filename=f"uploads/{heatmap_filename}")
            else:
                heatmap_url = None
                print("[WARNING] Could not find target layer for Grad-CAM.")
                
        except Exception as e:
            print(f"[ERROR] Error generating Grad-CAM: {e}")
            heatmap_url = None
        # ---------------------------

        # ---------------------------

        # Determine if healthy
        is_healthy = (predicted_class_en == "Tomato___healthy")

        return render_template(
            "result.html",
            is_healthy=is_healthy,
            image_path=url_for("static", filename=f"uploads/{filename}"),
            heatmap_path=heatmap_url,
            predicted_class=predicted_class,
            confidence=f"{conf * 100:.2f}%",
            class_probs=formatted_probs,
            disease_description=disease_info["description"],
            disease_treatment=disease_info["treatment"],
            disease_prevention=disease_info.get("prevention"),
            disease_journals=disease_info.get("journals"),
        )
    else:
        flash("Tipe file tidak didukung. Gunakan png/jpg/jpeg.")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
