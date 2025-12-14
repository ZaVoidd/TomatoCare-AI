import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a specific image and model.
    """
    # 1. Handle Nested Model (Transfer Learning case)
    # Check if the model has a nested 'densenet121' layer (or similar base model layer)
    # Why? Because in transfer learning, valid layers are often HIDDEN inside a Functional layer.
    
    target_layer = None
    base_model_layer = None
    
    # Try to find the layer in the main model first
    try:
        target_layer = model.get_layer(last_conv_layer_name)
        working_model = model
    except ValueError:
        # If not found, check if it's inside a nested base model (common in Transfer Learning)
        # We look for the first layer that IS a Model/Functional
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    target_layer = layer.get_layer(last_conv_layer_name)
                    base_model_layer = layer
                    print(f"[INFO] Found target layer '{last_conv_layer_name}' inside nested model '{layer.name}'")
                    break
                except ValueError:
                    continue
        
        if target_layer is None:
            raise ValueError(f"Could not find layer '{last_conv_layer_name}' in model or nested base model.")
            
    if base_model_layer:
        # COMPLEX CASE: The target layer is inside a nested model (densenet121).
        # We need to build a new graph that goes: Input -> Base Model -> (Target Layer Output, Base Model Output)
        # And then chain it to the top classifier.
        
        # Step A: Build a multi-output model from the BASE model
        # Input: Base Model Input
        # Outputs: [Target Layer Output, Base Model Final Output]
        base_multi_output_model = tf.keras.models.Model(
            [base_model_layer.inputs], 
            [base_model_layer.get_layer(last_conv_layer_name).output, base_model_layer.output]
        )
        
        # Step B: Build the full Grad-CAM model
        # Input: Original Model Input
        inputs = model.inputs
        
        # Run inputs through the base multi-output model
        # Note: We need to handle potential preprocessing/shape diffs if any, but usually inputs are passed transparently
        conv_output, base_output = base_multi_output_model(inputs)
        
        # Run base_output through the REST of the main model (Classifier part)
        # We need to find which layers come AFTER the base model layer
        
        # FIX: Ensure x is a single tensor.
        # Sometimes unpacking returns a list if the underlying layer is complex.
        # Since for DenseNet, conv_output IS the final output of the base model, we can safely use it.
        x = conv_output
        if isinstance(x, list):
            x = x[0]
            
        base_layer_index = model.layers.index(base_model_layer)
        
        for layer in model.layers[base_layer_index+1:]:
             x = layer(x)
             
        final_output = x
        
        grad_model = tf.keras.models.Model(inputs, [conv_output, final_output])
        
    else:
        # SIMPLE CASE: Target layer is directly in the main model
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

    # 2. Rekam operasi untuk menghitung gradien
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. Hitung gradien dari kelas prediksi terhadap output layer konvolusi
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Global Average Pooling pada gradien (untuk mendapatkan bobot setiap filter)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Kalikan output layer konvolusi dengan bobot gradien
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalisasi heatmap (Hanya ambil nilai positif dengan ReLU)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Overlay heatmap on original image and save it.
    """
    # 1. Load image asli
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV loads BGR, convert to RGB

    # 2. Rescale heatmap ke ukuran image asli (0-255)
    heatmap = np.uint8(255 * heatmap)

    # 3. Gunakan colormap jet (Biru = rendah, Merah = tinggi)
    jet = cm.get_cmap("jet")
    
    # Ambil nilai warna RGB dari colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # 4. Resize heatmap agar sama dengan ukuran gambar asli
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    
    # Konversi ke format uint8 (0-255)
    jet_heatmap =  np.uint8(255 * jet_heatmap)
    
    # Karena OpenCV pakai BGR, kita balik urutan warnanya
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Balikkan lagi img ke BGR untuk save

    # 5. Gabungkan gambar asli dengan heatmap
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    # 6. Simpan hasil
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path

def find_target_layer(model):
    """
    Mencari layer konvolusi terakhir secara otomatis.
    Mendukung pencarian dalam Nested Model (Transfer Learning).
    """
    # Helper recursive function
    def search_in_model(current_model):
        # 1. Coba cari layer dengan nama spesifik 'relu' (Ini output terakhir DenseNet121)
        try:
            layer = current_model.get_layer('relu')
            print(f"[INFO] Grad-CAM Target Layer found (Hardcoded): {layer.name}")
            return layer.name
        except ValueError:
            pass 

        # 2. Cari manual dari belakang
        for layer in reversed(current_model.layers):
            try:
                # Jika layer adalah model lain (Nested), cari di dalamnya
                if isinstance(layer, tf.keras.Model):
                     found_name = search_in_model(layer)
                     if found_name:
                         return found_name
                
                # Pastikan layer memiliki properti output_shape
                if not hasattr(layer, 'output_shape'):
                    continue
                    
                output_shape = layer.output_shape
                if isinstance(output_shape, list):
                    output_shape = output_shape[0]
                
                # Cari layer 4D. Prioritas: relu > conv > concat
                if len(output_shape) == 4:
                    if 'relu' in layer.name:
                        return layer.name
                    if 'conv' in layer.name:
                        return layer.name
                    if 'concat' in layer.name:
                        return layer.name
            except Exception:
                continue
        return None

    # Start search
    return search_in_model(model)
