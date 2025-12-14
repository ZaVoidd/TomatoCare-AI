"""Test untuk modul inference."""
import os
import json
import tempfile
import numpy as np
import pytest
import tensorflow as tf
from unittest.mock import Mock, patch
from src.inference import load_model_and_labels, predict_image


def test_load_model_and_labels_file_not_found():
    """Test error handling untuk model/label map tidak ditemukan."""
    with pytest.raises(FileNotFoundError):
        load_model_and_labels("model_tidak_ada.h5", "label_tidak_ada.json")


@patch('src.inference.tf.keras.models.load_model')
def test_load_model_and_labels_success(mock_load_model):
    """Test load model dan label map berhasil."""
    # Mock model
    mock_model = Mock()
    mock_load_model.return_value = mock_model
    
    # Buat label map temporary
    label_map = {"0": "Class_A", "1": "Class_B"}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(label_map, tmp)
        label_path = tmp.name
    
    # Buat dummy model file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        model_path = tmp.name
    
    try:
        model, class_names = load_model_and_labels(model_path, label_path)
        
        assert model == mock_model
        assert len(class_names) == 2
        assert class_names[0] == "Class_A"
    finally:
        if os.path.exists(label_path):
            os.unlink(label_path)
        if os.path.exists(model_path):
            os.unlink(model_path)


@patch('src.inference.load_and_preprocess')
def test_predict_image(mock_preprocess):
    """Test prediksi gambar."""
    # Mock preprocessing
    mock_img = np.random.rand(224, 224, 3).astype(np.float32)
    mock_preprocess.return_value = mock_img
    
    # Mock model
    mock_model = Mock()
    # Simulasi prediksi: class 1 dengan confidence 0.9
    mock_pred = np.array([[0.1, 0.9, 0.0]])
    mock_model.predict.return_value = mock_pred
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img_path = tmp.name
    
    try:
        pred_idx, conf, probs = predict_image(mock_model, img_path)
        
        assert pred_idx == "1"
        assert conf == pytest.approx(0.9)
        assert len(probs) == 3
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)

