"""Test untuk modul preprocessing."""
import os
import tempfile
import numpy as np
import cv2
import pytest
from src.preprocess import preprocess_image_bgr, load_and_preprocess


def test_preprocess_image_bgr_basic():
    """Test preprocessing dasar tanpa CLAHE."""
    # Buat gambar dummy BGR (100x100, 3 channel)
    img_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = preprocess_image_bgr(img_bgr, target_size=(224, 224), use_clahe=False)
    
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_image_bgr_with_clahe():
    """Test preprocessing dengan CLAHE."""
    img_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = preprocess_image_bgr(img_bgr, target_size=(224, 224), use_clahe=True)
    
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32


def test_load_and_preprocess():
    """Test load dan preprocess dari file."""
    # Buat file gambar temporary
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(tmp.name, img_bgr)
        tmp_path = tmp.name
    
    try:
        result = load_and_preprocess(tmp_path, target_size=(224, 224))
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32
    finally:
        os.unlink(tmp_path)


def test_load_and_preprocess_file_not_found():
    """Test error handling untuk file tidak ditemukan."""
    with pytest.raises(FileNotFoundError):
        load_and_preprocess("file_yang_tidak_ada.jpg")

