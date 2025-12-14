"""Test untuk modul utils."""
import os
import json
import tempfile
import pytest
from src.utils import save_label_map, load_label_map, set_seed


def test_save_and_load_label_map():
    """Test save dan load label map."""
    class_names = ["Class_A", "Class_B", "Class_C"]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        save_label_map(class_names, tmp_path)
        
        # Verifikasi file ada
        assert os.path.exists(tmp_path)
        
        # Load dan verifikasi
        loaded = load_label_map(tmp_path)
        assert loaded == class_names
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_load_label_map_string_keys():
    """Test load label map dengan string keys (format JSON)."""
    mapping = {"0": "Class_A", "1": "Class_B", "2": "Class_C"}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(mapping, tmp)
        tmp_path = tmp.name
    
    try:
        loaded = load_label_map(tmp_path)
        assert len(loaded) == 3
        assert loaded[0] == "Class_A"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_set_seed():
    """Test set seed untuk reproducibility."""
    # Test bahwa fungsi tidak error
    set_seed(42)
    set_seed(123)

