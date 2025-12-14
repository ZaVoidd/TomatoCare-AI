"""Test untuk Flask app."""
import os
import json
import tempfile
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from flask import Flask
from werkzeug.datastructures import FileStorage

# Import app setelah setup
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Setup Flask test client."""
    # Import app di dalam fixture untuk menghindari side effects
    from app import app as flask_app
    flask_app.config['TESTING'] = True
    flask_app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.fixture
def mock_model_and_labels():
    """Mock model dan class names."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([[0.1, 0.2, 0.7]])  # Prediksi class 2 dengan conf 0.7
    
    class_names = ["Class_A", "Class_B", "Class_C"]
    return mock_model, class_names


@pytest.fixture
def sample_image():
    """Buat sample image file untuk testing."""
    import cv2
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        yield tmp.name
    
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


def test_index_route(client):
    """Test route index (GET /)."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Tomato Leaf Disease Identification' in response.data


def test_predict_no_file(client):
    """Test predict tanpa file upload."""
    response = client.post('/predict')
    assert response.status_code == 302  # Redirect
    # Follow redirect untuk cek flash message
    response = client.post('/predict', follow_redirects=True)
    assert b'Tidak ada file yang dikirim' in response.data or b'No file part' in response.data


def test_predict_empty_filename(client):
    """Test predict dengan filename kosong."""
    response = client.post('/predict', data={'file': (None, '')}, follow_redirects=True)
    assert response.status_code == 200
    assert b'Silakan pilih file gambar' in response.data or b'No selected file' in response.data


def test_predict_invalid_file_type(client):
    """Test predict dengan file type tidak valid."""
    data = {'file': (tempfile.NamedTemporaryFile(suffix='.txt', delete=False), 'test.txt')}
    response = client.post('/predict', data=data, follow_redirects=True)
    assert response.status_code == 200
    assert b'Tipe file tidak didukung' in response.data or b'Unsupported file type' in response.data


@patch('app._get_model_and_labels')
def test_predict_model_not_found(mock_get_model, client, sample_image):
    """Test predict ketika model tidak ditemukan."""
    mock_get_model.side_effect = FileNotFoundError("Model tidak ditemukan")
    
    with open(sample_image, 'rb') as f:
        data = {'file': (f, 'test.jpg')}
        response = client.post('/predict', data=data, content_type='multipart/form-data', follow_redirects=True)
    
    assert response.status_code == 200
    assert b'Model tidak ditemukan' in response.data


@patch('app._get_model_and_labels')
@patch('app.predict_image')
def test_predict_success(mock_predict, mock_get_model, client, mock_model_and_labels, sample_image):
    """Test predict berhasil."""
    mock_model, class_names = mock_model_and_labels
    mock_get_model.return_value = (mock_model, class_names)
    mock_predict.return_value = ("2", 0.7, np.array([0.1, 0.2, 0.7]))
    
    with open(sample_image, 'rb') as f:
        data = {'file': (f, 'test.jpg')}
        response = client.post('/predict', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200
    assert b'Class_C' in response.data or b'Prediction Result' in response.data
    assert b'70.00%' in response.data  # Confidence


@patch('app.load_model_and_labels')
@patch('app.predict_image')
def test_model_caching(mock_predict, mock_load_model, client, mock_model_and_labels, sample_image):
    """Pastikan loader (load_model_and_labels) hanya dipanggil sekali karena caching."""
    import app as flask_app

    flask_app._MODEL = None
    flask_app._CLASS_NAMES = None

    mock_model, class_names = mock_model_and_labels
    mock_load_model.return_value = (mock_model, class_names)
    mock_predict.return_value = ("1", 0.5, np.array([0.3, 0.5, 0.2]))

    for _ in range(2):
        with open(sample_image, 'rb') as f:
            data = {'file': (f, 'test.jpg')}
            client.post('/predict', data=data, content_type='multipart/form-data')

    assert mock_load_model.call_count == 1
def test_allowed_file_extension():
    """Test fungsi allowed_file."""
    from app import allowed_file
    
    assert allowed_file('test.jpg') == True
    assert allowed_file('test.jpeg') == True
    assert allowed_file('test.png') == True
    assert allowed_file('test.txt') == False
    assert allowed_file('test.pdf') == False
    assert allowed_file('test.JPG') == True  # Case insensitive check
    assert allowed_file('test') == False

