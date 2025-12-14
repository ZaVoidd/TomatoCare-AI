import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')

# Files
MODEL_PATH = os.path.join(MODELS_DIR, 'densenet121_best.keras')
LABEL_MAP_PATH = os.path.join(MODELS_DIR, 'label_map.json')

# Training defaults
DEFAULT_IMG_SIZE = (192, 192)  # Sesuai dengan training di Colab
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42

# Flask
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create dirs
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
