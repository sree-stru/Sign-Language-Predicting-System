"""
config.py — Configuration for Sign Language Project
"""
import os

# Project Roots
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r"C:\Users\ANJANA\Downloads\data\data"

# Data Specs
IMG_SIZE = (128, 128)
IMG_CHANNELS = 3
NUM_CLASSES = 35
TEST_RATIO = 0.1
VAL_RATIO = 0.1
TRAIN_RATIO = 0.8
BATCH_SIZE = 16
EPOCHS = 15

# Training Results
MODELS_DIR = os.path.join(BASE_DIR, "models")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "gesture_model.keras")

# Landmarks Pivot Settings
LANDMARKS_DATA_FILE = "landmarks_data.npy"
LANDMARKS_LABELS_FILE = "landmarks_labels.npy"
LANDMARK_MODEL_PATH = os.path.join(MODELS_DIR, "sign_landmark_model.keras")
LANDMARK_INPUT_SIZE = 42 # 21 landmarks * (x, y)

# Inference Settings
CONFIDENCE_THRESHOLD = 0.60
PREDICTION_DELAY_FRAMES = 10
MAX_HANDS = 2

# Gemini AI Integration
# SECURITY: Use an environment variable for the API key instead of hardcoding it.
import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
