"""Configuration centrale du projet de reconnaissance faciale."""

import os
from pathlib import Path

# Chemins de base
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Chemins des modèles
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n-face.pt"
YOLO_MODEL_URL = "https://github.com/michrafnabil/yolo_face/raw/main/yolov8n-f.pt"

# Chemins des données
WHITELIST_DIR = r"C:\Users\DELL\Downloads\whitelist_preprocessed"
DATASET_FACES_DIR = r"C:\Users\DELL\Downloads\dataset_faces"
CAPTURE_DIR = r"C:\Users\DELL\Pictures\visages_webcam"

# Chemins des prototypes (dans data/prototypes)
PROTOTYPES_PATH = DATA_DIR / "prototypes" / "whitelist_proto.npz"
REFERENCES_PATH = DATA_DIR / "prototypes" / "whitelist_refs.npz"

# Chemin des résultats
RESULTS_DIR = DATA_DIR / "results"

# Paramètres YOLO
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45

# Paramètres de détection
FACE_MARGIN = 0.25  # 25% marge autour du visage détecté
TARGET_SIZE = (224, 224)  # Taille pour CNN
JPEG_QUALITY = 95

# Paramètres FaceNet
FACENET_SIZE = 160  # Taille d'entrée FaceNet
EMBEDDING_DIM = 512

# Paramètres de reconnaissance
RECOGNITION_THRESHOLD = 0.25  # Seuil de distance cosinus
PREPROCESSING_MARGIN = 0.15  # Marge pour preprocessing

# Paramètres de preprocessing
MIN_FACE_SIZE = 80  # Taille minimale du visage
MAX_IMAGES_PER_PERSON = 100

# Durée de capture webcam (secondes)
CAPTURE_DURATION = 0.5
