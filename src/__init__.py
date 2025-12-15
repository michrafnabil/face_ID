"""Package principal de reconnaissance faciale."""

import os
# Désactiver la détection Git pour éviter l'erreur FileNotFoundError
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

from src.utils.detection import FaceDetector
from src.utils.embeddings import FaceEmbedder
from src.utils.preprocessing import WhitelistPreprocessor
from src.utils.recognition import FaceRecognizer

__all__ = [
    'FaceDetector',
    'FaceEmbedder',
    'WhitelistPreprocessor',
    'FaceRecognizer'
]
