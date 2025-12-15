"""Module de détection de visages avec YOLO."""

import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO
from pathlib import Path

from src.config import (
    YOLO_MODEL_PATH,
    YOLO_MODEL_URL,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    FACE_MARGIN,
    TARGET_SIZE,
    JPEG_QUALITY
)


class FaceDetector:
    """Détecteur de visages utilisant YOLO."""
    
    def __init__(self):
        """Initialise le détecteur YOLO."""
        self.model = self._load_model()
    
    def _load_model(self):
        """Charge le modèle YOLO pour la détection de visages."""
        # Télécharger le modèle s'il n'existe pas
        if not YOLO_MODEL_PATH.exists():
            print(f"📥 Téléchargement du modèle YOLO...")
            YOLO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(YOLO_MODEL_URL, str(YOLO_MODEL_PATH))
            print(f"✅ Modèle téléchargé: {YOLO_MODEL_PATH}")
        
        model = YOLO(str(YOLO_MODEL_PATH))
        print("✅ Modèle YOLO chargé avec succès!")
        return model
    
    def detect_faces(self, image, verbose=False):
        """
        Détecte les visages dans une image.
        
        Args:
            image: Image BGR (numpy array)
            verbose: Afficher les détails de détection
            
        Returns:
            results: Résultats de détection YOLO
        """
        results = self.model(
            image,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            verbose=verbose
        )
        return results
    
    def crop_face(self, image, boxes, index=0, target_size=None, margin=None):
        """
        Découpe et redimensionne un visage détecté.
        
        Args:
            image: Image BGR
            boxes: Boîtes de détection YOLO
            index: Index du visage à découper
            target_size: Taille cible (largeur, hauteur)
            margin: Marge autour du visage (pourcentage)
            
        Returns:
            Face redimensionnée ou None si erreur
        """
        if target_size is None:
            target_size = TARGET_SIZE
        if margin is None:
            margin = FACE_MARGIN
        
        if len(boxes) == 0:
            print("Aucun visage détecté!")
            return None
        
        if index >= len(boxes):
            print(f"Index {index} invalide. Seulement {len(boxes)} visage(s) détecté(s).")
            return None
        
        # Récupérer les coordonnées de la boîte
        box = boxes[index].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        
        # Ajouter une marge
        h, w = image.shape[:2]
        width = x2 - x1
        height = y2 - y1
        
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        x1_marge = max(0, x1 - margin_x)
        y1_marge = max(0, y1 - margin_y)
        x2_marge = min(w, x2 + margin_x)
        y2_marge = min(h, y2 + margin_y)
        
        # Découper l'image
        visage_decoupe = image[y1_marge:y2_marge, x1_marge:x2_marge]
        
        # Redimensionner
        visage_resize = cv2.resize(
            visage_decoupe,
            target_size,
            interpolation=cv2.INTER_LANCZOS4
        )
        
        return visage_resize
    
    def save_face(self, face, output_path):
        """
        Sauvegarde un visage découpé.
        
        Args:
            face: Image du visage
            output_path: Chemin de sauvegarde
        """
        cv2.imwrite(
            output_path,
            face,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
