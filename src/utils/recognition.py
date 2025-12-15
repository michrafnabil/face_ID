"""Module de reconnaissance faciale."""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from src.config import RECOGNITION_THRESHOLD, RESULTS_DIR
from src.utils.detection import FaceDetector
from src.utils.embeddings import FaceEmbedder


class FaceRecognizer:
    """Système de reconnaissance faciale."""
    
    def __init__(self, prototypes=None, threshold=None):
        """
        Initialise le système de reconnaissance.
        
        Args:
            prototypes: Dictionnaire {nom: embedding}
            threshold: Seuil de distance pour la reconnaissance
        """
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.prototypes = prototypes or {}
        self.threshold = threshold or RECOGNITION_THRESHOLD
    
    def load_prototypes(self):
        """Charge les prototypes depuis le fichier."""
        self.prototypes = self.embedder.load_prototypes()
        return len(self.prototypes) > 0
    
    def recognize_face(self, face_bgr):
        """
        Reconnaît un visage en comparant son embedding aux prototypes.
        
        Args:
            face_bgr: Image du visage en BGR
            
        Returns:
            Tuple (nom, distance)
        """
        # Convertir BGR → RGB → PIL
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # Obtenir l'embedding
        emb = self.embedder.get_embedding(face_pil)
        
        # Comparer avec tous les prototypes
        best_name = "Inconnu"
        best_dist = float('inf')
        
        for name, proto in self.prototypes.items():
            # Distance cosinus : 1 - similarité
            dist = 1.0 - np.dot(emb, proto)
            if dist < best_dist:
                best_dist = dist
                best_name = name
        
        # Décision finale
        if best_dist > self.threshold:
            best_name = "Inconnu"
        
        return best_name, best_dist
    
    def recognize_from_image(self, image, annotate=True):
        """
        Détecte et reconnaît tous les visages dans une image.
        
        Args:
            image: Image BGR
            annotate: Annoter l'image avec les résultats
            
        Returns:
            Tuple (image_annotée, résultats)
            résultats = liste de dicts avec les infos de chaque visage
        """
        # Détection YOLO
        results = self.detector.detect_faces(image, verbose=False)
        num_faces = len(results[0].boxes)
        
        if num_faces == 0:
            return image, []
        
        face_results = []
        annotated = image.copy() if annotate else image
        
        for i, box in enumerate(results[0].boxes):
            # Extraire les coordonnées
            coords = box.xyxy[0].cpu().numpy()
            confidence_yolo = box.conf[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            
            # Extraire et redimensionner le visage
            face_crop = self.detector.crop_face(
                image,
                results[0].boxes,
                index=i,
                target_size=(160, 160),
                margin=0.15
            )
            
            if face_crop is None:
                continue
            
            # Reconnaissance
            name, distance = self.recognize_face(face_crop)
            
            # Stocker les résultats
            face_results.append({
                'index': i + 1,
                'name': name,
                'distance': distance,
                'bbox': (x1, y1, x2, y2),
                'confidence': float(confidence_yolo),
                'recognized': name != "Inconnu"
            })
            
            # Annoter l'image
            if annotate:
                color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                label = f"{name} ({distance:.3f})"
                
                # Background pour le texte
                (text_w, text_h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w, y1),
                    color,
                    -1
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        return annotated, face_results
    
    def recognize_from_webcam(self, duration=2, save_result=True, output_path=None):
        """
        Capture une image de la webcam et reconnaît les visages.
        
        Args:
            duration: Temps d'attente avant capture (secondes)
            save_result: Sauvegarder le résultat
            output_path: Chemin de sauvegarde personnalisé
            
        Returns:
            Tuple (image_annotée, résultats)
        """
        import time
        from datetime import datetime
        
        print("📷 Ouverture de la webcam...")
        cap = cv2.VideoCapture(0)
        time.sleep(0.5)
        
        print(f"⏱️  Capture dans {duration} secondes...")
        time.sleep(duration)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Erreur: Impossible de capturer depuis la webcam")
            return None, []
        
        print("✅ Image capturée")
        
        # Reconnaissance
        annotated, results = self.recognize_from_image(frame)
        
        # Afficher les résultats
        print(f"\n{'='*60}")
        print(f"🔍 {len(results)} visage(s) détecté(s)")
        print(f"{'='*60}\n")
        
        for res in results:
            print(f"Visage {res['index']}:")
            print(f"  👤 Identité: {res['name']}")
            print(f"  📏 Distance: {res['distance']:.4f}")
            print(f"  🎯 Confiance YOLO: {res['confidence']:.3f}")
            print(f"  ✓ Statut: {'RECONNU' if res['recognized'] else 'INCONNU'}")
            print()
        
        # Sauvegarder
        if save_result:
            if output_path is None:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = RESULTS_DIR / f"detection_{timestamp}.jpg"
            
            cv2.imwrite(str(output_path), annotated)
            print(f"📁 Résultat sauvegardé: {output_path}\n")
        
        return annotated, results
