"""Module de preprocessing de la whitelist."""

import os
import glob
import cv2
import numpy as np
from pathlib import Path

from src.config import (
    DATASET_FACES_DIR,
    WHITELIST_DIR,
    FACENET_SIZE,
    PREPROCESSING_MARGIN,
    MIN_FACE_SIZE,
    JPEG_QUALITY
)
from src.utils.detection import FaceDetector


class WhitelistPreprocessor:
    """Prétraitement des images de la whitelist."""
    
    def __init__(self, use_yolo_crop=True, delete_ignored=True):
        """
        Initialise le preprocessor.
        
        Args:
            use_yolo_crop: Utiliser YOLO pour cropper les visages
            delete_ignored: Supprimer les fichiers ignorés
        """
        self.use_yolo_crop = use_yolo_crop
        self.delete_ignored = delete_ignored
        self.detector = FaceDetector() if use_yolo_crop else None
        self.exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    
    def _ensure_dir(self, path):
        """Créer un répertoire s'il n'existe pas."""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    def _is_too_small(self, img, min_side=None):
        """Vérifie si l'image est trop petite."""
        if min_side is None:
            min_side = MIN_FACE_SIZE
        h, w = img.shape[:2]
        return min(h, w) < min_side
    
    def _crop_with_yolo(self, bgr_img):
        """
        Utilise YOLO pour détecter et cropper le visage principal.
        
        Args:
            bgr_img: Image BGR
            
        Returns:
            Crop du visage ou None si aucun visage détecté
        """
        results = self.detector.detect_faces(bgr_img, verbose=False)
        
        if len(results[0].boxes) == 0:
            return None
        
        # Prendre le visage avec le meilleur score
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax().item()
        box = boxes[best_idx].xyxy[0].cpu().numpy()
        
        x1, y1, x2, y2 = map(int, box)
        
        # Ajouter une marge
        h, w = bgr_img.shape[:2]
        width = x2 - x1
        height = y2 - y1
        
        margin_x = int(width * PREPROCESSING_MARGIN)
        margin_y = int(height * PREPROCESSING_MARGIN)
        
        x1_marge = max(0, x1 - margin_x)
        y1_marge = max(0, y1 - margin_y)
        x2_marge = min(w, x2 + margin_x)
        y2_marge = min(h, y2 + margin_y)
        
        face_crop = bgr_img[y1_marge:y2_marge, x1_marge:x2_marge]
        return face_crop
    
    def preprocess_whitelist(self, src_dir=None, dst_dir=None):
        """
        Prétraite toutes les images de la whitelist.
        
        Args:
            src_dir: Répertoire source
            dst_dir: Répertoire destination
            
        Returns:
            Statistiques de preprocessing
        """
        if src_dir is None:
            src_dir = DATASET_FACES_DIR
        if dst_dir is None:
            dst_dir = WHITELIST_DIR
        
        self._ensure_dir(dst_dir)
        
        total_in, total_out, total_skipped = 0, 0, 0
        total_deleted = 0
        no_face_detected = 0
        
        persons = [
            d for d in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, d))
        ]
        persons.sort()
        
        print(f"🔍 Traitement avec détection YOLO: {'OUI' if self.use_yolo_crop else 'NON'}")
        print(f"📁 Source: {src_dir}")
        print(f"📁 Destination: {dst_dir}")
        print(f"🗑️  Suppression des fichiers ignorés: {'OUI' if self.delete_ignored else 'NON'}\n")
        
        for person in persons:
            in_dir = os.path.join(src_dir, person)
            out_dir = os.path.join(dst_dir, person)
            self._ensure_dir(out_dir)
            
            paths = []
            for ext in self.exts:
                paths += glob.glob(os.path.join(in_dir, f"*{ext}"))
            paths.sort()
            
            person_out = 0
            person_skipped = 0
            person_deleted = 0
            saved_count = 0
            
            for p in paths:
                total_in += 1
                img_bgr = cv2.imread(p)
                
                if img_bgr is None:
                    total_skipped += 1
                    person_skipped += 1
                    if self.delete_ignored:
                        try:
                            os.remove(p)
                            total_deleted += 1
                            person_deleted += 1
                        except Exception as e:
                            print(f"⚠️ Erreur suppression {os.path.basename(p)}: {e}")
                    continue
                
                # Crop via YOLO
                if self.use_yolo_crop:
                    face_crop = self._crop_with_yolo(img_bgr)
                    if face_crop is None:
                        total_skipped += 1
                        person_skipped += 1
                        no_face_detected += 1
                        if self.delete_ignored:
                            try:
                                os.remove(p)
                                total_deleted += 1
                                person_deleted += 1
                            except Exception as e:
                                print(f"⚠️ Erreur suppression {os.path.basename(p)}: {e}")
                        continue
                else:
                    face_crop = img_bgr
                
                # Filtre qualité
                if (face_crop is None or face_crop.ndim != 3 or
                    face_crop.shape[2] != 3 or self._is_too_small(face_crop)):
                    total_skipped += 1
                    person_skipped += 1
                    continue
                
                # Resize à 160x160
                img_resized = cv2.resize(
                    face_crop,
                    (FACENET_SIZE, FACENET_SIZE),
                    interpolation=cv2.INTER_AREA
                )
                
                # Sauvegarder
                saved_count += 1
                out_path = os.path.join(out_dir, f"{saved_count:04d}.jpg")
                
                success = cv2.imwrite(
                    out_path,
                    img_resized,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
                
                if success:
                    total_out += 1
                    person_out += 1
                else:
                    print(f"⚠️ Erreur sauvegarde: {out_path}")
            
            if person_deleted > 0:
                print(f"✓ {person}: {person_out} visages extraits | {person_deleted} fichiers supprimés")
            else:
                print(f"✓ {person}: {person_out} visages extraits")
        
        # Afficher les statistiques
        print(f"\n{'='*60}")
        print(f"✅ Preprocessing terminé")
        print(f"📊 Statistiques:")
        print(f"   • Entrées totales: {total_in}")
        print(f"   • Visages extraits: {total_out}")
        print(f"   • Images ignorées: {total_skipped}")
        print(f"   • Aucun visage détecté: {no_face_detected}")
        if self.delete_ignored:
            print(f"   • Fichiers supprimés: {total_deleted}")
        print(f"📁 Dossier prêt: {dst_dir}")
        
        return {
            'total_in': total_in,
            'total_out': total_out,
            'total_skipped': total_skipped,
            'no_face_detected': no_face_detected,
            'total_deleted': total_deleted
        }
