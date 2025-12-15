"""Module de génération d'embeddings avec FaceNet."""

import os
import glob
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import traceback

from src.config import (
    FACENET_SIZE,
    EMBEDDING_DIM,
    MAX_IMAGES_PER_PERSON,
    WHITELIST_DIR,
    PROTOTYPES_PATH,
    REFERENCES_PATH
)


class FaceEmbedder:
    """Génère des embeddings de visages avec FaceNet."""
    
    def __init__(self):
        """Initialise l'embedder FaceNet."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Device: {self.device}")
        
        self.embedder = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(self.device)
        
        print("✅ Modèle FaceNet chargé avec succès!")
    
    def _preprocess_crop(self, pil_img):
        """
        Prétraite une image pour FaceNet.
        
        Args:
            pil_img: Image PIL
            
        Returns:
            Tensor PyTorch normalisé
        """
        img = pil_img.convert("RGB")
        
        # Resize to 160x160
        if img.size != (FACENET_SIZE, FACENET_SIZE):
            img = img.resize((FACENET_SIZE, FACENET_SIZE), Image.LANCZOS)
        
        x = np.asarray(img).astype(np.float32)
        x = (x - 127.5) / 128.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1,3,160,160]
        return x
    
    @torch.no_grad()
    def get_embedding(self, pil_img):
        """
        Génère un embedding à partir d'une image.
        
        Args:
            pil_img: Image PIL
            
        Returns:
            Embedding normalisé (512D)
        """
        x = self._preprocess_crop(pil_img).to(self.device)
        emb = self.embedder(x).cpu().numpy().flatten()  # 512D
        emb = emb / (np.linalg.norm(emb) + 1e-12)  # L2 normalize
        return emb
    
    def build_whitelist(self, dataset_dir=None, max_imgs_per_person=None):
        """
        Construit la whitelist à partir d'un dataset.
        
        Args:
            dataset_dir: Répertoire du dataset
            max_imgs_per_person: Nombre max d'images par personne
            
        Returns:
            Tuple (prototypes, embeddings_references)
        """
        if dataset_dir is None:
            dataset_dir = WHITELIST_DIR
        if max_imgs_per_person is None:
            max_imgs_per_person = MAX_IMAGES_PER_PERSON
        
        prototypes = {}
        all_embs = {}
        
        # Vérifier si le dossier existe
        if not os.path.exists(dataset_dir):
            print(f"❌ Erreur: Le dossier {dataset_dir} n'existe pas!")
            return prototypes, all_embs
        
        for person in sorted(os.listdir(dataset_dir)):
            pdir = os.path.join(dataset_dir, person)
            if not os.path.isdir(pdir):
                continue
            
            paths = sorted(glob.glob(os.path.join(pdir, "*.*")))[:max_imgs_per_person]
            if not paths:
                print(f"⚠️ Aucune image trouvée pour {person}")
                continue
            
            embs = []
            errors = 0
            for p in paths:
                try:
                    img = Image.open(p)
                    img.load()
                    
                    if img.mode not in ['RGB', 'L', 'RGBA']:
                        print(f"⚠️ Mode image invalide pour {os.path.basename(p)}: {img.mode}")
                        errors += 1
                        continue
                    
                    embs.append(self.get_embedding(img))
                    
                except Exception as e:
                    errors += 1
                    print(f"⚠️ Erreur avec {os.path.basename(p)}: {str(e)}")
                    if errors == 1:
                        traceback.print_exc()
                    continue
            
            if len(embs) == 0:
                print(f"⚠️ Aucun embedding valide pour {person} ({errors} erreurs)")
                continue
            
            embs = np.stack(embs)
            proto = embs.mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-12)
            
            prototypes[person] = proto
            all_embs[person] = embs[::max(1, len(embs)//10)]
            print(f"✓ {person}: {len(embs)} images traitées ({errors} erreurs)")
        
        if prototypes:
            # Sauvegarder
            PROTOTYPES_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(PROTOTYPES_PATH), **prototypes)
            np.savez(str(REFERENCES_PATH), **all_embs)
            print(f"\n✅ OK: {len(prototypes)} personnes enrôlées.")
            print(f"📁 Prototypes sauvegardés: {PROTOTYPES_PATH}")
            print(f"📁 Références sauvegardées: {REFERENCES_PATH}")
        else:
            print(f"\n❌ Aucune personne n'a pu être enrôlée.")
        
        return prototypes, all_embs
    
    def load_prototypes(self, proto_path=None):
        """
        Charge les prototypes depuis un fichier.
        
        Args:
            proto_path: Chemin du fichier de prototypes
            
        Returns:
            Dictionnaire {nom: embedding}
        """
        if proto_path is None:
            proto_path = PROTOTYPES_PATH
        
        try:
            proto_data = np.load(str(proto_path))
            prototypes = {name: proto_data[name] for name in proto_data.files}
            print(f"✅ {len(prototypes)} prototypes chargés")
            return prototypes
        except FileNotFoundError:
            print(f"❌ Fichier introuvable: {proto_path}")
            return {}
