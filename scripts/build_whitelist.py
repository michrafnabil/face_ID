"""Script de construction de la whitelist (preprocessing + embeddings)."""

import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import sys
import time
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import WhitelistPreprocessor
from src.utils.embeddings import FaceEmbedder


def main():
    """Construit la whitelist complète."""
    start_time = time.time()
    
    print("="*60)
    print("🔨 CONSTRUCTION DE LA WHITELIST")
    print("="*60)
    print()
    
    # Étape 1: Preprocessing
    print("📋 Étape 1/2: Preprocessing des images...")
    preprocessor = WhitelistPreprocessor(
        use_yolo_crop=True,
        delete_ignored=False
    )
    stats = preprocessor.preprocess_whitelist()
    print()
    
    # Étape 2: Génération des embeddings
    print("📋 Étape 2/2: Génération des embeddings...")
    embedder = FaceEmbedder()
    prototypes, refs = embedder.build_whitelist()
    
    # Résumé
    end_time = time.time()
    execution_time = end_time - start_time
    
    print()
    print("="*60)
    print("✅ Whitelist construite avec succès!")
    print(f"📊 {len(prototypes)} personnes enrôlées")
    print(f"⏱️  Temps d'exécution: {execution_time:.2f} secondes")
    print("💡 Vous pouvez maintenant lancer: python scripts/run_detection.py")
    print("="*60)


if __name__ == "__main__":
    main()
