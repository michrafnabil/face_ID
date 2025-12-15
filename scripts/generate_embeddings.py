"""Script de génération d'embeddings uniquement."""

import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import sys
import time
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.embeddings import FaceEmbedder


def main():
    """Lance la génération des embeddings."""
    start_time = time.time()
    
    print("="*60)
    print("🧠 GÉNÉRATION DES EMBEDDINGS")
    print("="*60)
    print()
    
    embedder = FaceEmbedder()
    prototypes, refs = embedder.build_whitelist()
    
    # Résumé
    end_time = time.time()
    execution_time = end_time - start_time
    
    print()
    print("="*60)
    print("✅ Embeddings générés!")
    print(f"📊 {len(prototypes)} personnes enrôlées")
    print(f"⏱️  Temps d'exécution: {execution_time:.2f} secondes")
    print("="*60)


if __name__ == "__main__":
    main()
