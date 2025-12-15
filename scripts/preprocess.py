"""Script de preprocessing uniquement."""

import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import sys
import time
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import WhitelistPreprocessor


def main():
    """Lance le preprocessing du dataset."""
    start_time = time.time()
    
    print("="*60)
    print("🔧 PREPROCESSING DU DATASET")
    print("="*60)
    print()
    
    preprocessor = WhitelistPreprocessor(
        use_yolo_crop=True,
        delete_ignored=True
    )
    stats = preprocessor.preprocess_whitelist()
    
    # Résumé
    end_time = time.time()
    execution_time = end_time - start_time
    
    print()
    print("="*60)
    print("✅ Preprocessing terminé!")
    print(f"⏱️  Temps d'exécution: {execution_time:.2f} secondes")
    print("="*60)


if __name__ == "__main__":
    main()
