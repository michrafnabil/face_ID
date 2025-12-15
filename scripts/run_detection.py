"""Script de détection et reconnaissance de visages depuis la webcam."""

import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import sys
import time
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.recognition import FaceRecognizer
from src.config import PROTOTYPES_PATH


def main():
    """Lance la détection et reconnaissance depuis la webcam."""
    start_time = time.time()
    
    print("="*60)
    print("🚀 DÉTECTION ET RECONNAISSANCE FACIALE")
    print("="*60)
    print()
    
    # Initialiser le système
    recognizer = FaceRecognizer()
    
    # Charger les prototypes
    print("📂 Chargement des prototypes...")
    if not PROTOTYPES_PATH.exists():
        print("❌ Aucun prototype trouvé!")
        print("💡 Lancez d'abord: python scripts/build_whitelist.py")
        return
    
    if not recognizer.load_prototypes():
        print("❌ Erreur lors du chargement des prototypes")
        return
    
    print(f"✅ {len(recognizer.prototypes)} personnes dans la whitelist:")
    for name in recognizer.prototypes.keys():
        print(f"   • {name}")
    print()
    
    # Lancer la reconnaissance
    print("🎥 Lancement de la détection...")
    annotated, results = recognizer.recognize_from_webcam(
        duration=0.5,
        save_result=True
    )
    
    if annotated is not None:
        print("✅ Détection terminée!")
    else:
        print("❌ Échec de la détection")
    
    # Temps d'exécution
    end_time = time.time()
    execution_time = end_time - start_time
    print()
    print("="*60)
    print(f"⏱️  Temps d'exécution total: {execution_time:.2f} secondes")
    print("="*60)


if __name__ == "__main__":
    main()
