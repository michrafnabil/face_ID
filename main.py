"""Point d'entrée principal du système de reconnaissance faciale."""

import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import sys
import argparse
import time
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.recognition import FaceRecognizer
from src.utils.embeddings import FaceEmbedder
from src.utils.preprocessing import WhitelistPreprocessor
from src.config import PROTOTYPES_PATH


def detect_faces():
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
        print("💡 Lancez d'abord: python main.py --build-whitelist")
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


def build_whitelist():
    """Construit la whitelist (preprocessing + embeddings)."""
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
    print("💡 Vous pouvez maintenant lancer: python main.py")
    print("="*60)


def preprocess_only():
    """Lance uniquement le preprocessing."""
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


def generate_embeddings_only():
    """Lance uniquement la génération d'embeddings."""
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


def main():
    """Point d'entrée principal avec gestion des arguments."""
    parser = argparse.ArgumentParser(
        description="🚀 Système de reconnaissance faciale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python main.py                    # Détection depuis webcam (par défaut)
  python main.py --build-whitelist  # Construire la whitelist
  python main.py --preprocess       # Preprocessing uniquement
  python main.py --embeddings       # Embeddings uniquement

Ou utilisez les scripts séparés:
  python scripts/run_detection.py
  python scripts/build_whitelist.py
  python scripts/preprocess.py
  python scripts/generate_embeddings.py
        """
    )
    
    parser.add_argument(
        '--build-whitelist',
        action='store_true',
        help='Construire la whitelist (preprocessing + embeddings)'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Lancer uniquement le preprocessing'
    )
    parser.add_argument(
        '--embeddings',
        action='store_true',
        help='Lancer uniquement la génération d\'embeddings'
    )
    
    args = parser.parse_args()
    
    if args.build_whitelist:
        build_whitelist()
    elif args.preprocess:
        preprocess_only()
    elif args.embeddings:
        generate_embeddings_only()
    else:
        # Par défaut: détection
        detect_faces()


if __name__ == "__main__":
    main()
