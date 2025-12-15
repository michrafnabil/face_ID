# 🚀 Système de Reconnaissance Faciale 

Système de reconnaissance faciale en temps réel utilisant YOLO pour la détection et FaceNet pour la reconnaissance.

## 📁 Structure du Projet

```
face_recognition_project_v2/
├── main.py                     # Point d'entrée principal
├── requirements.txt            # Dépendances Python
├── README.md                   # Cette documentation
│
├── data/                       # Données du projet
│   ├── whitelist/             # Images de référence (optionnel)
│   ├── prototypes/            # Prototypes générés
│   │   ├── whitelist_proto.npz
│   │   └── whitelist_refs.npz
│   └── results/               # Résultats de détection
│
├── models/                     # Modèles de deep learning
│   └── yolov8n-face.pt        # Modèle YOLO (téléchargé automatiquement)
│
├── src/                        # Code source
│   ├── __init__.py
│   ├── config.py              # Configuration centrale
│   └── utils/                 # Modules utilitaires
│       ├── __init__.py
│       ├── detection.py       # Détection YOLO
│       ├── embeddings.py      # Embeddings FaceNet
│       ├── preprocessing.py   # Preprocessing
│       └── recognition.py     # Reconnaissance
│
└── scripts/                    # Scripts d'exécution
    ├── run_detection.py       # Lancer la détection
    ├── build_whitelist.py     # Construire la whitelist
    ├── preprocess.py          # Prétraiter uniquement
    └── generate_embeddings.py # Générer embeddings uniquement
```




## ⚙️ Configuration

Modifiez les paramètres dans `src/config.py`:

```python
# Chemins des datasets (à adapter)
DATASET_FACES_DIR = r"C:\Users\DELL\Downloads\dataset_faces"
WHITELIST_DIR = r"C:\Users\DELL\Downloads\whitelist_preprocessed"

# Seuil de reconnaissance (plus bas = plus strict)
RECOGNITION_THRESHOLD = 0.25

# Paramètres YOLO
YOLO_CONF_THRESHOLD = 0.5

# Taille des images
FACENET_SIZE = 160
```

## 📊 Fonctionnalités

### ✨ Détection de visages
- **YOLO v8** optimisé pour la détection faciale
- Détection multi-visages avec scores de confiance
- Cropping intelligent avec marges ajustables

### ✨ Reconnaissance faciale
- **FaceNet** (InceptionResnetV1) pour les embeddings
- Reconnaissance par distance cosinus
- Gestion de plusieurs personnes simultanément

### ✨ Preprocessing
- Détection automatique des visages dans les images
- Recadrage et redimensionnement standardisé
- Filtrage des images de mauvaise qualité

### ✨ Résultats
- Images annotées avec noms et distances
- Sauvegarde automatique avec timestamps
- Statistiques d'exécution détaillées

## 🔧 Workflow Complet

```
1. Préparer le dataset
   ├── dataset_faces/
   │   ├── Personne1/
   │   │   ├── image1.jpg
   │   │   └── image2.jpg
   │   └── Personne2/
   │       └── image1.jpg

2. Construire la whitelist
   └─> python main.py --build-whitelist
       ├── Preprocessing (YOLO crop + resize)
       └── Génération embeddings (FaceNet)

3. Lancer la détection
   └─> python main.py
       ├── Capture webcam
       ├── Détection visages
       ├── Reconnaissance
       └── Sauvegarde résultat
```

## 📈 Optimisations

### ⚡ Performance
- Utilisation GPU si disponible (CUDA)
- Chargement unique des modèles
- Embeddings pré-calculés et sauvegardés

### 💾 Stockage
- Prototypes compressés (.npz)
- Images JPEG haute qualité (95%)
- Résultats horodatés




---

**⏱️ Temps d'exécution typiques:**
- Preprocessing: ~5-10 secondes (50 images)
- Embeddings: ~3-5 secondes (4 personnes)
- Détection: ~2-3 secondes (avec webcam)

