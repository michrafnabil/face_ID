# 🚀 Système de Reconnaissance Faciale - Version 2

Système de reconnaissance faciale en temps réel utilisant YOLO pour la détection et FaceNet pour la reconnaissance, avec une architecture modulaire améliorée.

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

## 🎯 Avantages de cette Structure

### ✅ Organisation Modulaire
- **Séparation claire** entre configuration, utilitaires et scripts
- **Réutilisabilité** des modules dans différents contextes
- **Maintenance facilitée** avec des fichiers spécialisés

### ✅ Gestion des Données
- **data/prototypes/** : Stockage centralisé des modèles entraînés
- **data/results/** : Historique des détections avec timestamps
- **models/** : Modèles de deep learning séparés des données

### ✅ Scripts Dédiés
- Un script par tâche pour une utilisation claire
- Temps d'exécution affiché pour chaque script
- Messages informatifs et structurés

## 🚀 Installation

### 1. Cloner le projet
```bash
cd C:\Users\DELL\Downloads\face_recognition_project_v2
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 📖 Utilisation

### Méthode 1: Via main.py (Recommandé)

#### 1️⃣ Construire la whitelist (première utilisation)
```bash
python main.py --build-whitelist
```
**Fait:** Preprocessing + Génération d'embeddings en une seule commande

#### 2️⃣ Lancer la détection
```bash
python main.py
```
**Fait:** Détection depuis webcam + Reconnaissance + Sauvegarde résultat

#### 3️⃣ Autres options
```bash
python main.py --preprocess    # Preprocessing seul
python main.py --embeddings    # Embeddings seuls
python main.py --help          # Aide complète
```

### Méthode 2: Via les Scripts Dédiés

#### 🔨 Construction de la whitelist
```bash
python scripts/build_whitelist.py
```

#### 🎥 Détection et reconnaissance
```bash
python scripts/run_detection.py
```

#### 🔧 Preprocessing uniquement
```bash
python scripts/preprocess.py
```

#### 🧠 Embeddings uniquement
```bash
python scripts/generate_embeddings.py
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

### 🛡️ Robustesse
- Gestion des erreurs Git (Windows)
- Validation des images chargées
- Messages d'erreur informatifs

## 🆚 Différences avec la V1

| Aspect | V1 (Beta) | V2 (Nouvelle) |
|--------|-----------|---------------|
| **Structure** | Fichiers plats | Modulaire organisée |
| **Data** | models/ mixte | data/ séparé |
| **Scripts** | main.py unique | Scripts dédiés |
| **Résultats** | Racine du projet | data/results/ |
| **Temps d'exec** | ✅ Ajouté | ✅ Tous les scripts |
| **Imports** | src.* | src.utils.* |

## 🐛 Dépannage

### Problème: FileNotFoundError avec Git
**Solution:** Déjà corrigé via `os.environ['GIT_PYTHON_REFRESH'] = 'quiet'`

### Problème: Aucun prototype trouvé
**Solution:** Exécutez d'abord `python main.py --build-whitelist`

### Problème: Webcam ne fonctionne pas
**Solution:** Vérifiez que la webcam est connectée et autorisée

## 📄 License

Projet éducatif - Libre d'utilisation

## 👤 Auteur

Développé pour un système de reconnaissance faciale modulaire et professionnel.

---

**⏱️ Temps d'exécution typiques:**
- Preprocessing: ~5-10 secondes (50 images)
- Embeddings: ~3-5 secondes (4 personnes)
- Détection: ~2-3 secondes (avec webcam)
