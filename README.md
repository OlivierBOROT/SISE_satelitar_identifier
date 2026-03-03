# SISE Satellite Identifier

**Détection de bâtiments sur imagerie satellitaire avec YOLOv5 Nano**

Projet réalisé dans le cadre du cours *Deep Learning for Computer Vision* — Master SISE, Université Lumière Lyon 2 (2025/2026).

Olivier BOROT - Marin NAGY
---

## Objectif

Détecter et classifier automatiquement **6 types de structures** sur des images satellites haute résolution :

| ID | Classe | Description |
|----|--------|-------------|
| 0 | `ferme` | Exploitation agricole |
| 1 | `immeuble` | Immeuble résidentiel/bureaux |
| 2 | `maison` | Maison individuelle |
| 3 | `piscine` | Piscine |
| 4 | `usine` | Bâtiment industriel |
| 5 | `villa` | Villa / grande maison |

---

## Architecture du projet

```
├── consignes.txt              # Consignes du projet
├── pyproject.toml             # Dépendances Python
├── README.md
│
├── data/                      # Données brutes et traitées
│   ├── raw_images/            # Images satellitaires brutes (512×542)
│   ├── cropped_images/        # Images recadrées (512×512, sans filigrane)
│   ├── raw_polygons/          # Polygones OSM bruts (GeoJSON)
│   ├── cleaned_polygons/      # Polygones nettoyés + labels YOLO
│   ├── annotated_images/      # Images annotées (visualisation)
│   └── scraped_images/        # Images supplémentaires
│
├── dataset/                   # Dataset final (format Roboflow/YOLO)
│   ├── data.yaml              # Configuration classes + chemins
│   ├── train/images/          # Images d'entraînement (640×640)
│   └── train/labels/          # Labels YOLO correspondants
│
├── miscellaneous/             # Modules utilitaires Python
│   ├── ask_mapbox_for_image.py
│   ├── clean_overlapping_bboxes.py
│   ├── create_bbox_from_coordinates.py
│   ├── create_bbox_from_city_coordinates.py
│   ├── format_yolo_labels.py
│   ├── get_random_points_in_bbox.py
│   └── make_a_gif_for_coordinates.py
│
├── notebooks/                 # Notebooks Jupyter
│   ├── 00 - Pipeline_pour_5_coordonnées.ipynb   # Prototype (5 images)
│   ├── 01 - Pipeline_complète.ipynb              # Pipeline complète
│   ├── 02 - Training.ipynb                       # Entraînement YOLOv5n
│   ├── 03 - Testing_model.ipynb                  # Inférence et évaluation
│   └── yolov5n_custom.pt                         # Poids du modèle entraîné
│
├── rapport/                   # Rapport LaTeX
│   └── rapport.tex
│
├── tests/                     # Scripts de test/expérimentation
│   ├── Test.py … Test5.py
│
└── yolov5/                    # Configuration YOLOv5
    └── hyp.scratch-low.yaml   # Hyperparamètres d'entraînement
```

---

## Pipeline de données

```
Coordonnées GPS (20 villes + 100 zones rurales)
        │
        ▼
Bounding box (Lambert 93, 0.4m/px, 512×542)
        │
        ├──► Mapbox Static API ──► Image satellite brute
        │                                │
        │                                ▼
        │                         Recadrage (512×512)
        │
        └──► OSM (osmnx) ──► Polygones bâtiments
                                    │
                                    ▼
                            Nettoyage chevauchements
                                    │
                                    ▼
                            Labels YOLO (normalisés)
                                    │
                                    ▼
                         Roboflow (640×640 + vérification)
```

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/<user>/SISE_satelitar_identifier.git
cd SISE_satelitar_identifier

# Créer un environnement virtuel (Python ≥ 3.13)
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate  # Linux/macOS

# Installer les dépendances
pip install -e .
```

### Variables d'environnement

Créer un fichier `.env` à la racine :
```env
MAPBOX_ACCESS_TOKEN=pk.xxxxxxxxxxxxxxxxxxxxxxx
```

---

## Utilisation

### 1. Collecte de données
Exécuter le notebook **`01 - Pipeline_complète.ipynb`** pour télécharger les images satellites et générer les annotations YOLO.

### 2. Entraînement
Exécuter le notebook **`02 - Training.ipynb`** pour fine-tuner YOLOv5n sur le dataset. Les poids sont sauvegardés dans `yolov5n_custom.pt`.

### 3. Inférence
Exécuter le notebook **`03 - Testing_model.ipynb`** pour lancer la détection sur de nouvelles images.

---

## Technologies

- **PyTorch** + **YOLOv5** — Détection d'objets
- **Mapbox Static API** — Imagerie satellitaire
- **OSMnx / OpenStreetMap** — Annotations de bâtiments
- **Roboflow** — Gestion et vérification du dataset
- **GeoPandas / Shapely / PyProj** — Traitement géospatial

---

## Licence

Dataset : [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Modèle YOLOv5 : [AGPL-3.0](https://github.com/ultralytics/yolov5/blob/master/LICENSE)
