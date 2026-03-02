"""
Script 1/3 : Génère les figures pour le rapport.
Exécuter depuis le dossier rapport/ :
    cd rapport
    python generate_figures.py

Produit dans rapport/figures/ :
  - loss_curve.png          (courbe de perte)
  - class_distribution.png  (distribution des classes)
  - sample_grid.png         (grille 2x3 d'images du dataset)
"""

import os, sys, collections
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset", "train")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CLASS_NAMES = ["ferme", "immeuble", "maison", "piscine", "usine", "villa"]
COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#00bcd4", "#9b59b6", "#f39c12"]

# ═══════════════════════════════════════════════════════
# 1. COURBE DE PERTE (training loss)
# ═══════════════════════════════════════════════════════
epochs = list(range(1, 51))
losses = [
    6.4510,
    5.4788,
    5.4477,
    5.0659,
    4.7124,
    4.7821,
    4.8435,
    4.5914,
    4.8182,
    4.5946,
    4.1841,
    4.2157,
    4.0422,
    4.1000,
    3.9791,
    4.1211,
    4.0705,
    3.7376,
    3.8286,
    3.6782,
    3.4651,
    3.2229,
    3.2462,
    3.5401,
    3.3791,
    3.1496,
    2.8646,
    2.9618,
    2.8494,
    2.7162,
    2.7798,
    2.7207,
    2.5218,
    2.4808,
    2.3099,
    2.2893,
    2.2209,
    2.2245,
    2.0517,
    1.9901,
    2.0030,
    1.9554,
    1.8214,
    1.8090,
    1.7708,
    1.7239,
    1.7095,
    1.6708,
    1.6832,
    1.6114,
]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(epochs, losses, color="#e74c3c", linewidth=2)
ax.fill_between(epochs, losses, alpha=0.15, color="#e74c3c")
ax.set_xlabel("Époque", fontsize=12)
ax.set_ylabel("Loss (moyenne par batch)", fontsize=12)
ax.set_title("Évolution de la perte d'entraînement", fontsize=14)
ax.set_xlim(1, 50)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "loss_curve.png"), dpi=200)
plt.close(fig)
print("[OK] loss_curve.png")

# ═══════════════════════════════════════════════════════
# 2. DISTRIBUTION DES CLASSES
# ═══════════════════════════════════════════════════════
labels_dir = os.path.join(DATASET_DIR, "labels")
counts = collections.Counter()
for fname in os.listdir(labels_dir):
    if not fname.endswith(".txt"):
        continue
    with open(os.path.join(labels_dir, fname)) as f:
        for line in f:
            line = line.strip()
            if line:
                cls = int(line.split()[0])
                counts[cls] += 1

class_counts = [counts.get(i, 0) for i in range(6)]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(CLASS_NAMES, class_counts, color=COLORS, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, class_counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 15,
        str(val),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
ax.set_xlabel("Classe", fontsize=12)
ax.set_ylabel("Nombre d'annotations", fontsize=12)
ax.set_title("Distribution des annotations par classe", fontsize=14)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=200)
plt.close(fig)
print("[OK] class_distribution.png")

# ═══════════════════════════════════════════════════════
# 3. GRILLE D'EXEMPLES (2x3 = 6 images du dataset)
# ═══════════════════════════════════════════════════════
images_dir = os.path.join(DATASET_DIR, "images")
all_imgs = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
# Pick 6 spaced samples
step = max(1, len(all_imgs) // 6)
selected = [all_imgs[i * step] for i in range(6)]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, img_name in zip(axes.flat, selected):
    img = Image.open(os.path.join(images_dir, img_name))
    ax.imshow(img)
    ax.set_title(img_name.split("_png")[0].replace("image_", "Image "), fontsize=10)
    ax.axis("off")
fig.suptitle("Exemples d'images du jeu d'entraînement", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "sample_grid.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("[OK] sample_grid.png")

# ═══════════════════════════════════════════════════════
# 4. GRILLE D'EXEMPLES AVEC BBOXES ANNOTÉES
# ═══════════════════════════════════════════════════════
from PIL import ImageDraw

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
bbox_colors = {i: c for i, c in enumerate(COLORS)}

for ax, img_name in zip(axes.flat, selected):
    img = Image.open(os.path.join(images_dir, img_name)).copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # find matching label
    label_name = img_name.replace(".jpg", ".txt")
    label_path = os.path.join(labels_dir, label_name)
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    xc, yc, bw, bh = [float(x) for x in parts[1:]]
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    color = bbox_colors.get(cls, "white")
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    ax.imshow(img)
    ax.set_title(img_name.split("_png")[0].replace("image_", "Image "), fontsize=10)
    ax.axis("off")

# legend
import matplotlib.patches as mpatches

legend_patches = [
    mpatches.Patch(color=COLORS[i], label=CLASS_NAMES[i]) for i in range(6)
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=6,
    fontsize=10,
    bbox_to_anchor=(0.5, -0.02),
)
fig.suptitle("Exemples d'images annotées (ground truth)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(
    os.path.join(FIGURES_DIR, "annotated_grid.png"), dpi=200, bbox_inches="tight"
)
plt.close(fig)
print("[OK] annotated_grid.png")

print(f"\n✅ Toutes les figures sont dans : {FIGURES_DIR}")
