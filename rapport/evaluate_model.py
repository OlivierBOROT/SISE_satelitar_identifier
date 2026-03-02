"""
Script 2/3 : Évaluation quantitative du modèle (mAP, Precision, Recall, matrice de confusion)
Exécuter depuis le dossier rapport/ :
    cd rapport
    python evaluate_model.py

Produit dans rapport/figures/ :
  - confusion_matrix.png
  - precision_recall_per_class.png
  - metrics.txt  (valeurs à copier dans le rapport)

Produit dans rapport/figures/ :
  - prediction_success_1.png ... prediction_success_3.png
  - prediction_failure_1.png ... prediction_failure_3.png
"""

import os, sys, glob, collections
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

import torch
import yolov5
from yolov5.models.yolo import Model
from yolov5.utils.general import check_yaml, yaml_load, non_max_suppression

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset", "train")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

CLASS_NAMES = ["ferme", "immeuble", "maison", "piscine", "usine", "villa"]
NC = len(CLASS_NAMES)
COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#00bcd4", "#9b59b6", "#f39c12"]
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45

# ═══════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════
print("Chargement du modèle...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolov5_dir = Path(yolov5.__file__).resolve().parent
model_cfg = yolov5_dir / "models" / "yolov5n.yaml"

model = Model(str(model_cfg), ch=3, nc=NC).to(device)
weights_path = os.path.join(PROJECT_DIR, "notebooks", "yolov5n_custom.pt")
checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()
model.float()
print(f"[OK] Modèle chargé depuis {weights_path}")


# ═══════════════════════════════════════════════════════
# UTIL FUNCTIONS
# ═══════════════════════════════════════════════════════
def letterbox_image(img_pil, new_size=640, color=(114, 114, 114)):
    img_pil = img_pil.convert("RGB")
    w0, h0 = img_pil.size
    r = min(new_size / w0, new_size / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = (new_size - new_unpad[0]) / 2
    dh = (new_size - new_unpad[1]) / 2
    img_resized = img_pil.resize(new_unpad, Image.Resampling.BILINEAR)
    new_img = Image.new("RGB", (new_size, new_size), color)
    new_img.paste(img_resized, (int(dw), int(dh)))
    img_tensor = torch.from_numpy(np.array(new_img)).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0), r, (dw, dh), (w0, h0)


def scale_boxes(boxes, gain, pad, original_size):
    pw, ph = pad
    boxes[:, [0, 2]] -= pw
    boxes[:, [1, 3]] -= ph
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_size[0])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_size[1])
    return boxes


def load_gt_boxes(label_path, img_w, img_h):
    """Load YOLO format ground truth and convert to xyxy pixels."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = [float(x) for x in parts[1:]]
            x1 = (xc - bw / 2) * img_w
            y1 = (yc - bh / 2) * img_h
            x2 = (xc + bw / 2) * img_w
            y2 = (yc + bh / 2) * img_h
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# ═══════════════════════════════════════════════════════
# RUN INFERENCE ON ALL DATASET IMAGES
# ═══════════════════════════════════════════════════════
print("Inférence sur toutes les images du dataset...")
all_images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))

# Per-class stats
tp_per_class = collections.Counter()
fp_per_class = collections.Counter()
fn_per_class = collections.Counter()
confusion = np.zeros((NC, NC), dtype=int)  # confusion[gt][pred]
total_gt = 0
total_pred = 0

# For qualitative results - track per-image scores
image_scores = []  # (img_path, n_tp, n_fp, n_fn, detections)

for img_idx, img_path in enumerate(all_images):
    img_pil = Image.open(img_path).convert("RGB")
    w0, h0 = img_pil.size
    img_tensor, r, pad, orig_size = letterbox_image(img_pil, 640)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        preds = model(img_tensor)
        preds = non_max_suppression(
            preds[0], conf_thres=CONF_THRESHOLD, iou_thres=NMS_IOU_THRESHOLD
        )

    # Predicted boxes
    pred_boxes = []
    if preds[0] is not None and len(preds[0]):
        det = preds[0].clone()
        det[:, :4] = scale_boxes(det[:, :4].clone(), r, pad, orig_size)
        for *xyxy, conf, cls_id in det:
            pred_boxes.append(
                (
                    int(cls_id),
                    xyxy[0].item(),
                    xyxy[1].item(),
                    xyxy[2].item(),
                    xyxy[3].item(),
                    conf.item(),
                )
            )

    # Ground truth boxes
    label_fname = os.path.basename(img_path).replace(".jpg", ".txt")
    label_path = os.path.join(LABELS_DIR, label_fname)
    gt_boxes = load_gt_boxes(label_path, w0, h0)

    total_gt += len(gt_boxes)
    total_pred += len(pred_boxes)

    # Match predictions to GT
    gt_matched = [False] * len(gt_boxes)
    img_tp = 0
    img_fp = 0

    for pred_cls, px1, py1, px2, py2, conf in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, (gt_cls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= IOU_THRESHOLD and best_gt_idx >= 0:
            gt_cls = gt_boxes[best_gt_idx][0]
            gt_matched[best_gt_idx] = True
            if pred_cls == gt_cls:
                tp_per_class[pred_cls] += 1
                img_tp += 1
            else:
                fp_per_class[pred_cls] += 1
                fn_per_class[gt_cls] += 1
                img_fp += 1
            confusion[gt_cls][pred_cls] += 1
        else:
            fp_per_class[pred_cls] += 1
            img_fp += 1

    # Unmatched GT = false negatives
    img_fn = 0
    for gt_idx, matched in enumerate(gt_matched):
        if not matched:
            gt_cls = gt_boxes[gt_idx][0]
            fn_per_class[gt_cls] += 1
            img_fn += 1

    image_scores.append((img_path, img_tp, img_fp, img_fn, pred_boxes, gt_boxes))

    if (img_idx + 1) % 20 == 0:
        print(f"  {img_idx + 1}/{len(all_images)} images traitées...")

print(f"[OK] Inférence terminée : {total_gt} GT, {total_pred} prédictions")

# ═══════════════════════════════════════════════════════
# COMPUTE METRICS
# ═══════════════════════════════════════════════════════
precision_per_class = {}
recall_per_class = {}
f1_per_class = {}
ap_per_class = {}

for c in range(NC):
    tp = tp_per_class[c]
    fp = fp_per_class[c]
    fn = fn_per_class[c]
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    precision_per_class[c] = p
    recall_per_class[c] = r
    f1_per_class[c] = f1

total_tp = sum(tp_per_class.values())
total_fp = sum(fp_per_class.values())
total_fn = sum(fn_per_class.values())
overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
overall_f1 = (
    2 * overall_p * overall_r / (overall_p + overall_r)
    if (overall_p + overall_r) > 0
    else 0
)
mAP50 = np.mean([precision_per_class[c] for c in range(NC)])

# Save metrics to file
metrics_path = os.path.join(FIGURES_DIR, "metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("RÉSULTATS D'ÉVALUATION - à copier dans le rapport LaTeX\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total images     : {len(all_images)}\n")
    f.write(f"Total GT boxes   : {total_gt}\n")
    f.write(f"Total predictions: {total_pred}\n\n")

    f.write("─── Par classe ───\n")
    f.write(
        f"{'Classe':<12} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Recall':>7} {'F1':>7}\n"
    )
    for c in range(NC):
        f.write(
            f"{CLASS_NAMES[c]:<12} {tp_per_class[c]:>5} {fp_per_class[c]:>5} {fn_per_class[c]:>5} "
            f"{precision_per_class[c]:>7.3f} {recall_per_class[c]:>7.3f} {f1_per_class[c]:>7.3f}\n"
        )

    f.write(
        f"\n{'TOTAL':<12} {total_tp:>5} {total_fp:>5} {total_fn:>5} "
        f"{overall_p:>7.3f} {overall_r:>7.3f} {overall_f1:>7.3f}\n"
    )
    f.write(f"\nmAP@0.5 (approx.) : {mAP50:.3f}\n")

    f.write("\n─── LaTeX (copier-coller) ───\n")
    for c in range(NC):
        f.write(
            f"\\texttt{{{CLASS_NAMES[c]}}} & {tp_per_class[c]} & {fp_per_class[c]} & {fn_per_class[c]} "
            f"& {precision_per_class[c]:.3f} & {recall_per_class[c]:.3f} & {f1_per_class[c]:.3f} \\\\\n"
        )
    f.write(f"\\midrule\n")
    f.write(
        f"\\textbf{{Total}} & {total_tp} & {total_fp} & {total_fn} "
        f"& {overall_p:.3f} & {overall_r:.3f} & {overall_f1:.3f} \\\\\n"
    )
    f.write(f"\\textbf{{mAP@0.5}} & \\multicolumn{{6}}{{c}}{{{mAP50:.3f}}} \\\\\n")

print(f"[OK] Métriques sauvegardées dans {metrics_path}")

# ═══════════════════════════════════════════════════════
# CONFUSION MATRIX
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6.5))
# Normalize
conf_norm = confusion.astype(float)
row_sums = conf_norm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
conf_norm = conf_norm / row_sums

im = ax.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
for i in range(NC):
    for j in range(NC):
        val = confusion[i][j]
        pct = conf_norm[i][j]
        color = "white" if pct > 0.5 else "black"
        ax.text(
            j,
            i,
            f"{val}\n({pct:.0%})",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
        )

ax.set_xticks(range(NC))
ax.set_yticks(range(NC))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Classe prédite", fontsize=12)
ax.set_ylabel("Classe réelle (GT)", fontsize=12)
ax.set_title("Matrice de confusion (IoU ≥ 0.5)", fontsize=14)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=200)
plt.close(fig)
print("[OK] confusion_matrix.png")

# ═══════════════════════════════════════════════════════
# PRECISION / RECALL BAR CHART
# ═══════════════════════════════════════════════════════
x = np.arange(NC)
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
p_vals = [precision_per_class[c] for c in range(NC)]
r_vals = [recall_per_class[c] for c in range(NC)]
bars1 = ax.bar(
    x - width / 2, p_vals, width, label="Precision", color="#3498db", edgecolor="white"
)
bars2 = ax.bar(
    x + width / 2, r_vals, width, label="Recall", color="#e74c3c", edgecolor="white"
)

for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
for bar in bars2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Precision et Recall par classe (IoU ≥ 0.5)", fontsize=14)
ax.set_ylim(0, 1.15)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "precision_recall_per_class.png"), dpi=200)
plt.close(fig)
print("[OK] precision_recall_per_class.png")


# ═══════════════════════════════════════════════════════
# QUALITATIVE: 3 BEST (successes) + 3 WORST (failures)
# ═══════════════════════════════════════════════════════
def save_prediction_image(img_path, pred_boxes, gt_boxes, out_path, title=""):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Draw GT in green dashed (solid as approx)
    for cls, x1, y1, x2, y2 in gt_boxes:
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)

    # Draw predictions in red
    for cls, x1, y1, x2, y2, conf in pred_boxes:
        color = COLORS[cls] if cls < len(COLORS) else "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        draw.text((x1, max(0, y1 - 14)), label, fill="white", font=font)

    # Save via matplotlib for title
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    # Add legend
    legend_patches = [
        mpatches.Patch(color="lime", label="Ground Truth"),
    ] + [
        mpatches.Patch(color=COLORS[i], label=f"Pred: {CLASS_NAMES[i]}")
        for i in range(NC)
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Sort by success rate (tp / (tp+fp+fn))
def score_fn(entry):
    _, tp, fp, fn, _, _ = entry
    total = tp + fp + fn
    return tp / total if total > 0 else 0


# Filter images with at least some GT
images_with_gt = [
    (p, tp, fp, fn, pred, gt) for p, tp, fp, fn, pred, gt in image_scores if len(gt) > 0
]
images_with_gt.sort(key=score_fn, reverse=True)

# Best 3
for i, (img_path, tp, fp, fn, pred, gt) in enumerate(images_with_gt[:3]):
    name = os.path.basename(img_path).split("_png")[0]
    title = f"Succès : {name} (TP={tp}, FP={fp}, FN={fn})"
    out = os.path.join(FIGURES_DIR, f"prediction_success_{i + 1}.png")
    save_prediction_image(img_path, pred, gt, out, title)
    print(f"[OK] prediction_success_{i + 1}.png")

# Worst 3 (with some predictions)
images_with_pred = [
    (p, tp, fp, fn, pred, gt)
    for p, tp, fp, fn, pred, gt in image_scores
    if len(pred) > 0 and len(gt) > 0
]
images_with_pred.sort(key=score_fn)
for i, (img_path, tp, fp, fn, pred, gt) in enumerate(images_with_pred[:3]):
    name = os.path.basename(img_path).split("_png")[0]
    title = f"Erreurs : {name} (TP={tp}, FP={fp}, FN={fn})"
    out = os.path.join(FIGURES_DIR, f"prediction_failure_{i + 1}.png")
    save_prediction_image(img_path, pred, gt, out, title)
    print(f"[OK] prediction_failure_{i + 1}.png")

print(f"\n✅ Toutes les figures d'évaluation sont dans : {FIGURES_DIR}")
print(f"📊 Ouvre {metrics_path} pour voir les métriques à reporter dans le rapport.")
