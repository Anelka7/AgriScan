"""Entraînement MobileNetV2 (transfer learning) : fertile vs non_fertile.

Usage :  python modele\\train_model.py   (depuis n'importe quel dossier)
Le script doit se trouver dans un sous-dossier de la racine du projet
(à côté du dossier "dataset").
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)

# ---------- Configuration ----------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent                      # racine du projet (contient dataset/)
DATA = ROOT / "dataset"
MODEL_PATH = ROOT / "modele_fertilite_sol.h5"
MODEL_KERAS = SCRIPT_DIR / "modele_fertilite_sol.keras"
REPORT = SCRIPT_DIR / "rapport"
REPORT.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG_SIZE = (224, 224)
BATCH = 32
CLASSES = ["fertile", "non_fertile"]          # 0 = fertile, 1 = non_fertile

keras.utils.set_random_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE

if not DATA.exists():
    raise SystemExit(f"Dossier introuvable : {DATA}\nLance d'abord prepare_dataset.py")


# ---------- Données ----------
def load_split(name, shuffle=False):
    path = DATA / name
    if not path.exists():
        return None
    ds = keras.utils.image_dataset_from_directory(
        path, labels="inferred", label_mode="binary", class_names=CLASSES,
        image_size=IMG_SIZE, batch_size=BATCH, shuffle=shuffle, seed=SEED)
    # Pixels dans [0, 1] : identique au /255.0 de la vue Django
    return ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)


def count_images(split, cls):
    return len(list((DATA / split / cls).glob("*.jpg")))


# Augmentation appliquée UNIQUEMENT sur le train, hors du modèle sauvegardé
augment = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.2, value_range=(0, 1)),
    layers.RandomContrast(0.2, value_range=(0, 1)),
])

train_ds = (load_split("train", shuffle=True)
            .cache()
            .map(lambda x, y: (augment(x, training=True), y),
                 num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE))
val_ds = load_split("val").cache().prefetch(AUTOTUNE)
test_ds = load_split("test").prefetch(AUTOTUNE)
terrain_ds = load_split("test_terrain")

n0, n1 = count_images("train", "fertile"), count_images("train", "non_fertile")
total = n0 + n1
class_weight = {0: total / (2 * n0), 1: total / (2 * n1)}
print(f"\nTrain : fertile={n0}, non_fertile={n1} -> poids de classes = {class_weight}\n")


# ---------- Modèle ----------
def build_model():
    inputs = keras.Input(shape=(*IMG_SIZE, 3))             # pixels dans [0, 1]
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)   # [0,1] -> [-1,1] (MobileNetV2)

    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)                            # BatchNorm en mode inférence
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="agriscan_mobilenetv2"), base


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])


model, base = build_model()

# ---------- Phase 1 : entraînement de la tête ----------
print("=== PHASE 1 : tête de classification (base gelée) ===")
compile_model(model, 1e-3)
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True)
h1 = model.fit(train_ds, validation_data=val_ds, epochs=20,
               class_weight=class_weight, callbacks=[early_stop])

# ---------- Phase 2 : fine-tuning ----------
print("\n=== PHASE 2 : fine-tuning des 30 dernières couches ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
compile_model(model, 1e-5)
early_stop_ft = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True)
h2 = model.fit(train_ds, validation_data=val_ds, epochs=15,
               class_weight=class_weight, callbacks=[early_stop_ft])

# ---------- Sauvegarde IMMEDIATE (avant toute évaluation) ----------
model.save(MODEL_KERAS)
print(f"\nSauvegarde native : {MODEL_KERAS}")

try:
    model.save(MODEL_PATH)
    print(f"Modèle .h5 sauvegardé : {MODEL_PATH}")
    reloaded = keras.models.load_model(MODEL_PATH, compile=False)
    x_batch, _ = next(iter(test_ds))
    diff = np.abs(model.predict(x_batch, verbose=0)
                  - reloaded.predict(x_batch, verbose=0)).max()
    print(f"Contrôle rechargement .h5 : écart max = {diff:.2e} (doit être ~0)")
except Exception as e:
    print("ÉCHEC de la sauvegarde .h5 :", type(e).__name__, e)


# ---------- Courbes d'apprentissage ----------
def merged(key):
    return h1.history[key] + h2.history.get(key, [])


boundary = len(h1.history["loss"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (k, title) in zip(axes, [("accuracy", "Précision"), ("loss", "Perte")]):
    ax.plot(merged(k), label="train")
    ax.plot(merged("val_" + k), label="validation")
    ax.axvline(boundary - 1, color="gray", linestyle="--", label="début fine-tuning")
    ax.set_title(title)
    ax.set_xlabel("Époque")
    ax.legend()
plt.tight_layout()
plt.savefig(REPORT / "courbes_apprentissage.png", dpi=150)
plt.close()


# ---------- Évaluation ----------
def evaluate(ds, name):
    y_true, y_prob = [], []
    for x, y in ds:
        y_prob.extend(model.predict(x, verbose=0).ravel())
        y_true.extend(y.numpy().ravel())
    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)

    print(f"\n=== ÉVALUATION : {name} ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Greens", values_format="d")
    plt.title(f"Matrice de confusion : {name}")
    plt.savefig(REPORT / f"confusion_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    return classification_report(y_true, y_pred, target_names=CLASSES,
                                 output_dict=True)


metrics = {"test": evaluate(test_ds, "test")}
if terrain_ds is not None:
    metrics["test_terrain"] = evaluate(terrain_ds, "test_terrain")
(REPORT / "metriques.json").write_text(json.dumps(metrics, indent=2))
print(f"\nRapport (courbes, matrices, métriques) : {REPORT}")