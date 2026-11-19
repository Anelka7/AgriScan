"""Compare le pipeline de production (TFLite + numpy) au modèle Keras .h5 de référence."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from diagnostic import ia

ref = keras.models.load_model(ROOT / "modele_fertilite_sol.h5", compile=False)
n = ok_prod = ok_ref = identiques = 0
ecart_max = 0.0

for verite, classe in enumerate(["fertile", "non_fertile"]):
    for chemin in sorted((ROOT / "dataset" / "test" / classe).glob("*.jpg")):
        img = Image.open(chemin).convert("RGB")
        p_prod = ia.predire(img)

        arr = tf.image.resize(np.asarray(img, dtype="float32"), (224, 224)).numpy() / 255.0
        p_ref = float(ref.predict(arr[None], verbose=0)[0][0])

        n += 1
        ok_prod += int((p_prod >= 0.5) == bool(verite))
        ok_ref += int((p_ref >= 0.5) == bool(verite))
        identiques += int((p_prod >= 0.5) == (p_ref >= 0.5))
        ecart_max = max(ecart_max, abs(p_prod - p_ref))

print(f"Images testées            : {n}")
print(f"Exactitude Keras (.h5)    : {ok_ref / n:.3f}")
print(f"Exactitude production     : {ok_prod / n:.3f}")
print(f"Décisions identiques      : {identiques}/{n}")
print(f"Écart maximal de sortie   : {ecart_max:.4f}")