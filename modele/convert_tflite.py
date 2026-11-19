"""Convertit le modèle .h5 en TFLite pour le déploiement."""
from pathlib import Path

import keras
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
modele = keras.models.load_model(ROOT / "modele_fertilite_sol.h5", compile=False)

try:
    convertisseur = tf.lite.TFLiteConverter.from_keras_model(modele)
    contenu = convertisseur.convert()
except Exception as e:
    print("Conversion directe impossible, passage par SavedModel :", e)
    modele.export(str(ROOT / "modele" / "saved_model_tmp"))
    convertisseur = tf.lite.TFLiteConverter.from_saved_model(
        str(ROOT / "modele" / "saved_model_tmp"))
    contenu = convertisseur.convert()

sortie = ROOT / "modele_fertilite_sol.tflite"
sortie.write_bytes(contenu)
print(f"OK : {sortie} ({sortie.stat().st_size / 1e6:.1f} Mo)")