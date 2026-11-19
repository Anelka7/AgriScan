"""Chargement du modèle et prédiction. TensorFlow est importé à la demande."""
import threading

import numpy as np
from django.conf import settings

TAILLE = (224, 224)
COTE_MAX_PREALABLE = 512   # même réduction préalable que prepare_dataset.py

_model = None
_lock = threading.Lock()


def get_model():
    """Charge le modèle une seule fois (thread-safe)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import keras
                _model = keras.models.load_model(settings.MODELE_PATH, compile=False)
    return _model


def preparer(img):
    """Image PIL RGB -> tableau (1, 224, 224, 3) avec pixels dans [0, 1].

    Reproduit le pipeline d'entraînement : réduction à 512 px max,
    puis redimensionnement bilinéaire TensorFlow, puis /255.
    """
    import tensorflow as tf

    img = img.copy()
    img.thumbnail((COTE_MAX_PREALABLE, COTE_MAX_PREALABLE))
    arr = np.asarray(img, dtype="float32")
    arr = tf.image.resize(arr, TAILLE, method="bilinear").numpy() / 255.0
    return np.expand_dims(arr, axis=0)


def predire(img):
    """Retourne P(non_fertile) entre 0 et 1 (classe 1 = non fertile)."""
    x = preparer(img)
    return float(get_model().predict(x, verbose=0)[0][0])