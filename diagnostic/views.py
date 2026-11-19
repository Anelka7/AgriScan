import io
import logging
import time
import uuid
from pathlib import Path

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from PIL import Image, ImageOps, UnidentifiedImageError

from . import ia

logger = logging.getLogger(__name__)

EXTENSIONS_OK = {".jpg", ".jpeg", ".png"}
FORMATS_OK = {"JPEG", "PNG"}
SEUIL_INCERTAIN = 70.0            # en dessous : "résultat peu sûr"
DUREE_CONSERVATION_S = 3600       # photos supprimées après 1 h

CONSEILS = {
    True: ("Le sol paraît sombre et riche en matière organique, un bon signe pour "
           "de nombreuses cultures. Entretenez-le avec du compost et gardez-le "
           "couvert (paillage) pour conserver son humidité."),
    False: ("Le sol paraît sec ou pauvre en matière organique. Apportez du compost "
            "ou du fumier, couvrez le sol (paillage) et commencez par des cultures "
            "peu exigeantes ou enrichissantes comme l'arachide ou le niébé."),
}


def _stockage():
    return FileSystemStorage(
        location=settings.MEDIA_ROOT / "uploads",
        base_url=settings.MEDIA_URL + "uploads/",
    )


def _purger_anciennes_photos(fs):
    dossier = Path(fs.location)
    if not dossier.exists():
        return
    limite = time.time() - DUREE_CONSERVATION_S
    for f in dossier.glob("*.jpg"):
        try:
            if f.stat().st_mtime < limite:
                f.unlink()
        except OSError:
            pass


def _lire_image(photo):
    """Valide le fichier envoyé. Retourne (image PIL RGB, None) ou (None, message)."""
    msg_format = "Format non pris en charge. Choisissez une photo JPG ou PNG."

    if Path(photo.name).suffix.lower() not in EXTENSIONS_OK:
        return None, msg_format
    if photo.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"Photo trop lourde (maximum {settings.MAX_UPLOAD_MB} Mo)."
    try:
        img = Image.open(photo)
        if img.format not in FORMATS_OK:
            return None, msg_format
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None, "Ce fichier n'est pas une image valide. Reprenez la photo."

    img = ImageOps.exif_transpose(img).convert("RGB")
    return img, None


def _sauver_apercu(fs, img):
    """Enregistre un aperçu JPEG léger avec un nom unique. Retourne son URL."""
    apercu = img.copy()
    apercu.thumbnail((800, 800))
    tampon = io.BytesIO()
    apercu.save(tampon, format="JPEG", quality=85)
    nom = fs.save(f"{uuid.uuid4().hex}.jpg", ContentFile(tampon.getvalue()))
    return fs.url(nom)


def analyser_sol(request):
    contexte = {}

    if request.method == "POST":
        photo = request.FILES.get("photo_sol")

        if photo is None:
            contexte["erreur"] = ("Veuillez prendre ou choisir une photo du sol "
                                  "avant de lancer l'analyse.")
        else:
            img, erreur = _lire_image(photo)
            if erreur:
                contexte["erreur"] = erreur
            else:
                try:
                    proba_pauvre = ia.predire(img)
                except Exception:
                    logger.exception("Échec de la prédiction")
                    contexte["erreur"] = ("L'analyse n'a pas pu aboutir. "
                                          "Réessayez dans un instant.")
                else:
                    est_fertile = proba_pauvre < 0.5
                    confiance = round((1 - proba_pauvre if est_fertile
                                       else proba_pauvre) * 100, 1)
                    fs = _stockage()
                    _purger_anciennes_photos(fs)
                    contexte = {
                        "resultat": True,
                        "est_fertile": est_fertile,
                        "statut": "Sol Fertile" if est_fertile else "Sol Non Fertile",
                        "confiance": confiance,
                        "incertain": confiance < SEUIL_INCERTAIN,
                        "conseil": CONSEILS[est_fertile],
                        "image_url": _sauver_apercu(fs, img),
                    }

    return render(request, "diagnostic/index.html", contexte)