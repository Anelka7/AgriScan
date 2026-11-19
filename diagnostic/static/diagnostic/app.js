(function () {
  "use strict";

  const form = document.getElementById("form-scan");
  const inCamera = document.getElementById("photo_camera");
  const inGalerie = document.getElementById("photo_galerie");
  const zone = document.getElementById("zone");
  const zoneContenu = document.getElementById("zone-contenu");
  const apercu = document.getElementById("apercu-local");
  const nomEl = document.getElementById("nom-fichier");
  const erreurEl = document.getElementById("message-erreur");
  const btn = document.getElementById("btn-analyser");
  const btnTxt = btn.querySelector(".txt");

  const FORMATS_OK = ["image/jpeg", "image/png"];
  const COTE_MAX = 1280;          // réduction avant envoi (économie de données)
  const SEUIL_REDUCTION = 1.5 * 1024 * 1024;
  const TEXTE_BTN = btnTxt.textContent;

  let urlApercu = null;
  let traitement = Promise.resolve();   // réduction en cours

  function montrerErreur(msg) {
    erreurEl.textContent = msg;
    erreurEl.hidden = false;
  }
  function cacherErreur() {
    erreurEl.hidden = true;
    erreurEl.textContent = "";
  }

  function reinitialiser() {
    if (urlApercu) { URL.revokeObjectURL(urlApercu); urlApercu = null; }
    apercu.hidden = true;
    apercu.removeAttribute("src");
    zoneContenu.hidden = false;
    zone.classList.remove("avec-photo");
    nomEl.textContent = "Aucune photo choisie";
    nomEl.classList.remove("ok");
  }

  async function reduire(fichier) {
    if (!window.createImageBitmap) return null;
    const bmp = await createImageBitmap(fichier, { imageOrientation: "from-image" });
    const echelle = Math.min(1, COTE_MAX / Math.max(bmp.width, bmp.height));
    if (echelle === 1 && fichier.size < SEUIL_REDUCTION) { bmp.close(); return null; }

    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * echelle);
    canvas.height = Math.round(bmp.height * echelle);
    canvas.getContext("2d").drawImage(bmp, 0, 0, canvas.width, canvas.height);
    bmp.close();

    const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.9));
    if (!blob) return null;
    const base = fichier.name.replace(/\.[^.]+$/, "") || "photo";
    return new File([blob], base + ".jpg", { type: "image/jpeg" });
  }

  async function traiterChoix(input, autre) {
    autre.value = "";               // une seule photo envoyée à la fois
    cacherErreur();

    const fichier = input.files[0];
    if (!fichier) { reinitialiser(); return; }

    if (!FORMATS_OK.includes(fichier.type)) {
      input.value = "";
      reinitialiser();
      montrerErreur("Format non pris en charge. Choisissez une photo JPG ou PNG.");
      return;
    }

    // Affichage immédiat : nom du fichier + aperçu
    if (urlApercu) URL.revokeObjectURL(urlApercu);
    urlApercu = URL.createObjectURL(fichier);
    apercu.src = urlApercu;
    apercu.hidden = false;
    zoneContenu.hidden = true;
    zone.classList.add("avec-photo");
    nomEl.textContent = "📷 " + fichier.name;
    nomEl.classList.add("ok");

    // Réduction silencieuse ; en cas d'échec on envoie l'original
    traitement = (async () => {
      try {
        const reduit = await reduire(fichier);
        if (reduit) {
          const dt = new DataTransfer();
          dt.items.add(reduit);
          input.files = dt.files;
        }
      } catch (e) { /* on garde le fichier d'origine */ }
    })();
  }

  inCamera.addEventListener("change", () => traiterChoix(inCamera, inGalerie));
  inGalerie.addEventListener("change", () => traiterChoix(inGalerie, inCamera));

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const fichier = inCamera.files[0] || inGalerie.files[0];
    if (!fichier) {
      montrerErreur("Veuillez prendre ou choisir une photo du sol avant de lancer l'analyse.");
      zone.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    cacherErreur();
    btn.disabled = true;
    btn.classList.add("charge");
    btnTxt.textContent = "Analyse en cours…";
    traitement.then(() => form.submit());
  });

  // Retour arrière du navigateur : on remet le bouton à l'état normal
  window.addEventListener("pageshow", () => {
    btn.disabled = false;
    btn.classList.remove("charge");
    btnTxt.textContent = TEXTE_BTN;
  });

  // Après une analyse, on amène le résultat à l'écran
  const resultat = document.getElementById("resultat");
  if (resultat) resultat.scrollIntoView({ behavior: "smooth", block: "start" });
})();