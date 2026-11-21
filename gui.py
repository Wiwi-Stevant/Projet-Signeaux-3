from contextlib import redirect_stdout
from io import StringIO
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from detecteur_piece2 import detecter_pieces, obtenir_taux_twd


def afficher_image(image, panneau):
    apercu = ImageOps.contain(image, (820, 480))
    photo = ImageTk.PhotoImage(apercu)
    panneau.configure(image=photo, text="")
    panneau.photo = photo


def effacer_image(panneau, texte):
    panneau.configure(image="", text=texte)
    panneau.photo = None


def afficher_details(texte):
    details.configure(state="normal")
    details.delete("1.0", "end")
    details.insert("1.0", texte or "Aucun détail disponible.")
    details.configure(state="disabled")


def choisir_image():
    chemin = filedialog.askopenfilename(
        title="Choisir une photo de pièces",
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Tous les fichiers", "*.*"),
        ],
    )
    if not chemin:
        return

    total.set("Total : —")
    equivalent_eur.set("Équivalent en euros : —")
    statut.set("")
    afficher_details("")
    effacer_image(panneau_original, "Aucune image")
    effacer_image(panneau_resultat, "Analyse en cours…")

    try:
        with Image.open(chemin) as fichier:
            originale = ImageOps.exif_transpose(fichier).convert("RGB")
    except (OSError, ValueError):
        effacer_image(panneau_resultat, "En attente d’une image")
        statut.set("Impossible d’ouvrir cette image.")
        return

    afficher_image(originale, panneau_original)
    fenetre.update_idletasks()

    sortie = StringIO()
    try:
        image_bgr = cv2.cvtColor(np.asarray(originale), cv2.COLOR_RGB2BGR)
        with redirect_stdout(sortie):
            image_annotee, total_twd, non_classees, ambigu = detecter_pieces(image_bgr)
    except (ValueError, cv2.error) as erreur:
        afficher_details(sortie.getvalue())
        effacer_image(panneau_resultat, "Analyse impossible")
        statut.set(str(erreur))
        return

    afficher_image(Image.fromarray(image_annotee), panneau_resultat)
    libelle = "Sous-total" if non_classees else "Total"
    total.set(f"{libelle} : {total_twd} TWD")

    messages = []
    if non_classees:
        messages.append(f"{non_classees} pièce(s) sans valeur attribuée.")
    if ambigu:
        messages.append("Plusieurs classements possibles. Vérifie les valeurs.")

    try:
        taux, date_mise_a_jour = obtenir_taux_twd()
        montant_eur = total_twd * taux["EUR"]
        libelle_eur = "Équivalent du sous-total" if non_classees else "Équivalent en euros"
        equivalent_eur.set(
            f"{libelle_eur} : {montant_eur:.2f} EUR "
            f"(taux mis à jour le {date_mise_a_jour})"
        )
        details_conversion = (
            f"\nÉquivalent en euros : {montant_eur:.2f} EUR\n"
            f"Taux mis à jour le : {date_mise_a_jour}"
        )
    except (OSError, RuntimeError, ValueError) as erreur:
        equivalent_eur.set("Conversion en euros indisponible.")
        details_conversion = f"\nConversion en euros impossible : {erreur}"
        messages.append(f"Conversion en euros impossible : {erreur}")

    afficher_details(sortie.getvalue() + details_conversion)
    statut.set(" ".join(messages))


fenetre = tk.Tk()
fenetre.title("Reconnaissance de pièces")
fenetre.geometry("980x720")
fenetre.minsize(880, 600)
fenetre.columnconfigure((0, 1), weight=1, uniform="images")
fenetre.rowconfigure(1, weight=1)

total = tk.StringVar(value="Total : —")
statut = tk.StringVar()

ttk.Button(fenetre, text="Choisir une image", command=choisir_image).grid(
    row=0, column=0, columnspan=2, pady=12
)

cadre_original = ttk.LabelFrame(fenetre, text="Image originale", padding=10)
cadre_original.grid(row=1, column=0, padx=10, sticky="nsew")

panneau_original = ttk.Label(cadre_original, text="Aucune image", anchor="center")
panneau_original.pack(fill="both", expand=True)

cadre_resultat = ttk.LabelFrame(fenetre, text="Pièces et valeurs", padding=10)
cadre_resultat.grid(row=1, column=1, padx=10, sticky="nsew")

panneau_resultat = ttk.Label(
    cadre_resultat, text="En attente d’une image", anchor="center"
)
panneau_resultat.pack(fill="both", expand=True)

ttk.Label(fenetre, textvariable=total, font=("TkDefaultFont", 18, "bold")).grid(
    row=2, column=0, columnspan=2, pady=(12, 4)
)
equivalent_eur = tk.StringVar(value="Équivalent en euros : —")
ttk.Label(fenetre, textvariable=equivalent_eur, font=("TkDefaultFont", 13, "bold")).grid(
    row=3, column=0, columnspan=2, pady=(0, 4)
)
ttk.Label(fenetre, textvariable=statut, wraplength=820).grid(
    row=4, column=0, columnspan=2, pady=(0, 10)
)

cadre_details = ttk.LabelFrame(fenetre, text="Détails de l’analyse", padding=8)
cadre_details.grid(row=5, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")
cadre_details.columnconfigure(0, weight=1)
cadre_details.rowconfigure(0, weight=1)

details = tk.Text(cadre_details, height=7, wrap="word", state="disabled")
details.grid(row=0, column=0, sticky="nsew")
defilement_details = ttk.Scrollbar(
    cadre_details, orient="vertical", command=details.yview
)
defilement_details.grid(row=0, column=1, sticky="ns")
details.configure(yscrollcommand=defilement_details.set)

fenetre.mainloop()