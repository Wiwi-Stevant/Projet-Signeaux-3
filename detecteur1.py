import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from dataclasses import dataclass
from scipy import ndimage
from skimage.segmentation import watershed

CHEMIN_IMAGE = "data/IMG_9238.jpg"

"""
liste des image qui marche +-
43fef775db14373494f2246ff3754a16_t.jpg
- pas de pieces de 50 donc il pense juste que les 10 sont des 50
image.jpg
image1.jpg
image2.jpg
image3.jpg
image5.jpg
image6.jpg
IMG_5674.jpg
- piece de 10 tres similaire a la piece de 50 donc il pense que c'est une piece de 50
IMG_5675.jpg
- pas de 50 et invente une piece de 1 donc il décale tout 
IMG_5676.jpg
- une piece de 1 qu'il surestime donc il pense que c'est une piece de 5
IMG_5677.jpg
- il y a de l'idee mais une piece de 1 -> 5 et une piece de 10 -> 50 (donc c'est plus subtil que ça)
IMG_5678.jpg
- pas de 5 donc une piece de 1 -> 5
IMG_5679.jpg
IMG_5680.jpg
- une piece de 1 -> 5
IMG_5681.jpg
- toutes les pieces de 1 -> 5
IMG_5682.jpg
- pieces de 5 -> 1
IMG_5688.jpg
- une piece de 10 non detectée
IMG_5689.jpg
- fausse piece a l'interieur d'une piece (chiant a justifier donc jsp si je la supprime ou pas) mais a part ça, elle fonctionne bien
IMG_5693.jpg
- une pieces de 1-> 5
IMG_5694.jpg
- une pieces de 5 -> 1
IMG_5695.jpg
IMG_5696.jpg
- une pieces de 10 -> 50
IMG_5698.jpg
- piece de 1 -> 5
IMG_5709.jpg
- detection ok mais quelques erreurs pour les valeur (10 -> 5)
IMG_9237.jpg
- 2 pieces de 10 -> 5

==================================================
exemple d'image qui ne marche pas
erreurs/IMG_5684.jpg
- imagine des pieces et n'arrive donc pas a estimer la valeur de certaine
erreurs/IMG_5690.jpg
- bordelique
erreurs/IMG_5691.jpg
- beaucoup de pieces pas detectées
erreurs/IMG_5706.jpg
- detecte juste un cercle dans une piece de 50
erreurs/IMG_5707.jpg
- c'est plus subtil que ça
erreurs/IMG_9238.jpg
- une pieces fantome qui decale tout 
===================================================
"""

TAILLES_VALEURS = [50, 10, 5, 1]
TAILLE_MAX_IMAGE = 1400
TOLERANCE_REGROUPEMENT = 0.08
RAPPORT_TAILLE_MANQUANTE = 1.25


@dataclass
class Piece:
    x: int
    y: int
    diametre_px: float
    lab_b: float
    valeur: int | None = None


def score_contour_circulaire(
    gradients_x: np.ndarray, gradients_y: np.ndarray,
    x: float, y: float, rayon: float,
) -> float:
    """Mesure la continuit? du bord d'un cercle dans l'image r?duite."""
    angles = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    score = np.zeros(angles.shape, dtype=np.float32)

    for decalage in (-2, 0, 2):
        xs = np.rint(x + (rayon + decalage) * np.cos(angles)).astype(int)
        ys = np.rint(y + (rayon + decalage) * np.sin(angles)).astype(int)
        valides = (xs >= 1) & (xs < gradients_x.shape[1] - 1) & (ys >= 1) & (ys < gradients_x.shape[0] - 1)
        xs = np.clip(xs, 1, gradients_x.shape[1] - 2)
        ys = np.clip(ys, 1, gradients_x.shape[0] - 2)
        gradient_radial = np.abs(
            gradients_x[ys, xs] * np.cos(angles)
            + gradients_y[ys, xs] * np.sin(angles)
        )
        score = np.maximum(score, np.where(valides, gradient_radial, 0))

    return float(np.mean(score))


def mesurer_couleur_chaude(image: np.ndarray, x: int, y: int, diametre: float) -> float:
    """Mesure le jaune/doré au centre de la pièce, sans inclure son arrière-plan."""
    rayon = diametre / 2
    marge = max(1, round(rayon * 0.7))
    x_min, x_max = max(0, x - marge), min(image.shape[1], x + marge + 1)
    y_min, y_max = max(0, y - marge), min(image.shape[0], y + marge + 1)
    zone = image[y_min:y_max, x_min:x_max]

    lignes, colonnes = np.ogrid[y_min:y_max, x_min:x_max]
    distance = np.sqrt((colonnes - x) ** 2 + (lignes - y) ** 2)
    masque = (distance >= rayon * 0.2) & (distance <= rayon * 0.68)
    lab = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)
    return float(np.median(lab[:, :, 2][masque]))


def detecter_pieces(image: np.ndarray) -> list[Piece]:
    """Repère les cercles et renvoie leur centre et leur diamètre en pixels."""
    hauteur, largeur = image.shape[:2]
    facteur = min(1.0, TAILLE_MAX_IMAGE / max(hauteur, largeur))

    if facteur < 1:
        image_reduite = cv2.resize(
            image, None, fx=facteur, fy=facteur, interpolation=cv2.INTER_AREA
        )
    else:
        image_reduite = image

    gris = cv2.cvtColor(image_reduite, cv2.COLOR_BGR2GRAY)
    flou = cv2.medianBlur(gris, 5)
    #plt.imshow(flou, cmap='gray')
    cote_court = min(flou.shape)

    bordure = np.concatenate((flou[0, :], flou[-1, :], flou[:, 0], flou[:, -1]))
    if float(np.median(bordure)) < 128:
        _, masque = cv2.threshold(
            gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, noyau)
        masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noyau)

        rayon_min = max(6, int(cote_court * 0.02))
        rayon_max = max(12, int(cote_court * 0.12))
        aire_min = math.pi * rayon_min**2 * 1.25
        aire_max = math.pi * rayon_max**2 * 1.6
        distance = cv2.distanceTransform(masque, cv2.DIST_L2, 5)
        taille_fenetre = max(15, int(cote_court * 0.04))
        maximum_distance = max(10, cote_court * 0.025)
        maxima = (
            (distance == ndimage.maximum_filter(distance, size=taille_fenetre))
            & (distance > maximum_distance)
            & (masque > 0)
        )
        etiquettes_maxima, nombre_maxima = ndimage.label(maxima)
        candidats: list[tuple[float, int, int]] = []

        for indice in range(1, nombre_maxima + 1):
            ys, xs = np.where(etiquettes_maxima == indice)
            if len(xs):
                position = int(np.argmax(distance[ys, xs]))
                candidats.append(
                    (
                        float(distance[ys[position], xs[position]]),
                        int(xs[position]),
                        int(ys[position]),
                    )
                )

        candidats.sort(reverse=True)
        distance_minimale = max(18, int(cote_court * 0.045))
        centres_pieces: list[tuple[float, int, int]] = []
        for candidat in candidats:
            _, x, y = candidat
            trop_proche = any(
                math.hypot(x - autre_x, y - autre_y) < distance_minimale
                for _, autre_x, autre_y in centres_pieces
            )
            if not trop_proche:
                centres_pieces.append(candidat)

        marqueurs = np.zeros(masque.shape, dtype=np.int32)
        for indice, (_, x, y) in enumerate(centres_pieces, start=1):
            marqueurs[y, x] = indice

        regions = watershed(-distance, marqueurs, mask=masque > 0)
        pieces_segmentees: list[Piece] = []

        for indice, (_, x, y) in enumerate(centres_pieces, start=1):
            aire = int(np.count_nonzero(regions == indice))
            if aire_min <= aire <= aire_max:
                pieces_segmentees.append(
                    Piece(
                        x=round(x / facteur),
                        y=round(y / facteur),
                        diametre_px=2 * math.sqrt(aire / math.pi) / facteur,
                        lab_b=0,
                    )
                )

        if pieces_segmentees and len(pieces_segmentees) <= 12:
            for piece in pieces_segmentees:
                piece.lab_b = mesurer_couleur_chaude(
                    image, piece.x, piece.y, piece.diametre_px
                )
            return pieces_segmentees

    if cote_court < 600:
        seuil_hough = 38
        rayon_min = max(6, int(cote_court * 0.015))
        rayon_max = max(12, int(cote_court * 0.12))
    else:
        seuil_hough = 45
        rayon_min = max(6, int(cote_court * 0.02))
        rayon_max = max(12, int(cote_court * 0.10))

    cercles = cv2.HoughCircles(
        flou,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(15, int(cote_court * 0.04)),
        param1=140,
        param2=seuil_hough,
        minRadius=rayon_min,
        maxRadius=rayon_max,
    )

    if cercles is None:
        return []

    pieces: list[Piece] = []
    for x, y, rayon in cercles[0]:
        piece = Piece(
            x=round(x / facteur),
            y=round(y / facteur),
            diametre_px=2 * rayon / facteur,
            lab_b=0,
        )

        # Hough peut proposer plusieurs cercles presque concentriques pour une pièce.
        doublon = any(
            math.hypot(piece.x - autre.x, piece.y - autre.y)
            < min(piece.diametre_px, autre.diametre_px) * 0.35
            and abs(piece.diametre_px - autre.diametre_px)
            / max(piece.diametre_px, autre.diametre_px)
            < 0.25
            for autre in pieces
        )
        if not doublon:
            pieces.append(piece)

    for piece in pieces:
        piece.lab_b = mesurer_couleur_chaude(
            image, piece.x, piece.y, piece.diametre_px
        )

    if len(pieces) > 12:
        seuil_contour = (
            105 if len(pieces) > 250
            else 90 if len(pieces) > 150
            else 78 if len(pieces) > 50
            else 88
        )
        gradients_x = cv2.Sobel(flou, cv2.CV_32F, 1, 0, ksize=3)
        gradients_y = cv2.Sobel(flou, cv2.CV_32F, 0, 1, ksize=3)
        pieces = [
            piece for piece in pieces
            if score_contour_circulaire(
                gradients_x, gradients_y, piece.x * facteur, piece.y * facteur,
                piece.diametre_px * facteur / 2,
            ) >= seuil_contour
        ]

    pieces = sorted(pieces, key=lambda piece: piece.diametre_px, reverse=True)
    pieces_filtrees: list[Piece] = []
    for piece in pieces:
        cercle_interieur = any(
            autre.diametre_px > piece.diametre_px * 1.25
            and math.hypot(piece.x - autre.x, piece.y - autre.y)
            < piece.diametre_px * 0.4
            for autre in pieces_filtrees
        )
        if not cercle_interieur:
            pieces_filtrees.append(piece)

    return pieces_filtrees


def regrouper_par_taille(pieces: list[Piece]) -> list[list[Piece]]:
    """Rassemble les diamètres proches en groupes, du plus petit au plus grand."""
    groupes: list[list[Piece]] = []

    for piece in sorted(pieces, key=lambda p: p.diametre_px):
        if not groupes:
            groupes.append([piece])
            continue

        diametre_median = float(
            np.median([membre.diametre_px for membre in groupes[-1]])
        )
        ecart_relatif = abs(piece.diametre_px - diametre_median) / diametre_median
        if ecart_relatif <= TOLERANCE_REGROUPEMENT:
            groupes[-1].append(piece)
        else:
            groupes.append([piece])

    return groupes


def attribuer_valeurs(groupes: list[list[Piece]]) -> None:
    """Utilise la couleur pour séparer les 50 des 10, puis compare les diamètres."""
    pieces = [piece for groupe in groupes for piece in groupe]
    plus_grand_diametre = max(piece.diametre_px for piece in pieces)
    grandes_pieces = [
        piece
        for piece in pieces
        if piece.diametre_px >= plus_grand_diametre * 0.9
    ]

    pieces_de_50: list[Piece] = []
    if len(grandes_pieces) >= 3:
        couleurs = sorted(piece.lab_b for piece in grandes_pieces)
        ecarts = [
            (couleurs[i + 1] - couleurs[i], i)
            for i in range(len(couleurs) - 1)
        ]
        ecart_max, position_ecart = max(ecarts)

        # Sur ces pièces, une surface dorée distingue les 50 des 10 argentées.
        if ecart_max >= 12:
            seuil_couleur = (
                couleurs[position_ecart] + couleurs[position_ecart + 1]
            ) / 2
            pieces_de_50 = [
                piece
                for piece in grandes_pieces
                if piece.lab_b > seuil_couleur
            ]

    for piece in pieces_de_50:
        piece.valeur = 50

    pieces_a_classer = [piece for piece in pieces if piece not in pieces_de_50]
    groupes_a_classer = regrouper_par_taille(pieces_a_classer)
    groupes_grands_vers_petits = list(reversed(groupes_a_classer))

    # Si les 50 ont été séparées par couleur, les autres tailles commencent à 10.
    indice_valeur = 1 if pieces_de_50 else 0

    for position, groupe in enumerate(groupes_grands_vers_petits):
        if position:
            diametre_precedent = float(
                np.median(
                    [piece.diametre_px for piece in groupes_grands_vers_petits[position - 1]]
                )
            )
            diametre_actuel = float(
                np.median([piece.diametre_px for piece in groupe])
            )
            rapport = diametre_precedent / diametre_actuel

            # Un grand écart peut signaler une ou plusieurs tailles absentes.
            if rapport > RAPPORT_TAILLE_MANQUANTE:
                pas = 2 + int(
                    math.log(
                        rapport / RAPPORT_TAILLE_MANQUANTE,
                        RAPPORT_TAILLE_MANQUANTE,
                    )
                )
            else:
                pas = 1
            indice_valeur += pas

            # Avec quatre groupes ou moins, la plus petite classe ne peut pas
            # être plus petite que la quatrième valeur prévue.
            if len(groupes) <= len(TAILLES_VALEURS):
                indice_valeur = min(indice_valeur, len(TAILLES_VALEURS) - 1)

        valeur = (
            TAILLES_VALEURS[indice_valeur]
            if indice_valeur < len(TAILLES_VALEURS)
            else None
        )
        for piece in groupe:
            piece.valeur = valeur



def main() -> None:
    image = cv2.imread(str(CHEMIN_IMAGE))
    if image is None:
        raise FileNotFoundError(f"Image introuvable ou illisible : {CHEMIN_IMAGE}")

    hauteur, largeur = image.shape[:2]
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Image originale - {largeur} x {hauteur} px")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.xticks(np.linspace(0, largeur - 1, num=min(9, largeur), dtype=int))
    plt.yticks(np.linspace(0, hauteur - 1, num=min(9, hauteur), dtype=int))
    plt.grid(color="white", alpha=0.3, linestyle="--")
    plt.tight_layout()

    pieces = detecter_pieces(image)
    if not pieces:
        print("Aucune pièce détectée. Vérifiez l'image et les paramètres de détection.")
        plt.show()
        return

    groupes = regrouper_par_taille(pieces)
    attribuer_valeurs(groupes)

    total = sum(piece.valeur or 0 for piece in pieces)
    inconnues = sum(piece.valeur is None for piece in pieces)

    print(f"Image : {CHEMIN_IMAGE}")
    print(f"Pièces détectées : {len(pieces)}")
    for numero, piece in enumerate(pieces, start=1):
        valeur = "indéterminée" if piece.valeur is None else str(piece.valeur)
        print(
            f"  Pièce {numero} : diamètre environ {piece.diametre_px:.0f} px, "
            f"valeur estimée = {valeur}"
        )
    print(f"Total estimé : {total}")

    if inconnues:
        print(
            f"Attention : {inconnues} pièce(s) n'ont pas pu être classées "
            "et ne sont pas incluses dans le total."
        )
    print(
        "Vérifiez les étiquettes sur l'image : le classement des tailles "
        "et les tailles manquantes sont des estimations."
    )

    rayons = [round(piece.diametre_px / 2) for piece in pieces]
    marge = max(32, round(max(piece.diametre_px for piece in pieces) * 0.2))
    x_min = max(0, min(piece.x - rayon for piece, rayon in zip(pieces, rayons)) - marge)
    y_min = max(0, min(piece.y - rayon for piece, rayon in zip(pieces, rayons)) - marge)
    x_max = min(largeur, max(piece.x + rayon for piece, rayon in zip(pieces, rayons)) + marge + 1)
    y_max = min(hauteur, max(piece.y + rayon for piece, rayon in zip(pieces, rayons)) + marge + 1)
    image_annotee = image[y_min:y_max, x_min:x_max].copy()
    for piece in pieces:
        centre = (piece.x - x_min, piece.y - y_min)
        rayon = round(piece.diametre_px / 2)
        cv2.circle(image_annotee, centre, rayon, (0, 255, 0), 4)
        cv2.circle(image_annotee, centre, 4, (0, 0, 255), -1)
        etiquette = "?" if piece.valeur is None else str(piece.valeur)
        cv2.putText(
            image_annotee,
            f"{round(piece.diametre_px)} px / {etiquette}",
            (centre[0] - rayon, centre[1] - rayon - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    hauteur_recadree, largeur_recadree = image_annotee.shape[:2]
    plt.figure(figsize=(14, 8))
    plt.imshow(
        cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB),
        extent=(x_min - 0.5, x_max - 0.5, y_max - 0.5, y_min - 0.5),
    )
    plt.title(
        f"Pièces détectées - total estimé : {total} "
        f"({largeur_recadree} x {hauteur_recadree} px, recadrée)"
    )
    plt.xlabel("X (pixels, coordonnées de l'image originale)")
    plt.ylabel("Y (pixels, coordonnées de l'image originale)")
    plt.xticks(np.linspace(x_min, x_max - 1, num=min(9, largeur_recadree), dtype=int))
    plt.yticks(np.linspace(y_min, y_max - 1, num=min(9, hauteur_recadree), dtype=int))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
