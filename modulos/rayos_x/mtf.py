"""
Módulo MTF — Rayos X Diagnóstico
Método: Slanted Edge (IEC 62220-1)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import (
    gaussian_filter, label,
    binary_fill_holes, binary_erosion, binary_dilation, uniform_filter
)
from scipy.fft import fft, fftfreq


# ── Parámetros del método ────────────────────────────────────────────────────
OVERSAMPLE      = 4
VENTANA_BORDE   = 30
SIGMA_SUAVIZADO = 0.8
CONTRASTE_MIN   = 80
ANCHO_MM        = 5.0   # mm hacia adentro del borde
LARGO_FRACCION  = 0.4   # fracción del largo del borde a usar


# ── Segmentación del cuadrado ────────────────────────────────────────────────
def segmentar_cuadrado(img):
    umbral_campo = np.percentile(img, 35)
    mask_campo   = img > umbral_campo
    mask_campo   = binary_fill_holes(mask_campo)
    mask_campo   = binary_erosion(mask_campo, iterations=10)
    fraccion_campo = mask_campo.sum() / img.size
    tiene_campo    = fraccion_campo > 0.05
    print(f"  Campo detectado: {fraccion_campo*100:.1f}% de la imagen")

    resultado = None

    # Estrategia A: región oscura dentro del campo
    if tiene_campo:
        fondo_local = uniform_filter(img.astype(float),
                                     size=int(img.shape[0] * 0.05))
        diff_oscuro = fondo_local - img.astype(float)
        diff_oscuro[~mask_campo] = 0

        if diff_oscuro[mask_campo].max() > 100:
            for pct in [95, 90, 85]:
                umbral   = np.percentile(diff_oscuro[mask_campo], pct)
                mask_raw = (diff_oscuro > umbral) & mask_campo
                mask_raw = binary_fill_holes(mask_raw)
                mask_raw = binary_erosion(mask_raw,  iterations=3)
                mask_raw = binary_dilation(mask_raw, iterations=5)

                labeled, n = label(mask_raw)
                if n == 0:
                    continue

                area_img    = img.shape[0] * img.shape[1]
                area_minima = area_img * 0.001
                area_maxima = area_img * 0.15

                mejor, mejor_score = None, 0
                for k in range(1, n + 1):
                    region = labeled == k
                    area   = region.sum()
                    if area < area_minima or area > area_maxima:
                        continue
                    ys, xs = np.where(region)
                    h = ys.max() - ys.min()
                    w = xs.max() - xs.min()
                    if h == 0 or w == 0:
                        continue
                    score = (min(h,w)/max(h,w)) * (area/(h*w)) * np.sqrt(area)
                    if score > mejor_score:
                        mejor_score = score
                        mejor       = region

                if mejor is not None:
                    print(f"  ✅ Estrategia A (pct={pct}): score={mejor_score:.0f}")
                    resultado = binary_dilation(mejor, iterations=8)
                    break

    # Estrategia B: región brillante respecto al fondo local
    if resultado is None:
        fondo_grande   = uniform_filter(img.astype(float),
                                        size=int(img.shape[0] * 0.15))
        diff_brillante = img.astype(float) - fondo_grande

        for pct in [97, 95, 93]:
            umbral   = np.percentile(diff_brillante, pct)
            mask_raw = diff_brillante > umbral
            mask_raw = binary_fill_holes(mask_raw)
            mask_raw = binary_erosion(mask_raw,  iterations=5)
            mask_raw = binary_dilation(mask_raw, iterations=5)

            labeled, n = label(mask_raw)
            if n == 0:
                continue

            area_img    = img.shape[0] * img.shape[1]
            area_minima = area_img * 0.001
            area_maxima = area_img * 0.20

            mejor, mejor_score = None, 0
            for k in range(1, n + 1):
                region = labeled == k
                area   = region.sum()
                if area < area_minima or area > area_maxima:
                    continue
                ys, xs = np.where(region)
                h = ys.max() - ys.min()
                w = xs.max() - xs.min()
                if h == 0 or w == 0:
                    continue
                score = (min(h,w)/max(h,w)) * (area/(h*w)) * np.sqrt(area)
                if score > mejor_score:
                    mejor_score = score
                    mejor       = region

            if mejor is not None:
                print(f"  ✅ Estrategia B (pct={pct}): score={mejor_score:.0f}")
                resultado = binary_dilation(mejor, iterations=8)
                break

    if resultado is None:
        raise RuntimeError(
            "No se encontró el cuadrado metálico. "
            "Verifica que el fantoma esté dentro del campo de radiación."
        )

    return resultado


# ── Extracción de lados ──────────────────────────────────────────────────────
def extraer_lados(mask):
    contorno = mask & ~binary_erosion(mask, iterations=3)
    ys, xs   = np.where(contorno)
    cy, cx   = ys.mean(), xs.mean()
    angulos  = np.degrees(np.arctan2(ys - cy, xs - cx))

    rangos = {
        "top":    (angulos >= -160) & (angulos <  -20),
        "right":  (angulos >= -20)  & (angulos <   70),
        "bottom": (angulos >=  20)  & (angulos <  160),
        "left":   (angulos >=  110) | (angulos < -110),
    }

    lados = {}
    for nombre, m in rangos.items():
        if m.sum() < 15:
            continue
        yb, xb       = ys[m], xs[m]
        med_y, med_x = np.median(yb), np.median(xb)
        dist = np.sqrt((yb - med_y)**2 + (xb - med_x)**2)
        ok   = dist < np.percentile(dist, 85)
        yb, xb = yb[ok], xb[ok]
        if len(yb) < 10:
            continue

        if nombre in ("top", "bottom"):
            coef   = np.polyfit(xb, yb, 1)
            angulo = np.degrees(np.arctan(coef[0]))
        else:
            coef   = np.polyfit(yb, xb, 1)
            angulo = np.degrees(np.arctan(1.0 / (coef[0] + 1e-10)))

        lados[nombre] = {
            "ys": yb, "xs": xb, "coef": coef,
            "angulo":   angulo,
            "centro_y": int(np.median(yb)),
            "centro_x": int(np.median(xb)),
            "largo":    max(yb.max() - yb.min(), xb.max() - xb.min()),
        }
    return lados


# ── Construir ROI hacia el interior del cuadrado ─────────────────────────────
def roi_sobre_borde(img, lado, tipo, nombre_lado, cy_cuad, cx_cuad,
                    px_mm=0.15, ancho_mm=ANCHO_MM, largo_fraccion=LARGO_FRACCION):
    """
    El ROI queda DENTRO del cuadrado, pegado al borde por adentro.
    ancho_mm : cuántos mm hacia el interior del cuadrado
    """
    ancho = int(ancho_mm / px_mm)   # mm → px
    cy    = lado["centro_y"]
    cx    = lado["centro_x"]
    largo = int(lado["largo"] * largo_fraccion)
    H, W  = img.shape

    if tipo == "H":
        # Decidir si el interior está arriba o abajo
        if nombre_lado == "top":
            # Borde superior → interior hacia ABAJO
            r0 = max(0, cy)
            r1 = min(H, cy + ancho)
        else:
            # Borde inferior → interior hacia ARRIBA
            r0 = max(0, cy - ancho)
            r1 = min(H, cy)
        c0 = max(0, cx - largo // 2)
        c1 = min(W, cx + largo // 2)

    else:
        # Decidir si el interior está a la izquierda o derecha
        if nombre_lado == "left":
            # Borde izquierdo → interior hacia la DERECHA
            c0 = max(0, cx)
            c1 = min(W, cx + ancho)
        else:
            # Borde derecho → interior hacia la IZQUIERDA
            c0 = max(0, cx - ancho)
            c1 = min(W, cx)
        r0 = max(0, cy - largo // 2)
        r1 = min(H, cy + largo // 2)

    return r0, r1, c0, c1


# ── Cálculo MTF ──────────────────────────────────────────────────────────────
def calcular_mtf(roi, px_mm, orientacion="H"):
    nyquist = 1 / (2 * px_mm)

    if orientacion == "H":
        n  = roi.shape[1]
        pf = lambda k: roi[:, k].astype(float)
    else:
        n  = roi.shape[0]
        pf = la
