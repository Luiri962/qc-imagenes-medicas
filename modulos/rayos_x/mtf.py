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

    # Estrategia A: región oscura de
