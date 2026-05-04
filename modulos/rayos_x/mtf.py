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
def roi_sobre_borde(img, lado, tipo, nombre_lado, cy_cuad, cx_cuad, px_mm=0.15):
    """
    ROI delgado a lo largo del borde y profundo hacia el interior.
    profundo : cuánto entra hacia adentro (perpendicular al borde)
    ancho    : qué tan largo es a lo largo del borde
    """

    # Después (correcto)
    profundo = int(lado["largo"] * 0.15)          # 10 mm hacia adentro
    margen   = int(-4.0  / px_mm)          # 3 mm de separación desde el borde
    ancho    = int(10.0 / px_mm)  # 35% del largo del borde
    cy       = lado["centro_y"]
    cx       = lado["centro_x"]
    H, W     = img.shape

    if tipo == "H":
        # Estrecho a lo largo del borde (ancho), profundo hacia adentro
        c0 = max(0, cx - ancho // 2)
        c1 = min(W, cx + ancho // 2)
        if nombre_lado == "top":
            r0 = max(0, cy + margen)
            r1 = min(H, cy + margen + profundo)
        else:  # bottom
            r0 = max(0, cy - margen - profundo)
            r1 = min(H, cy - margen)

    else:
        # Estrecho a lo largo del borde (ancho), profundo hacia adentro
        r0 = max(0, cy - ancho // 2)
        r1 = min(H, cy + ancho // 2)
        if nombre_lado == "left":
            c0 = max(0, cx + margen)
            c1 = min(W, cx + margen + profundo)
        else:  # right
            c0 = max(0, cx - margen - profundo)
            c1 = min(W, cx - margen)

    return r0, r1, c0, c1


# ── Cálculo MTF ──────────────────────────────────────────────────────────────
def calcular_mtf(roi, px_mm, orientacion="H"):
    from scipy.optimize import curve_fit
    from scipy.special import erf as sci_erf

    nyquist = 1 / (2 * px_mm)

    # ── 1. ESF = promedio directo (igual que ImageJ) ──────────────────────────
    if orientacion == "H":
        esf_raw = roi.mean(axis=1).astype(float)  # promedio por fila
    else:
        esf_raw = roi.mean(axis=0).astype(float)  # promedio por columna

    # Normalizar 0-1
    v_min = esf_raw.min()
    v_max = esf_raw.max()
    esf   = (esf_raw - v_min) / (v_max - v_min + 1e-10)
    x_px  = np.arange(len(esf), dtype=float)
    x_mm  = x_px * px_mm

    # ── 2. Detectar ángulo del borde ──────────────────────────────────────────
    if orientacion == "H":
        n  = roi.shape[1]
        pf = lambda k: roi[:, k].astype(float)
    else:
        n  = roi.shape[0]
        pf = lambda k: roi[k, :].astype(float)

    posiciones = []
    for k in range(n):
        p = pf(k)
        if p.max() - p.min() < CONTRASTE_MIN: continue
        grad = gaussian_filter(np.abs(np.diff(p)), sigma=0.8)
        imax = np.argmax(grad)
        if grad[imax] < 30: continue
        s = slice(max(0,imax-4), min(len(grad),imax+5))
        g = grad[s]; xg = np.arange(s.start, s.stop, dtype=float)
        posiciones.append(np.sum(xg*g)/(np.sum(g)+1e-10))

    angulo = 0.0
    if len(posiciones) >= 5:
        posiciones = np.array(posiciones)
        ok = np.abs(posiciones - np.median(posiciones)) < 8
        posiciones = posiciones[ok]
        angulo = np.degrees(
            np.arctan(np.polyfit(np.arange(len(posiciones)), posiciones, 1)[0])
        )

    # ── 3. Ajustar erf a la ESF (igual que ImageJ) ───────────────────────────
    def esf_func(x, a, b, c, sigma):
        return a + b * sci_erf((x - c) / (np.sqrt(2) * abs(sigma)))

    pos_borde = float(x_mm[np.argmax(np.abs(np.diff(esf)))])

    try:
        popt, _ = curve_fit(
            esf_func, x_mm, esf,
            p0=[0.0, 0.5, pos_borde, px_mm * 2],
            bounds=([-0.1, 0.1, x_mm.min(), px_mm*0.1],
                    [ 0.5, 1.0, x_mm.max(), px_mm*20]),
            maxfev=10000
        )
        sigma_mm = abs(popt[3])   # sigma en mm
        esf_fit  = esf_func(x_mm, *popt)
        fwhm_mm  = 2.355 * sigma_mm

        # ── 4. MTF analítica desde sigma (Gaussiana) ─────────────────────────
        # MTF(f) = exp(-2π²σ²f²) — idéntico a lo que calcula ImageJ
        freqs  = np.linspace(0, nyquist * 1.05, 2000)
        mtf    = np.exp(-2 * (np.pi * sigma_mm * freqs)**2)

        # LSF para graficar (Gaussiana centrada)
        x_lsf = x_mm - popt[2]
        lsf   = np.exp(-x_lsf**2 / (2 * sigma_mm**2))
        lsf  /= lsf.max()

        print(f"  Ajuste erf OK | sigma={sigma_mm:.4f} mm | FWHM={fwhm_mm:.4f} mm")

    except Exception as e:
        print(f"  Ajuste erf falló: {e} — usando derivada numérica")
        esf_fit = gaussian_filter(esf.astype(float), sigma=1.5)
        lsf_raw = np.diff(esf_fit)
        if abs(lsf_raw.min()) > abs(lsf_raw.max()): lsf_raw = -lsf_raw
        lsf_raw -= lsf_raw.min(); lsf_raw /= (lsf_raw.max() + 1e-10)
        lsf   = lsf_raw
        N     = len(lsf); pad = 16
        mtf_r = np.abs(fft(lsf * np.hanning(N), n=N*pad))[:N*pad//2]
        mtf_r /= mtf_r[0]
        freqs = fftfreq(N*pad, d=px_mm)[:N*pad//2]
        m     = (freqs >= 0) & (freqs <= nyquist * 1.05)
        freqs = freqs[m]; mtf = mtf_r[m]
        fwhm_mm = None

    # ── 5. MTF50 y MTF20 ─────────────────────────────────────────────────────
    def fu(f, m, u):
        idx = np.where(m <= u)[0]
        if not len(idx): return None
        i = idx[0]
        return float(f[i-1]+(f[i]-f[i-1])*(u-m[i-1])/(m[i]-m[i-1])) if i>0 else float(f[0])

    return {
        "freqs":   freqs, "mtf":     mtf,
        "mtf50":   fu(freqs, mtf, 0.50),
        "mtf20":   fu(freqs, mtf, 0.20),
        "angulo":  angulo,  "fwhm_mm": fwhm_mm,
        "esf_x":   x_mm,   "esf":     esf,
        "lsf":     lsf,    "bin_mm":  px_mm,
        "nyquist": nyquist,
    }
