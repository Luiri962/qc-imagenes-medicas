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
ANCHO_MM        = 7.0   # mm hacia adentro del borde
LARGO_FRACCION  = 0.15   # fracción del largo del borde a usar


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
                    px_mm=0.15, ancho_mm=5.0, largo_fraccion=0.4):
    ancho   = int(ancho_mm / px_mm)
    margen  = int(3 / px_mm)   # 3 mm extra para asegurar que queda dentro
    cy      = lado["centro_y"]
    cx      = lado["centro_x"]
    largo   = int(lado["largo"] * largo_fraccion)
    H, W    = img.shape

    if tipo == "H":
        if nombre_lado == "top":
            # Interior está ABAJO — empezar desde el borde + margen
            r0 = max(0, cy + margen)
            r1 = min(H, cy + margen + ancho)
        else:  # bottom
            # Interior está ARRIBA — terminar en el borde - margen
            r0 = max(0, cy - margen - ancho)
            r1 = min(H, cy - margen)
        c0 = max(0, cx - largo // 2)
        c1 = min(W, cx + largo // 2)
    else:
        if nombre_lado == "left":
            # Interior está a la DERECHA — empezar desde el borde + margen
            c0 = max(0, cx + margen)
            c1 = min(W, cx + margen + ancho)
        else:  # right
            # Interior está a la IZQUIERDA — terminar en borde - margen
            c0 = max(0, cx - margen - ancho)
            c1 = min(W, cx - margen)
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
        pf = lambda k: roi[k, :].astype(float)

    posiciones, idx_ok = [], []
    for k in range(n):
        p = pf(k)
        if p.max() - p.min() < CONTRASTE_MIN:
            continue
        grad = gaussian_filter(np.abs(np.diff(p)), sigma=0.8)
        imax = np.argmax(grad)
        if grad[imax] < 30:
            continue
        s = slice(max(0, imax - 4), min(len(grad), imax + 5))
        g = grad[s]; x = np.arange(s.start, s.stop, dtype=float)
        posiciones.append(np.sum(x * g) / (np.sum(g) + 1e-10))
        idx_ok.append(k)

    if len(posiciones) < 5:
        raise ValueError(
            f"Solo {len(posiciones)} perfiles válidos. "
            "Intenta aumentar ANCHO_MM o bajar CONTRASTE_MIN."
        )

    posiciones = np.array(posiciones)
    ok         = np.abs(posiciones - np.median(posiciones)) < 8
    posiciones = posiciones[ok]
    idx_ok     = np.array(idx_ok)[ok]

    angulo = np.degrees(
        np.arctan(np.polyfit(np.arange(len(posiciones)), posiciones, 1)[0])
    )

    offsets_all, valores_all = [], []
    for i, k in enumerate(idx_ok):
        p    = pf(k).astype(float)
        pos  = posiciones[i]
        v_lo = np.percentile(p, 10)
        v_hi = np.percentile(p, 90)
        pn   = np.clip((p - v_lo) / (v_hi - v_lo + 1e-10), 0, 1)
        for j, v in enumerate(pn):
            if abs(j - pos) <= VENTANA_BORDE:
                offsets_all.append((j - pos) * OVERSAMPLE)
                valores_all.append(v)

    offsets_all = np.array(offsets_all)
    valores_all = np.array(valores_all)
    orden       = np.argsort(offsets_all)
    offsets_all = offsets_all[orden]
    valores_all = valores_all[orden]

    n_bins = int((offsets_all.max() - offsets_all.min()) / 1.0) + 1
    edges  = np.linspace(offsets_all.min(), offsets_all.max(), n_bins + 1)
    esf = np.zeros(n_bins); cnt = np.zeros(n_bins)
    ib  = np.clip(np.searchsorted(edges, offsets_all) - 1, 0, n_bins - 1)
    np.add.at(esf, ib, valores_all)
    np.add.at(cnt, ib, 1)
    mb  = cnt > 0
    esf[mb] /= cnt[mb]
    bc  = (edges[:-1] + edges[1:]) / 2
    esf = np.interp(bc, bc[mb], esf[mb])

    bin_mm = (1.0 / OVERSAMPLE) * px_mm
    esf_x  = bc / OVERSAMPLE * px_mm

    esf_s = gaussian_filter(esf.astype(float), sigma=SIGMA_SUAVIZADO)
    lsf   = np.diff(esf_s)
    if abs(lsf.min()) > abs(lsf.max()):
        lsf = -lsf
    lsf -= lsf.min()
    lsf /= (lsf.max() + 1e-10)

    pi_ = np.argmax(lsf)
    hl  = np.where((np.arange(len(lsf)) < pi_) & (lsf >= 0.5))[0]
    hr  = np.where((np.arange(len(lsf)) > pi_) & (lsf >= 0.5))[0]
    fwhm_mm = (hr[-1] - hl[0]) * bin_mm if (len(hl) and len(hr)) else None

    N   = len(lsf); pad = 16
    mtf_raw = np.abs(fft(lsf * np.hanning(N), n=N * pad))[:N * pad // 2]
    mtf_raw /= mtf_raw[0]
    freqs = fftfreq(N * pad, d=bin_mm)[:N * pad // 2]
    m     = (freqs >= 0) & (freqs <= nyquist * 1.05)

    def fu(f, m, u):
        idx = np.where(m <= u)[0]
        if not len(idx):
            return None
        i = idx[0]
        return float(
            f[i-1] + (f[i]-f[i-1]) * (u-m[i-1]) / (m[i]-m[i-1])
        ) if i > 0 else float(f[0])

    fr = freqs[m]; mr = mtf_raw[m]
    return {
        "freqs":   fr,   "mtf":     mr,
        "mtf50":   fu(fr, mr, 0.50),
        "mtf20":   fu(fr, mr, 0.20),
        "angulo":  angulo,   "fwhm_mm": fwhm_mm,
        "esf_x":   esf_x,   "esf":     esf,
        "lsf":     lsf,     "bin_mm":  bin_mm,
        "nyquist": nyquist,
    }


# ── Figura completa ──────────────────────────────────────────────────────────
def figura_completa(img, mask, rois, res_H, res_V, equipo, fecha, px_mm):
    ys_m, xs_m = np.where(mask)
    mg  = 150
    rv0 = max(0, ys_m.min() - mg); rv1 = min(img.shape[0], ys_m.max() + mg)
    cv0 = max(0, xs_m.min() - mg); cv1 = min(img.shape[1], xs_m.max() + mg)
    zona = img[rv0:rv1, cv0:cv1]

    r0H, r1H, c0H, c1H = rois["H"]
    r0V, r1V, c0V, c1V = rois["V"]

    fig = plt.figure(figsize=(20, 14), facecolor="#F4F6F8")
    fig.suptitle(
        f"MTF — Control de Calidad  |  {equipo}  |  "
        f"Pixel spacing: {px_mm} mm  |  {fecha}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    ax1 = fig.add_axes([0.03, 0.54, 0.28, 0.40])
    ax1.imshow(zona, cmap="gray", aspect="auto",
               vmin=np.percentile(zona, 1), vmax=np.percentile(zona, 99))
    for (r0, r1, c0, c1), color in [
        ((r0H, r1H, c0H, c1H), "#E65100"),
        ((r0V, r1V, c0V, c1V), "#1565C0"),
    ]:
        ax1.add_patch(patches.Rectangle(
            (c0-cv0, r0-rv0), c1-c0, r1-r0,
            lw=2.5, edgecolor=color, facecolor=color, alpha=0.15))
        ax1.add_patch(patches.Rectangle(
            (c0-cv0, r0-rv0), c1-c0, r1-r0,
            lw=2.5, edgecolor=color, facecolor="none"))
    ax1.text(c0H-cv0+6, r0H-rv0-10,
             f"ROI H  ({res_H['angulo']:.1f}°)",
             color="#FF6D00", fontsize=8, fontweight="bold")
    ax1.text(c0V-cv0+6, r1V-rv0+14,
             f"ROI V  ({res_V['angulo']:.1f}°)",
             color="#1565C0", fontsize=8, fontweight="bold")
    ax1.set_title("Objeto borde — ROIs detectados", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_axes([0.36, 0.54, 0.18, 0.40])
    ax2.plot(res_H["esf_x"], res_H["esf"], color="#BDBDBD", lw=1, alpha=0.5)
    ax2.plot(res_H["esf_x"],
             gaussian_filter(res_H["esf"].astype(float), sigma=0.8),
             color="#E65100", lw=2)
    ax2.axvline(0, color="gray", ls=":", lw=1)
    ax2.set_title("ESF — Horizontal", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Posición relativa (mm)"); ax2.set_ylabel("ESF norm.")
    ax2.grid(True, alpha=0.2)

    ax3 = fig.add_axes([0.58, 0.54, 0.18, 0.40])
    ax3.plot(res_V["esf_x"], res_V["esf"], color="#BDBDBD", lw=1, alpha=0.5)
    ax3.plot(res_V["esf_x"],
             gaussian_filter(res_V["esf"].astype(float), sigma=0.8),
             color="#1565C0", lw=2)
    ax3.axvline(0, color="gray", ls=":", lw=1)
    ax3.set_title("ESF — Vertical", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Posición relativa (mm)"); ax3.set_ylabel("ESF norm.")
    ax3.grid(True, alpha=0.2)

    ax4 = fig.add_axes([0.80, 0.54, 0.17, 0.40])
    lH = (np.arange(len(res_H["lsf"])) * res_H["bin_mm"]
          - len(res_H["lsf"]) // 2 * res_H["bin_mm"])
    lV = (np.arange(len(res_V["lsf"])) * res_V["bin_mm"]
          - len(res_V["lsf"]) // 2 * res_V["bin_mm"])
    ax4.plot(lH, res_H["lsf"], color="#E65100", lw=2,
             label=f"H  {res_H['fwhm_mm']:.3f}mm" if res_H["fwhm_mm"] else "H")
    ax4.plot(lV, res_V["lsf"], color="#1565C0", lw=2,
             label=f"V  {res_V['fwhm_mm']:.3f}mm" if res_V["fwhm_mm"] else "V")
    ax4.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
    ax4.set_title("LSF", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Posición (mm)"); ax4.set_ylabel("Amplitud norm.")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.2)

    def panel_mtf(ax, res, titulo, color):
        ax.fill_between(res["freqs"], res["mtf"], alpha=0.08, color=color)
        ax.plot(res["freqs"], res["mtf"], color=color, lw=2.8, label="MTF medida")
        ax.axhline(0.50, color="#FF6F00", ls="--", lw=1.8, alpha=0.9, label="MTF = 50%")
        ax.axhline(0.20, color="#6A1B9A", ls="--", lw=1.8, alpha=0.9, label="MTF = 20%")
        ny = res["nyquist"]
        ax.axvline(ny, color="#546E7A", ls=":", lw=1.5, label=f"Nyquist={ny:.2f}")
        if res["mtf50"]:
            ax.axvline(res["mtf50"], color="#FF6F00", lw=2.2, alpha=0.9)
            ax.annotate(
                f"MTF50 = {res['mtf50']:.3f} lp/mm",
                xy=(res["mtf50"], 0.50),
                xytext=(res["mtf50"] + ny*0.07, 0.63),
                fontsize=10, fontweight="bold", color="#BF360C",
                arrowprops=dict(arrowstyle="->", color="#FF6F00", lw=2),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E1",
                          alpha=0.97, edgecolor="#FF6F00"),
            )
        if res["mtf20"]:
            ax.axvline(res["mtf20"], color="#6A1B9A", lw=2.2, alpha=0.9)
            ax.annotate(
                f"MTF20 = {res['mtf20']:.3f} lp/mm",
                xy=(res["mtf20"], 0.20),
                xytext=(res["mtf20"] + ny*0.07, 0.33),
                fontsize=10, fontweight="bold", color="#4A148C",
                arrowprops=dict(arrowstyle="->", color="#6A1B9A", lw=2),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F3E5F5",
                          alpha=0.97, edgecolor="#6A1B9A"),
            )
        ax.set_xlim([0, ny*1.05]); ax.set_ylim([0, 1.08])
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_xlabel("Frecuencia espacial (lp/mm)", fontsize=11)
        ax.set_ylabel("MTF", fontsize=11)
        ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.2)

    ax5 = fig.add_axes([0.06, 0.06, 0.40, 0.42])
    panel_mtf(ax5, res_H, "MTF — Dirección Horizontal", "#BF360C")

    ax6 = fig.add_axes([0.55, 0.06, 0.40, 0.42])
    panel_mtf(ax6, res_V, "MTF — Dirección Vertical", "#0D47A1")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ── Función principal ─────────────────────────────────────────────────────────
def run(img, ds):
    px_mm     = float(getattr(ds, "PixelSpacing", [0.15, 0.15])[0])
    equipo    = getattr(ds, "ManufacturerModelName", "N/D")
    fecha_raw = getattr(ds, "StudyDate", "")
    try:
        fecha = f"{fecha_raw[6:8]}/{fecha_raw[4:6]}/{fecha_raw[0:4]}"
    except Exception:
        fecha = fecha_raw

    mask  = segmentar_cuadrado(img)
    lados = extraer_lados(mask)

    # Centro del cuadrado
    ys_m, xs_m = np.where(mask)
    cy_cuad = int(np.median(ys_m))
    cx_cuad = int(np.median(xs_m))

    cands_H = {k: lados[k] for k in ("top", "bottom") if k in lados}
    cands_V = {k: lados[k] for k in ("left", "right")  if k in lados}
    if not cands_H or not cands_V:
        raise RuntimeError("No se detectaron los bordes H o V del cuadrado.")

    nombre_H = max(cands_H, key=lambda k: cands_H[k]["largo"])
    nombre_V = max(cands_V, key=lambda k: cands_V[k]["largo"])
    lado_H   = cands_H[nombre_H]
    lado_V   = cands_V[nombre_V]

    r0H, r1H, c0H, c1H = roi_sobre_borde(
        img, lado_H, "H", nombre_H, cy_cuad, cx_cuad, px_mm=px_mm)
    r0V, r1V, c0V, c1V = roi_sobre_borde(
        img, lado_V, "V", nombre_V, cy_cuad, cx_cuad, px_mm=px_mm)
    rois = {"H": (r0H, r1H, c0H, c1H), "V": (r0V, r1V, c0V, c1V)}

    res_H = calcular_mtf(img[r0H:r1H, c0H:c1H], px_mm, "H")
    res_V = calcular_mtf(img[r0V:r1V, c0V:c1V], px_mm, "V")

    fig = figura_completa(img, mask, rois, res_H, res_V, equipo, fecha, px_mm)

    return {
        "figura":   fig,
        "mtf50_h":  res_H["mtf50"],  "mtf20_h":  res_H["mtf20"],
        "mtf50_v":  res_V["mtf50"],  "mtf20_v":  res_V["mtf20"],
        "angulo_h": res_H["angulo"], "angulo_v": res_V["angulo"],
        "fwhm_h":   res_H["fwhm_mm"],"fwhm_v":   res_V["fwhm_mm"],
        "nyquist":  res_H["nyquist"],"px_mm":    px_mm,
        "equipo":   equipo,          "fecha":    fecha,
    }
