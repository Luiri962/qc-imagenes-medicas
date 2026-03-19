"""
Generador de reportes PDF para QC
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image as RLImage, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER
from datetime import datetime
import matplotlib.pyplot as plt
import io


def _estilo(nombre, **kw):
    sty = getSampleStyleSheet()
    return ParagraphStyle(nombre, parent=sty["Normal"], **kw)


def generar_pdf_mtf(resultado: dict, fig: plt.Figure) -> bytes:
    """
    Recibe el dict de resultados y la figura matplotlib.
    Devuelve los bytes del PDF listo para descargar.
    """
    buffer_pdf = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer_pdf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )

    st_tit  = _estilo("T",  fontSize=15, fontName="Helvetica-Bold",
                      textColor=colors.HexColor("#1A237E"),
                      alignment=TA_CENTER, spaceAfter=4)
    st_sub  = _estilo("Su", fontSize=9,  fontName="Helvetica",
                      textColor=colors.HexColor("#455A64"),
                      alignment=TA_CENTER, spaceAfter=8)
    st_h1   = _estilo("H1", fontSize=11, fontName="Helvetica-Bold",
                      textColor=colors.HexColor("#1A237E"),
                      spaceBefore=10, spaceAfter=4)
    st_body = _estilo("B",  fontSize=9,  fontName="Helvetica",
                      leading=13, spaceAfter=4)
    st_nota = _estilo("N",  fontSize=8,  fontName="Helvetica-Oblique",
                      textColor=colors.HexColor("#607D8B"), spaceAfter=4)

    # Guardar figura como imagen en memoria
    buf_img = io.BytesIO()
    fig.savefig(buf_img, format="png", dpi=120, bbox_inches="tight",
                facecolor="#F4F6F8")
    buf_img.seek(0)

    story = []
    story.append(Paragraph("CONTROL DE CALIDAD — RESOLUCIÓN ESPACIAL (MTF)", st_tit))
    story.append(Paragraph(
        f"Equipo: {resultado['equipo']}  |  "
        f"Pixel spacing: {resultado['px_mm']} mm  |  "
        f"Fecha: {resultado['fecha']}", st_sub))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#1A237E"), spaceAfter=10))

    # Tabla info
    story.append(Paragraph("1. Información del equipo", st_h1))
    info = [
        ["Parámetro", "Valor"],
        ["Equipo / Modelo",  resultado["equipo"]],
        ["Pixel Spacing",    f"{resultado['px_mm']} mm"],
        ["Nyquist",          f"{resultado['nyquist']:.2f} lp/mm"],
        ["Fecha estudio",    resultado["fecha"]],
    ]
    t_info = Table(info, colWidths=[5.5*cm, 10*cm])
    t_info.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1A237E")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),
         [colors.HexColor("#F5F5F5"), colors.white]),
        ("GRID",        (0, 0), (-1,-1), 0.5, colors.HexColor("#BDBDBD")),
        ("LEFTPADDING", (0, 0), (-1,-1), 8),
        ("TOPPADDING",  (0, 0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 0.4*cm))

    # Metodología
    story.append(Paragraph("2. Metodología", st_h1))
    story.append(Paragraph(
        "Metodo del <b>borde inclinado (Slanted Edge)</b> segun IEC 62220-1. "
        "Segmentacion automatica del cuadrado metalico por contraste local. "
        "ESF por superposicion de perfiles con oversampling x4. "
        "LSF por derivacion de la ESF suavizada (sigma=0.8). "
        "MTF por FFT con ventana de Hanning y zero-padding x16.", st_body))

    # Figura grande
    img_fig = RLImage(buf_img, width=16*cm, height=11*cm)
    t_fig = Table([[img_fig]], colWidths=[16*cm])
    t_fig.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER")]))
    story.append(t_fig)
    story.append(Spacer(1, 0.4*cm))

    # Resultados
    story.append(Paragraph("3. Resultados", st_h1))

    def fmt(v): return f"{v:.3f} lp/mm" if v else "—"

    res_data = [
        ["Direccion", "MTF50 (lp/mm)", "MTF20 (lp/mm)", "FWHM LSF", "Angulo"],
        ["Horizontal",
         fmt(resultado["mtf50_h"]), fmt(resultado["mtf20_h"]),
         f"{resultado['fwhm_h']:.3f} mm" if resultado["fwhm_h"] else "—",
         f"{resultado['angulo_h']:.2f} deg"],
        ["Vertical",
         fmt(resultado["mtf50_v"]), fmt(resultado["mtf20_v"]),
         f"{resultado['fwhm_v']:.3f} mm" if resultado["fwhm_v"] else "—",
         f"{resultado['angulo_v']:.2f} deg"],
    ]
    t_res = Table(res_data, colWidths=[3.2*cm, 3.2*cm, 3.2*cm, 4*cm, 2.8*cm])
    t_res.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), colors.HexColor("#1A237E")),
        ("TEXTCOLOR",   (0,0),(-1,0), colors.white),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",    (0,1),(0,-1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 9.5),
        ("GRID",        (0,0),(-1,-1), 0.5, colors.HexColor("#BDBDBD")),
        ("ALIGN",       (0,0),(-1,-1), "CENTER"),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0),(-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.HexColor("#FFF3E0"), colors.HexColor("#E3F2FD")]),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 0.4*cm))

    # Notas
    story.append(Paragraph("4. Notas tecnicas", st_h1))
    story.append(Paragraph(
        f"Nyquist: {resultado['nyquist']:.3f} lp/mm  |  "
        "Oversampling: 4x  |  Sigma suavizado ESF: 0.8  |  "
        "Ventana borde: 30 px", st_nota))
    story.append(Paragraph(
        f"Reporte generado automaticamente con Python / Streamlit  |  "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')}", st_nota))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#90A4AE"), spaceBefore=8))
    story.append(Paragraph(
        "Control de Calidad — Fisica Medica  |  Generado automaticamente",
        st_nota))

    doc.build(story)
    buffer_pdf.seek(0)
    return buffer_pdf.read()
