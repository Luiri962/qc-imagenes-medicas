"""
app.py — Aplicación principal de Control de Calidad en Imágenes Médicas
Ejecutar con: streamlit run app.py
"""
import streamlit as st
import pydicom
import numpy as np
import io

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="QC — Imágenes Médicas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos personalizados ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1A237E, #1565C0);
        padding: 1.2rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .metric-card {
        background: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Encabezado ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style="margin:0; font-size:1.6rem;">🏥 Control de Calidad — Imágenes Médicas</h2>
    <p style="margin:0.3rem 0 0; opacity:0.85; font-size:0.9rem;">
        Análisis automatizado de resolución espacial (MTF) y otros parámetros de QC
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1A237E/FFFFFF?text=QC+App",
             use_column_width=True)

    st.markdown("### ⚙️ Configuración")

    equipo = st.selectbox(
        "Equipo",
        ["Rayos X Diagnóstico", "Mamografía", "TC", "Fluoroscopía"],
        help="Selecciona el tipo de equipo a evaluar",
    )

    pruebas_por_equipo = {
        "Rayos X Diagnóstico": ["MTF — Resolución Espacial"],
        "Mamografía":          ["Próximamente"],
        "TC":                  ["Próximamente"],
        "Fluoroscopía":        ["Próximamente"],
    }

    prueba = st.selectbox(
        "Prueba de QC",
        pruebas_por_equipo[equipo],
        help="Selecciona la prueba que deseas calcular",
    )

    st.markdown("---")
    st.markdown("### 📋 Instrucciones")
    st.markdown("""
1. Selecciona el **equipo** y la **prueba**
2. Sube la imagen **DICOM**
3. Haz clic en **Calcular**
4. Descarga el **reporte PDF**
    """)

    st.markdown("---")
    st.markdown(
        "<small>Control de Calidad — Física Médica<br>"
        "Versión 1.0</small>",
        unsafe_allow_html=True,
    )


# ── Contenido principal ───────────────────────────────────────────────────────
col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.markdown("### 📂 Subir imagen DICOM")

    archivo = st.file_uploader(
        "Arrastra o selecciona el archivo",
        type=None,
        help="Formato DICOM (.dcm) — imagen del fantoma de QC",
    )

    if archivo:
        # Leer DICOM
        try:
            ds  = pydicom.dcmread(io.BytesIO(archivo.read()))
            img = ds.pixel_array.astype(float)
        except Exception as e:
            st.error(f"Error al leer el archivo DICOM: {e}")
            st.stop()

        px_mm = float(getattr(ds, "PixelSpacing", [0.15, 0.15])[0])

        # Información del archivo
        st.success("Imagen cargada correctamente")
        with st.expander("Metadatos DICOM", expanded=False):
            st.markdown(f"""
| Parámetro | Valor |
|---|---|
| Equipo | `{getattr(ds, 'ManufacturerModelName', 'N/D')}` |
| Modalidad | `{getattr(ds, 'Modality', 'N/D')}` |
| Tamaño | `{img.shape[1]} × {img.shape[0]} px` |
| Pixel spacing | `{px_mm} mm` |
| Nyquist | `{1/(2*px_mm):.2f} lp/mm` |
| Fecha | `{getattr(ds, 'StudyDate', 'N/D')}` |
""")

        # Previsualización
        st.markdown("**Previsualización:**")
        tab1, tab2 = st.tabs(["Ventana normal", "Ventana estrecha"])
        with tab1:
            st.image(
                _normalizar(img, 1, 99),
                caption="Vista general",
                use_column_width=True,
            )
        with tab2:
            st.image(
                _normalizar(img, 30, 70),
                caption="Ventana estrecha (revela el cuadrado)",
                use_column_width=True,
            )

        # Botón calcular
        st.markdown("---")
        calcular = st.button(
            "▶ Calcular QC",
            type="primary",
            use_container_width=True,
        )


# ── Función auxiliar de normalización ────────────────────────────────────────
def _normalizar(img, p_low=1, p_high=99):
    """Normaliza la imagen entre 0 y 1 usando percentiles."""
    v_min = np.percentile(img, p_low)
    v_max = np.percentile(img, p_high)
    return np.clip((img - v_min) / (v_max - v_min + 1e-10), 0, 1)


# ── Panel de resultados ───────────────────────────────────────────────────────
with col_der:
    if not archivo:
        # Pantalla de bienvenida
        st.markdown("""
### 👈 Empieza aquí

1. Selecciona el **equipo** en el panel izquierdo
2. Sube tu imagen **DICOM**
3. Haz clic en **Calcular QC**

---

#### Módulos disponibles

| Equipo | Prueba | Estado |
|---|---|---|
| Rayos X Diagnóstico | MTF — Resolución Espacial | ✅ Disponible |
| Mamografía | CNR, MTF | 🔜 Próximamente |
| TC | HU, Uniformidad, Ruido | 🔜 Próximamente |
| Fluoroscopía | MTF, CNR | 🔜 Próximamente |
        """)

    elif archivo and "calcular" in dir() and calcular:
        # Ejecutar módulo correspondiente
        with st.spinner("🔍 Procesando imagen..."):

            if equipo == "Rayos X Diagnóstico" and "MTF" in prueba:
                try:
                    from modulos.rayos_x.mtf import run
                    from utils.reporte_pdf import generar_pdf_mtf

                    resultado = run(img, ds)

                    # Mostrar figura
                    st.markdown("### 📊 Resultados")
                    st.pyplot(resultado["figura"])
                    plt = resultado["figura"]

                    # Métricas destacadas
                    st.markdown("### 🎯 Métricas principales")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.metric("MTF50 Horizontal",
                                  f"{resultado['mtf50_h']:.3f}" if resultado["mtf50_h"] else "—",
                                  "lp/mm")
                    with c2:
                        st.metric("MTF20 Horizontal",
                                  f"{resultado['mtf20_h']:.3f}" if resultado["mtf20_h"] else "—",
                                  "lp/mm")
                    with c3:
                        st.metric("MTF50 Vertical",
                                  f"{resultado['mtf50_v']:.3f}" if resultado["mtf50_v"] else "—",
                                  "lp/mm")
                    with c4:
                        st.metric("MTF20 Vertical",
                                  f"{resultado['mtf20_v']:.3f}" if resultado["mtf20_v"] else "—",
                                  "lp/mm")
                    with c5:
                        st.metric("Nyquist",
                                  f"{resultado['nyquist']:.2f}",
                                  "lp/mm")

                    # Tabla detallada
                    with st.expander("📋 Tabla completa de resultados", expanded=True):
                        import pandas as pd
                        df = pd.DataFrame({
                            "Dirección":    ["Horizontal", "Vertical"],
                            "MTF50 (lp/mm)":[
                                f"{resultado['mtf50_h']:.3f}" if resultado["mtf50_h"] else "—",
                                f"{resultado['mtf50_v']:.3f}" if resultado["mtf50_v"] else "—",
                            ],
                            "MTF20 (lp/mm)":[
                                f"{resultado['mtf20_h']:.3f}" if resultado["mtf20_h"] else "—",
                                f"{resultado['mtf20_v']:.3f}" if resultado["mtf20_v"] else "—",
                            ],
                            "FWHM LSF":[
                                f"{resultado['fwhm_h']:.3f} mm" if resultado["fwhm_h"] else "—",
                                f"{resultado['fwhm_v']:.3f} mm" if resultado["fwhm_v"] else "—",
                            ],
                            "Ángulo borde":[
                                f"{resultado['angulo_h']:.2f}°",
                                f"{resultado['angulo_v']:.2f}°",
                            ],
                        })
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    # Descarga del reporte PDF
                    st.markdown("### 📄 Reporte PDF")
                    pdf_bytes = generar_pdf_mtf(resultado, resultado["figura"])
                    st.download_button(
                        label="⬇️ Descargar reporte PDF",
                        data=pdf_bytes,
                        file_name=f"QC_MTF_{resultado['equipo'].replace(' ','_')}_{resultado['fecha'].replace('/','')}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"❌ Error durante el cálculo: {e}")
                    st.exception(e)

            else:
                st.info("🔜 Este módulo está en desarrollo. Próximamente disponible.")

    elif archivo:
        st.info("👈 Haz clic en **Calcular QC** para procesar la imagen.")
