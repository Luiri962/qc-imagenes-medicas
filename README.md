# QC Imágenes Médicas — App Web

Aplicación web para control de calidad de equipos de imágenes médicas.

## Estructura del proyecto

```
qc_app/
├── app.py                        ← App principal (Streamlit)
├── requirements.txt              ← Dependencias Python
├── modulos/
│   └── rayos_x/
│       └── mtf.py                ← Módulo MTF Rayos X
└── utils/
    └── reporte_pdf.py            ← Generador de reportes PDF
```

---

## Cómo desplegarlo GRATIS (sin saber programación)

### Paso 1 — Crear cuenta en GitHub (gratis)
1. Ve a https://github.com
2. Crea una cuenta con tu email

### Paso 2 — Crear un repositorio
1. Haz clic en el botón verde **"New"**
2. Nombre del repositorio: `qc-imagenes-medicas`
3. Selecciona **"Public"**
4. Haz clic en **"Create repository"**

### Paso 3 — Subir los archivos
1. En tu repositorio vacío, haz clic en **"uploading an existing file"**
2. Arrastra TODOS los archivos de esta carpeta (manteniendo la estructura)
3. Haz clic en **"Commit changes"**

### Paso 4 — Desplegar en Streamlit (gratis)
1. Ve a https://share.streamlit.io
2. Inicia sesión con tu cuenta de GitHub
3. Haz clic en **"New app"**
4. Selecciona tu repositorio `qc-imagenes-medicas`
5. En "Main file path" escribe: `app.py`
6. Haz clic en **"Deploy!"**

En 2-3 minutos tendrás una URL pública como:
`https://tu-nombre-qc-imagenes-medicas.streamlit.app`

---

## Cómo ejecutarlo localmente (en tu computador)

1. Instala Python desde https://python.org (versión 3.10 o superior)
2. Abre una terminal en la carpeta del proyecto
3. Ejecuta:
```bash
pip install -r requirements.txt
streamlit run app.py
```
4. Se abrirá automáticamente en tu navegador

---

## Cómo agregar nuevos módulos

Para agregar, por ejemplo, un módulo de CNR para mamografía:

1. Crea el archivo `modulos/mamografia/cnr.py`
2. Implementa una función `run(img, ds)` que retorne un dict con los resultados
3. En `app.py`, agrega la opción en el `selectbox` de pruebas
4. Llama al módulo con `from modulos.mamografia.cnr import run`

---

## Módulos disponibles

| Equipo | Prueba | Estado |
|---|---|---|
| Rayos X Diagnóstico | MTF (Resolución espacial) | ✅ Listo |
| Mamografía | CNR, MTF | 🔜 En desarrollo |
| TC | HU, Uniformidad, Ruido | 🔜 En desarrollo |
| Fluoroscopía | MTF, CNR | 🔜 En desarrollo |
