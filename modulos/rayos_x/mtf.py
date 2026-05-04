def roi_sobre_borde(img, lado, tipo, nombre_lado, cy_cuad, cx_cuad,
                    px_mm=0.15, ancho_mm=ANCHO_MM, largo_fraccion=LARGO_FRACCION):

    profundo = int(10.0 / px_mm)     # ← cuánto entra hacia adentro del cuadrado
    margen   = int(2.0  / px_mm)     # ← separación desde el borde
    ancho    = int(lado["largo"] * 0.3)  # ← qué tan ancho a lo largo del borde
    cy       = lado["centro_y"]
    cx       = lado["centro_x"]
    H, W     = img.shape

    if tipo == "H":
        # Delgado a lo largo del borde (ancho), profundo hacia adentro (profundo)
        if nombre_lado == "top":
            r0 = max(0, cy + margen)
            r1 = min(H, cy + margen + profundo)  # ← entra hacia abajo
        else:
            r0 = max(0, cy - margen - profundo)
            r1 = min(H, cy - margen)              # ← entra hacia arriba
        c0 = max(0, cx - ancho // 2)             # ← estrecho horizontalmente
        c1 = min(W, cx + ancho // 2)

    else:
        # Delgado a lo largo del borde (ancho), profundo hacia adentro (profundo)
        if nombre_lado == "left":
            c0 = max(0, cx + margen)
            c1 = min(W, cx + margen + profundo)  # ← entra hacia la derecha
        else:
            c0 = max(0, cx - margen - profundo)
            c1 = min(W, cx - margen)              # ← entra hacia la izquierda
        r0 = max(0, cy - ancho // 2)             # ← estrecho verticalmente
        r1 = min(H, cy + ancho // 2)

    return r0, r1, c0, c1
