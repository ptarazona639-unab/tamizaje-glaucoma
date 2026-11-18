"""Cálculo de las 33 variables del modelo a partir de las medidas de la segmentación.

Las medidas de entrada están en píxeles de la imagen original (ORIGA, 3072 x 2048), como en la base del proyecto:
grosores del anillo neurorretiniano por sector (I, S, N, T), áreas del disco y de la copa, y diámetros horizontal (X)
y vertical (Y) del disco y de la copa. Con ellas se calculan exactamente las mismas columnas de la base
`dataset-glaucoma-crudo:crudo-v3` (verificado en `pruebas/prueba_variables.py`).
"""
import math

import numpy as np
import pandas as pd

# Medidas que el usuario entrega (nombre de columna, etiqueta para la interfaz)
MEDIDAS = [
    ("I", "Grosor inferior del anillo, I (px)"),
    ("S", "Grosor superior del anillo, S (px)"),
    ("N", "Grosor nasal del anillo, N (px)"),
    ("T", "Grosor temporal del anillo, T (px)"),
    ("disc_area", "Área del disco (px²)"),
    ("cup_area", "Área de la copa (px²)"),
    ("Disco_X", "Diámetro horizontal del disco (px)"),
    ("Disco_Y", "Diámetro vertical del disco (px)"),
    ("Excavacion_X", "Diámetro horizontal de la copa (px)"),
    ("Excavacion_Y", "Diámetro vertical de la copa (px)"),
]
NOMBRES_MEDIDAS = [m for m, _ in MEDIDAS]

# Orden de las 33 variables de entrada del proyecto (el mismo de la base y de los notebooks)
VARIABLES_33 = [
    "Tipo_Ojo", "I", "S", "N", "T", "ISNT", "I/S", "N/S", "I^2", "S^2", "N^2", "T^2",
    "I*S", "I*N", "I*T", "S*N", "S*T", "N*T", "Sen(I)", "Sen(S)", "Sen(N)", "Sen(T)",
    "disc_area", "cup_area", "rim_area", "Cup_Disc_Area_ratio", "Rim_Disc_Area_ratio",
    "Disco_X", "Disco_Y", "Excavacion_X", "Excavacion_Y", "VCDR", "HCDR",
]


def codificar_ojo(valor):
    """IZQUIERDO = 1, DERECHO = 0 (codificación de los notebooks). Acepta texto o número."""
    if isinstance(valor, str):
        v = valor.strip().upper()
        if v.startswith("IZQ") or v in ("L", "OS", "LEFT"):
            return 1
        if v.startswith("DER") or v in ("R", "OD", "RIGHT"):
            return 0
        raise ValueError(f"Ojo no reconocido: {valor!r} (use DERECHO o IZQUIERDO)")
    v = int(valor)
    if v not in (0, 1):
        raise ValueError("Tipo_Ojo numérico debe ser 0 (derecho) o 1 (izquierdo)")
    return v


def codificar_isnt(valor):
    """ISNT de la base: 'Sí' = 1, 'No' = 0."""
    if isinstance(valor, str):
        return 1 if valor.strip().lower() in ("sí", "si", "s", "1", "true") else 0
    return int(bool(valor))


def validar_medidas(medidas):
    """Revisa que las medidas tengan sentido físico. Devuelve una lista de problemas (vacía si todo está bien)."""
    problemas = []
    for nombre in NOMBRES_MEDIDAS:
        if nombre not in medidas or medidas[nombre] is None or (isinstance(medidas[nombre], float) and math.isnan(medidas[nombre])):
            problemas.append(f"Falta {nombre}")
        elif medidas[nombre] < 0:
            problemas.append(f"{nombre} no puede ser negativo")
    if problemas:
        return problemas
    if medidas["S"] == 0:
        problemas.append("S no puede ser 0 (se divide por S en I/S y N/S)")
    for a, b in (("disc_area", "Área del disco"), ("Disco_X", "Diámetro horizontal del disco"),
                 ("Disco_Y", "Diámetro vertical del disco")):
        if medidas[a] == 0:
            problemas.append(f"{b} no puede ser 0")
    if medidas["cup_area"] > medidas["disc_area"]:
        problemas.append("El área de la copa no puede ser mayor que la del disco")
    if medidas["Excavacion_X"] > medidas["Disco_X"] or medidas["Excavacion_Y"] > medidas["Disco_Y"]:
        problemas.append("Un diámetro de la copa no puede ser mayor que el del disco")
    return problemas


def calcular_variables(medidas, ojo):
    """Calcula las 33 variables de un ojo.

    medidas: dict con las 10 medidas de `MEDIDAS`.
    ojo: 'DERECHO' / 'IZQUIERDO' (o 0 / 1).
    Devuelve un dict {variable: valor} con las 33 variables en el orden de `VARIABLES_33`.
    """
    problemas = validar_medidas(medidas)
    if problemas:
        raise ValueError("; ".join(problemas))
    I, S, N, T = (float(medidas[k]) for k in ("I", "S", "N", "T"))
    disco, copa = float(medidas["disc_area"]), float(medidas["cup_area"])
    dx, dy = float(medidas["Disco_X"]), float(medidas["Disco_Y"])
    ex, ey = float(medidas["Excavacion_X"]), float(medidas["Excavacion_Y"])
    seno = lambda grados: math.sin(math.radians(grados))  # la base usa el seno de la medida en grados
    razon_copa = copa / disco
    v = {
        "Tipo_Ojo": codificar_ojo(ojo),
        "I": I, "S": S, "N": N, "T": T,
        "ISNT": int(I >= S >= N >= T),
        "I/S": I / S, "N/S": N / S,
        "I^2": I ** 2, "S^2": S ** 2, "N^2": N ** 2, "T^2": T ** 2,
        "I*S": I * S, "I*N": I * N, "I*T": I * T, "S*N": S * N, "S*T": S * T, "N*T": N * T,
        "Sen(I)": seno(I), "Sen(S)": seno(S), "Sen(N)": seno(N), "Sen(T)": seno(T),
        "disc_area": disco, "cup_area": copa, "rim_area": disco - copa,
        "Cup_Disc_Area_ratio": razon_copa, "Rim_Disc_Area_ratio": 1.0 - razon_copa,
        "Disco_X": dx, "Disco_Y": dy, "Excavacion_X": ex, "Excavacion_Y": ey,
        "VCDR": ey / dy, "HCDR": ex / dx,
    }
    return {k: v[k] for k in VARIABLES_33}


def tabla_desde_medidas(df):
    """Calcula las 33 variables para cada fila de una tabla con las 10 medidas y una columna `Tipo_Ojo`.

    Devuelve (variables, errores): un DataFrame con las 33 variables (filas válidas) y una lista
    (índice de fila, mensaje) con las filas que no se pudieron calcular.
    """
    faltan = [c for c in NOMBRES_MEDIDAS + ["Tipo_Ojo"] if c not in df.columns]
    if faltan:
        raise ValueError("Faltan columnas en el archivo: " + ", ".join(faltan))
    filas, indices, errores = [], [], []
    for i, fila in df.iterrows():
        try:
            medidas = {c: float(fila[c]) for c in NOMBRES_MEDIDAS}
            filas.append(calcular_variables(medidas, fila["Tipo_Ojo"]))
            indices.append(i)
        except (ValueError, TypeError) as e:
            errores.append((i, str(e)))
    return pd.DataFrame(filas, index=indices, columns=VARIABLES_33), errores


def preparar_base(df):
    """Convierte una tabla con el formato de la base del proyecto (Tipo_Ojo y ISNT como texto) a numérica."""
    X = df[VARIABLES_33].copy()
    X["Tipo_Ojo"] = X["Tipo_Ojo"].map(codificar_ojo)
    X["ISNT"] = X["ISNT"].map(codificar_isnt)
    return X.astype(float)


def objetivo(df):
    """Clase de la base: positivo = 1 (glaucoma), negativo = 0."""
    return (df["class"].astype(str).str.lower() == "positivo").astype(int).to_numpy()
