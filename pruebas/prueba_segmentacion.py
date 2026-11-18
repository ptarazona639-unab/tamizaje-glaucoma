"""Segmenta fotografías de ORIGA y compara las 10 medidas con las de la base del proyecto (deben ser idénticas).

Uso: python pruebas/prueba_segmentacion.py <carpeta con imágenes .jpg de ORIGA> <base .xlsx o .csv con Archivo y Tipo_Ojo> [n]
Los pesos deben estar en pesos/TrainAll_Maghrabi84_50iteration_SWIN.pth.tar.
"""
import glob
import os
import sys
import warnings

import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from tamizaje_glaucoma.segmentacion import NOMBRE_PESOS, Segmentador  # noqa: E402
from tamizaje_glaucoma.variables import NOMBRES_MEDIDAS  # noqa: E402

carpeta, ruta_base = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 6
base = (pd.read_csv(ruta_base) if ruta_base.endswith(".csv") else pd.read_excel(ruta_base)).set_index("Archivo")
seg = Segmentador(os.path.join(RAIZ, "pesos", NOMBRE_PESOS))
iguales, total = 0, 0
for ruta in sorted(glob.glob(os.path.join(carpeta, "**", "*.jpg"), recursive=True))[:n]:
    nombre = os.path.basename(ruta)
    if nombre not in base.index:
        continue
    medidas, _, _ = seg.analizar(Image.open(ruta), base.loc[nombre, "Tipo_Ojo"])
    difs = {c: medidas[c] - float(base.loc[nombre, c]) for c in NOMBRES_MEDIDAS}
    total += 1
    if all(abs(d) < 1e-6 for d in difs.values()):
        iguales += 1
    else:
        print(nombre, {k: v for k, v in difs.items() if abs(v) >= 1e-6})
print(f"{iguales} de {total} imágenes con las 10 medidas idénticas a la base")
assert iguales == total
