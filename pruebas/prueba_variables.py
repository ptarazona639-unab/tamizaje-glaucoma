"""Comprueba que las 33 variables calculadas desde las 10 medidas coinciden con las de la base del proyecto."""
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from tamizaje_glaucoma.variables import VARIABLES_33, preparar_base, tabla_desde_medidas  # noqa: E402

df = pd.read_csv(os.path.join(RAIZ, "datos", "desarrollo_particion_v4.csv"))
calculadas, errores = tabla_desde_medidas(df)
assert not errores, errores
base = preparar_base(df)
dif = (calculadas[VARIABLES_33] - base[VARIABLES_33]).abs()
peor = dif.max().sort_values(ascending=False)
print("Mayor diferencia por variable (las 5 peores):")
print(peor.head().to_string())
assert np.allclose(calculadas.values, base.values, rtol=1e-9, atol=1e-6), "Las variables no coinciden con la base"
print(f"OK: {len(df)} filas, las 33 variables coinciden con la base")
