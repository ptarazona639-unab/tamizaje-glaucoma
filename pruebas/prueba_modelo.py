"""Carga el modelo provisional, predice el conjunto de desarrollo y comprueba la interfaz del paquete."""
import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from tamizaje_glaucoma.modelo import Modelo  # noqa: E402
from tamizaje_glaucoma.variables import calcular_variables, objetivo, tabla_desde_medidas  # noqa: E402

modelo = Modelo.cargar(os.path.join(RAIZ, "modelos", "modelo_provisional.joblib"))
print(modelo.resumen())
df = pd.read_csv(os.path.join(RAIZ, "datos", "desarrollo_particion_v4.csv"))
variables, _ = tabla_desde_medidas(df)
pred = modelo.predecir(variables)
assert set(pred.columns) == {"prediccion", "probabilidad"}
assert pred["prediccion"].isin([0, 1]).all() and pred["probabilidad"].between(0, 1).all()
y = objetivo(df)
print(f"Sobre los datos de entrenamiento (no es una medida de desempeño): "
      f"{(pred['prediccion'].to_numpy() == y).mean():.3f} de acierto")

# Un ojo suelto, con el formato de la interfaz
un_ojo = calcular_variables({"I": 80, "S": 56, "N": 40, "T": 64, "disc_area": 144256, "cup_area": 67264,
                             "Disco_X": 408, "Disco_Y": 440, "Excavacion_X": 304, "Excavacion_Y": 304}, "DERECHO")
print("Ejemplo 390.jpg:", modelo.predecir(pd.DataFrame([un_ojo])).iloc[0].to_dict())

# Errores que debe rechazar la validación
for malo in ({"cup_area": 200000}, {"S": 0}, {"Disco_Y": 0}):
    medidas = {"I": 80, "S": 56, "N": 40, "T": 64, "disc_area": 144256, "cup_area": 67264,
               "Disco_X": 408, "Disco_Y": 440, "Excavacion_X": 304, "Excavacion_Y": 304}
    medidas.update(malo)
    try:
        calcular_variables(medidas, "DERECHO")
        raise AssertionError(f"No rechazó {malo}")
    except ValueError as e:
        print("Rechazado como se esperaba:", e)
print("OK: el modelo carga y predice")
