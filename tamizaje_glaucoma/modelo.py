"""Paquete de modelo intercambiable.

Un modelo se guarda como un archivo `.joblib` con un diccionario. Así, cuando se elija el modelo final, basta con
entrenarlo con `entrenar_modelo.py` (o exportarlo desde un notebook con `guardar_paquete`) y cargarlo en la
aplicación, sin tocar la interfaz.

Claves del diccionario:
    formato            "tamizaje-glaucoma/1"
    nombre             nombre corto que muestra la interfaz
    descripcion        texto libre
    provisional        True si es un modelo de demostración
    clasificador       "SVM", "XGBoost", "ANN", ...
    metodo_seleccion   "SFS", "FSCNCA", "Elastic Net", "MWVA", "Ninguno", ...
    alfa               alfa del Score Combinado con el que se eligió
    variables          lista de variables de entrada (subconjunto de VARIABLES_33, en el orden que espera el modelo)
    pipeline           objeto ya entrenado con .predict(X) (y opcionalmente .predict_proba(X)); recibe un DataFrame
                       con las columnas de `variables` y hace él mismo la normalización
    hiperparametros    dict
    entrenado_con      descripción de los datos de entrenamiento
    metricas_cv        dict con las métricas de validación cruzada (si las hay)
    fecha, versiones   trazabilidad
"""
import datetime
import platform

import joblib
import numpy as np
import pandas as pd

from .variables import VARIABLES_33

FORMATO = "tamizaje-glaucoma/1"
CLAVES_OBLIGATORIAS = ("formato", "nombre", "variables", "pipeline")


def versiones_actuales():
    import sklearn
    v = {"python": platform.python_version(), "scikit-learn": sklearn.__version__}
    try:
        import imblearn
        v["imbalanced-learn"] = imblearn.__version__
    except ImportError:
        pass
    return v


def guardar_paquete(ruta, pipeline, variables, nombre, **metadatos):
    """Guarda un modelo entrenado con el formato que lee la aplicación."""
    desconocidas = [v for v in variables if v not in VARIABLES_33]
    if desconocidas:
        raise ValueError("Variables que no existen en el proyecto: " + ", ".join(desconocidas))
    paquete = {
        "formato": FORMATO,
        "nombre": nombre,
        "variables": list(variables),
        "pipeline": pipeline,
        "fecha": datetime.date.today().isoformat(),
        "versiones": versiones_actuales(),
    }
    paquete.update(metadatos)
    joblib.dump(paquete, ruta, compress=3)
    return ruta


class Modelo:
    """Envoltura de un paquete cargado desde disco."""

    def __init__(self, paquete, ruta=None):
        faltan = [k for k in CLAVES_OBLIGATORIAS if k not in paquete]
        if faltan:
            raise ValueError("El archivo no es un modelo de esta aplicación (faltan: " + ", ".join(faltan) + ")")
        if paquete["formato"] != FORMATO:
            raise ValueError(f"Formato de modelo no compatible: {paquete['formato']}")
        desconocidas = [v for v in paquete["variables"] if v not in VARIABLES_33]
        if desconocidas:
            raise ValueError("El modelo usa variables desconocidas: " + ", ".join(desconocidas))
        self.paquete = paquete
        self.ruta = ruta

    @classmethod
    def cargar(cls, ruta):
        return cls(joblib.load(ruta), ruta=ruta)

    def __getattr__(self, clave):
        try:
            return self.__dict__["paquete"][clave]
        except KeyError:
            raise AttributeError(clave) from None

    def get(self, clave, defecto=None):
        return self.paquete.get(clave, defecto)

    @property
    def tiene_probabilidad(self):
        return hasattr(self.paquete["pipeline"], "predict_proba")

    def predecir(self, variables):
        """variables: DataFrame con (al menos) las columnas que usa el modelo.

        Devuelve un DataFrame con `prediccion` (1 = sospecha de glaucoma) y, si el modelo lo permite,
        `probabilidad` (probabilidad estimada de glaucoma).
        """
        X = variables[self.paquete["variables"]].astype(float)
        pipe = self.paquete["pipeline"]
        salida = pd.DataFrame(index=variables.index)
        umbral = self.get("umbral")
        if self.tiene_probabilidad:
            salida["probabilidad"] = np.asarray(pipe.predict_proba(X))[:, 1]
        if umbral is not None and self.tiene_probabilidad:
            salida["prediccion"] = (salida["probabilidad"] >= umbral).astype(int)
        else:
            salida["prediccion"] = np.asarray(pipe.predict(X)).astype(int)
        return salida[["prediccion"] + (["probabilidad"] if self.tiene_probabilidad else [])]

    def resumen(self, completo=False):
        """Texto para la interfaz: corto para el panel, completo para la ventana de detalle."""
        nombre = self.paquete["nombre"] + ("  [MODELO PROVISIONAL DE DEMOSTRACIÓN]" if self.get("provisional") else "")
        partes = [f"Clasificador: {self.get('clasificador', '?')}",
                  f"Selección: {self.get('metodo_seleccion', '?')}",
                  f"{len(self.paquete['variables'])} variables"]
        if self.get("alfa") is not None:
            partes.append(f"α = {self.get('alfa')}")
        if self.get("umbral") is not None:
            partes.append(f"umbral {self.get('umbral'):.0%}")
        lineas = [nombre, " · ".join(partes)]
        if completo:
            if self.get("descripcion"):
                lineas.append(self.get("descripcion"))
            lineas.append("Variables: " + ", ".join(self.paquete["variables"]))
            if self.get("hiperparametros"):
                lineas.append("Hiperparámetros: " + ", ".join(f"{k} = {v}" for k, v in self.get("hiperparametros").items()))
            if self.get("entrenado_con"):
                lineas.append("Entrenado con: " + self.get("entrenado_con"))
            cv = self.get("metricas_cv") or {}
            if cv:
                lineas.append("Validación cruzada (5 folds, sin anidar): " + ", ".join(f"{k} {v:.3f}" for k, v in cv.items()))
            if self.get("fecha"):
                lineas.append(f"Fecha: {self.get('fecha')} · Versiones: " +
                              ", ".join(f"{k} {v}" for k, v in (self.get("versiones") or {}).items()))
        return "\n".join(lineas)
