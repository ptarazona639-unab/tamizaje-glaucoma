"""Entrena un modelo y lo guarda como paquete `.joblib` para la aplicación.

Reproduce el preprocesamiento del protocolo: normalización Min-Max y balanceo con ADASYN (vecinos con árbol k-d)
dentro de cada fold, y elección de hiperparámetros con el Score Combinado medio de una validación cruzada
estratificada por ojo y clase (5 folds, semilla 42).

Ejemplos:
    # Modelo provisional: SVM con las 33 variables y la malla de 74 combinaciones
    python entrenar_modelo.py --salida modelos/modelo_provisional.joblib --provisional

    # Modelo elegido por el director: variables e hiperparámetros fijos
    python entrenar_modelo.py --variables "Cup_Disc_Area_ratio,VCDR,I" --clasificador svm \
        --hiperparametros '{"kernel": "rbf", "C": 4, "gamma": 0.25}' --metodo "Elastic Net" \
        --nombre "SVM + Elastic Net (alfa 0.5)" --salida modelos/modelo_final.joblib

El SuperTest no se usa aquí: el archivo de datos por defecto es solo el conjunto de desarrollo (552 imágenes).
"""
import argparse
import warnings
import itertools
import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from tamizaje_glaucoma.modelo import guardar_paquete
from tamizaje_glaucoma.variables import VARIABLES_33, objetivo, preparar_base

SEMILLA = 42


def malla_svm():
    """Las 74 combinaciones del protocolo: lineal 7, RBF 49 (Vabalas et al., 2019) y polinómico 18."""
    malla = [{"kernel": "linear", "C": 2.0 ** c} for c in range(1, 8)]
    malla += [{"kernel": "rbf", "C": 2.0 ** c, "gamma": 2.0 ** -g} for c in range(1, 8) for g in range(1, 8)]
    malla += [{"kernel": "poly", "C": 2.0 ** c, "gamma": 2.0 ** -g, "degree": d, "coef0": 0.0}
              for c, g, d in itertools.product((1, 3, 5), (1, 3, 5), (2, 3))]
    assert len(malla) == 74
    return malla


def construir_clasificador(nombre, hiper, probabilidad=False):
    if nombre == "svm":
        return SVC(max_iter=10_000, probability=probabilidad, random_state=SEMILLA, **hiper)
    if nombre == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_jobs=1, tree_method="hist", random_state=SEMILLA, eval_metric="logloss", **hiper)
    raise ValueError(f"Clasificador no soportado: {nombre}")


def construir_pipeline(clasificador, hiper, probabilidad=False):
    vecinos = NearestNeighbors(n_neighbors=6, algorithm="kd_tree")
    return Pipeline([
        ("minmax", MinMaxScaler()),
        ("adasyn", ADASYN(random_state=SEMILLA, n_neighbors=vecinos)),
        ("clasificador", construir_clasificador(clasificador, hiper, probabilidad)),
    ])


def score_combinado(y, p, alfa):
    tp = np.sum((y == 1) & (p == 1)); fn = np.sum((y == 1) & (p == 0))
    tn = np.sum((y == 0) & (p == 0)); fp = np.sum((y == 0) & (p == 1))
    sens = tp / (tp + fn) if tp + fn else 0.0
    esp = tn / (tn + fp) if tn + fp else 0.0
    return alfa * sens + (1 - alfa) * esp, sens, esp


def validar(X, y, estratos, clasificador, hiper, alfa):
    folds = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
    sc, se, es = [], [], []
    for tr, va in folds.split(X, estratos):
        pipe = construir_pipeline(clasificador, hiper).fit(X.iloc[tr], y[tr])
        s, a, b = score_combinado(y[va], pipe.predict(X.iloc[va]), alfa)
        sc.append(s); se.append(a); es.append(b)
    return float(np.mean(sc)), float(np.mean(se)), float(np.mean(es))


def elegir_umbral(X, y, estratos, clasificador, hiper, alfa):
    """Umbral sobre la probabilidad que maximiza el Score Combinado con predicciones fuera de fold.

    Así la clase y la probabilidad que muestra la aplicación nunca se contradicen, y el umbral responde al alfa.
    """
    folds = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
    prob = np.zeros(len(y))
    for tr, va in folds.split(X, estratos):
        pipe = construir_pipeline(clasificador, hiper, probabilidad=True).fit(X.iloc[tr], y[tr])
        prob[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    candidatos = np.unique(np.round(prob, 4))
    puntajes = [score_combinado(y, (prob >= u).astype(int), alfa)[0] for u in candidatos]
    return float(candidatos[int(np.argmax(puntajes))])


def main():
    warnings.filterwarnings("ignore", message="Solver terminated early")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default="datos/desarrollo_particion_v4.csv")
    ap.add_argument("--clasificador", choices=["svm", "xgb"], default="svm")
    ap.add_argument("--variables", default="todas", help="'todas' o lista separada por comas")
    ap.add_argument("--alfa", type=float, default=0.5)
    ap.add_argument("--hiperparametros", default=None, help="JSON con hiperparámetros fijos (omite la búsqueda)")
    ap.add_argument("--metodo", default="Ninguno", help="método de selección que eligió las variables")
    ap.add_argument("--nombre", default="SVM con las 33 variables")
    ap.add_argument("--provisional", action="store_true")
    ap.add_argument("--salida", default="modelos/modelo_provisional.joblib")
    a = ap.parse_args()

    df = pd.read_csv(a.datos) if a.datos.endswith(".csv") else pd.read_excel(a.datos)
    variables = VARIABLES_33 if a.variables == "todas" else [v.strip() for v in a.variables.split(",")]
    X = preparar_base(df)[variables]
    y = objetivo(df)
    estratos = df["Tipo_Ojo"].astype(str) + "_" + df["class"].astype(str)
    print(f"Datos: {len(df)} filas, {int(y.sum())} positivos, {len(variables)} variables")

    if a.hiperparametros:
        mejor = json.loads(a.hiperparametros)
        sc, se, es = validar(X, y, estratos, a.clasificador, mejor, a.alfa)
    else:
        if a.clasificador != "svm":
            raise SystemExit("La búsqueda automática solo está para SVM; para XGBoost pase --hiperparametros")
        resultados = []
        for i, h in enumerate(malla_svm()):
            resultados.append((*validar(X, y, estratos, "svm", h, a.alfa), i, h))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/74 combinaciones evaluadas")
        sc, se, es, _, mejor = max(resultados, key=lambda r: (round(r[0], 6), -r[3]))
    print(f"Hiperparámetros: {mejor}")
    print(f"Validación cruzada (5 folds): SC {sc:.3f}, sensibilidad {se:.3f}, especificidad {es:.3f}")

    umbral = elegir_umbral(X, y, estratos, a.clasificador, mejor, a.alfa)
    print(f"Umbral sobre la probabilidad (máximo SC con predicciones fuera de fold): {umbral:.3f}")
    final = construir_pipeline(a.clasificador, mejor, probabilidad=(a.clasificador == "svm")).fit(X, y)
    guardar_paquete(
        a.salida, final, variables, a.nombre,
        descripcion=("Min-Max + ADASYN + " + a.clasificador.upper() + ", entrenado con todo el conjunto de desarrollo. "
                     "La probabilidad sale del escalado de Platt de scikit-learn y la clase, de comparar esa probabilidad con el umbral."),
        provisional=a.provisional, clasificador={"svm": "SVM", "xgb": "XGBoost"}[a.clasificador],
        metodo_seleccion=a.metodo, alfa=a.alfa, hiperparametros=mejor,
        entrenado_con=f"{a.datos} ({len(df)} imágenes de desarrollo, partición v4; SuperTest no usado)",
        metricas_cv={"SC": sc, "sensibilidad": se, "especificidad": es}, umbral=umbral,
    )
    print(f"Guardado en {a.salida}")


if __name__ == "__main__":
    main()
