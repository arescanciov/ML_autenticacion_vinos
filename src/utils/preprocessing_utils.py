"""
preprocessing_utils.py
-----------------------
Feature engineering para el dataset de autenticación de variedad de vino.
Se separa del notebook para garantizar que la misma transformación se
aplica de forma idéntica en entrenamiento y en inferencia.
"""

import pandas as pd


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Añade variables derivadas a partir de las features químicas originales
    del dataset Wine. Las features de entrada están ya escaladas
    (StandardScaler); los ratios e interacciones se calculan igualmente
    porque capturan relaciones relativas entre compuestos, potencialmente
    más estables entre cultivares que los valores absolutos.

    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame con, al menos, las columnas originales del dataset Wine.

    Returns
    -------
    pd.DataFrame
        Copia de `frame` con las columnas nuevas añadidas.
    """
    frame = frame.copy()

    # Proporción de fenoles totales que no son flavonoides: distintos
    # cultivares pueden tener perfiles fenólicos muy distintos.
    frame["phenols_flavanoid_ratio"] = frame["total_phenols"] / (frame["flavanoids"] + 1e-6)

    # Interacción alcohol x intensidad de color: combinación relevante en
    # la caracterización sensorial y química de un vino.
    frame["alcohol_color_interaction"] = frame["alcohol"] * frame["color_intensity"]

    # Ratio flavonoides / tono (hue): relación observada como discriminante
    # en la literatura de autenticación de vinos por análisis químico.
    frame["flavanoid_hue_ratio"] = frame["flavanoids"] / (frame["hue"] + 1e-6)

    return frame
