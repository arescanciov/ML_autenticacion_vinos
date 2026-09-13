"""
eda_utils.py
------------
Funciones auxiliares para el EDA dirigido al modelado del proyecto de
autenticación de variedad de vino. Se mantienen fuera del notebook para
poder reutilizarlas y para no saturar el pipeline principal de código
repetitivo.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.feature_selection import f_classif, mutual_info_classif


def iqr_outlier_summary(df: pd.DataFrame) -> pd.Series:
    """Porcentaje de valores fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR] por columna."""
    result = {}
    for col in df.select_dtypes(include=np.number).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        result[col] = ((df[col] < low) | (df[col] > high)).mean() * 100
    return pd.Series(result).sort_values(ascending=False)


def zscore_outlier_counts(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Nº de observaciones con |z-score| > threshold, por columna numérica."""
    numeric_df = df.select_dtypes(include=np.number)
    z = np.abs(stats.zscore(numeric_df))
    counts = (z > threshold).sum(axis=0)
    return pd.Series(counts, index=numeric_df.columns).sort_values(ascending=False)


def mahalanobis_outliers(df: pd.DataFrame, alpha: float = 0.025) -> pd.Series:
    """
    Detección de outliers MULTIVARIANTE mediante distancia de Mahalanobis.

    A diferencia de IQR/Z-score (que miran cada variable por separado), la
    distancia de Mahalanobis detecta observaciones "raras" en conjunto, es
    decir, combinaciones de valores poco habituales aunque cada variable
    individual esté dentro de rango.

    Devuelve una Serie con la distancia de Mahalanobis de cada fila y marca
    como outlier las que superan el percentil correspondiente de una
    chi-cuadrado con `df.shape[1]` grados de libertad.
    """
    numeric_df = df.select_dtypes(include=np.number)
    cov = np.cov(numeric_df.values, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    mean_vec = numeric_df.mean().values

    distances = numeric_df.apply(
        lambda row: mahalanobis(row.values, mean_vec, inv_cov), axis=1
    )
    threshold = np.sqrt(stats.chi2.ppf(1 - alpha, df=numeric_df.shape[1]))
    is_outlier = distances > threshold
    return distances, is_outlier, threshold


def anova_feature_ranking(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Ranking de features según el estadístico F de ANOVA respecto al target
    (multiclase). Un F alto indica que las medias de la variable difieren
    mucho entre clases en relación a la varianza interna de cada clase.
    """
    f_values, _ = f_classif(X, y)
    return pd.Series(f_values, index=X.columns).sort_values(ascending=False)


def mutual_information_ranking(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.Series:
    """
    Ranking de features según información mutua con el target. A diferencia
    de ANOVA (que solo detecta relaciones lineales entre medias), la
    información mutua también puede capturar relaciones no lineales.
    """
    mi_values = mutual_info_classif(X, y, random_state=random_state)
    return pd.Series(mi_values, index=X.columns).sort_values(ascending=False)


def high_correlation_pairs(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Pares de variables numéricas con correlación absoluta >= threshold."""
    corr = df.select_dtypes(include=np.number).corr().abs()
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    pairs = pairs[pairs["correlation"] >= threshold].sort_values("correlation", ascending=False)
    return pairs.reset_index(drop=True)
