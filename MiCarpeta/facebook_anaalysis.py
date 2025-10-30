# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1) Carga de datos
path = Path("Facebook Live Sellers en Tailandia.csv")
df = pd.read_csv(path, encoding="utf-8", engine="python")

# 2) Resumen general de columnas
summary = pd.DataFrame({
    "tipo_dato": df.dtypes.astype(str),
    "n_no_nulos": df.notna().sum(),
    "n_nulos": df.isna().sum(),
    "%_nulos": (df.isna().mean() * 100).round(2),
    "n_unicos": df.nunique(),
})
print("\nResumen general de columnas:")
print(summary)

# 3) Selección de variables numéricas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 4) Histogramas
for col in num_cols:
    plt.figure()
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f"Histograma: {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

# 5) Diagramas de caja y bigotes
for col in num_cols:
    plt.figure()
    plt.boxplot(df[col].dropna(), vert=True)
    plt.title(f"Caja y bigotes: {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# 6) Mapa de calor de correlación
if len(num_cols) >= 2:
    corr = df[num_cols].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation='nearest')
    plt.title("Mapa de calor de correlación (variables numéricas)")
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# 7) Variables con poca varianza o muchos nulos
varianza = df[num_cols].var(numeric_only=True)
std = df[num_cols].std(numeric_only=True)
diagnostico_vars = pd.DataFrame({
    "varianza": varianza,
    "std": std,
    "%_nulos": df[num_cols].isna().mean().round(4),
})
print("\nDiagnóstico de variables numéricas:")
print(diagnostico_vars)

# 8) Detección de outliers mediante IQR
def iqr_outlier_count(s):
    s = s.dropna()
    if s.empty:
        return 0, np.nan, np.nan, np.nan, np.nan
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    count = ((s < low) | (s > high)).sum()
    return count, q1, q3, low, high

out_stats = []
for col in num_cols:
    count, q1, q3, low, high = iqr_outlier_count(df[col])
    out_stats.append([col, count, q1, q3, low, high, df[col].min(), df[col].max()])
outliers_df = pd.DataFrame(out_stats, columns=["variable", "n_outliers", "Q1", "Q3", "lim_inf", "lim_sup", "min", "max"])
print("\nResumen de outliers:")
print(outliers_df)

# 9) Comparación de rangos
rangos = df[num_cols].agg(['min', 'max', 'mean', 'std']).T
rangos["rango_(max-min)"] = rangos["max"] - rangos["min"]
print("\nRangos por variable numérica:")
print(rangos)

# 10) Agrupamiento (K-Means)
usable = [c for c in num_cols if df[c].isna().mean() < 0.3]
df_imp = df[usable].copy()

if len(usable) >= 2 and df_imp.shape[0] > 5:
    for c in usable:
        df_imp[c] = df_imp[c].fillna(df_imp[c].median())

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = df_imp.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k, best_score = None, -1
    best_km, best_labels = None, None
    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        if score > best_score:
            best_k, best_score = k, score
            best_km = km
            best_labels = labels

    # Centroides y tamaños
    centroids_scaled = best_km.cluster_centers_
    centroids = pd.DataFrame(scaler.inverse_transform(centroids_scaled), columns=usable)
    centroids["cluster"] = range(best_k)
    print(f"\nCentroides por clúster (k={best_k}, silhouette={best_score:.3f}):")
    print(centroids)

    sizes = pd.Series(best_labels).value_counts().sort_index()
    print("\nTamaño de cada clúster:")
    print(sizes)
else:
    print("\nNo hay suficientes columnas numéricas utilizables para clustering o pocas filas.")
