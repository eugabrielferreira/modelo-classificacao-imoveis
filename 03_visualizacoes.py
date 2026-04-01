import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── Reproduz o modelo ────────────────────────────────────────────────────────
df = pd.read_csv("housing.csv")
mediana = df["median_house_value"].median()
df["caro"] = (df["median_house_value"] > mediana).astype(int)

features = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income"
]

X = df[features].copy()
y = df["caro"].copy()
X["total_bedrooms"] = X["total_bedrooms"].fillna(X["total_bedrooms"].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train_scaled, y_train)
y_pred = modelo.predict(X_test_scaled)

# ── Gráfico 1 — Matriz de confusão visual ───────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Abaixo da mediana", "Acima da mediana"]
)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Matriz de Confusão — Regressão Logística\n83% de acurácia em dados de teste", pad=12)
plt.tight_layout()
plt.savefig("grafico1_matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico 1 salvo.")

# ── Gráfico 2 — Importância das features (coeficientes) ─────────────────────
coef = pd.Series(modelo.coef_[0], index=features).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
cores = ["#E63946" if v < 0 else "#2A9D8F" for v in coef.values]
ax.barh(coef.index, coef.values, color=cores)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Quais variáveis mais influenciam o preço?\nCoeficientes da Regressão Logística", pad=12)
ax.set_xlabel("Coeficiente (positivo = aumenta chance de ser caro)")
plt.tight_layout()
plt.savefig("grafico2_coeficientes.png", dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico 2 salvo.")

# ── Gráfico 3 — Distribuição geográfica dos erros ───────────────────────────
X_test_original = X_test.copy()
X_test_original["real"] = y_test.values
X_test_original["previsto"] = y_pred
X_test_original["erro"] = (X_test_original["real"] != X_test_original["previsto"]).astype(int)

fig, ax = plt.subplots(figsize=(9, 6))
acertos = X_test_original[X_test_original["erro"] == 0]
erros = X_test_original[X_test_original["erro"] == 1]

ax.scatter(acertos["longitude"], acertos["latitude"],
           c="#2A9D8F", alpha=0.3, s=5, label="Acerto")
ax.scatter(erros["longitude"], erros["latitude"],
           c="#E63946", alpha=0.6, s=8, label="Erro")

ax.set_title("Onde o modelo erra geograficamente?\nDistribuição dos erros na Califórnia", pad=12)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig("grafico3_erros_geograficos.png", dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico 3 salvo.")