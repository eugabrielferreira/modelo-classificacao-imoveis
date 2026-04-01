import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ── 1. Carrega os dados ──────────────────────────────────────────────────────
df = pd.read_csv("housing.csv")

# ── 2. Cria a variável alvo ──────────────────────────────────────────────────
# Pergunta: este bloco está acima da mediana de preços?
mediana = df["median_house_value"].median()
df["caro"] = (df["median_house_value"] > mediana).astype(int)

print(f"Mediana de preços: US$ {mediana:,.0f}")
print(f"Imóveis acima da mediana (classe 1): {df['caro'].sum()}")
print(f"Imóveis abaixo da mediana (classe 0): {(df['caro'] == 0).sum()}")

# ── 3. Seleciona as features ─────────────────────────────────────────────────
# Removemos: median_house_value (é de onde veio o alvo)
# Removemos: ocean_proximity (texto — trataríamos com encoding, por ora simplificamos)
features = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income"
]

X = df[features].copy()
y = df["caro"].copy()

# ── 4. Trata nulos ───────────────────────────────────────────────────────────
# total_bedrooms tem 207 nulos — preenchemos com a mediana da coluna
X["total_bedrooms"] = X["total_bedrooms"].fillna(X["total_bedrooms"].median())

print(f"\nNulos após tratamento: {X.isnull().sum().sum()}")

# ── 5. Divide em treino e teste ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTreino: {len(X_train)} registros")
print(f"Teste:  {len(X_test)} registros")

# ── 6. Escala as features ────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 7. Treina o modelo ───────────────────────────────────────────────────────
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train_scaled, y_train)

print("\nModelo treinado.")

# ── 8. Avalia ────────────────────────────────────────────────────────────────
y_pred = modelo.predict(X_test_scaled)

print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_test, y_pred, target_names=["Abaixo da mediana", "Acima da mediana"]))

print("=== MATRIZ DE CONFUSÃO ===")
print(confusion_matrix(y_test, y_pred))