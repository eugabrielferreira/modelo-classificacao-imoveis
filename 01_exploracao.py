import pandas as pd

# Carrega o dataset
df = pd.read_csv("housing.csv")

# Estrutura básica
print("=== SHAPE ===")
print(df.shape)

print("\n=== PRIMEIRAS LINHAS ===")
print(df.head())

print("\n=== TIPOS E NULOS ===")
print(df.info())

print("\n=== ESTATÍSTICAS ===")
print(df.describe())

print("\n=== VALORES ÚNICOS - ocean_proximity ===")
print(df["ocean_proximity"].value_counts())