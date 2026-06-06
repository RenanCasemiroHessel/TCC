"""
Geração do Dataset Filtrado
Gera dataset_features_leves.csv a partir do dataset completo:
  - Remove lote B3 (dados clonados)
  - Mantém apenas bovinos leves (peso < 250 kg)
  - Remove linhas duplicadas
"""

import os
import pandas as pd

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_IN   = os.path.join(SAVE_DIR, "dataset_features_completo.csv")
CSV_OUT  = os.path.join(SAVE_DIR, "dataset_features_leves.csv")

print("\n" + "="*60)
print("  Filtragem do Dataset")
print("="*60 + "\n")

if not os.path.exists(CSV_IN):
    print(f"  ERRO: {CSV_IN} não encontrado.")
    print("  Certifique-se de que o dataset_features_completo.csv existe.")
    exit(1)

df = pd.read_csv(CSV_IN)
print(f"  Total original            : {len(df)}")
print(f"  Lotes originais           : {df['lote'].value_counts().to_dict()}")
print(f"  Distribuição de peso      : min={df['peso_kg'].min():.0f}kg, max={df['peso_kg'].max():.0f}kg")

df = df[df["lote"] != "B3"].copy()
print(f"\n  Após remover B3           : {len(df)}")

df = df[df["peso_kg"] < 250].copy()
print(f"  Após filtrar leves (<250kg): {len(df)}")

before_dedup = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"  Após remover duplicatas   : {len(df)} (removidas: {before_dedup - len(df)})")

print(f"\n  Distribuição final:")
print(f"    Lotes  : {df['lote'].value_counts().to_dict()}")
print(f"    Vista  : {df['vista'].value_counts().to_dict()}")
print(f"    Sexo   : {df['sexo'].value_counts().to_dict()}")
print(f"    Peso   : min={df['peso_kg'].min():.0f}kg, max={df['peso_kg'].max():.0f}kg, média={df['peso_kg'].mean():.1f}kg")

df.to_csv(CSV_OUT, index=False)
print(f"\n  Salvo em: {CSV_OUT}")
print("="*60 + "\n")
