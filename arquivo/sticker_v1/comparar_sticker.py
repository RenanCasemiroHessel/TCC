"""
Comparação controlada: mesma base de dados, mesmo split, mesmos modelos —
com e sem a feature calibrada pelo sticker de referência.

Entrada: dataset_features_sticker.csv (só side-view, 2.402 imagens — o sticker
não aparece nas fotos traseiras, então a comparação fica restrita a lateral
pra ser justa dos dois lados).
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SAVE_DIR, "dataset_features_sticker.csv")

df = pd.read_csv(CSV_PATH)
df = df[df["vista"] == "side"].copy()  # sticker só existe em lateral

df_enc = pd.get_dummies(df, columns=["sexo"], drop_first=True)
if "sexo_M" in df_enc.columns:
    df_enc["sexo_M"] = df_enc["sexo_M"].astype(float)
else:
    df_enc["sexo_M"] = 0.0

# mesma engenharia de features do pipeline atual (train_xgboost.py / train_linear_regression.py)
df_enc["bbox_area"] = df_enc["altura_bbox"] * df_enc["largura_bbox"]
df_enc["fill_ratio"] = df_enc["area"] / (df_enc["bbox_area"] + 1e-8)
df_enc["volume_approx"] = df_enc["area"] ** 1.5
df_enc["volume_norm_approx"] = df_enc["area_norm"] ** 1.5

BASE_FEATURES = [
    "area", "altura_bbox", "largura_bbox", "proporcao", "area_norm",
    "bbox_area", "fill_ratio", "volume_approx", "volume_norm_approx", "sexo_M",
]
STICKER_FEATURES = ["sticker_ratio", "altura_cal", "largura_cal"]

y = df_enc["peso_kg"].values

configs = {
    "SEM sticker (baseline atual)": BASE_FEATURES,
    "COM sticker_ratio só":         BASE_FEATURES + ["sticker_ratio"],
    "COM sticker completo (+cal)":  BASE_FEATURES + STICKER_FEATURES,
}

def avaliar(nome_modelo, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    acc10 = float((np.abs(y_pred - y_true) / y_true <= 0.10).mean() * 100)
    return {"modelo": nome_modelo, "MAE": mae, "RMSE": rmse, "R2": r2, "pearson_r": pearson, "acc10": acc10}

resultados = []

for nome_config, features in configs.items():
    X = df_enc[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Regressão Linear (Ridge + PolynomialFeatures grau 2, igual train_linear_regression.py)
    lr = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        Ridge(alpha=10),
    )
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    r = avaliar(f"LR — {nome_config}", y_test, pred_lr)
    r["n_features"] = len(features)
    resultados.append(r)

    # XGBoost (mesmos hiperparâmetros do train_xgboost.py)
    xg = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=1.0, random_state=SEED,
    )
    xg.fit(X_train, y_train)
    pred_xg = xg.predict(X_test)
    r2 = avaliar(f"XGB — {nome_config}", y_test, pred_xg)
    r2["n_features"] = len(features)
    resultados.append(r2)

res_df = pd.DataFrame(resultados)[["modelo", "n_features", "MAE", "RMSE", "R2", "pearson_r", "acc10"]]
res_df = res_df.round({"MAE": 2, "RMSE": 2, "R2": 4, "pearson_r": 4, "acc10": 1})

print(f"\n  N amostras (side-view, com sticker visível): {len(df)}")
print(f"  Split: 80/20, seed={SEED}\n")
print(res_df.to_string(index=False))

out_path = os.path.join(SAVE_DIR, "resultados_comparacao_sticker.csv")
res_df.to_csv(out_path, index=False)
print(f"\n  Salvo em: {out_path}\n")
