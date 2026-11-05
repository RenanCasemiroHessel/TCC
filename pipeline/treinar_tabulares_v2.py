"""
Regressão Linear e XGBoost v2 — mesmos dados e mesmos folds da CNN Single-View v2.

Os scripts antigos (train_linear_regression.py, train_xgboost.py) não dão pra
comparar com a CNN v2: usam só B2+B4 < 250kg, um único split 80/20, e dividem
por FOTO — a lateral de um boi pode cair no treino e a traseira no teste.

Aqui:
  - lê o mesmo cache da CNN (preparar_cache_singleview.py), 7.471 imagens;
  - usa folds_por_animal() do pipeline_singleview: mesmo CSV + mesma seed =
    exatamente os mesmos 5 folds da CNN, por animal;
  - mesmo alvo da CNN (log do peso) e mesmas métricas (resumo_oof).

Hiperparâmetros: os mesmos dos scripts antigos (Ridge alpha=10 + polinômio
grau 2; XGBoost 100 árvores, profundidade 3, lr 0.05) — não foram re-tunados
para este dataset maior.

Uso:
  python treinar_tabulares_v2.py
  python treinar_tabulares_v2.py --sticker
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caminhos import DADOS, MODELOS
from utils_tcc import salvar_json
from pipeline_singleview import (
    ESCALAR_STICKER, TAMANHO_CACHE, folds_por_animal, imprimir_resumo, metricas, resumo_oof,
)

SEED = 42

FEATURES_BASE = ["log_area_norm", "log_altura_rel", "log_largura_rel", "fill_ratio", "proporcao",
                 "sexo_M", "vista_side"]


def modelos():
    return {
        "Regressão Linear": lambda: make_pipeline(
            StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(alpha=10),
        ),
        "XGBoost": lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=1.0,
            random_state=SEED,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-csv", default=os.path.join(DADOS, f"cache_singleview_{TAMANHO_CACHE}",
                                                        "dataset_singleview_cache.csv"))
    ap.add_argument("--saida", default=None,
                    help="padrão: modelos_tabulares_v2/ (ou modelos_tabulares_v2_sticker/ com --sticker)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sticker", action="store_true",
                    help="adiciona log_sticker_ratio (restringe a fotos laterais com sticker, igual à CNN)")
    args = ap.parse_args()
    if args.saida is None:
        args.saida = os.path.join(MODELOS, "modelos_tabulares_v2" + ("_sticker" if args.sticker else ""))
    os.makedirs(args.saida, exist_ok=True)

    if not os.path.exists(args.cache_csv):
        sys.exit(f"  ERRO: {args.cache_csv} não existe. Rode antes: python preparar_cache_singleview.py")

    df = pd.read_csv(args.cache_csv)
    features = list(FEATURES_BASE)
    if args.sticker:
        df = df[df[ESCALAR_STICKER].notna()].reset_index(drop=True)
        features = [f for f in features if f != "vista_side"] + [ESCALAR_STICKER]  # só lateral
    df["vista_side"] = (df["vista"] == "side").astype(float)

    print(f"  Tabulares v2 — {args.folds} folds por animal | {len(df)} imagens | {df['key'].nunique()} animais")
    print(f"  Features: {features}")

    todos = {}
    for nome, construir in modelos().items():
        reais, preds, por_fold = [], [], []
        for treino_df, teste_df in folds_por_animal(df, args.folds, SEED):
            m = construir()
            m.fit(treino_df[features].to_numpy(), np.log(treino_df["peso_kg"].to_numpy()))
            pred = np.exp(m.predict(teste_df[features].to_numpy()))
            real = teste_df["peso_kg"].to_numpy()
            reais.append(real)
            preds.append(pred)
            por_fold.append(metricas(real, pred))
        real, pred = np.concatenate(reais), np.concatenate(preds)
        resumo = resumo_oof(real, pred)

        print(f"\n{'=' * 60}\n  {nome} ({args.folds} folds, out-of-fold)\n{'=' * 60}")
        for i, mf in enumerate(por_fold, start=1):
            print(f"    Fold {i}: MAE {mf['mae']:6.2f}kg | R² {mf['r2']:.3f} | MAPE {mf['mape']:.1f}%")
        print()
        imprimir_resumo(resumo)

        slug = "lr" if nome.startswith("Regressão") else "xgboost"
        np.save(os.path.join(args.saida, f"predicoes_oof_{slug}.npy"), np.array([real, pred]))
        todos[nome] = {"features": features, "n_imagens": len(df), "oof": resumo, "por_fold": por_fold}

    for nome, dados in todos.items():
        salvar_json(os.path.join(args.saida, "resultados_tabulares_v2.json"),
                    nome + (" + sticker" if args.sticker else ""), dados)
    print(f"  Resultados em: {args.saida}")


if __name__ == "__main__":
    main()
