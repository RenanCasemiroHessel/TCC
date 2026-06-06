# Predição de Peso de Bovinos por Visão Computacional

**Trabalho de Conclusão de Curso — Centro Universitário FEI, 1º semestre/2026**

Autores: Gustavo Mendes Franco Lapin Atui · Rafael Takahagi Mendes · Renan Casemiro Hessel

---

## Resumo

Pipeline completo para estimar o peso corporal de bovinos a partir de imagens, sem contato físico com o animal. O sistema combina segmentação automática com YOLO11n-seg, extração de atributos morfométricos e oito configurações experimentais de modelos de regressão — quatro *single-view* (uma imagem por animal) e quatro *multi-view* (vista lateral + traseira do mesmo indivíduo). O melhor resultado obtido foi com a CNN Multi-View: **MAE = 20,75 kg · R² = 0,407 · Pearson r = 0,667**.

---

## Dataset

- **Fonte:** [BMGF-LivestockWeight-CV](https://www.kaggle.com/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k) — baixado automaticamente via `kagglehub`
- **Lotes utilizados:** B2 e B4 (B3 descartado por conter dados clonados)
- **Filtro aplicado:** 50 kg ≤ peso < 250 kg
- **Total:** 4.818 imagens válidas → 2.397 pares lateral+traseira
- **Anotações:** máscaras semânticas fornecidas pelo dataset (cor do corpo do bovino por lote)

---

## Estrutura do Pipeline

```
Imagem Bruta
    │
    ▼
YOLO11n-seg  →  Imagem Segmentada (fundo preto)
    │
    ▼
Extração de Features Morfométricas
(área, altura, largura, proporção, fill_ratio, vol_approx, sexo, vista)
    │
    ├─ Single-View (4.818 amostras)
    │       Regressão Linear · XGBoost · MLP · CNN ResNet18
    │
    └─ Multi-View (2.397 pares)
            Regressão Linear MV · XGBoost MV · MLP MV · CNN MV (backbone compartilhado)
    │
    ▼
Peso Estimado (kg)
```

---

## Resultados

| Modelo | MAE (kg) | RMSE (kg) | R² | Pearson r | Ac. ±10% |
|---|---|---|---|---|---|
| Regressão Linear | 25,90 | 32,29 | 0,157 | 0,398 | 38,1% |
| XGBoost | 26,00 | 32,51 | 0,146 | 0,383 | 38,4% |
| MLP (PCA) | 29,33 | 37,00 | −0,143 | 0,200 | 34,3% |
| CNN ResNet18 | 26,42 | 34,17 | 0,416 | — | — |
| LR Multi-View | 23,82 | 29,95 | 0,245 | 0,496 | 41,9% |
| XGBoost Multi-View | 23,84 | 29,93 | 0,246 | 0,497 | 42,3% |
| MLP Multi-View | 29,04 | 37,69 | −0,233 | 0,310 | 34,8% |
| **CNN Multi-View** | **20,75** | **26,09** | **0,407** | **0,667** | **48,1%** |

Acurácia ±10%: proporção de predições com \|ŷ − y\| ≤ 0,10 × y.

---

## Configuração do Ambiente

Requer Python 3.10+ e uma conta no Kaggle configurada para `kagglehub`.

```bash
# Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt
```

O dataset (~4 GB) é baixado automaticamente na primeira execução de qualquer script de treinamento.

---

## Descrição dos Scripts

### Pré-processamento

| Script | Descrição |
|---|---|
| `preparar_dataset_kaggle.py` | Baixa o dataset via kagglehub e extrai as features morfométricas de todas as imagens, gerando `dataset_features_completo.csv` |
| `filtrar_dataset.py` | Aplica os filtros de peso e lote, gerando `dataset_features_leves.csv` (arquivo base para todos os modelos) |
| `visualizar_segmentacao.py` | Gera mosaico visual Original · Máscara · Segmentada para amostras de cada lote |

### Modelos Single-View

| Script | Descrição |
|---|---|
| `train_linear_regression.py` | Regressão Ridge + PolynomialFeatures (grau 2) — baseline interpretável |
| `train_lr_multiview.py` | Versão Multi-View da Regressão Linear (features de ambas as vistas concatenadas) |
| `train_xgboost.py` | XGBoost com features alométricas e otimização por GridSearch |
| `tune_xgboost.py` | Busca de hiperparâmetros estendida para o XGBoost |
| `train_mlp.py` | MLP sobre pixels comprimidos por PCA(256); inclui cache `.npy` para acelerar reexecuções |
| `treinar_cnn.py` | CNN ResNet18 single-view com fine-tuning progressivo |

### Modelos Multi-View

| Script | Descrição |
|---|---|
| `train_lr_multiview.py` | Regressão Ridge com features de lateral + traseira concatenadas |
| `train_xgboost_multiview.py` | XGBoost com vetor de 17 features (9 por vista + sexo) |
| `train_mlp_multiview.py` | MLP sobre pixels concatenados de ambas as vistas, reduzidos por PCA(256) |
| `treinar_cnn_multiview.py` | **Melhor modelo:** CNN ResNet18 com backbone compartilhado processando ambas as vistas em paralelo; treinada com ConcatDataset 4× e fine-tuning progressivo |

### Comparação e Inferência

| Script | Descrição |
|---|---|
| `comparar_modelos.py` | Lê `resultados_modelos.json` e gera boxplots, histogramas e curvas de overfitting comparativos |
| `gerar_graficos_tcc.py` | Regenera todos os gráficos do TCC com fontes maiores a partir dos arquivos `.npy` e `.csv` salvos |
| `predizer_imagem.py` | Inferência em imagem única: segmenta automaticamente com YOLO11n-seg e estima o peso |
| `utils_tcc.py` | Utilitários compartilhados: salvamento seguro de arquivos (`safe_path`, `salvar_json`) |

---

## Ordem de Execução (pipeline completo do zero)

```bash
# 1. Gerar CSV com todas as features
python preparar_dataset_kaggle.py

# 2. Filtrar dataset (50–250 kg, lotes B2 e B4)
python filtrar_dataset.py

# 3. Treinar modelos single-view
python train_linear_regression.py
python train_xgboost.py
python train_mlp.py
python treinar_cnn.py

# 4. Treinar modelos multi-view
python train_lr_multiview.py
python train_xgboost_multiview.py
python train_mlp_multiview.py
python treinar_cnn_multiview.py   # requer GPU; ~30 épocas

# 5. Gerar comparativos e gráficos
python comparar_modelos.py
python gerar_graficos_tcc.py

# 6. Inferência em imagem nova (opcional)
python predizer_imagem.py --imagem caminho/para/foto.jpg
```

---

## Arquivos Gerados

Após execução completa, os seguintes arquivos são criados na raiz do projeto:

- `dataset_features_leves.csv` — dataset tabular filtrado (4.818 amostras)
- `predictions_*.npy` — predições de cada modelo (formato `[y_real, y_pred]`)
- `resultados_modelos.json` — métricas consolidadas de todos os modelos
- `graficos/` — todos os gráficos gerados em PNG (150 DPI)
- `yolo11n-seg.pt` — checkpoint do modelo de segmentação (baixado automaticamente)

---

## Dependências Principais

- Python 3.10+
- PyTorch + torchvision
- ultralytics (YOLO11)
- scikit-learn
- xgboost
- pandas / numpy / matplotlib / seaborn
- kagglehub
- Pillow

---

## Citação do Dataset

Islam, M. et al. "BMGF-LivestockWeight-CV: A benchmark dataset for livestock body weight estimation using computer vision." *Data in Brief*, v. 54, p. 110480, 2024. DOI: 10.1016/j.dib.2024.110480
