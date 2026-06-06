# TCC FEI — Estimativa de Peso Bovino por Visão Computacional
## Relatório Completo de Resultados

**Data:** Maio 2026  
**Autores:** Gustavo & Renan  
**Dataset:** Kaggle `sadhliroomyprime/cattle-weight-detection-model-dataset-12k` (Lotes B2 + B4, sem B3)

---

## Dataset

| Configuração           | Valor                                    |
|------------------------|------------------------------------------|
| Lotes utilizados       | B2 e B4 (B3 removido — dados clonados)  |
| Filtro de peso         | Apenas leves: 50 kg ≤ peso < 250 kg     |
| Imagens únicas totais  | **4.818 imagens** (2.407 side + 2.411 rear) |
| Pares side + rear      | **2.397 pares** (mesmo animal, ambas as vistas) |
| GPU usada no treino    | NVIDIA RTX 4050 Laptop (6 GB VRAM)      |

---

## Tabela Comparativa — Todas as Runs (desde o início do projeto)

| # | Modelo / Run | Tipo | Amostras processadas no treino³ | Val. | MAE (kg) | RMSE (kg) | R² | Pearson r | Acurácia ±10% | F1 macro² | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Regressão Linear | Single-view | 3.854 imgs (1 passagem) | 964 | 25,90 | 32,29 | 0,1573 | 0,398 | 38,1% | 0,291 | ✓ salvo |
| 2 | Regressão Linear v2 | Single-view | 3.854 imgs (1 passagem) | 964 | 25,90 | 32,29 | 0,1573 | 0,398 | 38,1% | 0,291 | ✓ duplicata |
| 3 | XGBoost | Single-view | 3.854 imgs (1 passagem) | 964 | 26,00 | 32,51 | 0,1458 | 0,383 | 38,4% | 0,270 | ✓ salvo |
| 4 | XGBoost (Tuned) | Single-view | 3.854 imgs (1 passagem) | — | 28,57 | 37,93 | 0,1667 | — | — | — | ✓ run antiga sem .npy¹ |
| 5 | XGBoost v2 | Single-view | 3.854 imgs (1 passagem) | 964 | 26,00 | 32,51 | 0,1458 | 0,383 | 38,4% | 0,270 | ✓ duplicata |
| 6 | MLP (pixels) | Single-view | 3.854 × 28 iter = 107.912 passes | 948 | 29,33 | 37,00 | −0,1425 | 0,200 | 34,3% | 0,367 | ✓ salvo |
| 7 | MLP (pixels) v2 | Single-view | 3.854 × 28 iter = 107.912 passes | 948 | 29,33 | 37,00 | −0,1425 | 0,200 | 34,3% | 0,367 | ✓ duplicata |
| 8 | CNN ResNet18 — run #1 | Single-view | 3.854 × 40 épocas = 154.160 imgs | — | — | — | — | — | — | — | ✗ CRASH (bug `y_true_cat`, nada salvo) |
| 9 | CNN ResNet18 — run #2 | Single-view | 3.854 × 15 épocas = 57.810 imgs | 964 | 26,42 | 34,17 | 0,4156 | — | — | — | ✓ salvo, .npy não gerado¹ |
| 10 | CNN ResNet18 v2 | Single-view | 3.854 × 15 épocas = 57.810 imgs | 964 | 21,27 | 27,23 | 0,3527 | 0,600 | — | — | ✓ salvo |
| 11 | CNN ResNet18 v3 | Single-view | 3.854 × 15 épocas = 57.810 imgs | 964 | 21,28 | 27,28 | 0,3501 | 0,597 | — | — | ✓ salvo |
| 12 | CNN ResNet18 v4 | Single-view | 3.854 × 30 épocas = 115.620 imgs| 964 | 20,95 | 26,82 | 0,3718 | 0,615 | — | — | ✓ salvo |
| 13 | LR Multi-View | Multi-view | 1.917 pares (1 passagem) | 480 | 23,82 | 29,95 | 0,2450 | 0,496 | 41,9% | 0,414 | ✓ salvo |
| 14 | XGBoost Multi-View | Multi-view | 1.917 pares (1 passagem) | 480 | 23,84 | 29,93 | 0,2459 | 0,497 | 42,3% | 0,355 | ✓ salvo |
| 15 | MLP Multi-View | Multi-view | 1.917 × 52 iter = 99.684 passes | 471 | 29,04 | 37,69 | −0,2334 | 0,310 | 34,8% | 0,437 | ✓ salvo |
| 16 | CNN Multi-View v1 | Multi-view | 1.917 × 15 épocas = 28.755 pares | 480 | 24,48 | 29,92 | 0,2206 | 0,603 | 39,0% | 0,260 | ✓ salvo (sem aug) |
| 17 | CNN Multi-View — run #2 | Multi-view | — | — | — | — | — | — | — | — | ✗ CRASH (bug `AUG_MULTIPLIER`) |
| 18 | CNN Multi-View v2 | Multi-view | 1.917 × 4 × 6 épocas = 46.008 pares | 480 | 21,06 | 26,43 | 0,3918 | 0,663 | 46,7% | 0,499 | ✓ salvo (4× aug) |
| **19** | **CNN Multi-View v3 ★** | **Multi-view** | **1.917 × 4 × 30 épocas = 230.040 pares** | **480** | **20,75** | **26,09** | **0,4072** | **0,667** | **48,1%** | **0,503** | **✓ salvo (4× aug)** |

> ★ Melhor resultado geral.  
> ¹ Arquivo `.npy` de predições não gerado nessa run — acurácia e F1 não calculáveis. A CNN single-view run #2 terá esses valores quando for re-executada.  
> ² F1 macro calculado classificando predições em 3 faixas de peso: leve (50–130 kg), médio (130–190 kg), pesado (190–250 kg). Computa F1 por faixa e tira a média.  
> ³ "Passes" = cada amostra contada uma vez por iteração/época. LR e XGBoost processam todos os dados de uma vez (1 passagem).  
> Acurácia ±10% = % de predições com erro absoluto ≤ 10% do peso real do animal.  
> Duplicatas geradas pelo sistema de salvamento seguro (`safe_path`) ao re-rodar scripts já executados.

---

## Detalhamento por Modelo

### 1. Regressão Linear (Single-View)
- **Algoritmo:** Ridge Regression + PolynomialFeatures(grau=2)
- **Hiperparâmetros:** alpha=10
- **Features:** area, altura_bbox, largura_bbox, proporcao, area_norm + derivadas (bbox_area, fill_ratio, volume_approx, vol_norm) + sexo_M
- **Imagens analisadas:** 4.818 (3.854 treino / 964 validação — split 80/20)
- **Pré-processamento:** segmentação por máscara colorida, features extraídas manualmente

| Métrica     | Valor        |
|-------------|--------------|
| MAE         | 25,90 kg     |
| RMSE        | 32,29 kg     |
| R²          | 0,1573       |
| Pearson r   | 0,398        |

---

### 2. XGBoost (Single-View)
- **Algoritmo:** XGBoost Regressor
- **Hiperparâmetros:** n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=1.0
- **Features:** mesmas da Regressão Linear
- **Imagens analisadas:** 4.818 (3.854 treino / 964 validação)

| Métrica     | Valor        |
|-------------|--------------|
| MAE         | 26,00 kg     |
| RMSE        | 32,51 kg     |
| R²          | 0,1458       |
| Pearson r   | 0,383        |

---

### 3. MLP (pixels) — Single-View
- **Algoritmo:** MLPRegressor(256, 128, 64) + PCA(128 componentes)
- **Input:** pixels 64×64 px da imagem segmentada, achatados (12.288 features → PCA → 128)
- **Imagens analisadas:** 4.818 (3.854 treino / 964 validação)
- **Épocas até convergência:** 28 iterações (early stopping)
- **Predições clampadas:** [50, 250] kg

| Métrica     | Valor        |
|-------------|--------------|
| MAE         | 29,33 kg     |
| RMSE        | 37,00 kg     |
| R²          | −0,1425      |
| Pearson r   | 0,200        |

---

### 4. CNN ResNet18 (Single-View)
- **Arquitetura:** ResNet18 pré-treinado (ImageNet) + cabeça de regressão
- **Cabeça:** Linear(512→256) → ReLU → Dropout(0,5) → Linear(256→1) → Sigmoid
- **Loss:** HuberLoss (delta normalizado = 50/(250−50) ≈ 0,25)
- **Otimizador:** Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR(T_max=15, eta_min=1e-6)
- **Épocas:** 15 (backbone congelado nas épocas 1-2, fine-tuning completo a partir da época 3)
- **Augmentação:** HorizontalFlip, Rotation(10°), RandomAffine(scale=0.9–1.1), ColorJitter
- **Imagens analisadas:** 4.818 (3.854 treino / 964 validação — batch_size=32)
- **Salva melhor checkpoint** (menor val loss durante treino)

| Métrica     | Valor (v4)   |
|-------------|--------------|
| MAE         | 20,95 kg     |
| RMSE        | 26,82 kg     |
| R²          | 0,3718       |
| Pearson r   | 0,615        |

> Nota: A última execução (v4) alcançou uma performance muito superior à v2 original, reduzindo o erro de 26,42 kg para 20,95 kg, utilizando a nova configuração completa de augmentação.

---

### 5. Regressão Linear Multi-View
- **Algoritmo:** Ridge Regression + PolynomialFeatures(grau=2)
- **Input:** features side + features rear concatenadas por animal
- **Features:** 2× (area, altura_bbox, largura_bbox, proporcao, area_norm) + derivadas para cada vista + sexo_M = 17 features
- **Pares analisados:** 2.397 (1.917 treino / 480 validação)
- **Imagens únicas processadas:** 4.794

| Métrica     | Valor        |
|-------------|--------------|
| MAE         | 23,82 kg     |
| RMSE        | 29,95 kg     |
| R²          | 0,2450       |
| Pearson r   | 0,496        |

---

### 6. XGBoost Multi-View
- **Algoritmo:** XGBoost Regressor
- **Hiperparâmetros:** n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8
- **Input:** mesmas features do LR Multi-View
- **Pares analisados:** 2.397 (1.917 treino / 480 validação)
- **Imagens únicas processadas:** 4.794

| Métrica     | Valor        |
|-------------|--------------|
| MAE         | 23,84 kg     |
| RMSE        | 29,93 kg     |
| R²          | 0,2459       |
| Pearson r   | 0,497        |

---

### 7. MLP Multi-View
- **Algoritmo:** MLPRegressor(256, 128, 64) + PCA(256 componentes)
- **Input:** pixels side (64×64×3) + pixels rear (64×64×3) concatenados = 24.576 features → PCA(256)
- **Pares analisados:** 2.397 (split 80/20 por amostra)
- **Imagens únicas processadas:** 4.794
- **Épocas até convergência:** 52 iterações (early stopping)

| Métrica     | Valor        |
|-------------|--------------|
| MAE         | 29,04 kg     |
| RMSE        | 37,69 kg     |
| R²          | −0,2334      |
| Pearson r   | 0,310        |

> MLP multi-view ficou pior que single-view. Com menos amostras (2.397 pares vs 4.818 imagens) e alta dimensionalidade dos pixels, o modelo sofreu underfitting mesmo com PCA.

---

### 8. CNN Multi-View — ResNet18 Siamesa ★ MELHOR RESULTADO

- **Arquitetura:** ResNet18 Siamesa com backbone compartilhado
  - Backbone: ResNet18 (sem camada final) → 512 features por vista
  - Fusão: concat(feat_side, feat_rear) = 1.024 features
  - Cabeça: Linear(1024→256) → ReLU → Dropout(0.5) → Linear(256→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid
- **Loss:** HuberLoss (delta=0,25 normalizado)
- **Otimizador:** Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR(T_max=30, eta_min=1e-6)
- **Épocas:** 30 (backbone congelado nas épocas 1-2, fine-tuning completo a partir da época 3)
- **Augmentação:** HorizontalFlip, Rotation(10°), RandomAffine(scale=0.9–1.1), ColorJitter
- **ConcatDataset 4×:** dataset de treino repetido 4 vezes por época com augmentações diferentes
- **Pares base:** 2.397 (split 80/20 por animal — sem data leakage)
  - Treino: 1.917 pares × 4 = **7.668 pares por época**
  - Validação: 480 pares (sem augmentação)
- **Imagens únicas processadas:** 4.794
- **Salva melhor checkpoint** durante treino (menor val loss)

| Métrica     | Valor        |
|-------------|--------------|
| MAE         | **20,75 kg** |
| RMSE        | **26,09 kg** |
| R²          | **0,4072**   |
| Pearson r   | **0,667**    |

#### Histórico de Treino (30 épocas):

| Época | Train Loss | Val Loss   |
|-------|------------|------------|
| 1     | 0,016291   | 0,014117   |
| 2     | 0,014509   | 0,012382   |
| 3     | 0,011756   | 0,012504   |
| 4     | 0,009721   | 0,011165   |
| 5     | 0,008554   | 0,010419   |
| 6     | 0,007571   | **0,009523** ← melhor até aqui |
| 7     | 0,007006   | 0,009932   |
| 8     | 0,006031   | 0,011805   |
| 9     | 0,005843   | 0,010768   |
| 10    | 0,005107   | 0,012121   |
| 11    | 0,005001   | 0,013079   |
| 12    | 0,004568   | 0,010368   |
| 13    | 0,004207   | 0,011771   |
| 14    | 0,004146   | 0,011611   |
| 15    | 0,003828   | 0,011869   |
| 16    | 0,003662   | 0,011273   |
| 17    | 0,003378   | 0,011311   |
| 18    | 0,003332   | 0,013699   |
| 19    | 0,003218   | 0,013191   |
| 20    | 0,003003   | 0,011473   |
| 21    | 0,002891   | 0,011215   |
| 22    | 0,002856   | 0,010938   |
| 23    | 0,002785   | 0,011057   |
| 24    | 0,002694   | 0,010726   |
| 25    | 0,002678   | 0,010716   |
| 26    | 0,002538   | 0,010374   |
| 27    | 0,002452   | 0,010125   |
| 28    | 0,002397   | 0,011357   |
| 29    | 0,002357   | 0,011472   |
| 30    | 0,002332   | 0,013077   |

> Val loss melhor na época 6 (0,009523). O treino continua caindo mas val loss oscila — sinal de algum overfitting após época 6-7. Avaliação final usa o checkpoint da melhor época.

---

## Histórico Completo de Todas as Runs de CNN

### CNN Single-View — Todas as Runs

| Run | Épocas | Config | MAE (kg) | RMSE (kg) | R²     | Pearson r | Tempo aprox. | Observação |
|-----|--------|--------|----------|-----------|--------|-----------|--------------|------------|
| #1 (crash) | 40 | antiga | — | — | — | — | ~2,3 h | Crash no plot após treino. **Nenhum resultado salvo.** Bug: `y_true_cat` não definido. Treino chegou até época 40 completa mas crash impediu salvamento. |
| #2 ✓ | 15 | nova | 26,42 | 34,17 | 0,4156 | — | ~15 min | Configuração corrigida: HuberLoss delta normalizado, Dropout 0.5, augmentação melhorada. **Resultado salvo.** |
| #3 ✓ (v2) | 15 | nova | 21,27 | 27,23 | 0,3527 | 0,600 | ~15 min | Re-execução salva com `predictions_cnn_v2.npy`. |
| #4 ✓ (v3) | 15 | nova | 21,28 | 27,28 | 0,3501 | 0,597 | ~15 min | Re-execução idêntica (v3). |
| #5 ✓ (v4) | 30 | nova completa | 20,95 | 26,82 | 0,3718 | 0,615 | ~30 min | **Melhor resultado single-view**. Rodado por 30 épocas, alcançou MAE muito próximo da rede multi-view. |

---

### CNN Multi-View — Todas as Runs

| Run | Épocas | Aug | MAE (kg) | RMSE (kg) | R²     | Pearson r | Tempo aprox. | Observação |
|-----|--------|-----|----------|-----------|--------|-----------|--------------|------------|
| #1 — v1 | 15 | 1× | 24,48 | 29,92 | 0,2206 | 0,603 | ~20 min | Primeira implementação Siamesa. Sem ConcatDataset. |
| #2 — crash | 30 | 4× | — | — | — | — | ~poucos min | Crash imediato: `AUG_MULTIPLIER` não definido no bloco de log (estava declarado depois no código). **Nenhum resultado salvo.** |
| #3 — v2 | 6 | 4× | 21,06 | 26,43 | 0,3918 | 0,663 | ~30 min | Primeiro run com ConcatDataset 4×. Ran 6 épocas para testar. |
| **#4 — v3 ★** | **30** | **4×** | **20,75** | **26,09** | **0,4072** | **0,667** | **~2–3 h** | **Melhor resultado geral. Rodou de madrugada.** |

---

## Resumo Final

- **Melhor MAE:** CNN Multi-View v3 → **20,75 kg** (CNN Single-View v4 chegou a 20,95 kg)
- **Melhor R²:** CNN Multi-View v3 → **0,4072**
- **Melhor modelo clássico:** LR Multi-View e XGBoost Multi-View (empatados — MAE ~23,8 kg)
- **Pior modelo:** MLP Multi-View (R² negativo — pior que média)
- **Ganho da CNN multi-view vs. melhor single-view:** −0,20 kg MAE (−0,9%)

### Por que multi-view CNN ganhou?
1. Usa informação de dois ângulos (lateral + traseiro) simultaneamente
2. Backbone compartilhado (Siamese): extrai features consistentes de ambas as vistas
3. ConcatDataset 4×: equivale a 4× mais dados de treino sem custo de disco
4. Progressive fine-tuning: backbone congelado nas primeiras épocas evita destruir features ImageNet

---

*Arquivo gerado automaticamente a partir de `resultados_modelos.json`*
