# Predição de Peso de Bovinos por Visão Computacional

**Trabalho de Conclusão de Curso — Centro Universitário FEI, 2º semestre/2026**

Autores: Gustavo Mendes Franco Lapin Atui · Rafael Takahagi Mendes · Renan Casemiro Hessel

---

## Resumo

Sistema que estima o peso corporal de bovinos a partir de uma foto, sem contato físico com o animal. O YOLO11n-seg separa o animal do fundo, e uma rede convolucional estima o peso a partir do recorte, de medidas relativas do animal na foto e do sexo.

No teste com animais fora do treino, o sistema completo teve **MAE de cerca de 25,5 kg**, com 42,0% das estimativas dentro de ±10% do peso real. Com máscara anotada e marcador de escala, o MAE ficou entre 21,5 e 22,4 kg.

---

## Dataset

- **Fonte:** [BMGF-LivestockWeight-CV](https://www.kaggle.com/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k), baixado via `kagglehub`
- **Lotes:** B2, B3 e B4
- **Total:** 7.465 imagens de 5.023 animais, de 36 a 420 kg (média de 163,5 kg)
- **Vistas:** 5.023 laterais e 2.442 traseiras
- **Anotações:** máscaras do animal e do marcador de escala, fornecidas pelo dataset

As 6 imagens acima de 430 kg ficaram fora do treino e são usadas só para testar extrapolação.

---

## Pipeline

```
Foto
  │
  ▼
Máscara do animal (anotada no treino; YOLO11n-seg no uso real)
  │
  ▼
Recorte quadrado com fundo preto + medidas relativas e sexo
  │
  ▼
CNN (ResNet18 ou EfficientNetV2-S) em cada dobra
  │
  ▼
Média das estimativas, com a foto original e espelhada
  │
  ▼
Peso estimado (kg)
```

---

## Resultados

### Teste final

Os resultados abaixo usam 15% dos animais, fora do treino e da validação.

| Modelo | Cenário | MAE (kg) | R² | Ac. ±10% |
|---|---|---:|---:|---:|
| ResNet18, 3 modelos, 224 px | Máscara anotada e marcador | 22,41 | 0,510 | 46,7% |
| ResNet18, 3 modelos, 224 px | **Uso real** (YOLO, sem marcador) | **25,53** | 0,353 | 42,0% |
| EfficientNetV2-S, 5 modelos, 384 px | Máscara anotada e marcador | 21,54 | 0,542 | 48,3% |
| EfficientNetV2-S, 5 modelos, 384 px | **Uso real** (YOLO, sem marcador) | **25,50** | 0,356 | 42,0% |
| ConvNeXt-Tiny, 5 modelos, 384 px | Máscara anotada e marcador | 21,72 | 0,540 | 49,2% |

Acurácia ±10%: proporção de estimativas com \|ŷ − y\| ≤ 0,10 × y. No uso real, o YOLO encontrou o animal em 1.113 das 1.117 imagens, com IoU médio de 0,922 em relação à máscara anotada.

A estimativa final é a média dos modelos treinados em cada dobra: 3 na ResNet18 e 5 na EfficientNetV2-S e na ConvNeXt-Tiny.

### CNN e modelos tabulares

Validação cruzada de 5 dobras por animal, com as mesmas divisões nos três modelos, em 5.016 imagens com marcador visível.

| Modelo | MAE (kg) | R² | Ac. ±10% |
|---|---:|---:|---:|
| **ResNet18** | **22,82** | 0,501 | 44,4% |
| Regressão Ridge | 25,10 | 0,414 | 40,2% |
| XGBoost | 25,11 | 0,413 | 39,9% |

### Limitações

- Os modelos valem para animais como os da base, principalmente abaixo de 250 kg. Eles não foram validados em Nelore.
- As estimativas são puxadas para o centro da base: animais pesados são subestimados, e a maior estimativa foi de 326 kg.
- O sistema serve para acompanhar tendências e não substitui a balança.

Os resultados iniciais do projeto, como o MAE de 20,75 kg da CNN Multi-View, foram substituídos. Eles usavam uma base menor e uma avaliação menos rigorosa.

---

## Configuração do Ambiente

Testado com Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision ultralytics scikit-learn xgboost pandas numpy matplotlib seaborn opencv-python pillow flask kagglehub
```

---

## Demo Web

Interface para testar o sistema com uma foto nova. O backend usa Flask, e o frontend usa Vite, React e TypeScript.

Os pesos treinados não ficam neste repositório. Para rodar a demo, coloque os arquivos `sv2_fold*.pth` em `modelos/modelos_final_singleview/`.

```bash
cd app/frontend
npm install
npm run build
cd ../..

python app/app.py
```

Acesse `http://localhost:5000`.

---

## Organização

| Pasta | Conteúdo |
|---|---|
| `app/` | Servidor Flask e interface web |
| `pipeline/` | Preparação dos dados, treino, busca, inferência e métricas |
| `dados/` | CSVs da base e imagens do teste de extrapolação |
| `modelos/` | Resultados, predições e logs dos treinos |
| `arquivo/` | Scripts e resultados das fases anteriores do projeto |

---

## Como Treinar

```bash
# 1. Listar as imagens da base
python pipeline/preparar_dataset_estendido.py

# 2. Gerar os recortes
python pipeline/preparar_cache_singleview.py

# 3. Treinar uma configuração
python pipeline/treinar_cnn_singleview_v2.py --backbone resnet18 --folds 5 --avaliar-holdout --saida modelos/meu_treino

# 4. Treinar os modelos tabulares
python pipeline/treinar_tabulares_v2.py --sticker
```

Se o treino for interrompido, o mesmo comando retoma do último checkpoint.

---

## Citação do Dataset

Acme AI. *Cattle Weight Detection Model + Dataset (12k) [BMGF-LivestockWeight-CV]*. Kaggle, 2024. Licença CC BY 4.0. Disponível em: https://www.kaggle.com/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k
