# Estimativa de Peso Bovino por Visão Computacional 🐄

Este repositório contém o código-fonte desenvolvido para o Trabalho de Conclusão de Curso (TCC) da FEI. O objetivo deste projeto é estimar o peso corporal de bovinos utilizando imagens aéreas, comparando o desempenho de algoritmos clássicos de regressão e modelos avançados de Redes Neurais Convolucionais (CNN).

---

## 📌 Arquitetura do Projeto

O pipeline preditivo é composto por duas abordagens principais:
1. **Modelos Tabulares (Estatísticos e Baseados em Árvore):** Utilizam um conjunto de features geométricas extraídas previamente (ex.: área e proporção da Bounding Box). Implementou-se Regressão Linear Clássica e XGBoost com otimização via GridSearch.
2. **Modelos de Visão Computacional (Deep Learning):** Processam as imagens diretamente, utilizando Redes Densas (MLP com PCA) e Redes Convolucionais com Transfer Learning (ResNet18).

Todas as falhas e depreciações do código original foram rigorosamente revisadas, garantindo um ambiente robusto, reproduzível e com total aproveitamento do *dataset*.

---

## 🚀 Guia de Configuração e Execução

### 1. Configuração do Ambiente Virtual
Recomenda-se a utilização de uma versão atualizada do Python (3.9+). Execute os comandos abaixo para inicializar o ambiente de dependências:

```bash
# Inicialização e ativação do ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalação das bibliotecas necessárias (KaggleHub, PyTorch, XGBoost, Scikit-Learn, etc.)
pip install -r requirements.txt
```

### 2. Execução Automatizada (Pipeline Completo)
O projeto conta com um script de orquestração (`rodar_tudo.sh`) que enfileira o treinamento e validação de todos os modelos, da menor para a maior complexidade computacional, finalizando com a geração gráfica das métricas.

```bash
bash rodar_tudo.sh
```

A ordem de execução padrão é:
1. `train_linear_regression.py`
2. `train_xgboost.py`
3. `tune_xgboost.py`
4. `train_mlp.py`
5. `treinar_cnn.py`
6. `comparar_modelos.py`

*(Observação: A primeira execução irá baixar o dataset via KaggleHub, totalizando ~4GB de arquivos de imagem. Uma conexão de banda larga é recomendada).*

---

## 🏗 Descritivo dos Módulos (Scripts)

| Módulo/Arquivo | Descrição Técnica e Funções Implementadas |
| :--- | :--- |
| `dataset_features_completo.csv` | *Dataset* tabular consolidado com variáveis geométricas pré-computadas de cada imagem. |
| `train_linear_regression.py` | Baseline linear preditivo. Usado para ilustrar as limitações da regressão plana em inferências volumétricas. |
| `train_xgboost.py` / `tune_xgboost.py` | Modelo impulsionado por árvores. Introduz Engenharia de Atributos, especificamente modelagem Alométrica ($Área^{1.5}$) e *Fill Ratios*. Otimizado via GridSearch. |
| `train_mlp.py` | Implementação do Perceptron Multicamadas (Fully Connected). Inclui *caching* automatizado e redução de dimensionalidade por Análise de Componentes Principais (PCA) para prevenir *overfitting*. |
| `treinar_cnn.py` | Modelo de estado da arte utilizando a arquitetura *ResNet18*. Aplica estratégias de *Fine-Tuning* progressivo (descongelamento do *backbone*) e métricas dinâmicas. O *dataset loader* agora pesquisa o diretório de forma inteiramente recursiva. |
| `comparar_modelos.py` | Script interpretador que consome a matriz de métricas (`resultados_modelos.json`) e renderiza histogramas, *boxplots* e curvas comparativas. |
| `visualizar_segmentacao.py` | Ferramenta visual auxiliar que renderiza a imagem real com a respectiva máscara de corte lado a lado. Útil para demonstrações acadêmicas. |

---

## 📊 Síntese dos Resultados Recentes

| Modelo | MAE ($kg$) | $R^2$ | Considerações Analíticas |
| :--- | :--- | :--- | :--- |
| Regressão Linear | ~31.0 | 4.0% | Baixo potencial de variância descritiva devido à não-linearidade geométrica. |
| XGBoost (Tuned) | ~28.6 | 16.6% | Excelente equilíbrio custo-benefício computacional graças às novas *features* alométricas inseridas. |
| **CNN (ResNet18)** | **~26.4** | **41.6%** | Maior poder de generalização anatômica, garantindo assertividade inclusive nas camadas mais escassas (bois médios/pesados). |

---

## 📢 Notas aos Desenvolvedores do Grupo

Para a manutenção correta da integridade deste repositório, favor atentar-se aos seguintes pontos arquiteturais que foram corrigidos nas últimas atualizações:

1. **Loader do Dataset:** O *scanner* original foi totalmente refatorado. Atualmente o código varre o diretório do Kaggle de forma *recursiva*. Isso garantiu que saltássemos de ~4.800 imagens originais (que possuíam pastas fixas no código) para a base real unificada de **7.509 imagens**, abrangendo ângulos negligenciados (como *Rear* e as variações secundárias *Back_2*). 
2. **Cache da MLP:** Para acelerar a extração iterativa, a MLP salvará um buffer de cache em `.npy` na primeira execução. Caso haja qualquer modificação no parâmetro `IMG_SIZE`, deve-se excluir esses arquivos no repositório para forçar uma nova extração matricial.
3. **Módulo de Geração Gráfica:** Todos os ativos gerados (.png) foram formatados nas resoluções padronizadas e unificados (ex.: `cnn_curva_loss.png`), de modo que possam ser prontamente anexados ao corpo do artigo. O código está estabilizado contra sub-classes ausentes durante *splits* aleatórios (ex.: erro do SciKit-Learn ao omitir pesos acima de 450kg na camada de validação).

Todo o código segue padrões limpos, metodologias atualizadas do Pandas/Scikit-Learn e está estruturado sob boas práticas da Engenharia de Software focada em Dados.
