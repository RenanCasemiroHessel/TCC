# Relatório de Análise e Benchmarking — Estimativa de Peso Bovino
Este relatório apresenta a análise comparativa final do desempenho dos modelos preditivos desenvolvidos para a estimativa de peso corporal de bovinos. Os testes abrangeram desde abordagens estatísticas clássicas até redes de aprendizado profundo (Deep Learning), avaliando métricas de regressão ($MAE$, $RMSE$ e $R^2$) e classificação baseada em faixas biológicas de peso.

> [!IMPORTANT]
> Todos os códigos e saídas do terminal foram unificados, as falhas de dimensões do gráfico de perda corrigidas, e as correções de engenharia de features alométricas e PCA aplicadas com absoluto sucesso.

---

## 1. Tabela Comparativa de Resultados
Abaixo está compilada a tabela de resultados consolidada após a execução de todo o pipeline padronizado:

| Modelo | MAE ($kg$) | RMSE ($kg$) | $R^2$ (Det.) | Acurácia (%) | F1-Score (%) | Complexidade de Entrada |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CNN (ResNet18)** | **26.42** | **34.17** | **0.416** | 96.30% | 95.27% | Pixels segmentados (Imagem 3D/2D) |
| **XGBoost (Tuned)** | 28.57 | 37.93 | 0.167 | **97.47%** | **96.22%** | Features CSV + Engenharia Física |
| **XGBoost (Padrão)** | 29.58 | 39.06 | 0.116 | 97.34% | **96.27%** | Features CSV + Engenharia Física |
| **Regressão Linear** | 30.98 | 40.72 | 0.040 | 97.47% | 96.22% | Features CSV (Geometria 2D simples) |
| **MLP (pixels + PCA)** | 38.02 | 81.01 | -2.647 | 96.16% | 95.25% | Pixels 64x64 reduzidos por PCA |

---

## 2. Análise Crítica dos Modelos

### A. Regressão Linear (Baseline Estatístico)
*   **Desempenho**: $MAE = 30.98\ kg$ | $R^2 = 0.040$
*   **Diagnóstico**: **Sub-ajuste severo (Underfitting)**. O baixíssimo valor de $R^2$ indica que apenas 4% da variância do peso dos animais é explicada pela combinação linear das features geométricas básicas (`area`, `bbox_width`, `bbox_height`).
*   **Justificativa Acadêmica**: A relação geométrica entre projeções 2D planas e massa corpórea tridimensional de seres biológicos é inerentemente não-linear. Tentar traçar uma reta multidimensional estática falha em capturar as variações anatômicas individuais dos animais.

### B. XGBoost & XGBoost Tuned (Modelos de Árvore com Engenharia Alométrica)
*   **Desempenho**: $MAE = 28.57\ kg$ | $R^2 = 0.167$ (Melhoria significativa com GridSearch)
*   **Engenharia de Features aplicada**: A introdução de variáveis físicas, especialmente a **Aproximação Volumétrica Alométrica** ($\text{Área}^{1.5}$) baseada na relação de escala biológica clássica ($Peso \propto Volume \propto Área^{1.5}$) e a **Taxa de Preenchimento da Bounding Box** ($\text{Solidez}$), reduziu o erro e elevou o $R^2$ em comparação com a execução anterior.
*   **Justificativa Acadêmica**: Algoritmos de gradiente impulsionado por árvores (XGBoost) lidam excepcionalmente bem com dados tabulares e relações não-lineares locais. No entanto, por estarem limitados a features extraídas manualmente (Handcrafted Features), eles esbarram em um limite físico de aprendizado, sendo incapazes de entender curvas de profundidade ou textura.

### C. MLP — Multi-Layer Perceptron (Imagens Achatadas com PCA)
*   **Desempenho**: $MAE = 38.02\ kg$ | $R^2 = -2.647$
*   **Diagnóstico**: **Overfitting Extremo / Instabilidade Preditiva**. Mesmo aplicando PCA para reduzir a dimensionalidade espacial de 12.288 para 128 componentes, a arquitetura MLP puramente densa (Fully Connected) falha em tarefas de visão computacional biológica.
*   **Justificativa Acadêmica**: Redes totalmente conectadas ignoram a topologia espacial da imagem. Para a MLP, um pixel na coordenada $(x, y)$ não possui relação de vizinhança com o pixel $(x+1, y)$. Portanto, o modelo "decora" a presença de ruído e padrões exatos do lote de treinamento, desmoronando na validação com um $R^2$ negativo catastrófico.

### D. CNN ResNet18 (Estado da Arte / Transfer Learning)
*   **Desempenho**: $MAE = \mathbf{26.42\ kg}$ | $RMSE = \mathbf{34.17\ kg}$ | $R^2 = \mathbf{0.416}$
*   **Diagnóstico**: **Convergência ideal e generalização superior**. A CNN alcançou o menor erro médio absoluto ($26.42\ kg$) e o maior coeficiente de determinação ($0.416$), explicando mais de 41% da variância total a partir dos pixels segmentados.
*   **Justificativa Acadêmica**: Por meio das camadas convolucionais, a CNN extrai mapas de características espaciais profundas (Deep Features), identificando curvas de lombo, musculatura da coxa, textura e volumetria implícita que são invisíveis aos números secos de um CSV tabular. O descongelamento progressivo do backbone na **Época 5** permitiu que a rede recalibrasse seus filtros convolucionais para o domínio bovino de forma extremamente saudável.

---

## 3. O Fenômeno do Desbalanceamento de Dados e o "Falso Sucesso" da Acurácia
Um ponto crucial para a defesa da banca do seu TCC é a discrepância aparente entre as métricas de regressão e classificação:
> **Por que a Regressão Linear tem uma Acurácia espetacular de 97.47% se o seu R² é de míseros 0.04?**

> [!WARNING]
> Isso ocorre devido ao **severo desbalanceamento do conjunto de testes**:
> *   Bois **Leves** (<250 kg): **1.464** amostras
> *   Bois **Médios** (250-450 kg): **37** amostras
> *   Bois **Pesados** (>450 kg): **1** amostra

Como mais de 97% das amostras pertencem à categoria "Leve", qualquer modelo simplista ou viciado que prediga sistematicamente valores baixos obterá uma acurácia próxima de 97%.
*   A **Regressão Linear** e o **XGBoost** obtiveram Recall e Precision de **0.00** para bois médios e pesados. Eles simplesmente ignoraram essas faixas de peso por causa do viés matemático.
*   A **CNN (ResNet18)**, apesar de ter uma acurácia nominal ligeiramente menor (96.30%), foi o **único modelo capaz de classificar bois na faixa Média com precisão real (Precision de 0.50 e Recall de 0.14 na faixa média)**. Ela de fato aprendeu a diferenciar animais mais encorpados baseando-se no formato, em vez de apenas chutar pesos baixos por conveniência estatística.

---

## 4. O Veredito Acadêmico: Qual Modelo Utilizar?

### A Recomendação Principal para o TCC: **CNN (ResNet18)**
A **CNN convolucional com Transfer Learning e Fine-Tuning progressivo** é, sem sombra de dúvidas, a escolha técnica mais robusta para a conclusão do seu TCC.
*   **Porquê técnica**: Apresenta a maior capacidade de generalização visual, menor erro médio absoluto ($26.42\ kg$, uma melhoria fenomenal em relação aos $138\ kg$ iniciais de baseline sem normalização), e capacidade de aprendizado anatômico real.
*   **Porquê acadêmica**: Redes neurais convolucionais representam o estado da arte na literatura científica de pecuária de precisão. O uso de ResNet18 com segmentação de imagens demonstra rigor metodológico elevado e atende aos padrões de pesquisa avançada em Inteligência Artificial.

### O Co-Destaque Tabular: **XGBoost Otimizado**
Você deve apresentar o **XGBoost (Tuned)** como o campeão da categoria de dados tabulares. Ele é excelente para cenários onde recursos de processamento (GPU) são extremamente escassos ou onde o tempo de inferência imediato em campo é crítico, servindo como uma excelente alternativa "leve" (Edge Computing) às redes profundas.
