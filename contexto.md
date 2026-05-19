Atue como um Engenheiro de Machine Learning e Visão Computacional Sénior. Preciso que analise o meu ambiente local e configure o treino da CNN utilizando obrigatoriamente as máscaras de segmentação. Execute as seguintes etapas de forma autónoma no meu terminal:

1. ETAPA DE VERIFICAÇÃO (MÁSCARAS ESTÃO PRESENTES?):
   - Inspecione a pasta onde o 'kagglehub' descarregou o dataset "sadhliroomyprime/cattle-weight-detection-model-dataset-12k".
   - Verifique e confirme se existem ficheiros de máscara (como '___fuse.png' ou pastas chamadas 'annotations') junto às imagens originais (.jpg) para os lotes B2, B3 e B4.
   - Analise se o Pandas DataFrame gerado pelo kagglehub já possui colunas que apontam para os caminhos das máscaras.
   - Antes de avançar para o código de treino, imprima no terminal uma mensagem clara confirmando se as máscaras foram localizadas com sucesso e qual é a estrutura de caminhos delas.

2. DESENVOLVIMENTO DO PIPELINE COM MÁSCARA (SE SEGMENTAÇÃO EXISTIR):
   - Crie ou atualize o ficheiro 'treinar_cnn.py'.
   - Implemente um Dataset customizado no PyTorch que carregue a imagem original e a sua respetiva máscara de segmentação.
   - Aplique a lógica de isolamento do bovino por cor em tempo de execução:
     * Lote B2: Filtre a cor Verde [0, 255, 0, 255].
     * Lotes B3 e B4: Filtre a cor Rosa/Magenta [255, 30, 249, 255].
   - Crie a máscara binária e aplique-a na imagem original, pintando todo o fundo de PRETO (0, 0, 0), deixando apenas o corpo do bovino visível para a rede.
   - Redimensione a imagem resultante (apenas o boi com fundo preto) para 224x224 RGB.

3. REGRAS DE TREINO E FILTROS:
   - Aplique o Filtro de Domínio: remova registos com peso menor que 50 kg ou maior que 1000 kg.
   - Separe os dados em 80% para Treino e 20% para Validação (random_state=42).
   - Treine uma ResNet18 pré-treinada adaptada para regressão (saída de 1 neurónio para prever o peso contínuo em kg). Configure para 3 épocas como teste rápido.

4. MÉTRICAS E GRÁFICOS (.PNG):
   - No final das épocas, calcule e exiba nos 20% de validação:
     * Regressão: MAE, RMSE e R².
     * Classificação (Faixas pós-predição: Leve < 250kg, Médio 250-450kg, Pesado > 450kg): Acurácia, Precisão, Recall e F1-Score.
   - Guarde na pasta atual os 4 gráficos: 'curva_loss.png', 'grafico_dispersao_r2.png', 'histograma_erros.png' e 'boxplot_violino_erros.png'.

Inicie o processo executando primeiro a verificação física das máscaras no meu diretório.