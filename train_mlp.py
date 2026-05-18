import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    csv_path = 'dataset_features_completo.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    print("Carregando dados...")
    df = pd.read_csv(csv_path)
    
    # Aplicando One-Hot Encoding nas colunas categóricas (sexo e vista)
    df_encoded = pd.get_dummies(df, columns=['sexo', 'vista'], drop_first=True)
    
    # Convertendo colunas booleanas resultantes do get_dummies para float
    for col in ['sexo_M', 'vista_side']:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(float)
            
    features = ['area', 'altura_bbox', 'largura_bbox', 'proporcao', 'area_norm']
    if 'sexo_M' in df_encoded.columns:
        features.append('sexo_M')
    if 'vista_side' in df_encoded.columns:
        features.append('vista_side')
        
    target = 'peso_kg'
    
    X = df_encoded[features]
    y = df_encoded[target]
    
    # Redes Neurais (MLP) exigem que os dados estejam em uma escala normalizada.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Divisão dos dados em treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Features utilizadas: {features}")
    print("Iniciando treinamento do Multi-Layer Perceptron (MLP)...")
    
    # Configuração base da MLP
    mlp = MLPRegressor(max_iter=1000, early_stopping=True, random_state=42)
    
    # Grade de parâmetros para testar (reduzida para otimizar tempo, mas mantendo excelentes combinações)
    param_grid = {
        'hidden_layer_sizes': [(64, 32), (128, 64, 32)],       # Camadas escondidas e neurônios
        'activation': ['relu'],                                # Função de ativação
        'alpha': [0.001, 0.01],                                # Termo de L2
        'learning_rate_init': [0.001, 0.01]                    # Taxa de aprendizado
    }
    
    print("Testando combinações de hiperparâmetros (GridSearch)...")
    grid_search = GridSearchCV(
        estimator=mlp, 
        param_grid=param_grid, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        verbose=1, 
        n_jobs=-1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"\nBusca e treinamento finalizados em {(end_time - start_time):.2f} segundos.")
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    # Previsões
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Métricas
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print("\n" + "="*50)
    print("--- RESULTADOS DO MLP OTIMIZADO COM RECURSOS EXTRAS ---")
    print("\n[MÉTRICAS DE TREINO]")
    print(f"MAE: {mae_train:.2f} kg")
    print(f"RMSE: {rmse_train:.2f} kg")
    print(f"R^2: {r2_train:.4f}")
    
    print("\n[MÉTRICAS DE TESTE (Generalização)]")
    print(f"MAE: {mae_test:.2f} kg")
    print(f"RMSE: {rmse_test:.2f} kg")
    print(f"R^2: {r2_test:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
