import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

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
    
    # Divisão dos dados em treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Features utilizadas: {features}")
    print("Iniciando Otimização de Hiperparâmetros (GridSearchCV)...")
    
    # Modelo base
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    
    # Grade de parâmetros simplificada para ser veloz mas eficiente
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Configura o GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model, 
        param_grid=param_grid, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        verbose=1, 
        n_jobs=-1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"\nBusca finalizada em {(end_time - start_time):.2f} segundos.")
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    # Previsões com o melhor modelo
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
    print("--- RESULTADOS COM XGBOOST OTIMIZADO (RECURSOS EXTRAS) ---")
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
