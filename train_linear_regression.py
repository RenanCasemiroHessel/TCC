import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    csv_path = 'dataset_features_completo.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    print("Carregando dados...")
    df = pd.read_csv(csv_path)
    
    features = ['area', 'altura_bbox', 'largura_bbox', 'proporcao', 'area_norm']
    target = 'peso_kg'
    
    X = df[features]
    y = df[target]
    
    # Padronização dos dados (Importante para regressão linear para avaliar coeficientes)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Divisão dos dados em treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("Treinando Regressão Linear Múltipla (Baseline)...")
    
    # Modelo Base
    model = LinearRegression()
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"\nTreinamento finalizado em {(end_time - start_time):.4f} segundos.")
    
    # Previsões
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métricas - Treino
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    # Métricas - Teste
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print("\n--- RESULTADOS DA REGRESSÃO LINEAR ---")
    print("\n[MÉTRICAS DE TREINO]")
    print(f"MAE: {mae_train:.2f} kg")
    print(f"RMSE: {rmse_train:.2f} kg")
    print(f"R^2: {r2_train:.4f}")
    
    print("\n[MÉTRICAS DE TESTE (Generalização)]")
    print(f"MAE: {mae_test:.2f} kg")
    print(f"RMSE: {rmse_test:.2f} kg")
    print(f"R^2: {r2_test:.4f}")
    
    # Importância (Peso) de cada coeficiente
    print("\n[COEFICIENTES DA REGRESSÃO]")
    for feat, coef in zip(features, model.coef_):
        print(f"{feat}: {coef:.4f}")

if __name__ == '__main__':
    main()
