import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    initial_len = len(df)
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Outliers removidos na coluna {column}: {initial_len - len(df_filtered)}")
    return df_filtered

def main():
    csv_path = 'dataset_features_completo.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    print("Carregando dados...")
    df = pd.read_csv(csv_path)
    
    df = remove_outliers_iqr(df, 'peso_kg')
    
    df_encoded = pd.get_dummies(df, columns=['sexo', 'vista'], drop_first=True)
    
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
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Features utilizadas (antes de polinomios): {features}")
    print("\nTreinando Regressão Linear Regularizada (Ridge) com Polinômios...")
    
    model = Ridge(alpha=10.0) 
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Treinamento finalizado em {(end_time - start_time):.4f} segundos.")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print("\n" + "="*50)
    print("--- RESULTADOS DA REGRESSÃO RIDGE COM RECURSOS EXTRAS ---")
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
