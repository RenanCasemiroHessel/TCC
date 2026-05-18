import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def main():
    csv_path = 'dataset_features_completo.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Aplicando One-Hot Encoding nas colunas categóricas (sexo e vista)
    # sexo: F -> sexo_M=0, M -> sexo_M=1
    # vista: back -> vista_side=0, side -> vista_side=1
    df_encoded = pd.get_dummies(df, columns=['sexo', 'vista'], drop_first=True)
    
    # Convertendo colunas booleanas resultantes do get_dummies para float (XGBoost requer numerico)
    for col in ['sexo_M', 'vista_side']:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(float)
            
    print(f"Columns after encoding: {df_encoded.columns.tolist()}")
    
    features = ['area', 'altura_bbox', 'largura_bbox', 'proporcao', 'area_norm']
    if 'sexo_M' in df_encoded.columns:
        features.append('sexo_M')
    if 'vista_side' in df_encoded.columns:
        features.append('vista_side')
        
    target = 'peso_kg'
    
    X = df_encoded[features]
    y = df_encoded[target]
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"Features utilized: {features}")
    
    # XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42, 
        objective='reg:squarederror'
    )
    
    print("Training model...")
    start_time = time.time()
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train, 
        eval_set=eval_set, 
        verbose=False
    )
    
    end_time = time.time()
    train_time = end_time - start_time
    
    # Predições
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métricas
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print("\n" + "="*50)
    print("--- XGBOOST COM RECURSOS EXTRAS ---")
    print(f"Training time: {train_time:.4f} seconds")
    print("\n[TRAIN METRICS]")
    print(f"MAE: {mae_train:.2f} kg")
    print(f"RMSE: {rmse_train:.2f} kg")
    print(f"R^2: {r2_train:.4f}")
    
    print("\n[TEST METRICS (Generalization)]")
    print(f"MAE: {mae_test:.2f} kg")
    print(f"RMSE: {rmse_test:.2f} kg")
    print(f"R^2: {r2_test:.4f}")
    
    print("\n[OVERFITTING ANALYSIS]")
    diff_mae = mae_test - mae_train
    print(f"Difference in MAE (Test - Train): {diff_mae:.2f} kg")
    if diff_mae > 5:
         print("Model shows some signs of overfitting.")
    else:
         print("Model generalizes well.")
         
    print("\n[FEATURE IMPORTANCE]")
    importances = model.feature_importances_
    for feat, imp in zip(features, importances):
        print(f"{feat}: {imp:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
