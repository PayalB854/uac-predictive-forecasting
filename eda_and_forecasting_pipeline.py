import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_pipeline():
    print("--- Running Data Pipeline ---")
    
    # 1. Dataset Generation/Loading
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    np.random.seed(42)
    care_load = 5000 + np.cumsum(np.random.normal(2, 15, len(dates)))
    df = pd.DataFrame({"Date": dates, "HHS_Care_Load": care_load})
    df.set_index("Date", inplace=True)
    
    # 2. Time Series Decomposition
    decomposition = seasonal_decompose(df["HHS_Care_Load"], model="additive", period=7)
    print("Decomposition completed: Trend and 7-day Seasonality extracted.")
    
    # 3. Feature Engineering
    df["lag_1"] = df["HHS_Care_Load"].shift(1)
    df["lag_7"] = df["HHS_Care_Load"].shift(7)
    df["rolling_7_mean"] = df["HHS_Care_Load"].rolling(7).mean()
    df.dropna(inplace=True)
    
    # 4. Train-Test Split (Walk-forward)
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    print(f"Train Dataset size: {len(train)} | Test Dataset size: {len(test)}")
    
    # 5. ML Baseline (Random Forest)
    X_train, y_train = train.drop(columns=["HHS_Care_Load"]), train["HHS_Care_Load"]
    X_test, y_test = test.drop(columns=["HHS_Care_Load"]), test["HHS_Care_Load"]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    
    # 6. Evaluation
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("\n--- Model Results ---")
    print(f"Random Forest MAE: {mae:.2f}")
    print(f"Random Forest RMSE: {rmse:.2f}")

if __name__ == "__main__":
    run_pipeline()