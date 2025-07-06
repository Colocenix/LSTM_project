# -*- coding: utf-8 -*-
"""
LSTM Technical Indicator Prediction for 5 Hong Kong Stocks Closing Prices (Simplified Version)
=================================================
Core Features:
1. Use only price data and technical indicators
2. Support walk-forward validation (optional, default 80/20 split)
3. Rolling z-score normalization (default)
4. Multi-step prediction structure reserved (default single step)
5. Optional quantile loss (default MSE)
6. Add up/down direction prediction accuracy output
7. Generate unified chart with prediction results for 5 stocks
8. Calculate MSE and variance
9. Predict July 2nd prices
10. Test different sliding window lengths {5, 10, 15, 30}

Run Method
--------
```bash
python LSTM1.py
```

Dependencies: pandas, numpy, torch, sklearn, matplotlib, seaborn, pandas_ta, pyarrow
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STOCK_LIST = ["0700.HK", "1810.HK", "3690.HK", "9618.HK", "9988.HK"]
FEATURES_BASE = [
    "Adj Close",
    "Volume",
    "MA5",
    "MA10",
    "RSI14",
    "MACD",
    "MACDh",
    "MACDs",
]
# 测试不同的滑窗长度
WINDOW_LENGTHS = [5, 10, 15, 30]  # 添加您提到的滑窗长度
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
ROLLING_NORM_WINDOW = 60  # About 3 months
MULTI_STEP = 1    # Future prediction steps, 1 for single step
USE_WALK_FORWARD = False # Whether to use walk-forward validation
USE_QUANTILE_LOSS = False # Whether to use quantile loss
QUANTILE = 0.5    # Quantile

ROOT = Path(__file__).resolve().parent
PRICES_PATH = ROOT / "prices.parquet"  # Assume already in same directory


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def add_technical_indicators(prices_long: pd.DataFrame) -> pd.DataFrame:
    """Add MA/RSI/MACD for each stock."""
    for stock in STOCK_LIST:
        idx = prices_long["Ticker"] == stock
        df = prices_long.loc[idx].sort_values("Date").copy()
        prices_long.loc[idx, "MA5"] = df["Adj Close"].rolling(5).mean().values
        prices_long.loc[idx, "MA10"] = df["Adj Close"].rolling(10).mean().values
        prices_long.loc[idx, "RSI14"] = ta.rsi(df["Adj Close"], length=14).values
        macd = ta.macd(df["Adj Close"])
        prices_long.loc[idx, "MACD"] = macd["MACD_12_26_9"].values
        prices_long.loc[idx, "MACDh"] = macd["MACDh_12_26_9"].values
        prices_long.loc[idx, "MACDs"] = macd["MACDs_12_26_9"].values
    return prices_long.fillna(0)


def load_prices() -> pd.DataFrame:
    """Load price data and add technical indicators"""
    prices = pd.read_parquet(PRICES_PATH)
    prices_long = prices.stack(level=0, future_stack=True).reset_index()
    prices_long.columns = ["Date", "Ticker", "Adj Close", "Volume"]
    prices_long["Date"] = pd.to_datetime(prices_long["Date"], utc=False)
    prices_long = add_technical_indicators(prices_long)
    return prices_long


# ---------------------------------------------------------------------------
# Rolling Standardization
# ---------------------------------------------------------------------------
def rolling_zscore(df: pd.DataFrame, feature_cols: List[str], window: int) -> pd.DataFrame:
    """Perform rolling standardization on features"""
    for col in feature_cols:
        df[col] = df.groupby("Ticker")[col].transform(lambda x: (x - x.rolling(window, min_periods=1).mean()) / (x.rolling(window, min_periods=1).std() + 1e-8))
    return df.fillna(0)


# ---------------------------------------------------------------------------
# Sequence Construction & Data Splitting
# ---------------------------------------------------------------------------

def create_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int, multi_step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create time series data with variable sequence length"""
    xs, ys = [], []
    for i in range(len(df) - seq_len - multi_step + 1):
        x = df.iloc[i : i + seq_len][feature_cols].values
        y = df.iloc[i + seq_len : i + seq_len + multi_step]["Adj Close"].values
        if len(y) < multi_step:
            continue
        xs.append(x)
        ys.append(y)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def split_train_test(x: np.ndarray, y: np.ndarray, train_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split training and test data"""
    split = int(len(x) * train_ratio)
    return x[:split], x[split:], y[:split], y[split:]


# Walk-forward validation
def walk_forward_split(x: np.ndarray, y: np.ndarray, test_size: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Walk-forward validation split"""
    splits = []
    n = len(x)
    for start in range(0, n - test_size, test_size):
        end = start + test_size
        x_train, y_train = x[:end], y[:end]
        x_test, y_test = x[end:end+test_size], y[end:end+test_size]
        if len(x_test) == 0:
            break
        splits.append((x_train, x_test, y_train, y_test))
    return splits


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# Quantile loss
class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super().__init__()
        self.q = quantile

    def forward(self, pred, target):
        diff = target - pred
        return torch.mean(torch.max(self.q * diff, (self.q - 1) * diff))


# ---------------------------------------------------------------------------
# Training + Evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_with_window(data: pd.DataFrame, feature_cols: List[str], seq_len: int, plot_title: str = "", multi_step: int = 1) -> Dict[str, Dict[str, float]]:
    """Train model and evaluate with specific sequence length, return metrics and prediction results"""
    metrics = {}
    predictions = {}  # Store prediction results for visualization
    
    for stock in STOCK_LIST:
        df_stock = (
            data[data["Ticker"] == stock]
            .sort_values("Date")
            .reset_index(drop=True)
        )
        df_stock = rolling_zscore(df_stock, feature_cols, ROLLING_NORM_WINDOW)
        x_all, y_all = create_sequences(df_stock, feature_cols, seq_len, multi_step)
        if USE_WALK_FORWARD:
            test_size = int(len(x_all) * (1 - TRAIN_RATIO))
            splits = walk_forward_split(x_all, y_all, test_size)
        else:
            splits = [(split_train_test(x_all, y_all, TRAIN_RATIO))]
        y_true_all, y_pred_all = [], []
        y_true_dir, y_pred_dir = [], []
        for x_train, x_test, y_train, y_test in splits:
            # torch data
            ds_train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
            dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False)
            model = StockLSTM(len(feature_cols), output_size=multi_step).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            loss_fn = QuantileLoss(QUANTILE) if USE_QUANTILE_LOSS else nn.MSELoss()
            model.train()
            for ep in range(EPOCHS):
                for xb, yb in dl_train:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            # Evaluation
            model.eval()
            with torch.no_grad():
                y_pred = model(torch.from_numpy(x_test).to(device)).cpu().numpy()
            y_true = y_test[:, 0] if multi_step == 1 else y_test[:, 0]
            y_pred1 = y_pred[:, 0] if multi_step == 1 else y_pred[:, 0]
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred1)
            # Up/down direction
            y_true_dir.extend((np.diff(np.concatenate([[y_train[-1,0]], y_true])) > 0).astype(int))
            y_pred_dir.extend((np.diff(np.concatenate([[y_train[-1,0]], y_pred1])) > 0).astype(int))
        # Evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mae = mean_absolute_error(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)
        acc = accuracy_score(y_true_dir, y_pred_dir)
        variance = np.var(y_pred_all)  # Add variance calculation
        metrics[stock] = {"RMSE": rmse, "MAE": mae, "R2": r2, "Direction Accuracy": acc, "Variance": variance}
        
        # Store prediction results for visualization
        predictions[stock] = {
            'dates': df_stock["Date"].iloc[-len(y_true_all):],
            'true': y_true_all,
            'pred': y_pred_all
        }
    
    # ----------- Unified Visualization -----------
    if plot_title:
        plt.figure(figsize=(20, 12))
        for i, stock in enumerate(STOCK_LIST, 1):
            plt.subplot(2, 3, i)
            plt.plot(predictions[stock]['dates'], predictions[stock]['true'], label='Actual Price', linewidth=2, color='blue')
            plt.plot(predictions[stock]['dates'], predictions[stock]['pred'], label='Predicted Price', linewidth=2, color='red')
            plt.title(f'{stock} - LSTM Prediction Results (Window={seq_len})')
            plt.xlabel('Date')
            plt.ylabel('Price (HKD)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        plt.suptitle(plot_title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    return metrics


def train_and_evaluate(data: pd.DataFrame, feature_cols: List[str], plot_title: str = "", multi_step: int = 1) -> Dict[str, Dict[str, float]]:
    """Original function for backward compatibility"""
    return train_and_evaluate_with_window(data, feature_cols, 30, plot_title, multi_step)


# ---------------------------------------------------------------------------
# Predict Future Date Prices
# ---------------------------------------------------------------------------
def predict_future_price_with_window(data: pd.DataFrame, feature_cols: List[str], seq_len: int, target_date: str = "2024-07-02", multi_step: int = 1) -> Dict[str, float]:
    """Predict stock prices for specified date with specific sequence length"""
    target_date = pd.to_datetime(target_date)
    predictions = {}
    
    for stock in STOCK_LIST:
        df_stock = (
            data[data["Ticker"] == stock]
            .sort_values("Date")
            .reset_index(drop=True)
        )
        
        # Save original prices for inverse standardization
        original_prices = df_stock["Adj Close"].values
        
        df_stock = rolling_zscore(df_stock, feature_cols, ROLLING_NORM_WINDOW)
        
        # Use all data to train model
        x_all, y_all = create_sequences(df_stock, feature_cols, seq_len, multi_step)
        
        # Train model
        ds_train = TensorDataset(torch.from_numpy(x_all), torch.from_numpy(y_all))
        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False)
        model = StockLSTM(len(feature_cols), output_size=multi_step).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = QuantileLoss(QUANTILE) if USE_QUANTILE_LOSS else nn.MSELoss()
        
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        
        # Predict future price
        model.eval()
        with torch.no_grad():
            # Use last seq_len days data for prediction
            last_sequence = df_stock.iloc[-seq_len:][feature_cols].values
            last_sequence = torch.from_numpy(last_sequence).float().unsqueeze(0).to(device)
            future_pred = model(last_sequence).cpu().numpy().flatten()
            
            # Inverse standardization: convert standardized prediction back to original price
            # Use price statistics from last ROLLING_NORM_WINDOW days
            recent_prices = original_prices[-ROLLING_NORM_WINDOW:]
            price_mean = np.mean(recent_prices)
            price_std = np.std(recent_prices)
            
            # Inverse standardization formula: original = normalized * std + mean
            predicted_price = float(future_pred[0] * price_std + price_mean)
            predictions[stock] = predicted_price
    
    return predictions


def predict_future_price(data: pd.DataFrame, feature_cols: List[str], target_date: str = "2024-07-02", multi_step: int = 1) -> Dict[str, float]:
    """Original function for backward compatibility"""
    return predict_future_price_with_window(data, feature_cols, 30, target_date, multi_step)


# ---------------------------------------------------------------------------
# Window Length Comparison
# ---------------------------------------------------------------------------

def compare_window_lengths(data: pd.DataFrame, feature_cols: List[str]) -> Dict[int, Dict[str, float]]:
    """Compare performance of different window lengths"""
    print("\n========== 测试不同滑窗长度性能 ==========")
    window_results = {}
    
    for window_len in WINDOW_LENGTHS:
        print(f"\n测试滑窗长度: {window_len} 天")
        try:
            metrics = train_and_evaluate_with_window(data, feature_cols, window_len, f"LSTM 滑窗长度 {window_len} 天预测结果", MULTI_STEP)
            
            # Calculate average metrics
            avg_rmse = np.mean([d['RMSE'] for d in metrics.values()])
            avg_mae = np.mean([d['MAE'] for d in metrics.values()])
            avg_r2 = np.mean([d['R2'] for d in metrics.values()])
            avg_dir_acc = np.mean([d['Direction Accuracy'] for d in metrics.values()])
            avg_variance = np.mean([d['Variance'] for d in metrics.values()])
            
            window_results[window_len] = {
                'RMSE': avg_rmse,
                'MAE': avg_mae,
                'R2': avg_r2,
                'Direction Accuracy': avg_dir_acc,
                'Variance': avg_variance
            }
            
            print(f"滑窗长度 {window_len} 天 - 平均指标:")
            print(f"  RMSE: {avg_rmse:.4f}")
            print(f"  MAE: {avg_mae:.4f}")
            print(f"  R²: {avg_r2:.4f}")
            print(f"  方向准确率: {avg_dir_acc:.4f}")
            print(f"  方差: {avg_variance:.4f}")
            
        except Exception as e:
            print(f"滑窗长度 {window_len} 天测试失败: {e}")
            continue
    
    return window_results


def find_best_window(window_results: Dict[int, Dict[str, float]]) -> Tuple[int, Dict[str, float]]:
    """Find the best window length based on R² score"""
    if not window_results:
        return None, {}
    
    # Find best window based on R² (higher is better)
    best_window = max(window_results.keys(), key=lambda w: window_results[w]['R2'])
    best_metrics = window_results[best_window]
    
    print(f"\n========== 最佳滑窗长度分析 ==========")
    print(f"基于R²分数，最佳滑窗长度: {best_window} 天")
    print(f"最佳性能指标:")
    print(f"  RMSE: {best_metrics['RMSE']:.4f}")
    print(f"  MAE: {best_metrics['MAE']:.4f}")
    print(f"  R²: {best_metrics['R2']:.4f}")
    print(f"  方向准确率: {best_metrics['Direction Accuracy']:.4f}")
    print(f"  方差: {best_metrics['Variance']:.4f}")
    
    return best_window, best_metrics


# ---------------------------------------------------------------------------
# Main Process
# ---------------------------------------------------------------------------

def main():
    print("Loading price data...")
    prices_long = load_prices()
    print(f"Data loading completed, total {len(prices_long)} records")

    # Test different window lengths
    window_results = compare_window_lengths(prices_long, FEATURES_BASE)
    
    # Find best window length
    best_window, best_metrics = find_best_window(window_results)
    
    if best_window:
        print(f"\n========== 使用最佳滑窗长度 {best_window} 天进行最终预测 ==========")
        final_metrics = train_and_evaluate_with_window(prices_long, FEATURES_BASE, best_window, f"LSTM 最佳滑窗长度 {best_window} 天预测结果", MULTI_STEP)

        # Predict July 2nd prices with best window
        print(f"\n========== 使用最佳滑窗长度预测 July 2nd 价格 ==========")
        future_prices = predict_future_price_with_window(prices_long, FEATURES_BASE, best_window, "2024-07-02", MULTI_STEP)
        for stock, price in future_prices.items():
            print(f"{stock}: {price:.2f} HKD")

        # Summary display
        def summary(title: str, m: Dict[str, Dict[str, float]]):
            print(f"\n{title}")
            print("-" * 80)
            print(f"{'Stock Code':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Direction Acc':<12} {'Variance':<10}")
            print("-" * 80)
            for s, d in m.items():
                print(f"{s:<10} {d['RMSE']:<10.2f} {d['MAE']:<10.2f} {d['R2']:<10.4f} {d['Direction Accuracy']:<12.3f} {d['Variance']:<10.4f}")
            print("-" * 80)
            print("Average Metrics: ", end="")
            print(
                f"RMSE={np.mean([d['RMSE'] for d in m.values()]):.2f}, "
                f"MAE={np.mean([d['MAE'] for d in m.values()]):.2f}, "
                f"R²={np.mean([d['R2'] for d in m.values()]):.4f}, "
                f"Direction Accuracy={np.mean([d['Direction Accuracy'] for d in m.values()]):.3f}, "
                f"Average Variance={np.mean([d['Variance'] for d in m.values()]):.4f}"
            )

        summary(f"LSTM 最佳滑窗长度 {best_window} 天预测结果", final_metrics)
    else:
        print("无法确定最佳滑窗长度，使用默认30天")
        # Fallback to original method
        print("\n========== LSTM Technical Indicator Prediction Model ==========")
        metrics = train_and_evaluate(prices_long, FEATURES_BASE, "LSTM Technical Indicator Prediction Results", MULTI_STEP)

        # Predict July 2nd prices
        print("\n========== Predict July 2nd Prices ==========")
        future_prices = predict_future_price(prices_long, FEATURES_BASE, "2024-07-02", MULTI_STEP)
        for stock, price in future_prices.items():
            print(f"{stock}: {price:.2f} HKD")

        # Summary display
        def summary(title: str, m: Dict[str, Dict[str, float]]):
            print(f"\n{title}")
            print("-" * 80)
            print(f"{'Stock Code':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Direction Acc':<12} {'Variance':<10}")
            print("-" * 80)
            for s, d in m.items():
                print(f"{s:<10} {d['RMSE']:<10.2f} {d['MAE']:<10.2f} {d['R2']:<10.4f} {d['Direction Accuracy']:<12.3f} {d['Variance']:<10.4f}")
            print("-" * 80)
            print("Average Metrics: ", end="")
            print(
                f"RMSE={np.mean([d['RMSE'] for d in m.values()]):.2f}, "
                f"MAE={np.mean([d['MAE'] for d in m.values()]):.2f}, "
                f"R²={np.mean([d['R2'] for d in m.values()]):.4f}, "
                f"Direction Accuracy={np.mean([d['Direction Accuracy'] for d in m.values()]):.3f}, "
                f"Average Variance={np.mean([d['Variance'] for d in m.values()]):.4f}"
            )

        summary("LSTM Technical Indicator Prediction Results", metrics)


if __name__ == "__main__":
    main()
