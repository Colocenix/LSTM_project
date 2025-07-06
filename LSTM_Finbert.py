# -*- coding: utf-8 -*-
"""
LSTM + FinBERT Sentiment Analysis Prediction Model
=================================================
Core Features:
1. Use price data, technical indicators and FinBERT sentiment analysis data
2. Support walk-forward validation (optional, default 80/20 split)
3. Rolling z-score normalization (default)
4. Multi-step prediction structure reserved (default single step)
5. Optional quantile loss (default MSE)
6. Add up/down direction prediction accuracy output
7. Generate unified chart with prediction results for 5 stocks
8. Calculate MSE and variance
9. Predict July 2nd prices

Run Method
--------
```bash
python LSTM_Finbert.py
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
STOCK_CODES = ["0700", "1810", "3690", "9618", "9988"]  # Corresponding stock codes

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

FEATURES_SENTIMENT = FEATURES_BASE + [
    "sentiment_score",
    "positive_prob",
    "negative_prob",
    "neutral_prob",
]

SEQ_LEN = 30
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


def load_sentiment_data() -> Dict[str, pd.DataFrame]:
    """Load sentiment analysis data from CSV files"""
    sentiment_data = {}
    
    # First try to load lstm_sentiment_series files
    sentiment_files = []
    for file in os.listdir(ROOT):
        if file.startswith("lstm_sentiment_series_") and file.endswith(".csv"):
            sentiment_files.append(file)
    
    if sentiment_files:
        print(f"Found {len(sentiment_files)} sentiment series files")
        
        # Load each sentiment file
        for file in sentiment_files:
            # Extract stock code from filename (e.g., lstm_sentiment_series_3690.csv -> 3690)
            stock_code = file.replace("lstm_sentiment_series_", "").replace(".csv", "")
            
            # Handle special case for 700 -> 0700
            if stock_code == "700":
                stock_code = "0700"
            
            if stock_code in STOCK_CODES:
                file_path = ROOT / file
                print(f"Loading sentiment data for {stock_code} from {file}")
                
                # Read sentiment data
                sentiment_df = pd.read_csv(file_path)
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                
                # Rename columns to match expected format
                sentiment_df = sentiment_df.rename(columns={
                    'score': 'sentiment_score',
                    'positive_prob': 'positive_prob',
                    'negative_prob': 'negative_prob',
                    'neutral_prob': 'neutral_prob'
                })
                
                sentiment_data[stock_code] = sentiment_df
                print(f"Loaded {len(sentiment_df)} records for {stock_code}")
    
    # If no sentiment series files found, try FinBERT files
    if not sentiment_data:
        print("No sentiment series files found, trying FinBERT files...")
        finbert_files = []
        for file in os.listdir(ROOT / "Today but not"):
            if file.startswith("chinese_finbert_sentiment_analysis_") and file.endswith(".csv"):
                finbert_files.append(file)
        
        if finbert_files:
            # Use latest file
            latest_file = sorted(finbert_files)[-1]
            finbert_path = ROOT / "Today but not" / latest_file
            print(f"Using FinBERT file: {latest_file}")
            
            # Read FinBERT data
            finbert_df = pd.read_csv(finbert_path)
            finbert_df['date'] = pd.to_datetime(finbert_df['date'])
            
            # Process by stock code
            for stock_code in STOCK_CODES:
                # Convert stock code to string format for matching
                stock_data = finbert_df[finbert_df['stock_code'] == str(stock_code)].copy()
                if len(stock_data) > 0:
                    # Aggregate sentiment data by date
                    daily_sentiment = stock_data.groupby(stock_data['date'].dt.date).agg({
                        'sentiment': lambda x: (x == 'positive').sum() - (x == 'negative').sum(),
                        'positive_prob': 'mean',
                        'negative_prob': 'mean', 
                        'neutral_prob': 'mean'
                    }).reset_index()
                    
                    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
                    # Calculate daily sentiment score
                    daily_counts = stock_data.groupby(stock_data['date'].dt.date).size()
                    daily_sentiment['sentiment_score'] = daily_sentiment['sentiment'] / daily_counts.values
                    
                    sentiment_data[stock_code] = daily_sentiment
    
    if not sentiment_data:
        print("No sentiment data found, using basic features")
    
    return sentiment_data


def load_prices() -> pd.DataFrame:
    """Load price data and add technical indicators"""
    prices = pd.read_parquet(PRICES_PATH)
    prices_long = prices.stack(level=0, future_stack=True).reset_index()
    prices_long.columns = ["Date", "Ticker", "Adj Close", "Volume"]
    prices_long["Date"] = pd.to_datetime(prices_long["Date"], utc=False)
    prices_long = add_technical_indicators(prices_long)
    return prices_long


def merge_sentiment_data(prices_long: pd.DataFrame, sentiment_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge price data and sentiment data"""
    if not sentiment_data:
        return prices_long
    
    # Add sentiment features for each stock
    for i, stock in enumerate(STOCK_LIST):
        stock_code = STOCK_CODES[i]
        if stock_code in sentiment_data:
            stock_sentiment = sentiment_data[stock_code]
            
            # Get price data for this stock
            stock_mask = prices_long["Ticker"] == stock
            stock_prices = prices_long[stock_mask].copy()
            
            # Merge sentiment data
            stock_prices = stock_prices.merge(
                stock_sentiment, 
                left_on='Date', 
                right_on='date', 
                how='left'
            )
            
            # Fill missing values
            sentiment_cols = ['sentiment_score', 'positive_prob', 'negative_prob', 'neutral_prob']
            for col in sentiment_cols:
                if col in stock_prices.columns:
                    stock_prices[col] = stock_prices[col].fillna(0)
                else:
                    stock_prices[col] = 0
            
            # Update original data
            prices_long.loc[stock_mask, sentiment_cols] = stock_prices[sentiment_cols].values
    
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

def create_sequences(df: pd.DataFrame, feature_cols: List[str], multi_step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create time series data"""
    xs, ys = [], []
    for i in range(len(df) - SEQ_LEN - multi_step + 1):
        x = df.iloc[i : i + SEQ_LEN][feature_cols].values
        y = df.iloc[i + SEQ_LEN : i + SEQ_LEN + multi_step]["Adj Close"].values
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

def train_and_evaluate(data: pd.DataFrame, feature_cols: List[str], plot_title: str = "", multi_step: int = 1) -> Dict[str, Dict[str, float]]:
    """Train model and evaluate, return metrics and prediction results"""
    metrics = {}
    predictions = {}  # Store prediction results for visualization
    
    for stock in STOCK_LIST:
        df_stock = (
            data[data["Ticker"] == stock]
            .sort_values("Date")
            .reset_index(drop=True)
        )
        df_stock = rolling_zscore(df_stock, feature_cols, ROLLING_NORM_WINDOW)
        x_all, y_all = create_sequences(df_stock, feature_cols, multi_step)
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
            plt.title(f'{stock} - LSTM + FinBERT Prediction Results')
            plt.xlabel('Date')
            plt.ylabel('Price (HKD)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        plt.suptitle(plot_title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    return metrics


# ---------------------------------------------------------------------------
# Predict Future Date Prices
# ---------------------------------------------------------------------------
def predict_future_price(data: pd.DataFrame, feature_cols: List[str], target_date: str = "2025-07-02", multi_step: int = 1) -> Dict[str, float]:
    """Predict stock prices for specified date"""
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
        x_all, y_all = create_sequences(df_stock, feature_cols, multi_step)
        
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
            # Use last SEQ_LEN days data for prediction
            last_sequence = df_stock.iloc[-SEQ_LEN:][feature_cols].values
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


# ---------------------------------------------------------------------------
# Main Process
# ---------------------------------------------------------------------------

def main():
    print("Loading price data...")
    prices_long = load_prices()
    print(f"Price data loading completed, total {len(prices_long)} records")

    print("Loading FinBERT sentiment analysis data...")
    sentiment_data = load_sentiment_data()
    if sentiment_data:
        print(f"Sentiment data loading completed, contains {len(sentiment_data)} stocks data")
        prices_long = merge_sentiment_data(prices_long, sentiment_data)
        feature_cols = FEATURES_SENTIMENT
        model_name = "LSTM + FinBERT (with sentiment features)"
    else:
        print("No sentiment data found, using basic features")
        feature_cols = FEATURES_BASE
        model_name = "LSTM Technical Indicator Prediction"

    print(f"\n========== {model_name} ==========")
    metrics = train_and_evaluate(prices_long, feature_cols, f"{model_name} Prediction Results", MULTI_STEP)

    # Predict July 2nd prices
    print("\n========== Predict July 2nd Prices ==========")
    future_prices = predict_future_price(prices_long, feature_cols, "2025-07-02", MULTI_STEP)
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

    summary(f"{model_name} Results", metrics)


if __name__ == "__main__":
    main()
