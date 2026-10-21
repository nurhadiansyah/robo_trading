"""
Feature Engineering Module for Deep Learning Market Prediction
Calculates technical indicators, stationary price features, scaling, and sliding windows.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def compute_technical_features(df: pd.DataFrame, use_kalman: bool = False) -> pd.DataFrame:
    """
    Compute stationary technical features and indicators from OHLCV data.
    Avoids raw price levels to maintain model stationarity across bull/bear regimes.
    Includes optional Kinematic Kalman Filter denoising.
    """
    data = df.copy()
    close = data['Close']
    open_p = data['Open']
    high = data['High']
    low = data['Low']
    vol = data['Volume']

    # 1. Price Action & Returns
    data['ret_1'] = close.pct_change(1)
    data['ret_3'] = close.pct_change(3)
    data['ret_5'] = close.pct_change(5)
    data['log_ret'] = np.log(close / close.shift(1))
    data['candle_body'] = (close - open_p) / open_p
    data['candle_range'] = (high - low) / close
    data['upper_shadow'] = (high - np.maximum(close, open_p)) / close
    data['lower_shadow'] = (np.minimum(close, open_p) - low) / close

    # 2. RSI (14)
    data['rsi_14'] = compute_rsi(close, 14) / 100.0  # Scale to [0, 1]

    # 3. Moving Averages Ratios
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    data['dist_ema9'] = (close - ema9) / ema9
    data['dist_ema21'] = (close - ema21) / ema21
    data['dist_ema50'] = (close - ema50) / ema50
    data['dist_ema200'] = (close - ema200) / ema200
    data['ema9_21_cross'] = (ema9 - ema21) / ema21

    # 4. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = (ema12 - ema26) / close
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    data['macd'] = macd_line
    data['macd_signal'] = macd_signal
    data['macd_hist'] = macd_line - macd_signal

    # 5. Bollinger Bands (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    data['bb_pct'] = (close - lower_bb) / (upper_bb - lower_bb + 1e-9)
    data['bb_width'] = (upper_bb - lower_bb) / (sma20 + 1e-9)

    # 6. Volatility & ATR
    atr = compute_atr(high, low, close, 14)
    data['atr_norm'] = atr / (close + 1e-9)

    # 7. Volume Features
    vol_sma20 = vol.rolling(20).mean()
    data['vol_ratio'] = vol / (vol_sma20 + 1e-9)
    data['vol_change'] = vol.pct_change(1).clip(-5.0, 5.0)

    # 8. Target variables for t+1
    # Next close return
    data['next_ret'] = close.shift(-1) / close - 1.0
    # Direction: 1 if Up (Bullish), 0 if Down (Bearish)
    data['target_dir'] = (data['next_ret'] > 0).astype(int)
    # Next Close price
    data['next_close'] = close.shift(-1)

    feature_cols = [
        'ret_1', 'ret_3', 'ret_5', 'log_ret', 'candle_body', 'candle_range',
        'upper_shadow', 'lower_shadow', 'rsi_14', 'dist_ema9', 'dist_ema21',
        'dist_ema50', 'dist_ema200', 'ema9_21_cross', 'macd', 'macd_signal',
        'macd_hist', 'bb_pct', 'bb_width', 'atr_norm', 'vol_ratio', 'vol_change'
    ]

    # 9. Kalman Filter Denoising & Kinematic Momentum
    if use_kalman and extract_kalman_features is not None:
        kf_feats = extract_kalman_features(df, close_col='Close')
        for col in kf_feats.columns:
            data[col] = kf_feats[col]
            if col not in feature_cols:
                feature_cols.append(col)

    clean_df = data.dropna().copy()
    return clean_df, feature_cols

def create_sequences(features_array: np.ndarray, targets_dir: np.ndarray,
                     targets_ret: np.ndarray, raw_close: np.ndarray,
                     seq_length: int = 30):
    """
    Generate sliding window time-series sequences of shape (N, seq_length, num_features).
    """
    X, y_dir, y_ret, close_vals = [], [], [], []
    num_samples = len(features_array) - seq_length

    for i in range(num_samples):
        X.append(features_array[i: i + seq_length])
        y_dir.append(targets_dir[i + seq_length])
        y_ret.append(targets_ret[i + seq_length])
        close_vals.append(raw_close[i + seq_length])

    return (
        np.array(X, dtype=np.float32),
        np.array(y_dir, dtype=np.int64),
        np.array(y_ret, dtype=np.float32),
        np.array(close_vals, dtype=np.float32)
    )

def prepare_dataset(df: pd.DataFrame, seq_length: int = 30,
                    train_ratio: float = 0.70, val_ratio: float = 0.15,
                    scaler_path: str = None, use_kalman: bool = False):
    """
    Full pipeline to compute features, split chronologically (Train/Val/Test),
    fit scaler on train set, and construct sequence arrays.
    """
    clean_df, feature_cols = compute_technical_features(df, use_kalman=use_kalman)
    n = len(clean_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = clean_df.iloc[:train_end]
    val_df = clean_df.iloc[train_end:val_end]
    test_df = clean_df.iloc[val_end:]

    # Penskalaan fitur secara kronologis (FIT HANYA PADA TRAIN untuk cegah kebocoran data)
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])

    if scaler_path:
        joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, scaler_path)
        print(f"[Features] Scaler & nama fitur disimpan ke {scaler_path}")

    # Transform masing-masing split
    X_train_raw = scaler.transform(train_df[feature_cols])
    X_val_raw = scaler.transform(val_df[feature_cols])
    X_test_raw = scaler.transform(test_df[feature_cols])

    # Buat sequence
    X_train, y_dir_train, y_ret_train, c_train = create_sequences(
        X_train_raw, train_df['target_dir'].values, train_df['next_ret'].values, train_df['Close'].values, seq_length
    )
    X_val, y_dir_val, y_ret_val, c_val = create_sequences(
        X_val_raw, val_df['target_dir'].values, val_df['next_ret'].values, val_df['Close'].values, seq_length
    )
    X_test, y_dir_test, y_ret_test, c_test = create_sequences(
        X_test_raw, test_df['target_dir'].values, test_df['next_ret'].values, test_df['Close'].values, seq_length
    )

    dates = {
        'train': (train_df.index.min(), train_df.index.max()),
        'val': (val_df.index.min(), val_df.index.max()),
        'test': (test_df.index.min(), test_df.index.max())
    }

    return {
        'train': (X_train, y_dir_train, y_ret_train, c_train),
        'val': (X_val, y_dir_val, y_ret_val, c_val),
        'test': (X_test, y_dir_test, y_ret_test, c_test),
        'feature_cols': feature_cols,
        'scaler': scaler,
        'dates': dates,
        'raw_df': clean_df
    }


try:
    from kalman import extract_kalman_features
except ImportError:
    try:
        from ml.kalman import extract_kalman_features
    except ImportError:
        extract_kalman_features = None


def compute_features_m1(df: pd.DataFrame, use_kalman: bool = False):
    """
    Kalkulasi fitur teknikal cepat untuk timeframe 1-menit (M1).
    Mendukung opsi Kalman Filter untuk meredam noise pasar mikro.
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    features = pd.DataFrame(index=df.index)
    
    # 1. Log Returns & Multi-Period Momentum
    features['log_ret'] = np.log(close / close.shift(1)).fillna(0)
    features['ret_3'] = ((close - close.shift(3)) / close.shift(3)).fillna(0)
    features['ret_5'] = ((close - close.shift(5)) / close.shift(5)).fillna(0)
    features['ret_15'] = ((close - close.shift(15)) / close.shift(15)).fillna(0)
    
    # 2. Exponential Moving Averages (EMA) Ratios
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()
    
    features['ema_9_ratio'] = (close - ema_9) / ema_9
    features['ema_21_ratio'] = (close - ema_21) / ema_21
    features['ema_50_ratio'] = (close - ema_50) / ema_50
    features['ema_200_ratio'] = (close - ema_200) / ema_200
    features['ema_trend'] = (ema_9 - ema_21) / ema_21
    
    # 3. RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    features['rsi_14'] = (100 - (100 / (1 + rs))) / 100.0
    
    # 4. MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features['macd'] = macd_line / close
    features['macd_signal'] = signal_line / close
    features['macd_hist'] = (macd_line - signal_line) / close
    
    # 5. Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)
    features['bb_pct'] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-9)
    features['bb_width'] = (bb_upper - bb_lower) / bb_mid
    
    # 6. ATR & Volatilitas M1
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    features['atr_norm'] = atr_14 / close
    features['hl_spread'] = hl / close
    
    # 7. Volume relative moving average
    vol_ma = volume.rolling(20).mean() + 1e-5
    features['vol_norm'] = np.log1p(volume / vol_ma)
    
    # 8. Kalman Filter Denoising & Kinematic Momentum
    if use_kalman and extract_kalman_features is not None:
        kf_feats = extract_kalman_features(df, close_col='Close')
        for col in kf_feats.columns:
            features[col] = kf_feats[col]
            
    features = features.bfill().ffill().fillna(0)
    feature_cols = list(features.columns)
    return features, feature_cols


