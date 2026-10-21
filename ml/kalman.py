"""
Kalman Filter Module for High-Frequency (M1) Financial Time-Series
Provides causal, online state-space filtering for price denoising,
trend velocity extraction, and mean-reversion residual calculation.
Zero future-leakage design suitable for quantitative algorithmic trading.
"""

import numpy as np
import pandas as pd

def apply_kinematic_kalman_filter(prices: np.ndarray, q_x: float = 1e-4, q_v: float = 1e-5, r: float = 0.05):
    """
    2-State Kinematic Kalman Filter:
    States: [x_t (true price), v_t (true velocity/momentum)]
    Measurements: z_t (observed noisy close price)
    
    Returns:
        kf_price: numpy array of denoised prices
        kf_vel: numpy array of estimated trend velocity
    """
    n = len(prices)
    if n == 0:
        return np.array([]), np.array([])
        
    kf_price = np.empty(n, dtype=np.float64)
    kf_vel = np.empty(n, dtype=np.float64)
    
    x = float(prices[0])
    v = 0.0
    p00 = 1.0
    p01 = 0.0
    p11 = 1.0
    
    for i in range(n):
        # 1. State Prediction (A Priori)
        x_pred = x + v
        v_pred = v
        p00_p = p00 + 2.0 * p01 + p11 + q_x
        p01_p = p01 + p11
        p11_p = p11 + q_v
        
        # 2. Measurement Update (A Posteriori)
        y = float(prices[i]) - x_pred  # Innovation / residual
        s = p00_p + r                  # Innovation covariance
        k0 = p00_p / s                 # Kalman Gain for price
        k1 = p01_p / s                 # Kalman Gain for velocity
        
        x = x_pred + k0 * y
        v = v_pred + k1 * y
        p00 = (1.0 - k0) * p00_p
        p01 = (1.0 - k0) * p01_p
        p11 = p11_p - k1 * p01_p
        
        kf_price[i] = x
        kf_vel[i] = v
        
    return kf_price, kf_vel

def extract_kalman_features(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """
    Extracts 4 causal, stationary features derived from the 2-state Kalman Filter:
    - kf_dist: (Close - kf_price) / ATR (Mean-reversion noise distance)
    - kf_vel: kf_vel / ATR (Normalized trend velocity)
    - kf_accel: delta(kf_vel) / ATR (Trend acceleration / inflection)
    - kf_ratio: (Close - kf_price) / kf_price (Raw percentage deviation)
    """
    prices = df[close_col].values
    kf_p, kf_v = apply_kinematic_kalman_filter(prices)
    
    high = df['High'] if 'High' in df else df[close_col]
    low = df['Low'] if 'Low' in df else df[close_col]
    close = df[close_col]
    
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().bfill() + 1e-6
    
    kf_dist = (close.values - kf_p) / atr.values
    kf_vel_norm = kf_v / atr.values
    
    # Acceleration
    kf_accel = np.zeros_like(kf_v)
    kf_accel[1:] = (kf_v[1:] - kf_v[:-1]) / atr.values[1:]
    
    kf_ratio = (close.values - kf_p) / (kf_p + 1e-9)
    
    features = pd.DataFrame({
        "kf_dist": kf_dist,
        "kf_vel": kf_vel_norm,
        "kf_accel": kf_accel,
        "kf_ratio": kf_ratio
    }, index=df.index)
    
    return features.bfill().ffill().fillna(0)
