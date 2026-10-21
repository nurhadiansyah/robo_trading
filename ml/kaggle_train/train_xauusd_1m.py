"""
=============================================================================
XAUUSD 1-Minute (M1) 5-Year Deep Learning Training Engine (Kaggle Cloud GPU)
Model: Bidirectional LSTM + Temporal Self-Attention Mechanism
Trained on NVIDIA Tesla T4 GPU with Mixed Precision (CUDA AMP)
=============================================================================
"""

import os
import sys
import glob
import json
import time
import math
import pickle
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# 1. Dataset Finder & Loader
# ---------------------------------------------------------------------------
def find_dataset_file(filename="XAU_1m_data.csv"):
    search_paths = [
        f"/kaggle/input/*/{filename}",
        f"/kaggle/input/*/*/{filename}",
        f"/kaggle/input/{filename}",
        f"./{filename}",
        f"../{filename}"
    ]
    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
            
    # Recursive search if standard patterns fail
    for root, _, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
            
    raise FileNotFoundError(f"File {filename} tidak ditemukan di direktori /kaggle/input.")

def load_5y_m1_data(filepath):
    print("=" * 70)
    print(f"[Data Loader] Memuat data dari: {filepath}")
    t0 = time.time()
    
    # Baca CSV dengan delimiter titik-koma
    df = pd.read_csv(filepath, sep=';', parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    total_bars = len(df)
    max_date = df['Date'].max()
    min_date_full = df['Date'].min()
    print(f"[Data Loader] Total seluruh bar tersedia: {total_bars:,} bar ({min_date_full.strftime('%Y-%m-%d')} s.d. {max_date.strftime('%Y-%m-%d')})")
    
    # Filter 5 tahun terakhir
    cutoff_date = max_date - pd.DateOffset(years=5)
    df_5y = df[df['Date'] >= cutoff_date].copy().reset_index(drop=True)
    
    print(f"[Data Loader] Data 5 Tahun Terakhir: {len(df_5y):,} candle M1")
    print(f"             Rentang Tanggal: {df_5y['Date'].min().strftime('%Y-%m-%d %H:%M')} s.d. {df_5y['Date'].max().strftime('%Y-%m-%d %H:%M')}")
    print(f"[Data Loader] Waktu baca CSV: {time.time() - t0:.2f} detik")
    print("=" * 70)
    return df_5y

def apply_kinematic_kalman_filter(prices, q_x=1e-4, q_v=1e-5, r=0.05):
    """
    2-State Kinematic Kalman Filter for causal price denoising and trend velocity extraction.
    Zero future-leakage state-space model for high-frequency M1 data.
    """
    n = len(prices)
    if n == 0:
        return np.array([]), np.array([])
    kf_price = np.empty(n, dtype=np.float64)
    kf_vel = np.empty(n, dtype=np.float64)
    x = float(prices[0])
    v = 0.0
    p00, p01, p11 = 1.0, 0.0, 1.0
    for i in range(n):
        # State prediction
        x_pred = x + v
        v_pred = v
        p00_p = p00 + 2.0 * p01 + p11 + q_x
        p01_p = p01 + p11
        p11_p = p11 + q_v
        
        # Measurement update
        y = float(prices[i]) - x_pred
        s = p00_p + r
        k0 = p00_p / s
        k1 = p01_p / s
        x = x_pred + k0 * y
        v = v_pred + k1 * y
        p00 = (1.0 - k0) * p00_p
        p01 = (1.0 - k0) * p01_p
        p11 = p11_p - k1 * p01_p
        kf_price[i] = x
        kf_vel[i] = v
    return kf_price, kf_vel

def compute_features_m1(df, use_kalman: bool = True):
    print(f"\n[Features] Menghitung Indikator Teknikal Timeframe 1-Menit (Kalman Filter: {'AKTIF' if use_kalman else 'NONAKTIF'})...")
    t0 = time.time()
    
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
    features['rsi_14'] = (100 - (100 / (1 + rs))) / 100.0 # Normalisasi 0-1
    
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
    if use_kalman:
        print("[Features] Menerapkan Causal Kinematic Kalman Filter pada harga...")
        kf_p, kf_v = apply_kinematic_kalman_filter(close.values)
        atr_series = atr_14.bfill() + 1e-6
        features['kf_dist'] = (close.values - kf_p) / atr_series.values
        features['kf_vel'] = kf_v / atr_series.values
        kf_accel = np.zeros_like(kf_v)
        kf_accel[1:] = (kf_v[1:] - kf_v[:-1]) / atr_series.values[1:]
        features['kf_accel'] = kf_accel
        features['kf_ratio'] = (close.values - kf_p) / (kf_p + 1e-9)
    
    # 8. Target Labeling (Prediksi Arah 3-Candle M1 ke Depan)
    # 3 menit ke depan memberi ruang pergerakan yang memadai untuk scalping
    target_window = 3
    future_close = close.shift(-target_window)
    future_return = (future_close - close) / close
    
    # Label Klasifikasi: 1 = BUY, 0 = SELL
    labels_cls = (future_return > 0.0).astype(int)
    labels_reg = (future_return * 100.0).astype(float) # persen return
    
    # Bersihkan NaN dari rolling windows dan shift
    valid_idx = ~(features.isna().any(axis=1) | labels_reg.isna())
    clean_features = features[valid_idx].copy().reset_index(drop=True)
    clean_cls = labels_cls[valid_idx].values
    clean_reg = labels_reg[valid_idx].values
    clean_close = close[valid_idx].values
    
    feature_cols = list(clean_features.columns)
    print(f"[Features] Selesai: {len(clean_features):,} bar valid, {len(feature_cols)} kolom fitur.")
    print(f"           Distribusi Label: BUY={np.mean(clean_cls==1)*100:.1f}%, SELL={np.mean(clean_cls==0)*100:.1f}%")
    print(f"           Waktu kalkulasi fitur: {time.time() - t0:.2f} detik")
    
    return clean_features.values, clean_cls, clean_reg, clean_close, feature_cols

# ---------------------------------------------------------------------------
# 3. High-Performance Memory-Efficient PyTorch Dataset
# ---------------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """Memory-efficient sliding window dataset without duplicating 3D tensors in RAM."""
    def __init__(self, X, y_cls, y_reg, seq_len=30):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.seq_len = seq_len
        self.valid_len = len(self.X) - self.seq_len
        
    def __len__(self):
        return self.valid_len
        
    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_len],
            self.y_cls[idx + self.seq_len],
            self.y_reg[idx + self.seq_len]
        )

# ---------------------------------------------------------------------------
# 4. Neural Network Architecture: Bi-LSTM + Temporal Self-Attention
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor):
        scores = self.query(x)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights

class MarketTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2, max_len=500, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.readout_norm = nn.LayerNorm(d_model)
        self.attention_pooling = TemporalAttentionPooling(d_model)

        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x_norm = self.input_norm(x)
        x_proj = self.input_proj(x_norm)
        x_pe = self.pos_encoder(x_proj)

        encoded = self.transformer_encoder(x_pe)
        encoded_norm = self.readout_norm(encoded)

        context, attn_weights = self.attention_pooling(encoded_norm)
        feat = self.shared_fc(context)

        logits = self.classifier(feat)
        exp_return = self.regressor(feat).squeeze(-1)
        return logits, exp_return, attn_weights

MarketAttentionLSTM = MarketTransformer

def train_model(use_kalman: bool = True):
    print("\n" + "=" * 70)
    print("   MEMULAI PELATIHAN MODEL DEEP LEARNING (M1 DATA 5 TAHUN)")
    print(f"   Kalman Filter Denoising: {'AKTIF' if use_kalman else 'NONAKTIF'}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Menggunakan Unit Pemrosesan: {device}")
    if torch.cuda.is_available():
        print(f"[Device] GPU Card: {torch.cuda.get_device_name(0)}")
        print(f"[Device] VRAM Tersedia: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
    # 1. Muat Data
    data_path = find_dataset_file("XAU_1m_data.csv")
    df = load_5y_m1_data(data_path)
    X_raw, y_cls, y_reg, close_prices, feature_cols = compute_features_m1(df, use_kalman=use_kalman)
    
    # 2. Pembagian Kronologis (Train 70%, Val 15%, Test 15%)
    n_total = len(X_raw)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    X_train_raw = X_raw[:n_train]
    y_cls_train = y_cls[:n_train]
    y_reg_train = y_reg[:n_train]
    
    X_val_raw = X_raw[n_train : n_train + n_val]
    y_cls_val = y_cls[n_train : n_train + n_val]
    y_reg_val = y_reg[n_train : n_train + n_val]
    
    X_test_raw = X_raw[n_train + n_val:]
    y_cls_test = y_cls[n_train + n_val:]
    y_reg_test = y_reg[n_train + n_val:]
    
    print(f"\n[Split] Pembagian Data:")
    print(f"        Train Set : {len(X_train_raw):,} candle ({len(X_train_raw)/n_total*100:.1f}%)")
    print(f"        Val Set   : {len(X_val_raw):,} candle ({len(X_val_raw)/n_total*100:.1f}%)")
    print(f"        Test Set  : {len(X_test_raw):,} candle ({len(X_test_raw)/n_total*100:.1f}%)")
    
    # 3. Penskalaan Fitur Kronologis (Hanya fit pada Train)
    print("\n[Scaler] Melakukan RobustScaler normalization...")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 4. Data Loaders
    SEQ_LEN = 30
    BATCH_SIZE = 512 if torch.cuda.is_available() else 128
    
    train_dataset = TimeSeriesDataset(X_train, y_cls_train, y_reg_train, seq_len=SEQ_LEN)
    val_dataset = TimeSeriesDataset(X_val, y_cls_val, y_reg_val, seq_len=SEQ_LEN)
    test_dataset = TimeSeriesDataset(X_test, y_cls_test, y_reg_test, seq_len=SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=2)
    
    # 5. Inisialisasi Model, Loss, Optimizer (Time-Series Transformer M1)
    input_dim = len(feature_cols)
    model = MarketTransformer(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2
    ).to(device)
    print(f"\n[Model Architecture] Inisialisasi MarketTransformer (d_model=64, nhead=4, layers=2, GELU FFN)...")
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    EPOCHS = 12
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 4
    
    best_weights_path = "/kaggle/working/model_m1.pth"
    history = []
    
    print("\n" + "=" * 70)
    print(f"[Training] Memulai Pelatihan Selama Maksimal {EPOCHS} Epoch (Batch: {BATCH_SIZE})...")
    print("=" * 70)
    start_train_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # --- TRAIN LOOP ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_cls, batch_reg in train_loader:
            batch_x = batch_x.to(device)
            batch_cls = batch_cls.to(device)
            batch_reg = batch_reg.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, exp_ret, _ = model(batch_x)
                loss_c = criterion_cls(logits, batch_cls)
                loss_r = criterion_reg(exp_return=exp_ret, target=batch_reg) if False else criterion_reg(exp_ret, batch_reg)
                loss = loss_c + 0.3 * loss_r
                
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            train_loss += loss.item() * len(batch_cls)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == batch_cls).sum().item()
            train_total += len(batch_cls)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_cls, batch_reg in val_loader:
                batch_x = batch_x.to(device)
                batch_cls = batch_cls.to(device)
                batch_reg = batch_reg.to(device)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits, exp_ret, _ = model(batch_x)
                    loss_c = criterion_cls(logits, batch_cls)
                    loss_r = criterion_reg(exp_ret, batch_reg)
                    loss = loss_c + 0.3 * loss_r
                    
                val_loss += loss.item() * len(batch_cls)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_cls).sum().item()
                val_total += len(batch_cls)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0
        scheduler.step(epoch_val_loss)
        
        duration = time.time() - epoch_start
        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 2),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 2),
            "duration_sec": round(duration, 1)
        })
        
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({duration:4.1f}s) | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
        # Save Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            print(f"   >>> [BEST MODEL SAVED] Val Loss berkurang menjadi {best_val_loss:.4f} (Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\n[Early Stopping] Pelatihan dihentikan lebih awal pada epoch {epoch} karena tidak ada perbaikan.")
                break
                
    total_training_time = time.time() - start_train_time
    print(f"\n[Training] Pelatihan selesai dalam: {total_training_time/60:.2f} menit.")
    
    # 6. Evaluasi Out-of-Sample Test Set
    print("\n" + "=" * 70)
    print("   EVALUASI OUT-OF-SAMPLE TEST SET (DATA M1 2025-2026)")
    print("=" * 70)
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_cls, _ in test_loader:
            batch_x = batch_x.to(device)
            logits, _, _ = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_cls.numpy())
            test_correct += (preds == batch_cls.numpy()).sum()
            test_total += len(batch_cls)
            
    test_acc = (test_correct / test_total) * 100.0
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    tp_buy = np.sum((all_preds == 1) & (all_targets == 1))
    fp_buy = np.sum((all_preds == 1) & (all_targets == 0))
    buy_precision = (tp_buy / (tp_buy + fp_buy + 1e-9)) * 100.0
    
    tp_sell = np.sum((all_preds == 0) & (all_targets == 0))
    fp_sell = np.sum((all_preds == 0) & (all_targets == 1))
    sell_precision = (tp_sell / (tp_sell + fp_sell + 1e-9)) * 100.0
    
    print(f"[Test Set Result] Out-of-Sample Accuracy: {test_acc:.2f}%")
    print(f"                  BUY Precision         : {buy_precision:.2f}%")
    print(f"                  SELL Precision        : {sell_precision:.2f}%")
    print(f"                  Total Evaluated Bars  : {test_total:,} candle")
    
    # 7. Simpan Semua Artefak ke /kaggle/working/
    os.makedirs("/kaggle/working", exist_ok=True)
    
    # Simpan Scaler
    with open("/kaggle/working/scaler_m1.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    # Simpan Konfigurasi Fitur & Arsitektur
    config = {
        "model_name": "MarketTransformer",
        "model_type": "MarketTransformer",
        "timeframe": "1m",
        "data_period": "5_years",
        "total_candles": len(df),
        "seq_len": SEQ_LEN,
        "input_dim": input_dim,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "use_kalman": use_kalman,
        "kalman_features": ["kf_dist", "kf_vel", "kf_accel", "kf_ratio"] if use_kalman else [],
        "feature_cols": feature_cols,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("/kaggle/working/config_m1.json", "w") as f:
        json.dump(config, f, indent=2)
        
    # Simpan Metrik
    metrics = {
        "status": "success",
        "use_kalman": use_kalman,
        "val_loss": round(best_val_loss, 4),
        "val_accuracy": round(best_val_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "buy_precision": round(buy_precision, 2),
        "sell_precision": round(sell_precision, 2),
        "total_bars_trained": n_train,
        "total_bars_evaluated": test_total,
        "training_time_minutes": round(total_training_time / 60, 2),
        "history": history
    }
    with open("/kaggle/working/metrics_m1.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("\n" + "=" * 70)
    print(">>> SELURUH ARTEFAK BERHASIL DISIMPAN KE /kaggle/working/ <<<")
    print(f"  - model_m1.pth (Input dim: {input_dim}, Kalman: {use_kalman})")
    print("  - scaler_m1.pkl")
    print("  - config_m1.json")
    print("  - metrics_m1.json")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 5-Year XAUUSD M1 Transformer with optional Kalman Filter")
    parser.add_argument("--kalman", dest="use_kalman", action="store_true", default=True, help="Aktifkan fitur Kinematic Kalman Filter (default: True)")
    parser.add_argument("--no-kalman", dest="use_kalman", action="store_false", help="Nonaktifkan fitur Kalman Filter")
    args = parser.parse_args()
    train_model(use_kalman=args.use_kalman)

