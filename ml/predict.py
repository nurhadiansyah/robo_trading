"""
Real-time Inference Engine for Market Prediction Deep Learning Pipeline
Supports both 1-Minute Scalping Model (M1 from Kaggle 5Y data) and Daily Model.
Loads checkpointed model, extracts features from latest MT5 market candles,
and produces actionable trading signals (BUY/SELL/HOLD), confidence, and TP/SL levels.
"""

import os
import sys
import json
import pickle
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data_loader import download_5y_data
from features import compute_technical_features, compute_features_m1
from model import MarketTransformer, MarketTransformerM1, MarketAttentionLSTM, MarketAttentionLSTM_M1, TemporalAttention, TemporalAttentionM1

def get_prediction(symbol: str = "GC=F", interval: str = "1d",
                   weights_dir: str = "ml/weights", threshold: float = 0.51,
                   as_json: bool = False):
    
    # Deteksi apakah Model M1 (5 Tahun Kaggle) diminta dan tersedia
    model_m1_path = os.path.join(weights_dir, "model_m1.pth")
    config_m1_path = os.path.join(weights_dir, "config_m1.json")
    scaler_m1_path = os.path.join(weights_dir, "scaler_m1.pkl")
    
    # Gunakan M1 jika interval spesifik 1m dan file M1 lengkap, atau jika model harian belum ada
    daily_model_path = os.path.join(weights_dir, "best_model.pth")
    daily_config_path = os.path.join(weights_dir, "config.json")
    
    is_m1_model = (interval == "1m" or not os.path.exists(daily_model_path)) and os.path.exists(model_m1_path) and os.path.exists(config_m1_path)

    if is_m1_model:
        # Gunakan Model M1 (5 Tahun Kaggle)
        with open(config_m1_path, "r") as f:
            config = json.load(f)
        with open(scaler_m1_path, "rb") as f:
            scaler = pickle.load(f)
        feature_cols = config.get("feature_cols", [])
        seq_length = config.get("seq_len", 30)
        timeframe_name = "1M (Kaggle 5-Year Data)"
        model_weights_file = model_m1_path
        model_is_state_dict_only = True
    else:
        # Gunakan Model Harian (MarketTransformer / Daily)
        config_path = daily_config_path
        model_path = daily_model_path
        scaler_path = os.path.join(weights_dir, "scaler.pkl")

        if not os.path.exists(config_path) or not os.path.exists(model_path):
            err = {"error": "Model weights tidak ditemukan. Silakan jalankan training terlebih dahulu."}
            if as_json:
                print(json.dumps(err))
            return err

        with open(config_path, "r") as f:
            config = json.load(f)
        scaler_data = joblib.load(scaler_path)
        scaler = scaler_data["scaler"]
        feature_cols = scaler_data["feature_cols"]
        seq_length = config.get("seq_length", 30)
        timeframe_name = "1D (5-Year Daily)"
        model_weights_file = model_path
        model_is_state_dict_only = False

    # 1. Unduh Data Candle Terbaru dari MT5
    df = None
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            tf = mt5.TIMEFRAME_M1 if is_m1_model else mt5.TIMEFRAME_D1
            rates = mt5.copy_rates_from_pos("XAUUSD", tf, 0, max(seq_length + 200, 300))
            if rates is not None and len(rates) >= seq_length + 20:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.set_index('time').rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'
                })[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        df = None

    if df is None:
        df = download_5y_data(symbol=symbol, interval=interval, force_refresh=True, silent=True)

    # 2. Hitung Fitur
    if is_m1_model:
        use_kalman = config.get("use_kalman", False) or any(k.startswith("kf_") for k in feature_cols)
        features_df, _ = compute_features_m1(df, use_kalman=use_kalman)
        clean_df = features_df
    else:
        clean_df, _ = compute_technical_features(df)

    if len(clean_df) < seq_length:
        raise ValueError(f"Jumlah bar tidak mencukupi untuk sequence {seq_length}")

    # Ambil sequence paling mutakhir (terbaru)
    recent_features_df = clean_df[feature_cols].iloc[-seq_length:]
    recent_scaled = scaler.transform(recent_features_df.values if hasattr(recent_features_df, 'values') else recent_features_df)

    # 3. Muat Arsitektur Model & Bobot
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config.get('model_type', config.get('model_name', ''))
    is_transformer = ('transformer' in model_type.lower()) or ('d_model' in config) or ('nhead' in config)

    if is_transformer:
        d_model = config.get('d_model', 64)
        nhead = config.get('nhead', 4)
        num_layers = config.get('num_layers', 2)
        model = MarketTransformer(
            input_dim=len(feature_cols),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(device)
    else:
        hidden_dim = config.get('hidden_dim', 64)
        num_layers = config.get('num_layers', 2)
        if is_m1_model:
            model = MarketAttentionLSTM_M1(
                input_dim=len(feature_cols),
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(device)
        else:
            model = MarketAttentionLSTM(
                input_dim=len(feature_cols),
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(device)

    if model_is_state_dict_only:
        loaded = torch.load(model_weights_file, map_location=device)
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            model.load_state_dict(loaded['model_state_dict'])
        else:
            model.load_state_dict(loaded)
    else:
        checkpoint = torch.load(model_weights_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # 4. Eksekusi Forward Pass
    with torch.no_grad():
        x_tensor = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        dir_logits, exp_ret_tensor, attn_weights = model(x_tensor)

        probs = F.softmax(dir_logits, dim=-1).squeeze(0).cpu().numpy()
        prob_down = float(probs[0])
        prob_up = float(probs[1])
        exp_return_pct = float(exp_ret_tensor.item())

    # 5. Tentukan Sinyal dan Level Trading Dinamis
    last_candle = df.iloc[-1]
    current_close = float(last_candle['Close'])
    last_time = str(df.index[-1])

    # Hitung estimasi volatilitas ATR dari data terbaru
    if 'atr_norm' in clean_df:
        atr_val = float(clean_df['atr_norm'].iloc[-1] * current_close)
    else:
        high_low = df['High'] - df['Low']
        atr_val = float(high_low.rolling(14).mean().iloc[-1])
        
    if is_m1_model:
        # Volatilitas M1 realistis untuk scalping: batas min $1.00, maks $8.00
        atr_val = max(1.0, min(atr_val, 8.0))
        sl_points = round(atr_val * 1.0, 2)
        tp_points = round(sl_points * 1.5, 2)
    else:
        atr_val = max(2.5, min(atr_val, 18.0))
        sl_points = round(atr_val * 1.0, 2)
        tp_points = round(sl_points * 1.5, 2)

    if prob_up >= threshold:
        signal = "BUY"
        confidence = prob_up
        direction = "BULLISH"
        tp = current_close + tp_points
        sl = current_close - sl_points
    elif prob_down >= threshold:
        signal = "SELL"
        confidence = prob_down
        direction = "BEARISH"
        tp = current_close - tp_points
        sl = current_close + sl_points
    else:
        signal = "HOLD"
        confidence = max(prob_up, prob_down)
        direction = "NEUTRAL"
        tp = current_close + tp_points
        sl = current_close - sl_points

    result = {
        "symbol": "XAUUSD" if symbol in ["GC=F", "XAUUSD"] else symbol,
        "time": last_time,
        "timeframe": timeframe_name,
        "model_type": config.get("model_type", config.get("model_name", "MarketTransformer")),
        "use_kalman": config.get("use_kalman", False) or any(k.startswith("kf_") for k in feature_cols),
        "current_price": round(current_close, 2),
        "signal": signal,
        "direction": direction,
        "confidence_pct": round(confidence * 100, 2),
        "prob_bullish": round(prob_up * 100, 2),
        "prob_bearish": round(prob_down * 100, 2),
        "expected_return_pct": round(exp_return_pct, 2),
        "atr_volatility": round(atr_val, 2),
        "auto_tp_points": round(tp_points, 2),
        "auto_sl_points": round(sl_points, 2),
        "suggested_tp": round(tp, 2),
        "suggested_sl": round(sl, 2),
        "risk_reward_ratio": "1:1.5"
    }

    # Simpan snapshot prediksi terbaru ke file JSON
    latest_pred_path = os.path.join(weights_dir, "latest_prediction.json")
    try:
        with open(latest_pred_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "="*50)
        print(f"HASIL PREDIKSI DEEP LEARNING ({timeframe_name}): {symbol}")
        print("="*50)
        print(f"Waktu Data Terakhir    : {last_time}")
        print(f"Harga Pasar Saat Ini   : {current_close:.2f}")
        print(f"Sinyal AI Rekomendasi  : {signal} ({direction})")
        print(f"Tingkat Keyakinan      : {result['confidence_pct']}%")
        print(f"Probabilitas Bullish   : {result['prob_bullish']}%")
        print(f"Probabilitas Bearish   : {result['prob_bearish']}%")
        print(f"Target Scalp TP        : +${result['auto_tp_points']} (Level: {result['suggested_tp']})")
        print(f"Batas Scalp SL         : -${result['auto_sl_points']} (Level: {result['suggested_sl']})")
        print("="*50)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Market Direction with Deep Learning")
    parser.add_argument("--symbol", type=str, default="GC=F", help="Ticker pasar (default: GC=F)")
    parser.add_argument("--interval", type=str, default="1d", help="Interval candle (default: 1d)")
    parser.add_argument("--threshold", type=float, default=0.51, help="Confidence threshold untuk sinyal")
    parser.add_argument("--json", action="store_true", help="Output format JSON murni")
    args = parser.parse_args()

    get_prediction(
        symbol=args.symbol,
        interval=args.interval,
        threshold=args.threshold,
        as_json=args.json
    )
