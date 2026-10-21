"""
Evaluation and Backtest Simulation Module for Deep Learning Market Model
Evaluates out-of-sample test data, computes classification metrics and simulated trading performance.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_loader import download_5y_data
from features import compute_technical_features, create_sequences
from model import MarketTransformer

def evaluate_model(weights_dir: str = "ml/weights", threshold: float = 0.53):
    config_path = os.path.join(weights_dir, "config.json")
    model_path = os.path.join(weights_dir, "best_model.pth")
    scaler_path = os.path.join(weights_dir, "scaler.pkl")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model belum dilatih. Jalankan train.py terlebih dahulu.")

    with open(config_path, "r") as f:
        config = json.load(f)

    scaler_data = joblib.load(scaler_path)
    scaler = scaler_data["scaler"]
    feature_cols = scaler_data["feature_cols"]

    # 1. Unduh Data & Ekstraksi Fitur
    symbol = config["symbol"]
    interval = config["interval"]
    seq_length = config["seq_length"]

    print(f"\n[Evaluate] Menyiapkan data evaluasi untuk {symbol} ({interval})...")
    df = download_5y_data(symbol=symbol, interval=interval)
    clean_df, _ = compute_technical_features(df)

    # Ambil 15% bagian akhir sebagai Test Set (Out-of-sample)
    test_start_idx = int(len(clean_df) * 0.85)
    test_df = clean_df.iloc[test_start_idx:].copy()

    X_scaled = scaler.transform(test_df[feature_cols])
    X_test, y_dir_test, y_ret_test, close_test = create_sequences(
        X_scaled, test_df['target_dir'].values, test_df['next_ret'].values, test_df['Close'].values, seq_length
    )

    test_dates = test_df.index[seq_length:]

    print(f"[Evaluate] Jumlah candle uji (Out-of-Sample): {len(X_test)}")
    print(f"[Evaluate] Periode Uji: {test_dates.min().strftime('%Y-%m-%d')} s/d {test_dates.max().strftime('%Y-%m-%d')}")

    # 2. Muat Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)

    model = MarketTransformer(
        input_dim=len(feature_cols),
        d_model=checkpoint.get('d_model', checkpoint.get('hidden_dim', 64)),
        nhead=checkpoint.get('nhead', 4),
        num_layers=checkpoint.get('num_layers', 2)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Inferensi pada Test Set
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits, exp_rets, attn = model(x_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        pred_dirs = np.argmax(probs, axis=1)

    # 4. Metrik Klasifikasi
    acc = accuracy_score(y_dir_test, pred_dirs)
    prec_bull = precision_score(y_dir_test, pred_dirs, pos_label=1, zero_division=0)
    rec_bull = recall_score(y_dir_test, pred_dirs, pos_label=1, zero_division=0)
    f1_bull = f1_score(y_dir_test, pred_dirs, pos_label=1, zero_division=0)

    prec_bear = precision_score(y_dir_test, pred_dirs, pos_label=0, zero_division=0)
    rec_bear = recall_score(y_dir_test, pred_dirs, pos_label=0, zero_division=0)
    f1_bear = f1_score(y_dir_test, pred_dirs, pos_label=0, zero_division=0)

    cm = confusion_matrix(y_dir_test, pred_dirs)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)

    # Akurasi pada sinyal berkeyakinan tinggi
    max_probs = np.max(probs, axis=1)
    high_conf_mask = max_probs >= threshold
    if np.sum(high_conf_mask) > 0:
        high_conf_acc = accuracy_score(y_dir_test[high_conf_mask], pred_dirs[high_conf_mask])
        high_conf_count = int(np.sum(high_conf_mask))
    else:
        high_conf_acc = acc
        high_conf_count = len(X_test)

    print("\n" + "="*55)
    print("      HASIL EVALUASI DETAIL OUT-OF-SAMPLE (DATA UJI)")
    print("="*55)
    print(f"Total Sampel Uji          : {len(X_test)} candle")
    print(f"Overall Direction Accuracy: {acc * 100:.2f}%")
    print(f"High-Confidence Accuracy  : {high_conf_acc * 100:.2f}% ({high_conf_count} candle dengan conf >= {threshold*100:.0f}%)")
    print("-" * 55)
    print(f"Bullish (BUY / Naik)     :")
    print(f"  - Precision             : {prec_bull * 100:.2f}%")
    print(f"  - Recall                : {rec_bull * 100:.2f}%")
    print(f"  - F1-Score              : {f1_bull:.4f}")
    print(f"Bearish (SELL / Turun)   :")
    print(f"  - Precision             : {prec_bear * 100:.2f}%")
    print(f"  - Recall                : {rec_bear * 100:.2f}%")
    print(f"  - F1-Score              : {f1_bear:.4f}")
    print("-" * 55)
    print(f"Matriks Kontingensi (Confusion Matrix):")
    print(f"  True Bearish (Pred Down, Riil Down): {tn}")
    print(f"  False Bullish(Pred Up, Riil Down)  : {fp}")
    print(f"  False Bearish(Pred Down, Riil Up)  : {fn}")
    print(f"  True Bullish (Pred Up, Riil Up)    : {tp}")

    # 5. Simulasi Backtest Sederhana
    prob_ups = probs[:, 1]
    prob_downs = probs[:, 0]

    signals = np.zeros(len(X_test))
    signals[prob_ups >= threshold] = 1.0     # BUY
    signals[prob_downs >= threshold] = -1.0  # SELL

    strat_returns = signals * y_ret_test
    cum_strat = np.cumprod(1 + strat_returns) - 1.0
    cum_bh = np.cumprod(1 + y_ret_test) - 1.0

    total_strat_return = cum_strat[-1] * 100 if len(cum_strat) > 0 else 0
    total_bh_return = cum_bh[-1] * 100 if len(cum_bh) > 0 else 0

    trades = signals != 0
    trade_returns = strat_returns[trades]
    win_trades = trade_returns > 0
    win_rate = (np.sum(win_trades) / len(trade_returns) * 100) if len(trade_returns) > 0 else 0

    print("\n" + "="*55)
    print("      HASIL SIMULASI TRADING STRATEGI AI")
    print("="*55)
    print(f"Total Transaksi Eksekusi : {np.sum(trades)} dari {len(X_test)} candle")
    print(f"Win Rate Sinyal          : {win_rate:.2f}%")
    print(f"Total Return AI Strategy : {total_strat_return:+.2f}%")
    print(f"Total Return Buy & Hold  : {total_bh_return:+.2f}%")
    print("="*55)

    # Visualisasi Matriks Kebingungan (Confusion Matrix Plot)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix: Prediksi vs Riil Pasar', fontsize=12, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Bearish (0)', 'Bullish (1)'])
    plt.yticks(tick_marks, ['Bearish (0)', 'Bullish (1)'])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0
            plt.text(j, i, format(val, 'd'),
                     horizontalalignment="center",
                     color="white" if val > thresh else "black",
                     fontsize=14, fontweight='bold')

    plt.ylabel('Kondisi Riil Pasar')
    plt.xlabel('Prediksi Model Deep Learning')
    plt.tight_layout()
    cm_path = os.path.join(weights_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 6. Simpan Visualisasi
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, cum_strat * 100, label=f'AI Deep Learning Model ({total_strat_return:+.1f}%)', color='#10b981', linewidth=2)
    plt.plot(test_dates, cum_bh * 100, label=f'Buy & Hold Benchmark ({total_bh_return:+.1f}%)', color='#94a3b8', linestyle='--', linewidth=1.5)
    plt.title(f'Simulasi Backtest Out-of-Sample: {symbol} ({interval})', fontsize=14, fontweight='bold')
    plt.xlabel('Tanggal', fontsize=11)
    plt.ylabel('Kumulatif Return (%)', fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(weights_dir, "backtest_simulation.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Evaluate] Grafik simulasi trading disimpan di: {plot_path}")

    # Simpan hasil dalam JSON
    eval_results = {
        "symbol": symbol,
        "test_period": f"{test_dates.min().strftime('%Y-%m-%d')} - {test_dates.max().strftime('%Y-%m-%d')}",
        "num_test_candles": int(len(X_test)),
        "accuracy": float(acc),
        "precision_bullish": float(prec_bull),
        "recall_bullish": float(rec_bull),
        "f1_bullish": float(f1_bull),
        "precision_bearish": float(prec_bear),
        "recall_bearish": float(rec_bear),
        "f1_bearish": float(f1_bear),
        "high_confidence_accuracy": float(high_conf_acc),
        "total_trades": int(np.sum(trades)),
        "win_rate": float(win_rate),
        "strat_return_pct": float(total_strat_return),
        "bh_return_pct": float(total_bh_return)
    }
    with open(os.path.join(weights_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    return eval_results

if __name__ == "__main__":
    evaluate_model()
