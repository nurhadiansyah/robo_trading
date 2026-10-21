"""
Training Engine for Market Prediction Deep Learning Pipeline
Includes Chronological TimeSeries Split, Multi-Task Loss, Early Stopping, and Model Checkpointing.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_loader import download_5y_data
from features import prepare_dataset
from model import MarketTransformer

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(symbol: str = "GC=F", interval: str = "1d", seq_length: int = 30,
                epochs: int = 60, batch_size: int = 32, lr: float = 1e-3,
                patience: int = 12, weights_dir: str = "ml/weights",
                use_kalman: bool = False):
    set_seed(42)
    os.makedirs(weights_dir, exist_ok=True)
    scaler_path = os.path.join(weights_dir, "scaler.pkl")

    # 1. Download 5-Year Data
    print(f"\n[1/5] Mengunduh & memvalidasi data historis 5 tahun ({symbol} - {interval})...")
    df = download_5y_data(symbol=symbol, interval=interval)

    # 2. Extract Features & Split
    print(f"\n[2/5] Menghitung indikator teknikal & membagi data kronologis (Kalman: {use_kalman})...")
    data_dict = prepare_dataset(
        df, seq_length=seq_length, train_ratio=0.70, val_ratio=0.15, scaler_path=scaler_path, use_kalman=use_kalman
    )

    X_train, y_dir_tr, y_ret_tr, c_tr = data_dict['train']
    X_val, y_dir_val, y_ret_val, c_val = data_dict['val']
    X_test, y_dir_test, y_ret_test, c_test = data_dict['test']
    feature_cols = data_dict['feature_cols']
    num_features = len(feature_cols)

    print(f"Data Split Summary:")
    print(f" - Train samples: {len(X_train)} (Rentang: {data_dict['dates']['train'][0].date()} s/d {data_dict['dates']['train'][1].date()})")
    print(f" - Val samples:   {len(X_val)} (Rentang: {data_dict['dates']['val'][0].date()} s/d {data_dict['dates']['val'][1].date()})")
    print(f" - Test samples:  {len(X_test)} (Rentang: {data_dict['dates']['test'][0].date()} s/d {data_dict['dates']['test'][1].date()})")
    print(f" - Fitur input:   {num_features} indikator teknikal")

    # DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_dir_tr), torch.tensor(y_ret_tr)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_dir_val), torch.tensor(y_ret_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model & Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[3/5] Menginisialisasi Model Deep Learning pada device: {device}")
    if device.type == 'cuda':
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")

    model = MarketTransformer(
        input_dim=num_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.25
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.HuberLoss(delta=0.01)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )

    # 4. Training Loop
    print(f"\n[4/5] Memulai Proses Training ({epochs} epoch, patience={patience})...")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    best_model_path = os.path.join(weights_dir, "best_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0

        for batch_x, batch_ydir, batch_yret in train_loader:
            batch_x = batch_x.to(device)
            batch_ydir = batch_ydir.to(device)
            batch_yret = batch_yret.to(device)

            optimizer.zero_grad()
            logits, pred_ret, _ = model(batch_x)

            loss_cls = criterion_cls(logits, batch_ydir)
            loss_reg = criterion_reg(pred_ret, batch_yret)
            loss = loss_cls + 5.0 * loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item() * len(batch_x)

        avg_train_loss = train_loss_total / len(train_dataset)

        # Validation
        model.eval()
        val_loss_total = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_ydir, batch_yret in val_loader:
                batch_x = batch_x.to(device)
                batch_ydir = batch_ydir.to(device)
                batch_yret = batch_yret.to(device)

                logits, pred_ret, _ = model(batch_x)
                loss_cls = criterion_cls(logits, batch_ydir)
                loss_reg = criterion_reg(pred_ret, batch_yret)
                loss = loss_cls + 5.0 * loss_reg

                val_loss_total += loss.item() * len(batch_x)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_ydir.cpu().numpy())

        avg_val_loss = val_loss_total / len(val_dataset)
        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)

        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if epoch % 5 == 0 or epoch == 1 or avg_val_loss < best_val_loss:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc*100:.1f}% | Val F1: {val_f1:.3f}")

        # Checkpoint Best Model
        is_best = (val_acc > best_val_acc) or (abs(val_acc - best_val_acc) < 1e-4 and avg_val_loss < best_val_loss)
        if is_best:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_dim': num_features,
                'd_model': 64,
                'nhead': 4,
                'num_layers': 2,
                'model_type': 'MarketTransformer',
                'seq_length': seq_length,
                'val_loss': avg_val_loss,
                'val_acc': val_acc
            }, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EarlyStopping] Stop di epoch {epoch} (tidak ada peningkatan loss val selama {patience} epoch).")
                break

    # Save Config
    config = {
        "model_type": "MarketTransformer",
        "symbol": symbol,
        "interval": interval,
        "seq_length": seq_length,
        "num_features": num_features,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "use_kalman": use_kalman,
        "feature_cols": feature_cols,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "data_range_start": str(df.index.min()),
        "data_range_end": str(df.index.max()),
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = os.path.join(weights_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # 5. Plot Training Curves
    print(f"\n[5/5] Menyimpan grafik performa ke {weights_dir}/training_curve.png...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#3b82f6')
    plt.plot(history['val_loss'], label='Val Loss', color='#ef4444')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot([a * 100 for a in history['val_acc']], label='Val Direction Acc (%)', color='#10b981')
    plt.title('Validation Directional Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(weights_dir, "training_curve.png"), dpi=150)
    plt.close()

    print(f"[Sukses] Model terbaik tersimpan di: {best_model_path}")
    return model, data_dict, config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deep Learning Market Model")
    parser.add_argument("--symbol", type=str, default="GC=F", help="Ticker pasar (default: GC=F Gold)")
    parser.add_argument("--interval", type=str, default="1d", help="Interval candle (default: 1d)")
    parser.add_argument("--epochs", type=int, default=60, help="Jumlah epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=30, help="Panjang sequence lookback")
    parser.add_argument("--kalman", dest="use_kalman", action="store_true", default=False, help="Aktifkan Kalman Filter Denoising")
    args = parser.parse_args()

    train_model(
        symbol=args.symbol,
        interval=args.interval,
        seq_length=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_kalman=args.use_kalman
    )
