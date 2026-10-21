"""
Full Automated Pipeline Orchestrator for Market Deep Learning Model
Runs Data Download (5 Years) -> Feature Engineering -> Training -> Evaluation -> Latest Prediction.
"""

import sys
import os
import time

# Pastikan direktori ml ada di sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_model
from evaluate import evaluate_model
from predict import get_prediction

def run_full_pipeline(symbol="GC=F", interval="1d", epochs=60, batch_size=32, use_kalman=False):
    start_time = time.time()
    print("=" * 65)
    print("  MEMULAI PIPELINE DEEP LEARNING: PREDIKSI PASAR (DATA 5 TAHUN)")
    print(f"  Target Aset : {symbol}")
    print(f"  Timeframe   : {interval}")
    print(f"  Kalman Filter : {'AKTIF' if use_kalman else 'NONAKTIF'}")
    print("=" * 65)

    # Langkah 1 & 2: Training
    print("\n>>> TAHAP 1: Training Model Deep Learning (Market Transformer)...")
    train_model(symbol=symbol, interval=interval, epochs=epochs, batch_size=batch_size, use_kalman=use_kalman)

    # Langkah 3: Evaluasi & Backtesting
    print("\n>>> TAHAP 2: Evaluasi Out-of-Sample & Simulasi Backtesting...")
    evaluate_model()

    # Langkah 4: Inferensi Real-time
    print("\n>>> TAHAP 3: Inferensi Real-time Pada Data Terkini...")
    get_prediction(symbol=symbol, interval=interval)

    elapsed = time.time() - start_time
    print(f"\n[SELESAI] Seluruh pipeline selesai dalam {elapsed:.1f} detik.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full Automated Pipeline Orchestrator")
    parser.add_argument("--symbol", type=str, default="GC=F", help="Ticker aset")
    parser.add_argument("--interval", type=str, default="1d", help="Interval timeframe")
    parser.add_argument("--epochs", type=int, default=60, help="Jumlah epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--kalman", dest="use_kalman", action="store_true", default=False, help="Gunakan Kalman Filter")
    args = parser.parse_args()

    run_full_pipeline(
        symbol=args.symbol,
        interval=args.interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_kalman=args.use_kalman
    )
