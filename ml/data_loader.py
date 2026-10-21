"""
Data Loader Module for Market Prediction Deep Learning Pipeline
Fetches 5 years of historical market data (OHLCV) via yfinance or MT5 CSV export.
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

DEFAULT_SYMBOL = "GC=F"  # Gold Futures (XAUUSD equivalent)
DEFAULT_INTERVAL = "1d"  # 1-day candles

def get_5y_date_range():
    """Calculate the 5-year start and end dates from current date."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365 + 10)  # Add padding for holidays
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def download_5y_data(symbol: str = DEFAULT_SYMBOL, interval: str = DEFAULT_INTERVAL,
                     cache_dir: str = "ml/data", force_refresh: bool = False,
                     silent: bool = False) -> pd.DataFrame:
    """
    Download 5 years of OHLCV historical market data.
    Caches the data to CSV to prevent unnecessary downloads.
    """
    os.makedirs(cache_dir, exist_ok=True)
    clean_sym = symbol.replace("=", "_").replace("^", "_").replace("/", "_")
    cache_path = os.path.join(cache_dir, f"{clean_sym}_{interval}_5y.csv")

    if os.path.exists(cache_path) and not force_refresh:
        if not silent:
            print(f"[DataLoader] Membaca data cache dari {cache_path}...")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if not silent:
            print(f"[DataLoader] Terbaca {len(df)} candle dari {df.index.min()} sampai {df.index.max()}.")
        return df

    start_date, end_date = get_5y_date_range()
    if not silent:
        print(f"[DataLoader] Mengunduh data 5 tahun untuk {symbol} ({interval}) dari {start_date} hingga {end_date}...")

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty or len(df) < 50:
        alt_sym = "GLD" if symbol == "GC=F" else symbol
        if not silent:
            print(f"[DataLoader] Data kosong untuk {symbol}, mencoba ticker alternatif: {alt_sym}...")
        ticker = yf.Ticker(alt_sym)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        raise ValueError(f"Gagal mengunduh data untuk ticker {symbol}. Periksa koneksi internet atau nama ticker.")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index().drop_duplicates()
    df = df.dropna()

    df.to_csv(cache_path)
    if not silent:
        print(f"[DataLoader] Berhasil mengunduh & menyimpan {len(df)} candle ke {cache_path}")
        print(f"[DataLoader] Rentang Waktu: {df.index.min().strftime('%Y-%m-%d')} s/d {df.index.max().strftime('%Y-%m-%d')}")
    return df

def load_from_csv(csv_path: str) -> pd.DataFrame:
    """Load historical OHLCV data from a custom CSV file (e.g. from MT5)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File CSV tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)
    # Standarisasi nama kolom
    col_map = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=col_map)

    time_col = None
    for candidate in ['Time', 'Date', 'Datetime']:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    required = ['Open', 'High', 'Low', 'Close']
    for req in required:
        if req not in df.columns:
            raise ValueError(f"Kolom {req} tidak ditemukan dalam file CSV.")

    if 'Volume' not in df.columns:
        df['Volume'] = 1.0

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index().dropna()
    print(f"[DataLoader] Terbaca {len(df)} bar dari file custom {csv_path}")
    return df

if __name__ == "__main__":
    df = download_5y_data()
    print("Contoh 5 baris pertama:")
    print(df.head())
    print("\nContoh 5 baris terakhir:")
    print(df.tail())
