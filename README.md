# robo_trading

Sistem Algorithmic & AI Trading terintegrasi untuk XAU/USD (Gold) yang menggabungkan model Deep Learning (LSTM & Kalman Filter), integrasi MetaTrader 5 (MT5 Bridge), serta Web Dashboard pemantauan real-time.

---

## 🌟 Fitur Utama

1. **AI / Deep Learning Prediction Engine (`ml/`)**:
   - Model LSTM untuk memprediksi arah tren dan pergerakan harga instrumen (XAU/USD).
   - Kalman Filter untuk meredam noise pasar.
   - Ekstraksi indikator teknikal otomatis (RSI, MACD, Bollinger Bands, ATR, dll.).
   - Pipeline pelatihan lokal & integrasi cloud training via Kaggle (`ml/kaggle_train/`).

2. **MetaTrader 5 Bridge (`MT5_Bridge.mq5` & `ml/bot_bridge.py`)**:
   - Skrip Expert Advisor (EA) MQL5 untuk mengirimkan tick / bar data dan mengeksekusi order langsung ke MT5.
   - Komunikasi socket real-time dua arah antara Python ML Engine dan terminal MT5.

3. **Web Dashboard Interaktif (`src/` & `web_server.py`)**:
   - Visualisasi candlestick interaktif berbasis TradingView Lightweight Charts.
   - Panel sinyal AI (probabilitas Buy/Sell/Hold, confidence score, dan level SL/TP).
   - Statistik performa dan log eksekusi trading real-time.

---

## 📂 Struktur Repositori

```text
robo-trading/
├── ml/                      # Machine Learning & AI Trading Engine
│   ├── bot_bridge.py        # Bridge engine antara AI dan MT5
│   ├── model.py             # Arsitektur Neural Network (LSTM)
│   ├── features.py          # Feature engineering & teknikal indikator
│   ├── kalman.py            # Filter peredam noise harga
│   ├── train.py             # Script pelatihan model lokal
│   ├── evaluate.py          # Evaluasi model & backtest
│   ├── sync_kaggle.py       # Sinkronisasi model dari Kaggle
│   └── weights/             # Bobot model terlatih & metrik
├── MT5_Bridge.mq5           # EA / Script MQL5 untuk MetaTrader 5
├── web_server.py            # Backend Python Web Server & API
├── src/                     # Frontend React + Vite Dashboard
│   ├── components/          # Komponen UI (TradingPanel, AIPanel, ChartArea)
│   └── App.jsx
├── package.json             # Dependensi frontend
└── README.md
```

---

## 🚀 Panduan Memulai

### 1. Menjalankan Backend & Bot Bridge
Pastikan dependensi Python terpasang:
```bash
pip install -r ml/requirements.txt
```

Jalankan bot bridge:
```bash
python ml/bot_bridge.py
```

### 2. Menjalankan Web Dashboard
Jalankan server web frontend:
```bash
# Menggunakan web_server.py
python web_server.py

# Atau menggunakan Vite dev server
npm install
npm run dev
```

### 3. Menghubungkan ke MetaTrader 5
1. Salin `MT5_Bridge.mq5` ke folder `MQL5/Experts` atau `MQL5/Scripts` di direktori MetaTrader 5 Anda.
2. Compile file di MetaEditor untuk menghasilkan `MT5_Bridge.ex5`.
3. Pasang EA ke chart XAUUSD pada timeframe yang diinginkan dan pastikan opsi **"Allow Algo Trading"** dan **"Allow WebRequest"** aktif.
