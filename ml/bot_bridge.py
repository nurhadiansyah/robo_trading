"""
Direct Python Automated Trading Bridge & Position Manager
Automates both OPENING and CLOSING positions in MetaTrader 5 using Deep Learning signals.
"""

import os
import sys

# Ensure ml directory is always in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import json
import argparse
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data_loader import download_5y_data
from features import compute_technical_features, compute_features_m1
from model import MarketTransformer, MarketAttentionLSTM, MarketAttentionLSTM_M1

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

def get_symbol_filling_mode(symbol):
    """Mendapatkan mode filling order yang didukung broker untuk simbol tertentu."""
    if not MT5_AVAILABLE:
        return 2
    info = mt5.symbol_info(symbol)
    if info:
        if info.filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        elif info.filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN

MAGIC_NUMBER = 234000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "ml", "weights")
FLAG_FILE = os.path.join(WEIGHTS_DIR, "bot_stop.flag")
PID_FILE = os.path.join(WEIGHTS_DIR, "bot.pid")

def get_open_positions(symbol="XAUUSD"):
    """Mengambil daftar posisi terbuka di MT5 untuk simbol tertentu."""
    if not MT5_AVAILABLE:
        return []
    pos = mt5.positions_get(symbol=symbol)
    return list(pos) if pos else []

def close_position(ticket, symbol="XAUUSD"):
    """Menutup 1 posisi tertentu di MT5 berdasarkan nomor tiket."""
    if not MT5_AVAILABLE:
        return False

    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return False
    p = pos[0]

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False

    close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask

    filling_mode = get_symbol_filling_mode(symbol)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(p.volume),
        "type": close_type,
        "position": int(p.ticket),
        "price": float(close_price),
        "deviation": 25,
        "magic": MAGIC_NUMBER,
        "comment": "RoboTrade AI Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        profit_usd = p.profit
        print(f">>> [MT5 POSISI DITUTUP] Ticket: {p.ticket} | Tipe: {'BUY' if p.type==0 else 'SELL'} | Profit: ${profit_usd:+.2f} USD")
        return True
    else:
        err = result.comment if result else mt5.last_error()
        print(f"[MT5 Gagal Tutup] Ticket {ticket}: {err}")
        return False

def close_all_positions(symbol="XAUUSD"):
    """Menutup seluruh posisi terbuka untuk simbol tertentu."""
    positions = get_open_positions(symbol)
    if not positions:
        print(f"[MT5] Tidak ada posisi terbuka untuk {symbol}.")
        return 0

    print(f"\n[MT5] Menutup {len(positions)} posisi terbuka...")
    count = 0
    for p in positions:
        if close_position(p.ticket, symbol):
            count += 1
    print(f"[MT5] Berhasil menutup {count} posisi.\n")
    return count

def open_market_order(symbol, order_type, volume, tp_points=4.0, sl_points=2.5, comment="RoboTrade Deep Learning"):
    """Membuka posisi pasar (BUY atau SELL) dengan TP & SL presisi."""
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"[Error] Gagal mendapatkan tick harga untuk {symbol}")
        return None

    is_buy = (order_type == "BUY")
    action_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    price = tick.ask if is_buy else tick.bid

    tp = round(price + tp_points, 2) if is_buy else round(price - tp_points, 2)
    sl = round(price - sl_points, 2) if is_buy else round(price + sl_points, 2)

    filling_mode = get_symbol_filling_mode(symbol)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": action_type,
        "price": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 25,
        "magic": MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print("=" * 60)
        print(f">>> [MT5 POSISI RESMI TERBUKA!] <<<")
        print(f"    Ticket Order : {result.order}")
        print(f"    Jenis        : {order_type}")
        print(f"    Volume       : {result.volume} Lot")
        print(f"    Harga Entry  : ${result.price:.2f}")
        print(f"    Take Profit  : ${tp:.2f}")
        print(f"    Stop Loss    : ${sl:.2f}")
        print("=" * 60)
        return result
    else:
        err = result.comment if result else mt5.last_error()
        print(f"[MT5 Error Order] Kode {result.retcode if result else 'N/A'}: {err}")
        return None

def run_trading_bot(symbol="GC=F", mt5_symbol="XAUUSD", interval="1d",
                     lot_size=0.01, threshold=0.52, check_interval_sec=10,
                     tp_points=5.0, sl_points=3.0, auto_tpsl=True, rr_ratio=1.5, paper_trade=False):
    print("=" * 65)
    print("   ROBOT TRADING DEEP LEARNING: OPEN & CLOSE OTOMATIS")
    print(f"   Target Pasar       : {symbol} / MT5: {mt5_symbol}")
    print(f"   Mode Eksekusi      : {'PAPER TRADING (Simulasi)' if paper_trade else '[LIVE REAL MT5 TRADING]'}")
    print(f"   Lot Volume         : {lot_size} Lot")
    print(f"   Mode TP / SL       : {'🤖 Otomatis AI (ATR Volatilitas R:R 1:' + str(rr_ratio) + ')' if auto_tpsl else f'Manual (TP +${tp_points} / SL -${sl_points})'}")
    print(f"   Confidence Ambang  : {threshold * 100:.1f}%")
    print("=" * 65)

    weights_dir = WEIGHTS_DIR
    model_m1_path = os.path.join(weights_dir, "model_m1.pth")
    config_m1_path = os.path.join(weights_dir, "config_m1.json")
    scaler_m1_path = os.path.join(weights_dir, "scaler_m1.pkl")
    is_m1 = os.path.exists(model_m1_path) and os.path.exists(config_m1_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_m1:
        with open(config_m1_path, "r") as f:
            config = json.load(f)
        with open(scaler_m1_path, "rb") as f:
            scaler = pickle.load(f)
        feature_cols = config.get("feature_cols", [])
        seq_length = config.get("seq_len", 30)
        timeframe_str = "1M (Kaggle 5Y)"
        m1_weights = torch.load(model_m1_path, map_location=device)
        is_m1_transformer = any('transformer_encoder' in k for k in m1_weights.keys()) if isinstance(m1_weights, dict) else False
        if is_m1_transformer:
            d_model = config.get('d_model', 64)
            nhead = config.get('nhead', 4)
            num_layers = config.get('num_layers', 2)
            model = MarketTransformer(
                input_dim=len(feature_cols),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers
            ).to(device)
            print(f"[Model] Menggunakan Model Time-Series Transformer 1-Menit (M1) Data 5 Tahun Kaggle GPU.")
        else:
            model = MarketAttentionLSTM_M1(
                input_dim=len(feature_cols),
                hidden_dim=config.get('hidden_dim', 64),
                num_layers=config.get('num_layers', 2)
            ).to(device)
            print(f"[Model] Menggunakan Model Deep Learning 1-Menit (M1 LSTM) Data 5 Tahun Kaggle.")
        model.load_state_dict(m1_weights)
        model.eval()
    else:
        config_path = os.path.join(weights_dir, "config.json")
        model_path = os.path.join(weights_dir, "best_model.pth")
        scaler_path = os.path.join(weights_dir, "scaler.pkl")

        if not os.path.exists(config_path) or not os.path.exists(model_path):
            print("[Error] Model belum dilatih. Jalankan 'python ml/run_pipeline.py' terlebih dahulu.")
            return

        with open(config_path, "r") as f:
            config = json.load(f)
        scaler_data = joblib.load(scaler_path)
        scaler = scaler_data["scaler"]
        feature_cols = scaler_data["feature_cols"]
        seq_length = config.get("seq_length", 30)
        timeframe_str = "1D (5Y Daily)"
        checkpoint = torch.load(model_path, map_location=device)
        model_type = checkpoint.get('model_type', config.get('model_type', ''))
        is_transformer = ('transformer' in model_type.lower()) or ('d_model' in checkpoint) or ('d_model' in config)
        if is_transformer:
            d_model = checkpoint.get('d_model', config.get('d_model', 64))
            nhead = checkpoint.get('nhead', config.get('nhead', 4))
            num_layers = checkpoint.get('num_layers', config.get('num_layers', 2))
            model = MarketTransformer(
                input_dim=len(feature_cols),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers
            ).to(device)
            print(f"[Model] Menggunakan Model Time-Series Transformer ({timeframe_str}).")
        else:
            model = MarketAttentionLSTM(
                input_dim=len(feature_cols),
                hidden_dim=checkpoint.get('hidden_dim', 64),
                num_layers=checkpoint.get('num_layers', 2)
            ).to(device)
            print(f"[Model] Menggunakan Model Harian (1D LSTM).")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    is_mt5_connected = False
    if MT5_AVAILABLE:
        if mt5.initialize():
            is_mt5_connected = True
            acc = mt5.account_info()
            print(f"[MT5 Live] Berhasil terhubung ke MetaTrader 5 Terminal!")
            if acc:
                print(f"[MT5 Live] Akun ID   : {acc.login} ({acc.company})")
                print(f"[MT5 Live] Server    : {acc.server}")
                print(f"[MT5 Live] Saldo     : ${acc.balance:,.2f} {acc.currency} (Equity: ${acc.equity:,.2f})")
        else:
            print(f"[MT5] Inisialisasi MetaTrader 5 gagal: {mt5.last_error()}")
            if not paper_trade:
                print("[MT5] Beralih otomatis ke mode Paper Trading.")
                paper_trade = True

    print("\n[Bot] Sistem aktif mengawasi pasar & posisi terbuka. Tekan Ctrl+C untuk berhenti.\n")

    try:
        # Bersihkan flag stop lama dari sesi sebelumnya jika ada
        if os.path.exists(FLAG_FILE):
            try:
                os.remove(FLAG_FILE)
            except Exception:
                pass

        # Tulis PID agar web server dan sistem dapat memonitor proses ini secara presisi
        try:
            with open(PID_FILE, "w") as f:
                f.write(str(os.getpid()))
        except Exception:
            pass

        while True:
            if os.path.exists(FLAG_FILE):
                print("[Bot] Stop flag terdeteksi. Menghentikan robot secara aman...")
                break

            now_str = time.strftime("%Y-%m-%d %H:%M:%S")

            # 1. Ambil data candle terbaru dari MT5 jika terhubung, atau yfinance
            df = None
            if is_mt5_connected:
                tf = mt5.TIMEFRAME_M1 if is_m1 else mt5.TIMEFRAME_D1
                rates = mt5.copy_rates_from_pos(mt5_symbol, tf, 0, max(seq_length + 200, 300))
                if rates is not None and len(rates) >= seq_length + 20:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df = df.set_index('time').rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'
                    })[['Open', 'High', 'Low', 'Close', 'Volume']]

            if df is None:
                df = download_5y_data(symbol=symbol, interval=interval, force_refresh=True, silent=True)
            
            if is_m1:
                use_kalman = config.get("use_kalman", False) or any(k.startswith("kf_") for k in feature_cols)
                clean_df, _ = compute_features_m1(df, use_kalman=use_kalman)
            else:
                clean_df, _ = compute_technical_features(df)

            recent_features_df = clean_df[feature_cols].iloc[-seq_length:]
            recent_scaled = scaler.transform(recent_features_df.values if hasattr(recent_features_df, 'values') else recent_features_df)

            # 2. Forward pass Deep Learning
            with torch.no_grad():
                x_tensor = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                logits, exp_ret_tensor, _ = model(x_tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                prob_down = float(probs[0])
                prob_up = float(probs[1])

            current_close = float(df['Close'].iloc[-1])

            signal = "HOLD"
            conf = max(prob_up, prob_down)
            if prob_up >= threshold:
                signal = "BUY"
            elif prob_down >= threshold:
                signal = "SELL"

            # Kalkulasi TP & SL Dinamis dari AI & Volatilitas ATR
            if 'atr_norm' in clean_df:
                current_atr = float(clean_df['atr_norm'].iloc[-1] * current_close)
            else:
                high_low = df['High'] - df['Low']
                current_atr = float(high_low.rolling(14).mean().iloc[-1])
                
            if is_m1:
                current_atr = max(1.0, min(current_atr, 8.0))
            else:
                current_atr = max(2.5, min(current_atr, 18.0))

            computed_sl = round(current_atr * 1.0, 2)
            computed_tp = round(computed_sl * rr_ratio, 2)

            active_tp = computed_tp if auto_tpsl else tp_points
            active_sl = computed_sl if auto_tpsl else sl_points

            # 3. Pantau Posisi Terbuka Saat Ini di MT5
            open_pos = get_open_positions(mt5_symbol) if is_mt5_connected else []
            has_buy_pos = any(p.type == mt5.ORDER_TYPE_BUY for p in open_pos) if is_mt5_connected else False
            has_sell_pos = any(p.type == mt5.ORDER_TYPE_SELL for p in open_pos) if is_mt5_connected else False

            status_pos_str = f"Posisi Aktif: {len(open_pos)}"
            if open_pos:
                floating_sum = sum(p.profit for p in open_pos)
                status_pos_str += f" (Floating: ${floating_sum:+.2f} USD)"

            print(f"[{now_str}] Harga: {current_close:.2f} | AI: {signal} ({conf*100:.1f}%) | TP/SL AI: +${active_tp:.2f}/-${active_sl:.2f} | {status_pos_str}")

            # 4. Manajemen PENUTUPAN & PEMBUKAAN POSISI

            # A. Jika Sinyal BUY:
            if signal == "BUY":
                # Tutup posisi SELL jika ada yang berlawanan arah (Reversal Exit)
                if has_sell_pos and not paper_trade and is_mt5_connected:
                    print(">>> [REVERSAL CLOSE] Sinyal AI BUY terdeteksi. Menutup posisi SELL yang berlawanan...")
                    for p in open_pos:
                        if p.type == mt5.ORDER_TYPE_SELL:
                            close_position(p.ticket, mt5_symbol)

                # Buka posisi BUY jika belum ada posisi BUY yang berjalan
                if not has_buy_pos:
                    if auto_tpsl:
                        print(f">>> [AI DYNAMIC TP/SL] ATR: ${current_atr:.2f} | TP: +${active_tp:.2f} | SL: -${active_sl:.2f} (R:R 1:{rr_ratio})")
                    if not paper_trade and is_mt5_connected:
                        open_market_order(mt5_symbol, "BUY", lot_size, tp_points=active_tp, sl_points=active_sl, comment="RoboTrade AI TP/SL")
                    else:
                        print(f">>> [Paper Trade] MEMBUKA POSISI BUY: {lot_size} Lot @ {current_close:.2f} (TP: {current_close+active_tp:.2f}, SL: {current_close-active_sl:.2f})")

            # B. Jika Sinyal SELL:
            elif signal == "SELL":
                # Tutup posisi BUY jika ada yang berlawanan arah (Reversal Exit)
                if has_buy_pos and not paper_trade and is_mt5_connected:
                    print(">>> [REVERSAL CLOSE] Sinyal AI SELL terdeteksi. Menutup posisi BUY yang berlawanan...")
                    for p in open_pos:
                        if p.type == mt5.ORDER_TYPE_BUY:
                            close_position(p.ticket, mt5_symbol)

                # Buka posisi SELL jika belum ada posisi SELL yang berjalan
                if not has_sell_pos:
                    if auto_tpsl:
                        print(f">>> [AI DYNAMIC TP/SL] ATR: ${current_atr:.2f} | TP: +${active_tp:.2f} | SL: -${active_sl:.2f} (R:R 1:{rr_ratio})")
                    if not paper_trade and is_mt5_connected:
                        open_market_order(mt5_symbol, "SELL", lot_size, tp_points=active_tp, sl_points=active_sl, comment="RoboTrade AI TP/SL")
                    else:
                        print(f">>> [Paper Trade] MEMBUKA POSISI SELL: {lot_size} Lot @ {current_close:.2f} (TP: {current_close-active_tp:.2f}, SL: {current_close+active_sl:.2f})")

            # Simpan snapshot ke latest_prediction.json agar web dashboard terupdate
            snapshot = {
                "symbol": mt5_symbol if is_mt5_connected else symbol,
                "time": now_str,
                "timeframe": timeframe_str,
                "current_price": round(current_close, 2),
                "signal": signal,
                "direction": "BULLISH" if prob_up > prob_down else "BEARISH",
                "confidence_pct": round(conf * 100, 2),
                "prob_bullish": round(prob_up * 100, 2),
                "prob_bearish": round(prob_down * 100, 2),
                "atr_volatility": round(current_atr, 2),
                "auto_tp_points": round(computed_tp, 2),
                "auto_sl_points": round(computed_sl, 2),
                "active_tp_points": round(active_tp, 2),
                "active_sl_points": round(active_sl, 2),
                "suggested_tp": round(current_close + active_tp if prob_up >= prob_down else current_close - active_tp, 2),
                "suggested_sl": round(current_close - active_sl if prob_up >= prob_down else current_close + active_sl, 2),
                "risk_reward_ratio": f"1:{rr_ratio}",
                "mode_tpsl": f"AI Dynamic (ATR R:R 1:{rr_ratio})" if auto_tpsl else f"Manual (TP +${tp_points} / SL -${sl_points})",
                "open_positions": len(open_pos),
                "mt5_connected": is_mt5_connected
            }
            try:
                with open(os.path.join(weights_dir, "latest_prediction.json"), "w") as f:
                    json.dump(snapshot, f, indent=2)
            except Exception:
                pass

            # Sleep responsif dengan interval 0.5 detik agar respon penghentian seketika
            stop_requested = False
            for _ in range(max(1, int(check_interval_sec * 2))):
                if os.path.exists(FLAG_FILE):
                    stop_requested = True
                    break
                time.sleep(0.5)
            if stop_requested:
                print("[Bot] Stop flag terdeteksi saat jeda. Menghentikan robot...")
                break

    except KeyboardInterrupt:
        print("\n[Bot] Robot dihentikan oleh pengguna.")
    finally:
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except Exception:
                pass
        if is_mt5_connected:
            mt5.shutdown()
        print("[Bot] Robot telah berhenti sepenuhnya.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Auto Open/Close Trading Bot")
    parser.add_argument("--symbol", type=str, default="GC=F", help="Ticker analisis (default: GC=F)")
    parser.add_argument("--mt5_symbol", type=str, default="XAUUSD", help="Ticker MT5 (default: XAUUSD)")
    parser.add_argument("--lot", type=float, default=0.01, help="Lot volume")
    parser.add_argument("--threshold", type=float, default=0.51, help="Ambang batas keyakinan sinyal")
    parser.add_argument("--tp", type=float, default=5.0, help="Jarak Take Profit manual ($)")
    parser.add_argument("--sl", type=float, default=3.0, help="Jarak Stop Loss manual ($)")
    parser.add_argument("--auto_tpsl", action="store_true", default=True, help="Gunakan kalkulasi TP & SL otomatis dari AI")
    parser.add_argument("--no_auto_tpsl", dest="auto_tpsl", action="store_false", help="Gunakan TP & SL manual")
    parser.add_argument("--rr_ratio", type=float, default=1.5, help="Rasio Risk:Reward AI (default: 1.5)")
    parser.add_argument("--interval_sec", type=int, default=10, help="Interval loop deteksi (detik)")
    parser.add_argument("--live", action="store_true", help="Aktifkan eksekusi riil ke MT5")
    parser.add_argument("--close_all", action="store_true", help="Tutup semua posisi aktif sekarang juga")
    args = parser.parse_args()

    if args.close_all:
        if mt5.initialize():
            close_all_positions(args.mt5_symbol)
            mt5.shutdown()
        else:
            print("[Error] Tidak dapat terhubung ke MT5")
        sys.exit(0)

    run_trading_bot(
        symbol=args.symbol,
        mt5_symbol=args.mt5_symbol,
        lot_size=args.lot,
        threshold=args.threshold,
        check_interval_sec=args.interval_sec,
        tp_points=args.tp,
        sl_points=args.sl,
        auto_tpsl=args.auto_tpsl,
        rr_ratio=args.rr_ratio,
        paper_trade=not args.live
    )

