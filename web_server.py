"""
RoboTrade Web Control Server (Flask + MT5 + Deep Learning)
Provides REST API and serves the Glassmorphic Dashboard on port 8000.
"""

import os
import sys
import json
import time
import subprocess
import datetime
import psutil
from flask import Flask, request, jsonify, send_from_directory

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

def get_symbol_filling_mode(symbol):
    """Mendapatkan mode filling order yang didukung broker untuk simbol tertentu."""
    if not MT5_AVAILABLE:
        return 2
    try:
        info = mt5.symbol_info(symbol)
        if info:
            if info.filling_mode & 2:
                return mt5.ORDER_FILLING_IOC
            elif info.filling_mode & 1:
                return mt5.ORDER_FILLING_FOK
    except Exception:
        pass
    return getattr(mt5, 'ORDER_FILLING_RETURN', 0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)

# Bot process management
bot_process = None
MAGIC_NUMBER = 234000
WEIGHTS_DIR = os.path.join(BASE_DIR, "ml", "weights")
PID_FILE = os.path.join(WEIGHTS_DIR, "bot.pid")
FLAG_FILE = os.path.join(WEIGHTS_DIR, "bot_stop.flag")

def get_mt5_status():
    """Mengambil status koneksi dan akun MT5."""
    if not MT5_AVAILABLE:
        return {"connected": False, "error": "Modul MetaTrader5 tidak terpasang"}
    
    if not mt5.initialize():
        return {"connected": False, "error": str(mt5.last_error())}
    
    acc = mt5.account_info()
    term = mt5.terminal_info()
    if not acc:
        return {"connected": False, "error": "Gagal membaca info akun MT5"}
    
    return {
        "connected": True,
        "login": acc.login,
        "company": acc.company,
        "server": acc.server,
        "balance": acc.balance,
        "equity": acc.equity,
        "currency": acc.currency,
        "trade_allowed": term.trade_allowed if term else False,
        "bot_running": is_bot_running()
    }

def find_bot_processes():
    """Mencari semua proses python yang menjalankan bot_bridge.py di sistem operasi."""
    procs = []
    my_pid = os.getpid()
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if p.pid == my_pid:
                continue
            cmdline = p.info.get('cmdline') or []
            cmd_str = " ".join(cmdline)
            if "bot_bridge.py" in cmd_str:
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs

def is_bot_running():
    global bot_process
    # 1. Cek in-memory handle
    if bot_process is not None and bot_process.poll() is None:
        return True
    # 2. Cek apakah ada proses bot_bridge.py yang aktif di OS
    active_procs = find_bot_processes()
    if len(active_procs) > 0:
        return True
    return False

def stop_bot_service():
    global bot_process
    # 1. Tulis stop flag
    try:
        with open(FLAG_FILE, "w") as f:
            f.write("stop")
    except Exception:
        pass

    # 2. Terminate in-memory handle jika ada
    if bot_process is not None:
        try:
            bot_process.terminate()
            bot_process.wait(timeout=1.0)
        except Exception:
            try:
                bot_process.kill()
            except Exception:
                pass
        bot_process = None

    # 3. Terminate semua proses bot_bridge.py yang terdeteksi di OS
    for p in find_bot_processes():
        try:
            p.terminate()
        except Exception:
            pass

    # Beri jeda 0.5 detik, lalu paksa kill jika masih ada yang tersisa
    time.sleep(0.5)
    for p in find_bot_processes():
        try:
            p.kill()
        except Exception:
            pass

    # 4. Hapus file PID jika masih tersisa
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except Exception:
            pass

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# 1. Serve index.html and static assets
@app.route('/')
@app.route('/index.html')
def serve_index():
    resp = send_from_directory(BASE_DIR, 'index.html')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(BASE_DIR, path)):
        return send_from_directory(BASE_DIR, path)
    return jsonify({"error": "File tidak ditemukan"}), 404

# 2. API Status
@app.route('/api/status', methods=['GET'])
def api_status():
    status = get_mt5_status()
    if MT5_AVAILABLE and mt5.initialize():
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_D1, 0, 2)
        if rates is not None and len(rates) >= 1:
            today = rates[-1]
            prev_close = rates[-2]['close'] if len(rates) >= 2 else today['open']
            curr_price = today['close']
            change_pct = ((curr_price - prev_close) / prev_close) * 100.0
            status["high_24h"] = round(float(today['high']), 2)
            status["low_24h"] = round(float(today['low']), 2)
            status["change_24h"] = round(float(change_pct), 2)
    status["bot_running"] = is_bot_running()
    return jsonify(status)

# 3. API Open Positions in MT5
@app.route('/api/positions', methods=['GET'])
def api_positions():
    if not MT5_AVAILABLE or not mt5.initialize():
        return jsonify({"positions": [], "total": 0, "floating_profit": 0.0, "connected": False})
    
    pos_raw = mt5.positions_get()
    positions = []
    total_profit = 0.0
    
    if pos_raw:
        for p in pos_raw:
            positions.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": p.volume,
                "price_open": round(p.price_open, 2),
                "price_current": round(p.price_current, 2),
                "sl": round(p.sl, 2),
                "tp": round(p.tp, 2),
                "profit": round(p.profit, 2),
                "time": datetime.datetime.fromtimestamp(p.time).strftime("%H:%M:%S")
            })
            total_profit += p.profit
            
    acc = mt5.account_info()
    if acc:
        account_floating = acc.equity - acc.balance
        if abs(account_floating) > 0.001 and len(positions) > 0:
            total_profit = account_floating

    return jsonify({
        "positions": positions,
        "total": len(positions),
        "floating_profit": round(total_profit, 2),
        "connected": True
    })

# 4. API Real Market Tick
@app.route('/api/market/tick', methods=['GET'])
def api_market_tick():
    symbol = request.args.get('symbol', 'XAUUSD')
    if not MT5_AVAILABLE or not mt5.initialize():
        return jsonify({"error": "MT5 tidak terhubung"}), 500
    
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return jsonify({"error": f"Gagal membaca tick {symbol}"}), 404
        
    return jsonify({
        "symbol": symbol,
        "bid": round(tick.bid, 2),
        "ask": round(tick.ask, 2),
        "price": round((tick.bid + tick.ask) / 2.0, 2),
        "spread": round(tick.ask - tick.bid, 2),
        "volume": tick.volume,
        "time": datetime.datetime.fromtimestamp(tick.time).strftime("%H:%M:%S")
    })

# 5. API Real Candles for Chart
@app.route('/api/market/candles', methods=['GET'])
def api_market_candles():
    symbol = request.args.get('symbol', 'XAUUSD')
    count = int(request.args.get('count', 60))
    if not MT5_AVAILABLE or not mt5.initialize():
        return jsonify({"candles": []})
        
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, count)
    if rates is None or len(rates) == 0:
        return jsonify({"candles": []})
        
    candles = []
    for r in rates:
        candles.append({
            "time": int(r['time']),
            "open": round(float(r['open']), 2),
            "high": round(float(r['high']), 2),
            "low": round(float(r['low']), 2),
            "close": round(float(r['close']), 2),
            "volume": int(r['tick_volume'])
        })
    return jsonify({"candles": candles})

# 6. API Manual Open Position
@app.route('/api/trade/open', methods=['POST'])
def api_trade_open():
    if not MT5_AVAILABLE or not mt5.initialize():
        return jsonify({"success": False, "error": "MT5 tidak aktif"}), 500
        
    data = request.get_json() or {}
    symbol = data.get('symbol', 'XAUUSD')
    order_type = data.get('type', 'BUY').upper()
    volume = float(data.get('volume', 0.01))
    use_ai_tpsl = data.get('use_ai_tpsl', True)
    
    # Ambil nilai presisi yang dikirim klien
    tp_points = float(data.get('tp_points', 0.0))
    sl_points = float(data.get('sl_points', 0.0))
    
    # Hanya jika belum ditentukan atau <= 0, cari fallback dari snapshot AI
    if tp_points <= 0 or sl_points <= 0:
        pred_path = os.path.join(BASE_DIR, "ml", "weights", "latest_prediction.json")
        if os.path.exists(pred_path):
            try:
                with open(pred_path, "r") as f:
                    snap = json.load(f)
                    if tp_points <= 0:
                        tp_points = float(snap.get('auto_tp_points', 27.0))
                    if sl_points <= 0:
                        sl_points = float(snap.get('auto_sl_points', 18.0))
            except Exception:
                pass
        if tp_points <= 0: tp_points = 27.0
        if sl_points <= 0: sl_points = 18.0

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return jsonify({"success": False, "error": f"Gagal membaca harga {symbol}"}), 400
        
    is_buy = (order_type == "BUY")
    action_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    price = tick.ask if is_buy else tick.bid
    
    tp = round(price + tp_points, 2) if is_buy else round(price - tp_points, 2)
    sl = round(price - sl_points, 2) if is_buy else round(price + sl_points, 2)
    
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": action_type,
        "price": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 25,
        "magic": MAGIC_NUMBER,
        "comment": "RoboTrade AI TP/SL" if use_ai_tpsl else "RoboTrade Manual Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_symbol_filling_mode(symbol),
    }
    
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return jsonify({
            "success": True,
            "order": result.order,
            "price": result.price,
            "volume": result.volume,
            "tp": tp,
            "sl": sl,
            "message": f"Order {order_type} {volume} Lot berhasil dibuka pada ${price:.2f}"
        })
    else:
        err = result.comment if result else str(mt5.last_error())
        return jsonify({"success": False, "error": f"Gagal eksekusi: {err}"}), 400

# 7. API Close Single Position by Ticket
@app.route('/api/trade/close', methods=['POST'])
def api_trade_close():
    if not MT5_AVAILABLE or not mt5.initialize():
        return jsonify({"success": False, "error": "MT5 tidak aktif"}), 500
        
    data = request.get_json() or {}
    ticket = int(data.get('ticket', 0))
    
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return jsonify({"success": False, "error": f"Posisi dengan tiket #{ticket} tidak ditemukan"}), 404
    p = pos[0]
    symbol = p.symbol
    
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return jsonify({"success": False, "error": f"Gagal membaca tick {symbol}"}), 400
        
    close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
    
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(p.volume),
        "type": close_type,
        "position": int(p.ticket),
        "price": float(close_price),
        "deviation": 25,
        "magic": MAGIC_NUMBER,
        "comment": "Web Manual Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_symbol_filling_mode(symbol),
    }
    
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return jsonify({
            "success": True,
            "ticket": ticket,
            "profit": p.profit,
            "message": f"Posisi #{ticket} berhasil ditutup! Profit: ${p.profit:+.2f} USD"
        })
    else:
        err = result.comment if result else str(mt5.last_error())
        return jsonify({"success": False, "error": f"Gagal tutup tiket #{ticket}: {err}"}), 400

# 8. API Close All Positions
@app.route('/api/trade/close_all', methods=['POST'])
def api_trade_close_all():
    if not MT5_AVAILABLE or not mt5.initialize():
        return jsonify({"success": False, "error": "MT5 tidak aktif"}), 500
        
    data = request.get_json() or {}
    symbol = data.get('symbol', 'XAUUSD')
    pos_raw = mt5.positions_get(symbol=symbol)
    if not pos_raw:
        return jsonify({"success": True, "closed_count": 0, "message": "Tidak ada posisi terbuka untuk ditutup."})
        
    closed = 0
    total_profit = 0.0
    for p in pos_raw:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(p.volume),
            "type": close_type,
            "position": int(p.ticket),
            "price": float(close_price),
            "deviation": 25,
            "magic": MAGIC_NUMBER,
            "comment": "Web Close All",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": get_symbol_filling_mode(symbol),
        }
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            closed += 1
            total_profit += p.profit
            
    return jsonify({
        "success": True,
        "closed_count": closed,
        "total_profit": round(total_profit, 2),
        "message": f"Berhasil menutup {closed} posisi! Total Profit: ${total_profit:+.2f} USD"
    })

# 9. API Latest AI Prediction
@app.route('/api/ai/latest', methods=['GET'])
def api_ai_latest():
    pred_path = os.path.join(BASE_DIR, "ml", "weights", "latest_prediction.json")
    if os.path.exists(pred_path):
        try:
            with open(pred_path, "r") as f:
                data = json.load(f)
                return jsonify(data)
        except Exception as e:
            pass
    return jsonify({
        "symbol": "XAUUSD",
        "signal": "HOLD",
        "direction": "BULLISH",
        "confidence_pct": 51.9,
        "prob_bullish": 51.9,
        "prob_bearish": 48.1,
        "current_price": 4307.0,
        "suggested_tp": 4312.0,
        "suggested_sl": 4304.0
    })

# 10. API Trigger Re-Inference
@app.route('/api/ai/predict', methods=['POST'])
def api_ai_predict():
    try:
        cmd = [sys.executable, os.path.join(BASE_DIR, "ml", "predict.py"), "--json"]
        out = subprocess.check_output(cmd, cwd=BASE_DIR, text=True, timeout=20)
        start_idx = out.find('{')
        if start_idx != -1:
            json_str = out[start_idx:]
            data = json.loads(json_str)
            return jsonify(data)
        else:
            return jsonify({"error": "Output bukan format JSON valid", "raw": out}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 11. API Bot Start / Stop
@app.route('/api/bot/toggle', methods=['POST'])
def api_bot_toggle():
    global bot_process
    data = request.get_json() or {}
    action = data.get('action', 'toggle') # 'start', 'stop', 'toggle'
    
    currently_running = is_bot_running()
    
    if action == 'stop' or (action == 'toggle' and currently_running):
        stop_bot_service()
        return jsonify({
            "status": "stopped",
            "running": False,
            "message": "Bot Auto-Trading AI berhasil dihentikan."
        })
    else:
        # Bersihkan proses lama terlebih dahulu agar tidak dobel
        stop_bot_service()
        if os.path.exists(FLAG_FILE):
            try: os.remove(FLAG_FILE)
            except Exception: pass
            
        lot = float(data.get('lot', 0.01))
        threshold = float(data.get('threshold', 0.51))
        auto_tpsl = data.get('auto_tpsl', True)
        rr_ratio = float(data.get('rr_ratio', 1.5))
        tp = float(data.get('tp', 27.0))
        sl = float(data.get('sl', 18.0))
        
        script_path = os.path.join(BASE_DIR, "ml", "bot_bridge.py")
        cmd = [
            sys.executable, "-u", script_path,
            "--live", "--mt5_symbol", "XAUUSD",
            "--threshold", str(threshold),
            "--lot", str(lot),
            "--rr_ratio", str(rr_ratio),
            "--tp", str(tp),
            "--sl", str(sl)
        ]
        if auto_tpsl:
            cmd.append("--auto_tpsl")
        else:
            cmd.append("--no_auto_tpsl")

        log_path = os.path.join(WEIGHTS_DIR, "bot.log")
        bot_log_file = open(log_path, "a", encoding="utf-8")
        bot_process = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=bot_log_file,
            stderr=bot_log_file
        )
        try:
            with open(PID_FILE, "w") as f:
                f.write(str(bot_process.pid))
        except Exception:
            pass

        time.sleep(0.5)
        if bot_process.poll() is not None:
            err_msg = "Bot gagal berjalan."
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
                    lines = lf.readlines()[-6:]
                    if lines:
                        err_msg += " Log: " + "".join(lines).strip()
            return jsonify({
                "status": "stopped",
                "running": False,
                "error": err_msg,
                "message": err_msg
            }), 500

        mode_str = f"Otomatis AI (ATR Volatilitas R:R 1:{rr_ratio})" if auto_tpsl else f"Manual (TP +${tp} / SL -${sl})"
        return jsonify({
            "status": "running",
            "running": True,
            "pid": bot_process.pid,
            "message": f"Bot Auto-Trading AI berhasil diaktifkan! Mode TP/SL: {mode_str}"
        })

def ensure_bridge_server():
    """Memastikan server port 5000 untuk EA MT5_Bridge aktif."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        s.connect(('127.0.0.1', 5000))
        s.close()
    except Exception:
        bridge_script = os.path.join(BASE_DIR, "server", "bridge_server.py")
        if os.path.exists(bridge_script):
            subprocess.Popen([sys.executable, bridge_script], cwd=BASE_DIR)
            print(">>> [Bridge Server] Port 5000 Bridge berhasil dijalankan otomatis.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    ensure_bridge_server()
    print(f">>> [RoboTrade Web Server] Berjalan pada http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

