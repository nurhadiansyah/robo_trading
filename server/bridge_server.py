"""
RoboTrade MT5 HTTP Bridge Server (Port 5000)
Handles incoming WebRequests from MT5_Bridge.mq5 / MT5_Bridge.ex5
Provides seamless connection so MT5 EA connects with HTTP 200 OK.
"""

import os
import sys
import json
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

# State
current_tick = {
    "symbol": "XAUUSD",
    "price": 0.0,
    "bid": 0.0,
    "ask": 0.0,
    "open": 0.0,
    "high": 0.0,
    "low": 0.0,
    "volume": 0,
    "time": int(time.time()),
    "last_seen": int(time.time())
}

mt5_history = []
pending_orders = []
execution_results = []

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "ml", "weights")

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# 1. Endpoint Ticks dari MT5 EA (POST /api/mt5)
@app.route('/api/mt5', methods=['POST'])
def receive_mt5_tick():
    global current_tick
    data = request.form
    if not data:
        data = request.get_json(silent=True) or {}
    
    bid = data.get('bid')
    if bid is not None:
        try:
            current_tick = {
                "symbol": data.get('symbol', 'XAUUSD'),
                "bid": float(bid),
                "ask": float(data.get('ask', bid)),
                "price": float(bid),
                "open": float(data.get('open', 0.0)),
                "high": float(data.get('high', 0.0)),
                "low": float(data.get('low', 0.0)),
                "volume": int(float(data.get('volume', 1))),
                "time": int(data.get('time', int(time.time()))),
                "last_seen": int(time.time())
            }
        except Exception as e:
            print(f"[Bridge 5000] Error parse tick: {e}")
    return "OK", 200

# 2. Endpoint Riwayat Candle M1 dari MT5 EA (POST & GET /api/mt5/history)
@app.route('/api/mt5/history', methods=['POST'])
def receive_mt5_history():
    global mt5_history
    data_str = request.form.get('data')
    if not data_str:
        json_body = request.get_json(silent=True) or {}
        data_str = json_body.get('data') if isinstance(json_body, dict) else None
        
    if data_str:
        try:
            mt5_history = json.loads(data_str)
            print(f"[Bridge 5000] M1 History diterima: {len(mt5_history)} candle.")
        except Exception as e:
            print(f"[Bridge 5000] Gagal parse history JSON: {e}")
    return "OK", 200

@app.route('/api/mt5/history', methods=['GET'])
def get_mt5_history():
    return jsonify(mt5_history)

# 3. Endpoint Perintah Trading ke MT5 EA (GET /api/mt5/trade)
@app.route('/api/mt5/trade', methods=['GET'])
def poll_mt5_trade():
    global pending_orders
    if pending_orders:
        order = pending_orders.pop(0)
        print(f"[Bridge 5000] Mengirim order ke MT5 EA: {order}")
        return jsonify({
            "status": "order",
            "action": order.get("action", "BUY"),
            "volume": float(order.get("volume", 0.01)),
            "tp": float(order.get("tp", 0.0)),
            "sl": float(order.get("sl", 0.0)),
            "price": float(order.get("price", 0.0)),
            "id": str(order.get("id", int(time.time() * 1000)))
        })
    return jsonify({"status": "none"})

# 4. Endpoint Laporan Hasil Eksekusi dari MT5 EA (POST /api/mt5/trade_result)
@app.route('/api/mt5/trade_result', methods=['POST'])
def receive_trade_result():
    data = request.form
    if not data:
        data = request.get_json(silent=True) or {}
    order_id = data.get('id')
    result = data.get('result')
    message = data.get('message')
    print(f"[Bridge 5000] Laporan Order #{order_id}: {result} | {message}")
    execution_results.append({
        "id": order_id,
        "result": result,
        "message": message,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    return "OK", 200

# 5. Endpoint Status Bridge
@app.route('/api/status', methods=['GET'])
def bridge_status():
    age = int(time.time()) - current_tick["last_seen"]
    is_ea_alive = (age < 5) and (current_tick["price"] > 0)
    return jsonify({
        "bridge": "Python MT5 Port 5000 Bridge",
        "status": "online",
        "ea_connected": is_ea_alive,
        "last_tick_age_seconds": age,
        "current_tick": current_tick,
        "history_bars": len(mt5_history),
        "pending_orders": len(pending_orders)
    })

# 6. Endpoint AI Prediction Snapshot
@app.route('/api/ai/deep_prediction', methods=['GET'])
def deep_prediction():
    pred_path = os.path.join(WEIGHTS_DIR, "latest_prediction.json")
    if os.path.exists(pred_path):
        try:
            with open(pred_path, "r") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Belum ada snapshot prediksi"}), 404

if __name__ == '__main__':
    port = int(os.environ.get("BRIDGE_PORT", 5000))
    print(f">>> [RoboTrade MT5 Port 5000 Bridge] Berjalan pada http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
