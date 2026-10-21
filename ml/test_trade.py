import time
import MetaTrader5 as mt5

def execute_test_order(symbol="XAUUSD", volume=0.01):
    if not mt5.initialize():
        print("[Error] Inisialisasi MetaTrader 5 gagal:", mt5.last_error())
        return

    print("=" * 60)
    print("      EKSEKUSI ORDER UJI COBA KE METATRADER 5")
    print(f"      Target Simbol : {symbol} | Volume: {volume} Lot")
    print("=" * 60)

    # Cek status AlgoTrading
    t_info = mt5.terminal_info()
    if not t_info.trade_allowed:
        print("\n[PERINGATAN] Tombol 'Algo Trading' di MT5 Anda saat ini masih NONAKTIF.")
        print(">>> Silakan buka aplikasi MT5 dan tekan tombol 'Algo Trading' di toolbar atas")
        print(">>> atau tekan tombol shortcut keyboard: Ctrl + E sekarang...")
        print("[Menunggu tombol Algo Trading diaktifkan (timeout 30 detik)]...")

        for i in range(30):
            time.sleep(1)
            t_info = mt5.terminal_info()
            if t_info.trade_allowed:
                print("\n[OK] Tombol Algo Trading TERDETEKSI AKTIF! Menembakkan order sekarang...")
                break
        else:
            print("\n[Batal] Waktu tunggu habis. Tombol Algo Trading belum diaktifkan di MT5.")
            mt5.shutdown()
            return

    # Ambil harga terkini
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"[Error] Gagal membaca harga tick untuk {symbol}")
        mt5.shutdown()
        return

    ask = tick.ask
    tp = round(ask + 3.0, 2)
    sl = round(ask - 2.0, 2)

    # Cek filling mode broker
    sym_info = mt5.symbol_info(symbol)
    filling = mt5.ORDER_FILLING_IOC
    if sym_info and sym_info.filling_mode == 1:
        filling = mt5.ORDER_FILLING_FOK
    elif sym_info and sym_info.filling_mode == 2:
        filling = mt5.ORDER_FILLING_IOC

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt5.ORDER_TYPE_BUY,
        "price": ask,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "RoboTrade Deep Learning",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    print(f"\n[Mengirim Order] BUY {volume} {symbol} @ {ask} | TP: {tp} | SL: {sl}...")
    result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("=" * 60)
        print(">>> [BERHASIL!] ORDER RESMI TERBUKA DI METATRADER 5 ANDA! <<<")
        print("=" * 60)
        print(f"Ticket Order  : {result.order}")
        print(f"Jenis Posisi  : BUY")
        print(f"Simbol        : {symbol}")
        print(f"Harga Masuk   : ${result.price}")
        print(f"Take Profit   : ${tp}")
        print(f"Stop Loss     : ${sl}")
        print(f"Volume        : {result.volume} Lot")
        print("Silakan cek tab 'Trade' di bagian bawah aplikasi MetaTrader 5 Anda!")
    else:
        print(f"[Gagal Eksekusi] Kode Error MT5: {result.retcode} ({result.comment})")
        if result.retcode == 10027:
            print("Penyebab: Tombol Algo Trading (Ctrl + E) belum menyala hijau di MT5.")

    mt5.shutdown()

if __name__ == "__main__":
    execute_test_order()
