//+------------------------------------------------------------------+
//|                                                   MT5_Bridge.mq5 |
//|                             RoboTrade XAUUSD Execution & Monitor |
//+------------------------------------------------------------------+
#property copyright "RoboTrade"
#property link      ""
#property version   "4.00"

#include <Trade\Trade.mqh>

// --- SETTINGS ---
input string BackendURL = "http://127.0.0.1:5000/api/mt5";
input string HistoryURL = "http://127.0.0.1:5000/api/mt5/history";
input string TradeURL = "http://127.0.0.1:5000/api/mt5/trade";
input string ResultURL = "http://127.0.0.1:5000/api/mt5/trade_result";
input int SendIntervalMs = 500; // Cek order dan kirim harga setiap 0.5 detik

CTrade trade;
bool historySent = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    EventSetMillisecondTimer(SendIntervalMs);
    Print("MT5 Bridge v4 (Dengan Riwayat Grafik M1) diinisialisasi.");
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
    EventKillTimer();
    Print("MT5 Bridge dimatikan.");
}

void OnTimer()
{
    if (!historySent) {
        SendHistoryToBackend();
    }
    
    SendDataToBackend();
    CheckPendingOrders();
}

//+------------------------------------------------------------------+
//| Mengirim Riwayat 100 Candle M1 ke Backend                        |
//+------------------------------------------------------------------+
void SendHistoryToBackend()
{
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    
    // Ambil 100 candle M1 terakhir
    if(CopyRates(Symbol(), PERIOD_M1, 0, 100, rates) <= 0) return;
    
    string json = "[";
    for(int i=99; i>=0; i--) {
        json += StringFormat("{\"time\":%d,\"open\":%f,\"high\":%f,\"low\":%f,\"close\":%f}",
                             rates[i].time, rates[i].open, rates[i].high, rates[i].low, rates[i].close);
        if(i > 0) json += ",";
    }
    json += "]";
    
    string postData = "data=" + json;
    char data[];
    StringToCharArray(postData, data, 0, WHOLE_ARRAY, CP_UTF8);
    ArrayResize(data, ArraySize(data) - 1);
    
    char result[];
    string resultHeaders;
    string headers = "Content-Type: application/x-www-form-urlencoded\r\n";
    
    int res = WebRequest("POST", HistoryURL, headers, 2000, data, result, resultHeaders);
    
    if (res == 200) {
        historySent = true;
        Print("Riwayat grafik M1 berhasil dikirim ke Web.");
    }
}

//+------------------------------------------------------------------+
//| Mengirim Harga ke Backend                                        |
//+------------------------------------------------------------------+
void SendDataToBackend()
{
    string symbol = Symbol();
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    
    double open[], high[], low[];
    long volume[];
    
    // Mengambil M1 candle saat ini agar grafik bergerak
    if(CopyOpen(symbol, PERIOD_M1, 0, 1, open) <= 0) return;
    if(CopyHigh(symbol, PERIOD_M1, 0, 1, high) <= 0) return;
    if(CopyLow(symbol, PERIOD_M1, 0, 1, low) <= 0) return;
    if(CopyTickVolume(symbol, PERIOD_M1, 0, 1, volume) <= 0) return;
    
    // Time candle M1 berjalan (untuk update grafik)
    datetime time = iTime(symbol, PERIOD_M1, 0);

    string postData = StringFormat("symbol=%s&bid=%f&ask=%f&open=%f&high=%f&low=%f&volume=%d&time=%d", 
                                   symbol, bid, ask, open[0], high[0], low[0], volume[0], time);
    
    char data[];
    StringToCharArray(postData, data, 0, WHOLE_ARRAY, CP_UTF8);
    ArrayResize(data, ArraySize(data) - 1);
    
    char result[];
    string resultHeaders;
    string headers = "Content-Type: application/x-www-form-urlencoded\r\n";
    
    WebRequest("POST", BackendURL, headers, 1000, data, result, resultHeaders);
}

//+------------------------------------------------------------------+
//| Helper Fungsi Ekstrak Angka JSON                                 |
//+------------------------------------------------------------------+
double GetDoubleFromJson(string json, string key)
{
    int pos = StringFind(json, "\"" + key + "\":");
    if (pos == -1) return 0.0;
    string sub = StringSubstr(json, pos + StringLen(key) + 3, 15);
    int comma = StringFind(sub, ",");
    int bracket = StringFind(sub, "}");
    if (comma != -1) sub = StringSubstr(sub, 0, comma);
    else if (bracket != -1) sub = StringSubstr(sub, 0, bracket);
    return StringToDouble(sub);
}

//+------------------------------------------------------------------+
//| Cek Perintah dari Backend                                        |
//+------------------------------------------------------------------+
void CheckPendingOrders()
{
    char data[];
    char result[];
    string resultHeaders;
    string headers = "Accept: application/json\r\n";
    
    int res = WebRequest("GET", TradeURL, headers, 1000, data, result, resultHeaders);
    
    if (res == 200 && ArraySize(result) > 0)
    {
        string json = CharArrayToString(result);
        
        if(StringFind(json, "\"status\":\"order\"") != -1)
        {
            string action = "";
            if(StringFind(json, "\"action\":\"BUY\"") != -1) action = "BUY";
            else if(StringFind(json, "\"action\":\"SELL\"") != -1) action = "SELL";
            else if(StringFind(json, "\"action\":\"BUYLIMIT\"") != -1) action = "BUYLIMIT";
            else if(StringFind(json, "\"action\":\"SELLLIMIT\"") != -1) action = "SELLLIMIT";
            else if(StringFind(json, "\"action\":\"CANCEL_LIMITS\"") != -1) action = "CANCEL_LIMITS";
            
            double volume = GetDoubleFromJson(json, "volume");
            if (volume == 0) volume = 0.01; 
            
            double tp = GetDoubleFromJson(json, "tp");
            double sl = GetDoubleFromJson(json, "sl");
            double price = GetDoubleFromJson(json, "price");

            string orderId = "0";
            int idPos = StringFind(json, "\"id\":\"");
            if (idPos != -1) {
                orderId = StringSubstr(json, idPos + 6, 13);
            }
            
            string symbol = Symbol();
            double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
            double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
            
            if (action == "BUY") {
                bool success = trade.Buy(volume, symbol, ask, sl, tp);
                SendResultToBackend(orderId, success ? "success" : "failed", trade.ResultRetcodeDescription());
            } 
            else if (action == "SELL") {
                bool success = trade.Sell(volume, symbol, bid, sl, tp);
                SendResultToBackend(orderId, success ? "success" : "failed", trade.ResultRetcodeDescription());
            }
            else if (action == "BUYLIMIT") {
                bool success = trade.BuyLimit(volume, price, symbol, sl, tp);
                SendResultToBackend(orderId, success ? "success" : "failed", trade.ResultRetcodeDescription());
            }
            else if (action == "SELLLIMIT") {
                bool success = trade.SellLimit(volume, price, symbol, sl, tp);
                SendResultToBackend(orderId, success ? "success" : "failed", trade.ResultRetcodeDescription());
            }
            else if (action == "CANCEL_LIMITS") {
                int total = OrdersTotal();
                int cancelled = 0;
                for (int i = total - 1; i >= 0; i--) {
                    ulong ticket = OrderGetTicket(i);
                    if (ticket > 0) {
                        long type = OrderGetInteger(ORDER_TYPE);
                        if (type == ORDER_TYPE_BUY_LIMIT || type == ORDER_TYPE_SELL_LIMIT) {
                            if (trade.OrderDelete(ticket)) cancelled++;
                        }
                    }
                }
                SendResultToBackend(orderId, "success", "Cancelled " + IntegerToString(cancelled) + " limits");
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Lapor Hasil ke Web                                               |
//+------------------------------------------------------------------+
void SendResultToBackend(string id, string status, string message)
{
    string postData = StringFormat("id=%s&result=%s&message=%s", id, status, message);
    char data[];
    StringToCharArray(postData, data, 0, WHOLE_ARRAY, CP_UTF8);
    ArrayResize(data, ArraySize(data) - 1);
    
    char result[];
    string resultHeaders;
    string headers = "Content-Type: application/x-www-form-urlencoded\r\n";
    
    WebRequest("POST", ResultURL, headers, 1000, data, result, resultHeaders);
}
