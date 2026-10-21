const express = require('express');
const cors = require('cors');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);

app.use(cors());
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

const io = new Server(server, {
  cors: { origin: '*', methods: ['GET', 'POST'] }
});

// State Global
let currentData = { price: 0, ask: 0, bid: 0, time: Date.now() };
let pendingOrders = [];

// Konfigurasi Algoritma Auto-Trading
let isAutoTradingEnabled = false;
let config = {
    volume: 0.01,
    tpPoints: 2.00, // Disarankan 2.00 untuk R:R lebih baik
    slPoints: 1.00,
    momentumThreshold: 0.30, // Sensitivitas (default $0.30)
    maxSpread: 0.80, // Batas maksimal spread (default $0.80)
    algoVersion: ['v3'], // Versi Algoritma (Bisa lebih dari 1)
    executionMode: 'independent' // 'independent' atau 'confluence'
};

// State Mesin Momentum
let priceHistory = [];
let lastTradeTime = 0;
let tickImbalance = { up: 0, down: 0 };
let lastTickPrice = 0;
let lastLimitTime = 0;

// State Data Grafik MT5 Asli
let mt5ChartHistory = [];
let cachedIndicators = {
    sma20: null,
    rsi14: null,
    macd: null,
    sr: null,
    maCrossover: null,
    fibo: null
};

// State Model Deep Learning (Bi-LSTM + Attention 5 Tahun)
let deepLearningState = null;

function loadDeepLearningPrediction() {
    const predPath = path.join(__dirname, '../ml/weights/latest_prediction.json');
    if (fs.existsSync(predPath)) {
        try {
            deepLearningState = JSON.parse(fs.readFileSync(predPath, 'utf8'));
            console.log(`[DeepLearning] Model 5-Tahun Termuat: Sinyal ${deepLearningState.signal} (${deepLearningState.confidence_pct}%)`);
            if (typeof io !== 'undefined') {
                io.emit('deep_ai_update', deepLearningState);
            }
        } catch (e) {
            console.error("[DeepLearning] Gagal membaca snapshot prediksi:", e.message);
        }
    }
}

// Muat awal saat server dimulai
loadDeepLearningPrediction();

// Auto-refresh inferensi deep learning setiap 10 menit
setInterval(() => {
    const pythonScript = path.join(__dirname, '../ml/predict.py');
    const projectRoot = path.join(__dirname, '..');
    exec(`python "${pythonScript}" --json`, { cwd: projectRoot }, (err, stdout) => {
        if (!err && stdout) {
            try {
                const jsonStart = stdout.indexOf('{');
                if (jsonStart !== -1) {
                    deepLearningState = JSON.parse(stdout.substring(jsonStart));
                    io.emit('deep_ai_update', deepLearningState);
                    console.log(`[DeepLearning Auto] Prediksi diperbarui: ${deepLearningState.signal} (${deepLearningState.confidence_pct}%)`);
                }
            } catch (e) {}
        }
    });
}, 10 * 60 * 1000);

function updateCachedIndicators() {
    cachedIndicators.sma20 = calculateSMA(20);
    cachedIndicators.rsi14 = calculateRSI(14);
    cachedIndicators.macd = calculateMACD();
    cachedIndicators.sr = calculateSupportResistance();
    
    if (mt5ChartHistory.length > 25) {
        const closes = mt5ChartHistory.map(c => c.close);
        const ema9 = calculateEMA(9, closes);
        const ema21 = calculateEMA(21, closes);
        const prevCloses = closes.slice(0, -1);
        const prevEma9 = calculateEMA(9, prevCloses);
        const prevEma21 = calculateEMA(21, prevCloses);
        cachedIndicators.maCrossover = { ema9, ema21, prevEma9, prevEma21 };
    } else {
        cachedIndicators.maCrossover = null;
    }
    
    if (mt5ChartHistory.length >= 50) {
        const recent = mt5ChartHistory.slice(-100);
        let maxHigh = -Infinity;
        let minLow = Infinity;
        for (let c of recent) {
            if (c.high > maxHigh) maxHigh = c.high;
            if (c.low < minLow) minLow = c.low;
        }
        const range = maxHigh - minLow;
        cachedIndicators.fibo = { maxHigh, minLow, range };
    } else {
        cachedIndicators.fibo = null;
    }
}

// Fungsi untuk menghitung SMA 20 dari history MT5
function calculateSMA(period = 20) {
    if (mt5ChartHistory.length < period) return null;
    
    // Ambil candle 20 terakhir
    const recentCandles = mt5ChartHistory.slice(-period);
    const sum = recentCandles.reduce((acc, curr) => acc + curr.close, 0);
    return sum / period;
}

// Fungsi untuk menghitung RSI 14 dari history MT5
function calculateRSI(period = 14) {
    if (mt5ChartHistory.length < period + 1) return null;
    
    let gains = 0;
    let losses = 0;
    
    const recent = mt5ChartHistory.slice(-(period + 1));
    for (let i = 1; i < recent.length; i++) {
        const diff = recent[i].close - recent[i - 1].close;
        if (diff >= 0) gains += diff;
        else losses -= diff;
    }
    
    if (losses === 0) return 100;
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    
    const rs = avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));
    return rsi;
}

// Fungsi menghitung Fibonacci Golden Zone dari cache
function getCachedFibonacciZone(currentPrice) {
    const fibo = cachedIndicators.fibo;
    if (!fibo || fibo.range === 0) return null;
    
    const { maxHigh, minLow, range } = fibo;
    const midPoint = minLow + (range / 2);
    let trendDir = currentPrice > midPoint ? 'UPTREND' : 'DOWNTREND';
    let fib50, fib618;
    
    if (trendDir === 'UPTREND') {
        fib50 = maxHigh - (0.500 * range);
        fib618 = maxHigh - (0.618 * range);
    } else {
        fib50 = minLow + (0.500 * range);
        fib618 = minLow + (0.618 * range);
    }
    
    return {
        upperBound: Math.max(fib50, fib618),
        lowerBound: Math.min(fib50, fib618),
        trendDir: trendDir
    };
}

// Fungsi untuk menghitung EMA
function calculateEMA(period, dataArray) {
    if (!dataArray || dataArray.length < period) return null;
    const k = 2 / (period + 1);
    
    let sum = 0;
    for (let i = 0; i < period; i++) sum += dataArray[i];
    let ema = sum / period;
    
    for (let i = period; i < dataArray.length; i++) {
        ema = (dataArray[i] * k) + (ema * (1 - k));
    }
    return ema;
}

// Fungsi untuk menghitung MACD
function calculateMACD() {
    if (mt5ChartHistory.length < 50) return null;
    
    const closes = mt5ChartHistory.map(c => c.close);
    const macdLineArr = [];
    
    for (let i = mt5ChartHistory.length - 30; i <= mt5ChartHistory.length; i++) {
        const sliced = closes.slice(0, i);
        const ema12 = calculateEMA(12, sliced);
        const ema26 = calculateEMA(26, sliced);
        if (ema12 && ema26) {
            macdLineArr.push(ema12 - ema26);
        }
    }
    
    if (macdLineArr.length < 10) return null;
    
    const currentMACD = macdLineArr[macdLineArr.length - 1];
    const prevMACD = macdLineArr[macdLineArr.length - 2];
    
    const signalLine = calculateEMA(9, macdLineArr);
    const prevSignalLine = calculateEMA(9, macdLineArr.slice(0, -1));
    
    return {
        macd: currentMACD,
        signal: signalLine,
        prevMacd: prevMACD,
        prevSignal: prevSignalLine
    };
}

// Fungsi Support & Resistance
function calculateSupportResistance() {
    if (mt5ChartHistory.length < 50) return null;
    const recent = mt5ChartHistory.slice(-50); // Cari S&R dari 50 candle terakhir (sekitar 1 jam)
    
    let support = Infinity;
    let resistance = -Infinity;
    
    for (let c of recent) {
        if (c.high > resistance) resistance = c.high;
        if (c.low < support) support = c.low;
    }
    
    return { support, resistance };
}

// Fungsi Algoritma Analisis (Dijalankan setiap ada tick harga baru)
function analyzeTick() {
    if (!isAutoTradingEnabled || currentData.price === 0) return;
    
    const now = Date.now();
    priceHistory.push({ price: currentData.price, time: now });
    
    // Update Tick Imbalance
    if (lastTickPrice) {
        if (currentData.price > lastTickPrice) tickImbalance.up += currentData.volume;
        else if (currentData.price < lastTickPrice) tickImbalance.down += currentData.volume;
    }
    lastTickPrice = currentData.price;
    tickImbalance.up *= 0.95;
    tickImbalance.down *= 0.95;
    
    // Simpan history harga selama 10 detik terakhir
    // Optimasi splice daripada filter setiap tick
    let cutoffIndex = 0;
    while (cutoffIndex < priceHistory.length && now - priceHistory[cutoffIndex].time > 10000) {
        cutoffIndex++;
    }
    if (cutoffIndex > 0) {
        priceHistory.splice(0, cutoffIndex);
    }
    
    // Cooldown: Jangan trading jika sudah trading di 15 detik terakhir
    if (now - lastTradeTime < 15000) return;
    
    // === V3.6 HFT Market Maker Logic ===
    if (config.algoVersion.includes('v3.6')) {
        if (now - lastLimitTime > 2000) {
            let totalImbalance = tickImbalance.up + tickImbalance.down;
            if (totalImbalance > 10) { 
                pendingOrders.push({ id: "AUTO_CANCEL_" + Date.now(), action: "CANCEL_LIMITS" });
                
                let buyPrice = currentData.bid - (config.maxSpread / 2);
                let sellPrice = currentData.ask + (config.maxSpread / 2);
                
                pendingOrders.push({
                    id: "AUTO_HFT_B_" + Date.now(),
                    action: "BUYLIMIT",
                    volume: config.volume,
                    tp: parseFloat((buyPrice + config.tpPoints).toFixed(2)),
                    sl: parseFloat((buyPrice - config.slPoints).toFixed(2)),
                    price: parseFloat(buyPrice.toFixed(2))
                });
                
                pendingOrders.push({
                    id: "AUTO_HFT_S_" + Date.now(),
                    action: "SELLLIMIT",
                    volume: config.volume,
                    tp: parseFloat((sellPrice - config.tpPoints).toFixed(2)),
                    sl: parseFloat((sellPrice + config.slPoints).toFixed(2)),
                    price: parseFloat(sellPrice.toFixed(2))
                });
                
                lastLimitTime = now;
                
                io.emit('ai_signal', {
                    type: tickImbalance.up > tickImbalance.down ? 'bullish' : 'bearish',
                    title: `V3.6 HFT Market Maker`,
                    time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute:'2-digit', second:'2-digit' }),
                    message: `Memperbarui Limit Order (BuyLimit: ${buyPrice.toFixed(2)}, SellLimit: ${sellPrice.toFixed(2)})`,
                    levels: { entry: currentData.price, target: buyPrice + config.tpPoints, stop: buyPrice - config.slPoints }
                });
            }
        }
    }

    if (priceHistory.length > 5) {
        const oldestPrice = priceHistory[0].price;
        const currentPrice = currentData.price;
        const priceDiff = currentPrice - oldestPrice;
        
        // Cek Tren Menggunakan Data Cache
        const sma20 = cachedIndicators.sma20;
        const rsi14 = cachedIndicators.rsi14;
        const currentSpread = currentData.ask - currentData.bid;
        
        // Logika Eksekusi: Harga bergerak melewati batas momentum threshold (dalam 10 detik)
        if (Math.abs(priceDiff) >= config.momentumThreshold) {
            const isSpikeUp = priceDiff > 0;
            const isSpikeDown = priceDiff < 0;
            
            // Filter: Spread Protection (Berlaku global jika bukan cuma V1)
            const isOnlyV1 = config.algoVersion.length === 1 && config.algoVersion[0] === 'v1';
            if (!isOnlyV1 && currentSpread > config.maxSpread) {
                const rejectMsg = `Spread terlalu tinggi ($${currentSpread.toFixed(2)} > $${config.maxSpread.toFixed(2)})`;
                console.log(`[GLOBAL] Sinyal Ditolak: ${rejectMsg}`);
                io.emit('ai_log', {
                    type: 'info',
                    time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute:'2-digit', second:'2-digit' }),
                    message: `Sinyal Ditolak: ${rejectMsg}`,
                    algoName: "GLOBAL FILTER"
                });
                return;
            }
            
            let approvals = [];
            let rejections = [];
            
            // Evaluasi semua algoritma yang aktif
            for (let v of config.algoVersion) {
                let action = null;
                let rejectReason = "";
                let algoName = "";
                
                if (v === 'v1') {
                    algoName = "V1 Aggressive";
                    action = isSpikeUp ? 'BUY' : 'SELL';
                } 
                else if (v === 'v2') {
                    algoName = "V2 Trend Sync";
                    if (sma20) {
                        if (isSpikeUp && currentPrice > sma20) action = 'BUY';
                        else if (isSpikeDown && currentPrice < sma20) action = 'SELL';
                        else rejectReason = isSpikeUp ? "Spike Naik tapi berlawanan arah Tren" : "Spike Turun tapi berlawanan arah Tren";
                    } else rejectReason = "Menghitung SMA...";
                } 
                else if (v === 'v3') {
                    algoName = "V3 Sniper";
                    if (sma20 && rsi14) {
                        if (isSpikeUp && currentPrice > sma20) {
                            if (rsi14 > 70) rejectReason = `Pasar Overbought (RSI: ${rsi14.toFixed(1)})`;
                            else action = 'BUY';
                        } else if (isSpikeDown && currentPrice < sma20) {
                            if (rsi14 < 30) rejectReason = `Pasar Oversold (RSI: ${rsi14.toFixed(1)})`;
                            else action = 'SELL';
                        } else rejectReason = "Berlawanan arah Tren";
                    } else rejectReason = "Menghitung SMA & RSI...";
                }
                else if (v === 'v3.1') {
                    algoName = "V3.1 Fibo Zone";
                    const fibZone = getCachedFibonacciZone(currentPrice);
                    if (fibZone) {
                        const isInZone = currentPrice >= fibZone.lowerBound && currentPrice <= fibZone.upperBound;
                        if (!isInZone) rejectReason = `Di luar Area Fibo (${fibZone.lowerBound.toFixed(2)} - ${fibZone.upperBound.toFixed(2)})`;
                        else if (rsi14) {
                            if (isSpikeUp && fibZone.trendDir === 'UPTREND') {
                                if (rsi14 > 70) rejectReason = `RSI Overbought (${rsi14.toFixed(1)})`;
                                else action = 'BUY';
                            } else if (isSpikeDown && fibZone.trendDir === 'DOWNTREND') {
                                if (rsi14 < 30) rejectReason = `RSI Oversold (${rsi14.toFixed(1)})`;
                                else action = 'SELL';
                            } else rejectReason = `Berlawanan dengan Tarikan Fibo (${fibZone.trendDir})`;
                        } else rejectReason = "Menghitung RSI...";
                    } else rejectReason = "Data Fibo belum cukup";
                }
                else if (v === 'v3.2') {
                    algoName = "V3.2 MA Crossover";
                    if (cachedIndicators.maCrossover) {
                        const { ema9, ema21, prevEma9, prevEma21 } = cachedIndicators.maCrossover;
                        if (ema9 && ema21 && prevEma9 && prevEma21) {
                            if (prevEma9 <= prevEma21 && ema9 > ema21) action = 'BUY';
                            else if (prevEma9 >= prevEma21 && ema9 < ema21) action = 'SELL';
                            else rejectReason = "Belum ada persilangan EMA 9 dan 21";
                        } else rejectReason = "Menghitung EMA...";
                    } else rejectReason = "Data EMA belum cukup";
                }
                else if (v === 'v3.3') {
                    algoName = "V3.3 RSI Reversal";
                    const currentRsi = cachedIndicators.rsi14;
                    const prevRsi = mt5ChartHistory.length > 15 && currentRsi ? currentRsi - (isSpikeUp ? 5 : (isSpikeDown ? -5 : 0)) : null; 
                    if (currentRsi && prevRsi) {
                        if (prevRsi < 30 && currentRsi >= 30) action = 'BUY';
                        else if (prevRsi > 70 && currentRsi <= 70) action = 'SELL';
                        else rejectReason = `RSI (${currentRsi.toFixed(1)}) tidak menunjukkan pembalikan dari 30/70`;
                    } else rejectReason = "Menghitung RSI...";
                }
                else if (v === 'v3.4') {
                    algoName = "V3.4 MACD Crossover";
                    const macdData = cachedIndicators.macd;
                    if (macdData && macdData.macd !== null && macdData.signal !== null) {
                        if (macdData.prevMacd <= macdData.prevSignal && macdData.macd > macdData.signal) action = 'BUY';
                        else if (macdData.prevMacd >= macdData.prevSignal && macdData.macd < macdData.signal) action = 'SELL';
                        else rejectReason = `MACD (${macdData.macd.toFixed(3)}) belum memotong Signal`;
                    } else rejectReason = "Data MACD belum cukup";
                }
                else if (v === 'v3.5') {
                    algoName = "V3.5 S&R Bounce";
                    const sr = cachedIndicators.sr;
                    if (sr) {
                        const buffer = (sr.resistance - sr.support) * 0.1;
                        if (currentPrice <= (sr.support + buffer) && isSpikeUp) action = 'BUY';
                        else if (currentPrice >= (sr.resistance - buffer) && isSpikeDown) action = 'SELL';
                        else rejectReason = `Harga berada di area abu-abu (S: ${sr.support.toFixed(2)}, R: ${sr.resistance.toFixed(2)})`;
                    } else rejectReason = "Mengumpulkan level S&R...";
                }
                else if (v === 'v4') {
                    algoName = "V4 Deep Learning (5Y LSTM+Attention)";
                    if (deepLearningState && deepLearningState.signal) {
                        const { signal, confidence_pct } = deepLearningState;
                        if (signal === 'BUY') {
                            if (isSpikeUp) action = 'BUY';
                            else rejectReason = `Model BUY (${confidence_pct}%), menunggu konfirmasi momentum naik`;
                        } else if (signal === 'SELL') {
                            if (isSpikeDown) action = 'SELL';
                            else rejectReason = `Model SELL (${confidence_pct}%), menunggu konfirmasi momentum turun`;
                        } else {
                            rejectReason = `Model dalam posisi HOLD / Netral (${confidence_pct}%)`;
                        }
                    } else {
                        rejectReason = "Menunggu pemuatan data inferensi Deep Learning...";
                    }
                }
                
                if (action) {
                    approvals.push({ algo: v, algoName, action });
                } else {
                    rejections.push({ algo: v, algoName, reason: rejectReason });
                }
            }
            
            // Evaluasi berdasarkan Mode Eksekusi
            let finalAction = null;
            let finalAlgoName = "";
            let finalMessage = "";
            
            if (config.executionMode === 'confluence') {
                // Semua algo harus setuju dan arahnya sama
                if (approvals.length === config.algoVersion.length && approvals.length > 0) {
                    const firstAction = approvals[0].action;
                    const allSame = approvals.every(a => a.action === firstAction);
                    if (allSame) {
                        finalAction = firstAction;
                        finalAlgoName = "KONFLUENSI " + approvals.map(a => a.algo.toUpperCase()).join("+");
                        finalMessage = `Seluruh ${approvals.length} Algoritma SEPAKAT mengeksekusi ${firstAction}.`;
                    } else {
                        rejections.push({ algo: 'SYSTEM', algoName: 'Confluence Check', reason: "Konflik arah antar algoritma" });
                    }
                } else {
                    rejections.push({ algo: 'SYSTEM', algoName: 'Confluence Check', reason: "Tidak semua algoritma setuju" });
                }
            } else {
                // Independen: Jika ada 1 yang setuju, langsung eksekusi.
                // Tapi batalkan jika ada konflik (Buy vs Sell) di antara yang setuju.
                if (approvals.length > 0) {
                    const hasBuy = approvals.some(a => a.action === 'BUY');
                    const hasSell = approvals.some(a => a.action === 'SELL');
                    
                    if (hasBuy && hasSell) {
                        rejections.push({ algo: 'SYSTEM', algoName: 'Independent Check', reason: "Konflik sinyal (BUY dan SELL bersamaan). Dibatalkan." });
                    } else {
                        finalAction = approvals[0].action; // Ambil aksi dari yang pertama setuju
                        finalAlgoName = approvals.map(a => a.algo.toUpperCase()).join(" / ");
                        finalMessage = `Disetujui secara independen oleh: ${approvals.map(a => a.algoName).join(', ')}`;
                    }
                }
            }
            
            if (!finalAction) {
                // Kirim log penolakan ke frontend untuk setiap algo
                for (let rej of rejections) {
                    io.emit('ai_log', {
                        type: 'info',
                        time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute:'2-digit', second:'2-digit' }),
                        message: `Ditolak: ${rej.reason}`,
                        algoName: rej.algoName
                    });
                }
                return;
            }
            
            // Hitung TP dan SL yang sangat presisi
            let tp = finalAction === 'BUY' ? currentData.ask + config.tpPoints : currentData.bid - config.tpPoints;
            let sl = finalAction === 'BUY' ? currentData.ask - config.slPoints : currentData.bid + config.slPoints;
            
            // Format 2 desimal untuk XAUUSD
            tp = parseFloat(tp.toFixed(2));
            sl = parseFloat(sl.toFixed(2));

            const isV3Advanced = approvals.some(a => ['v3.1', 'v3.2', 'v3.3', 'v3.4', 'v3.5', 'v4'].includes(a.algo));
            const numOrders = isV3Advanced ? 5 : 1;
            
            for (let i = 0; i < numOrders; i++) {
                const order = {
                    id: "AUTO_" + Date.now() + "_" + i,
                    action: finalAction,
                    volume: config.volume,
                    tp: tp,
                    sl: sl,
                    entryPrice: currentPrice
                };
                pendingOrders.push(order);
            }
            lastTradeTime = now;
            
            console.log(`[ALGO ${finalAlgoName}] ${finalAction} Terpicu! Entry: ${currentPrice} | TP: ${tp} | SL: ${sl}`);
            
            // Beritahu frontend bahwa AI baru saja mengambil posisi
            io.emit('ai_signal', {
                type: finalAction === 'BUY' ? 'bullish' : 'bearish',
                title: `Auto-Trade ${finalAction} (${finalAlgoName})`,
                time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute:'2-digit', second:'2-digit' }),
                message: `Momentum $${Math.abs(priceDiff).toFixed(2)} terdeteksi. ${finalMessage}`,
                levels: { entry: currentPrice, target: tp, stop: sl }
            });
        }
    }
}

// --- ENDPOINT UNTUK MT5 MENGIRIM RIWAYAT ---
app.post('/api/mt5/history', (req, res) => {
    const { data } = req.body;
    if (data) {
        try {
            mt5ChartHistory = JSON.parse(data);
            updateCachedIndicators();
            io.emit('mt5_history', mt5ChartHistory);
            console.log(`Riwayat grafik M1 diterima: ${mt5ChartHistory.length} candle`);
        } catch(e) {
            console.error("Gagal parse history", e);
        }
    }
    res.status(200).send('OK');
});

// --- ENDPOINT WEB UNTUK MEMINTA RIWAYAT ---
app.get('/api/mt5/history', (req, res) => {
    res.json(mt5ChartHistory);
});

// --- ENDPOINT DEEP LEARNING MODEL (5 TAHUN DATA) ---
app.get('/api/ai/deep_prediction', (req, res) => {
    const predPath = path.join(__dirname, '../ml/weights/latest_prediction.json');
    if (fs.existsSync(predPath)) {
        try {
            const data = JSON.parse(fs.readFileSync(predPath, 'utf8'));
            return res.json(data);
        } catch (e) {
            return res.status(500).json({ error: "Gagal membaca file prediksi: " + e.message });
        }
    }
    res.status(404).json({ error: "Belum ada hasil prediksi. Jalankan inferensi atau training terlebih dahulu." });
});

// Endpoint untuk memicu kalkulasi inferensi baru dari terminal Python
app.post('/api/ai/deep_prediction/refresh', (req, res) => {
    const pythonScript = path.join(__dirname, '../ml/predict.py');
    const projectRoot = path.join(__dirname, '..');
    
    exec(`python "${pythonScript}" --json`, { cwd: projectRoot }, (err, stdout, stderr) => {
        if (err) {
            console.error("[DeepLearning] Error refresh prediksi:", err.message);
            return res.status(500).json({ error: err.message });
        }
        try {
            const jsonStart = stdout.indexOf('{');
            if (jsonStart !== -1) {
                const parsed = JSON.parse(stdout.substring(jsonStart));
                io.emit('deep_ai_update', parsed);
                return res.json(parsed);
            }
            res.json({ message: "Eksekusi selesai", output: stdout });
        } catch (parseErr) {
            res.status(500).json({ error: "Gagal parsing output json", raw: stdout });
        }
    });
});

// --- ENDPOINT UNTUK MT5 MENGIRIM DATA HARGA ---
app.post('/api/mt5', (req, res) => {
    const { symbol, bid, ask, open, high, low, volume, time } = req.body;
    if (bid) {
        currentData = {
            symbol: symbol || 'XAUUSD',
            price: parseFloat(bid),
            bid: parseFloat(bid),
            ask: parseFloat(ask),
            open: parseFloat(open),
            high: parseFloat(high),
            low: parseFloat(low),
            volume: parseFloat(volume),
            candleTime: parseInt(time) || Math.floor(Date.now() / 1000), // Waktu candle MT5
            time: Date.now()
        };
        io.emit('mt5_tick', currentData);
        
        // Panggil mesin analisis AI
        analyzeTick();
    }
    res.status(200).send('OK');
});

// --- ENDPOINT UNTUK MT5 MENGECEK ORDER TERTUNDA ---
app.get('/api/mt5/trade', (req, res) => {
    if (pendingOrders.length > 0) {
        const order = pendingOrders.shift();
        res.json({ 
            status: 'order', 
            action: order.action, 
            volume: order.volume || 0.01, 
            tp: order.tp || 0,
            sl: order.sl || 0,
            id: order.id,
            price: order.price || 0
        });
    } else {
        res.json({ status: 'none' });
    }
});

// --- ENDPOINT UNTUK MT5 MELAPORKAN HASIL EKSEKUSI ---
app.post('/api/mt5/trade_result', (req, res) => {
    const { id, result, message } = req.body;
    io.emit('trade_result', { id, result, message });
    res.status(200).send('OK');
});

io.on('connection', (socket) => {
    console.log('React Frontend terhubung');
    socket.emit('mt5_tick', currentData);
    socket.emit('algo_status', { isEnabled: isAutoTradingEnabled, config });
    
    // Menerima perubahan setting atau perintah on/off dari Web
    socket.on('set_algo_config', (newConfig) => {
        config = { ...config, ...newConfig };
        io.emit('algo_status', { isEnabled: isAutoTradingEnabled, config });
    });

    socket.on('toggle_algo', (enabled) => {
        isAutoTradingEnabled = enabled;
        console.log(`Auto Trading: ${isAutoTradingEnabled ? 'ON' : 'OFF'}`);
        io.emit('algo_status', { isEnabled: isAutoTradingEnabled, config });
        if(!enabled) priceHistory = []; // Reset memori jika dimatikan
    });

    socket.on('disconnect', () => {
        console.log('React Frontend terputus');
    });
});

const PORT = 5000;
server.listen(PORT, () => {
    console.log(`AI Trading Backend berjalan di http://localhost:${PORT}`);
});
