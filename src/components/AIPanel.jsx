import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import { Cpu, Zap, TrendingUp, TrendingDown, Target, RefreshCw, BarChart2 } from 'lucide-react';

const AIPanel = () => {
  const [logs, setLogs] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(true);
  const [isAutoEnabled, setIsAutoEnabled] = useState(false);
  const [deepAi, setDeepAi] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    // 1. Ambil prediksi model deep learning terbaru
    fetch('http://localhost:5000/api/ai/deep_prediction')
      .then(res => res.json())
      .then(data => {
        if (!data.error) setDeepAi(data);
      })
      .catch(() => {});

    // 2. Hubungkan socket untuk real-time update
    const socket = io('http://localhost:5000');
    
    socket.on('algo_status', (status) => {
       setIsAutoEnabled(status.isEnabled);
       setIsAnalyzing(status.isEnabled);
    });

    socket.on('ai_signal', (newSignal) => {
      setLogs(prev => [{...newSignal, isSignal: true}, ...prev].slice(0, 50));
      setIsAnalyzing(false);
      setTimeout(() => setIsAnalyzing(true), 3000);
    });
    
    socket.on('ai_log', (logEntry) => {
      setLogs(prev => [{...logEntry, isSignal: false}, ...prev].slice(0, 50));
    });

    socket.on('deep_ai_update', (data) => {
      setDeepAi(data);
    });

    return () => socket.close();
  }, []);

  const handleRefreshDeepAi = async () => {
    setIsRefreshing(true);
    try {
      const res = await fetch('http://localhost:5000/api/ai/deep_prediction/refresh', { method: 'POST' });
      const data = await res.json();
      if (!data.error) setDeepAi(data);
    } catch (e) {
      console.error(e);
    } finally {
      setIsRefreshing(false);
    }
  };

  return (
    <div className="ai-panel glass-panel">
      <div className="ai-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div className="ai-icon-wrapper">
            <Cpu className="ai-icon" size={18} />
          </div>
          <div className="ai-title">Log Eksekusi AI & Deep Learning</div>
        </div>
        <button
          onClick={handleRefreshDeepAi}
          disabled={isRefreshing}
          title="Hitung ulang inferensi Deep Learning"
          style={{
            background: 'rgba(212, 175, 55, 0.15)',
            border: '1px solid var(--gold-primary)',
            color: 'var(--gold-primary)',
            borderRadius: '4px',
            padding: '4px 8px',
            cursor: isRefreshing ? 'wait' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            fontSize: '0.75rem'
          }}
        >
          <RefreshCw size={12} className={isRefreshing ? 'spin-anim' : ''} />
          <span>{isRefreshing ? 'Menghitung...' : 'Prediksi Ulang'}</span>
        </button>
      </div>

      {/* Deep Learning 5-Year Model Status Card */}
      {deepAi && (
        <div style={{
          margin: '10px 0',
          padding: '10px',
          background: 'rgba(15, 23, 42, 0.65)',
          border: '1px solid rgba(212, 175, 55, 0.3)',
          borderRadius: '6px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <BarChart2 size={13} className="text-gold" />
              Bi-LSTM + Attention (Data 5 Thn): <strong>{deepAi.symbol}</strong>
            </span>
            <span style={{
              fontSize: '0.75rem',
              fontWeight: 700,
              padding: '2px 8px',
              borderRadius: '4px',
              backgroundColor: deepAi.signal === 'BUY' ? 'rgba(16, 185, 129, 0.2)' : deepAi.signal === 'SELL' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(212, 175, 55, 0.2)',
              color: deepAi.signal === 'BUY' ? 'var(--bullish)' : deepAi.signal === 'SELL' ? 'var(--bearish)' : 'var(--gold-primary)',
              border: `1px solid ${deepAi.signal === 'BUY' ? 'var(--bullish)' : deepAi.signal === 'SELL' ? 'var(--bearish)' : 'var(--gold-primary)'}`
            }}>
              SINYAL: {deepAi.signal} ({deepAi.confidence_pct}%)
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', fontSize: '0.78rem', color: '#cbd5e1' }}>
            <div>Bullish: <span style={{ color: 'var(--bullish)', fontWeight: 600 }}>{deepAi.prob_bullish}%</span></div>
            <div>Bearish: <span style={{ color: 'var(--bearish)', fontWeight: 600 }}>{deepAi.prob_bearish}%</span></div>
            {deepAi.suggested_tp > 0 && deepAi.signal !== 'HOLD' && (
              <>
                <div>TP: <span style={{ color: 'var(--bullish)' }}>{deepAi.suggested_tp}</span></div>
                <div>SL: <span style={{ color: 'var(--bearish)' }}>{deepAi.suggested_sl}</span></div>
              </>
            )}
          </div>
        </div>
      )}
      
      <div className="ai-content" style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
        {!isAutoEnabled && logs.length === 0 && (
          <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: '20px' }}>
            [SYSTEM] Auto-Trading OFF. Aktifkan di panel kontrol.
          </div>
        )}

        {isAutoEnabled && logs.length === 0 && !isAnalyzing && (
          <div style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: '20px' }}>
            [SYSTEM] Menunggu data...
          </div>
        )}

        {logs.map((log, idx) => (
          <div key={idx} style={{ 
            marginBottom: '8px', 
            padding: '8px', 
            background: log.isSignal ? 'rgba(212, 175, 55, 0.1)' : 'rgba(255,255,255,0.02)',
            borderLeft: log.isSignal ? '2px solid var(--gold-primary)' : '2px solid rgba(255,255,255,0.1)',
            borderRadius: '4px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '4px' }}>
              <span>[{log.time}] {log.algoName || log.title}</span>
            </div>
            
            <div style={{ color: log.isSignal ? '#fff' : '#94a3b8' }}>
              {log.isSignal ? (
                <>
                  <strong className={log.type === 'bullish' ? 'text-bullish' : 'text-bearish'}>
                    {log.type === 'bullish' ? 'BUY' : 'SELL'} EXECUTED
                  </strong>
                  <br/> {log.message}
                </>
              ) : (
                log.message
              )}
            </div>
            
            {log.levels && (
              <div style={{ marginTop: '4px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Entry: {log.levels.entry.toFixed(2)} | TP: <span className="text-bullish">{log.levels.target.toFixed(2)}</span> | SL: <span className="text-bearish">{log.levels.stop.toFixed(2)}</span>
              </div>
            )}
          </div>
        ))}
        
        {isAnalyzing && (
          <div className="ai-indicator" style={{ borderTop: 'none', paddingTop: 0, marginTop: '8px' }}>
            <Zap size={14} className="text-gold" />
            <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Mesin Algoritma (Node.js) sedang mengamati MT5...</span>
            <div className="typing-indicator">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AIPanel;
