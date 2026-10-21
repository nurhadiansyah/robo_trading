import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import { Settings, CheckCircle, Power, PowerOff } from 'lucide-react';

const TradingPanel = () => {
  const [config, setConfig] = useState({ volume: 0.01, tpPoints: 2.0, slPoints: 1.0, momentumThreshold: 0.3, maxSpread: 0.80, algoVersion: ['v3'], executionMode: 'independent' });
  const [isEnabled, setIsEnabled] = useState(false);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('algo_status', (status) => {
      setIsEnabled(status.isEnabled);
      setConfig(status.config);
    });

    return () => newSocket.close();
  }, []);

  const handleToggle = () => {
    if (socket) {
      socket.emit('toggle_algo', !isEnabled);
    }
  };

  const updateConfig = (key, value) => {
    let val = value;
    if (key !== 'algoVersion' && key !== 'executionMode') {
      val = parseFloat(value);
      if (isNaN(val)) return;
    }
    
    // Khusus untuk algoVersion (Checkbox array)
    if (key === 'algoVersion') {
      let currentAlgos = [...(config.algoVersion || [])];
      if (currentAlgos.includes(value)) {
        currentAlgos = currentAlgos.filter(a => a !== value);
      } else {
        currentAlgos.push(value);
      }
      if (currentAlgos.length === 0) return; // Minimal 1 harus terpilih
      val = currentAlgos;
    }
    
    const newConfig = { ...config, [key]: val };
    setConfig(newConfig);
    
    if (socket) {
      socket.emit('set_algo_config', newConfig);
    }
  };

  return (
    <div className="trading-panel glass-panel" style={{ marginTop: '16px', borderColor: isEnabled ? 'rgba(16, 185, 129, 0.5)' : 'rgba(255,255,255,0.1)' }}>
      <div className="ai-header" style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <div className="ai-icon-wrapper" style={{ background: isEnabled ? 'var(--bullish)' : 'rgba(255,255,255,0.1)' }}>
          {isEnabled ? <Power size={18} color="white" /> : <PowerOff size={18} />}
        </div>
        <div className="ai-title">Kontrol Auto-Trading</div>
        
        <div style={{ marginLeft: 'auto' }}>
          <button 
            onClick={handleToggle}
            style={{ 
              background: isEnabled ? 'var(--bearish)' : 'var(--bullish)', 
              color: 'white', border: 'none', padding: '6px 16px', borderRadius: '20px', 
              fontSize: '0.875rem', fontWeight: 'bold', cursor: 'pointer' 
            }}>
            {isEnabled ? 'MATIKAN ROBOT' : 'AKTIFKAN ROBOT'}
          </button>
        </div>
      </div>

      <div className="trading-content" style={{ padding: '16px' }}>
        
        {isEnabled ? (
          <div style={{ padding: '12px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '8px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
             <CheckCircle size={16} className="text-bullish" />
             <span style={{ fontSize: '0.875rem', color: 'var(--bullish)' }}>AI sedang berjalan. Posisi akan dieksekusi secara otomatis jika momentum cocok.</span>
          </div>
        ) : (
          <div style={{ padding: '12px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '8px', marginBottom: '16px' }}>
             <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Robot mati. Tidak ada eksekusi otomatis yang dilakukan.</span>
          </div>
        )}

        <div style={{ display: 'flex', gap: '12px', marginBottom: '12px' }}>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: '4px' }}>Volume (Lot)</label>
            <input 
              type="number" step="0.01" value={config.volume} 
              onChange={(e) => updateConfig('volume', e.target.value)}
              disabled={isEnabled}
              style={{ width: '100%', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', padding: '8px', borderRadius: '4px', outline: 'none', opacity: isEnabled ? 0.5 : 1 }}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: '4px' }}>Sensitivitas ($)</label>
            <input 
              type="number" step="0.1" value={config.momentumThreshold} 
              onChange={(e) => updateConfig('momentumThreshold', e.target.value)}
              disabled={isEnabled}
              title="Pergerakan harga dalam 10 detik untuk memicu sinyal"
              style={{ width: '100%', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', padding: '8px', borderRadius: '4px', outline: 'none', opacity: isEnabled ? 0.5 : 1 }}
            />
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: '4px' }}>Maks Spread ($)</label>
            <input 
              type="number" step="0.05" value={config.maxSpread} 
              onChange={(e) => updateConfig('maxSpread', e.target.value)}
              disabled={isEnabled}
              title="Batas aman maksimal pelebaran spread broker (Mencegah kerugian saat News)"
              style={{ width: '100%', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', padding: '8px', borderRadius: '4px', outline: 'none', opacity: isEnabled ? 0.5 : 1 }}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: '4px' }}>Mode Keputusan Dewan AI</label>
            <select 
              value={config.executionMode || 'independent'}
              onChange={(e) => updateConfig('executionMode', e.target.value)}
              disabled={isEnabled}
              style={{ width: '100%', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', padding: '8px', borderRadius: '4px', outline: 'none', opacity: isEnabled ? 0.5 : 1 }}
            >
              <option value="independent">Mode Independen (Gaya Bebas)</option>
              <option value="confluence">Mode Konfluensi (Wajib Sepakat)</option>
            </select>
          </div>
        </div>

        <div style={{ marginBottom: '16px', background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
          <label style={{ display: 'block', color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: '8px', fontWeight: 'bold' }}>Pilih Spesialis Algoritma (Bisa Lebih dari 1)</label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', opacity: isEnabled ? 0.5 : 1 }}>
            {[
              { id: 'v1', label: 'V1 - Agresif' },
              { id: 'v2', label: 'V2 - Trend Sync' },
              { id: 'v3', label: 'V3 - Sniper (RSI+Spread)' },
              { id: 'v3.1', label: 'V3.1 - Fibo Golden Zone' },
              { id: 'v3.2', label: 'V3.2 - MA Crossover' },
              { id: 'v3.3', label: 'V3.3 - RSI Reversal' },
              { id: 'v3.4', label: 'V3.4 - MACD Crossover' },
              { id: 'v3.5', label: 'V3.5 - Support & Resistance' },
              { id: 'v3.6', label: 'V3.6 - HFT Market Maker' },
              { id: 'v4', label: '⚡ V4 - Deep Learning (5Y Model)' }
            ].map(algo => (
              <label key={algo.id} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', cursor: isEnabled ? 'not-allowed' : 'pointer' }}>
                <input 
                  type="checkbox"
                  checked={(config.algoVersion || []).includes(algo.id)}
                  onChange={() => updateConfig('algoVersion', algo.id)}
                  disabled={isEnabled}
                  style={{ cursor: isEnabled ? 'not-allowed' : 'pointer' }}
                />
                <span style={{ color: (config.algoVersion || []).includes(algo.id) ? 'var(--gold-primary)' : 'var(--text-muted)' }}>
                  {algo.label}
                </span>
              </label>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: '4px' }}>Take Profit ($)</label>
            <input 
              type="number" step="0.1" value={config.tpPoints} 
              onChange={(e) => updateConfig('tpPoints', e.target.value)}
              disabled={isEnabled}
              style={{ width: '100%', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', color: 'var(--bullish)', padding: '8px', borderRadius: '4px', outline: 'none', opacity: isEnabled ? 0.5 : 1 }}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: '4px' }}>Stop Loss ($)</label>
            <input 
              type="number" step="0.1" value={config.slPoints} 
              onChange={(e) => updateConfig('slPoints', e.target.value)}
              disabled={isEnabled}
              style={{ width: '100%', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: 'var(--bearish)', padding: '8px', borderRadius: '4px', outline: 'none', opacity: isEnabled ? 0.5 : 1 }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingPanel;
