import React, { useState, useEffect, useRef, memo } from 'react';
import { Activity, ArrowDownRight, ArrowUpRight, Maximize2, Settings } from 'lucide-react';

const TickRow = memo(({ tick, prevTick, isEven }) => {
  let color = '#fff';
  let Icon = null;
  
  if (prevTick) {
    if (tick.price > prevTick.price) {
      color = 'var(--bullish)';
      Icon = <ArrowUpRight size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '4px' }} />;
    } else if (tick.price < prevTick.price) {
      color = 'var(--bearish)';
      Icon = <ArrowDownRight size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '4px' }} />;
    }
  }

  return (
    <tr style={{ 
      background: isEven ? 'rgba(0,0,0,0.2)' : 'transparent',
      borderBottom: '1px solid rgba(255,255,255,0.02)'
    }}>
      <td style={{ padding: '8px', color: 'var(--text-muted)' }}>{tick.time}</td>
      <td style={{ padding: '8px', color: color, fontWeight: 600 }}>
        {Icon}
        {tick.price.toFixed(2)}
      </td>
      <td style={{ padding: '8px' }}>{tick.bid.toFixed(2)}</td>
      <td style={{ padding: '8px' }}>{tick.ask.toFixed(2)}</td>
      <td style={{ padding: '8px', color: tick.spread > 0.4 ? 'var(--bearish)' : 'var(--bullish)' }}>
        {tick.spread}
      </td>
      <td style={{ padding: '8px', color: 'var(--text-muted)' }}>{tick.volume}</td>
    </tr>
  );
});

const ChartArea = () => {
  const [ticks, setTicks] = useState([]);
  const listRef = useRef(null);

  useEffect(() => {
    const handleTick = (e) => {
      const { price, bid, ask, volume, candleTime } = e.detail;
      if (!price) return;

      const newTick = {
        id: Date.now() + Math.random(), // Unique ID for key
        time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute:'2-digit', second:'2-digit' }),
        price,
        bid,
        ask,
        spread: (ask - bid).toFixed(2),
        volume
      };

      setTicks(prev => {
        // Simpan 100 tick terakhir
        const updated = [newTick, ...prev];
        if (updated.length > 100) updated.pop();
        return updated;
      });
    };

    window.addEventListener('simulated-tick', handleTick);

    return () => {
      window.removeEventListener('simulated-tick', handleTick);
    };
  }, []);

  return (
    <div className="chart-container glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="chart-header" style={{ marginBottom: 0, paddingBottom: '16px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <div className="chart-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={18} className="text-gold" />
          <span>Raw Data Stream</span>
          <span className="text-muted" style={{ fontSize: '0.875rem' }}> / MT5 Tick Activity</span>
        </div>
        <div className="chart-tools">
          <button className="icon-btn"><Settings size={18} /></button>
          <button className="icon-btn"><Maximize2 size={18} /></button>
        </div>
      </div>
      
      <div 
        ref={listRef}
        style={{ 
          flex: 1, 
          overflowY: 'auto', 
          fontFamily: 'monospace', 
          fontSize: '0.875rem',
          padding: '16px 0'
        }}
      >
        <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
              <th style={{ padding: '8px' }}>Time</th>
              <th style={{ padding: '8px' }}>Price</th>
              <th style={{ padding: '8px' }}>Bid</th>
              <th style={{ padding: '8px' }}>Ask</th>
              <th style={{ padding: '8px' }}>Spread</th>
              <th style={{ padding: '8px' }}>Vol</th>
            </tr>
          </thead>
          <tbody>
            {ticks.length === 0 && (
              <tr>
                <td colSpan="6" style={{ textAlign: 'center', padding: '20px', color: 'var(--text-muted)' }}>
                  Menunggu aliran data dari MT5...
                </td>
              </tr>
            )}
            {ticks.map((tick, idx) => (
              <TickRow 
                key={tick.id} 
                tick={tick} 
                prevTick={ticks[idx + 1]} 
                isEven={idx % 2 === 0} 
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ChartArea;
