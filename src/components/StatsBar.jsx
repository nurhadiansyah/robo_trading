import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Activity, BarChart2 } from 'lucide-react';

const StatsBar = () => {
  const [stats, setStats] = useState({
    price: 2450.00,
    changePercent: 0.25,
    high: 2465.50,
    low: 2435.20,
    trend: 'Bullish'
  });

  useEffect(() => {
    // Listen to simulated tick events from ChartArea
    const handleTick = (e) => {
      const { price, open, high, low } = e.detail;
      
      const change = price - open;
      const changePercent = (change / open) * 100;
      
      setStats(prev => ({
        price: price,
        changePercent: changePercent,
        high: Math.max(prev.high, high),
        low: Math.min(prev.low, low),
        trend: changePercent >= 0 ? 'Bullish' : 'Bearish'
      }));
    };

    window.addEventListener('simulated-tick', handleTick);

    return () => {
      window.removeEventListener('simulated-tick', handleTick);
    };
  }, []);

  const isPositive = stats.changePercent >= 0;

  return (
    <div className="stats-bar">
      <div className="stat-card glass-panel">
        <div className="stat-label">Harga Terkini (XAU/USD)</div>
        <div className="stat-value text-gold">
          ${stats.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </div>
      </div>
      
      <div className="stat-card glass-panel">
        <div className="stat-label">Perubahan Sesi Ini</div>
        <div className="stat-value">
          <span className={`stat-change ${!isPositive ? 'negative' : ''}`}>
            {isPositive ? <TrendingUp size={16} style={{ marginRight: '4px' }} /> : <TrendingDown size={16} style={{ marginRight: '4px' }} />}
            {stats.changePercent > 0 ? '+' : ''}{stats.changePercent.toFixed(2)}%
          </span>
        </div>
      </div>
      
      <div className="stat-card glass-panel">
        <div className="stat-label">High / Low Sesi Ini</div>
        <div className="stat-value" style={{ fontSize: '1.25rem' }}>
          ${stats.high.toLocaleString('en-US', { minimumFractionDigits: 2 })} <span style={{ color: 'var(--text-muted)', margin: '0 8px', fontSize: '1rem' }}>/</span> ${stats.low.toLocaleString('en-US', { minimumFractionDigits: 2 })}
        </div>
      </div>
      
      <div className="stat-card glass-panel">
        <div className="stat-label">Tren Pasar (Sesi)</div>
        <div className={`stat-value ${isPositive ? 'text-bullish' : 'text-bearish'}`} style={{ fontSize: '1.25rem' }}>
          <Activity size={20} style={{ marginRight: '8px' }} />
          {stats.trend}
        </div>
      </div>
    </div>
  );
};

export default StatsBar;
