import React from 'react';
import { Activity, Bell, Settings } from 'lucide-react';

const Header = ({ status, isMT5Active }) => {
  return (
    <header className="header glass-panel">
      <div className="header-brand">
        <Activity className="logo-icon" size={28} />
        <span>RoboTrade <span className="text-gold">XAU</span></span>
      </div>
      
      <div className="header-actions">
        <div className="status-badge" style={{ border: isMT5Active ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(239, 68, 68, 0.3)' }}>
          <div className="status-dot" style={{ background: isMT5Active ? 'var(--bullish)' : 'var(--bearish)' }}></div>
          <span style={{ color: isMT5Active ? 'var(--bullish)' : 'var(--bearish)' }}>
             {isMT5Active ? 'MT5 Online (Live)' : 'MT5 Offline / Waiting...'}
          </span>
        </div>
        <button className="icon-btn" style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
          <Bell size={20} />
        </button>
        <button className="icon-btn" style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
          <Settings size={20} />
        </button>
      </div>
    </header>
  );
};

export default Header;
