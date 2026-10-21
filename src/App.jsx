import React, { useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import Header from './components/Header';
import StatsBar from './components/StatsBar';
import ChartArea from './components/ChartArea';
import AIPanel from './components/AIPanel';
import TradingPanel from './components/TradingPanel';
import './index.css'; 

function App() {
  const [connectionStatus, setConnectionStatus] = useState('Terputus');
  const [isMT5Active, setIsMT5Active] = useState(false);
  const lastTickTime = React.useRef(0);

  useEffect(() => {
    // Hubungkan ke Backend Lokal (Node.js) yang menerima data dari MT5
    const socket = io('http://localhost:5000');

    socket.on('connect', () => {
      setConnectionStatus('Terhubung ke MT5 Bridge');
    });

    socket.on('disconnect', () => {
      setConnectionStatus('Terputus');
    });

    // Mendengarkan data tick langsung dari MT5
    socket.on('mt5_tick', (data) => {
      lastTickTime.current = Date.now();
      setIsMT5Active(prev => prev ? prev : true);
      
      // Dispatch custom event agar StatsBar dan AIPanel menerima harga asli dari MT5
      const event = new CustomEvent('simulated-tick', {
        detail: {
          price: data.price,
          bid: data.bid,
          ask: data.ask,
          open: data.open,
          high: data.high,
          low: data.low,
          volume: data.volume,
          candleTime: data.candleTime
        }
      });
      window.dispatchEvent(event);
    });

    // Heartbeat checker: Jika tidak ada data dari MT5 lebih dari 3 detik, anggap offline
    const heartbeat = setInterval(() => {
        if (Date.now() - lastTickTime.current > 3000) {
            setIsMT5Active(false);
        }
    }, 1000);

    return () => {
      clearInterval(heartbeat);
      socket.disconnect();
    };
  }, []);

  return (
    <div className="app-container">
      <Header status={connectionStatus} isMT5Active={isMT5Active} />
      <div className="dashboard-main">
        <div className="left-column">
          <StatsBar />
          <ChartArea />
        </div>
        <div className="right-column">
          <AIPanel />
          <TradingPanel />
        </div>
      </div>
    </div>
  );
}

export default App;
