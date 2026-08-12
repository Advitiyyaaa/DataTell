import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import ReasoningTrace from './components/ReasoningTrace';
import { Sidebar, Sun, Moon } from 'lucide-react';
import './index.css';

function App() {
  const [traceData, setTraceData] = useState(null);
  const [isTraceOpen, setIsTraceOpen] = useState(false);
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  return (
    <>
      <div className="wavy-background" />
      <div className="app-container">
        <main className="main-content" style={{ marginRight: isTraceOpen ? '400px' : '0' }}>
          <div className="top-bar">
            <button 
              className="theme-toggle-btn"
              onClick={toggleTheme}
              title="Toggle Theme"
            >
              {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
            </button>
            <button 
              className="toggle-trace-btn"
              onClick={() => setIsTraceOpen(!isTraceOpen)}
              title="Toggle Reasoning Trace"
            >
              <Sidebar size={20} />
            </button>
          </div>
          <ChatInterface onTraceUpdate={(data) => {
            setTraceData(data);
            if (data && !isTraceOpen) {
              setIsTraceOpen(true);
            }
          }} />
        </main>
        <ReasoningTrace data={traceData} isOpen={isTraceOpen} onClose={() => setIsTraceOpen(false)} />
      </div>
    </>
  );
}

export default App;
