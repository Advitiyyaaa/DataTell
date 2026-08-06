import React, { useState } from 'react';
import ChatInterface from './components/ChatInterface';
import ReasoningTrace from './components/ReasoningTrace';
import { Sidebar } from 'lucide-react';
import './index.css';

function App() {
  const [traceData, setTraceData] = useState(null);
  const [isTraceOpen, setIsTraceOpen] = useState(false);

  return (
    <div className="app-container">
      <main className="main-content" style={{ marginRight: isTraceOpen ? '400px' : '0' }}>
        <button 
          className="toggle-trace-btn"
          onClick={() => setIsTraceOpen(!isTraceOpen)}
          title="Toggle Reasoning Trace"
        >
          <Sidebar size={20} />
        </button>
        <ChatInterface onTraceUpdate={(data) => {
          setTraceData(data);
          if (data && !isTraceOpen) {
            setIsTraceOpen(true);
          }
        }} />
      </main>
      <ReasoningTrace data={traceData} isOpen={isTraceOpen} onClose={() => setIsTraceOpen(false)} />
    </div>
  );
}

export default App;
