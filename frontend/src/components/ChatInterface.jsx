import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import ResultRenderer from './ResultRenderer';

function ChatInterface({ onTraceUpdate }) {
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hello! I am DataTell. How can I help you with the Olist dataset today?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const endOfMessagesRef = useRef(null);

  const scrollToBottom = () => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userMessage }]);
    setIsLoading(true);
    onTraceUpdate(null);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userMessage }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      
      setMessages(prev => [...prev, { 
        role: 'bot', 
        text: data.final_answer, 
        sqlResult: data.sql_result,
        route: data.route
      }]);
      
      onTraceUpdate(data);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'bot', text: 'Sorry, there was an error processing your request.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div style={{ flex: 1, paddingBottom: '2rem' }}>
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-header">{msg.role === 'user' ? 'You' : 'DataTell'}</div>
            <div className="message-bubble">
              <ResultRenderer text={msg.text} sqlResult={msg.sqlResult} route={msg.route} />
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message bot">
            <div className="message-header">DataTell</div>
            <div className="message-bubble">Thinking...</div>
          </div>
        )}
        <div ref={endOfMessagesRef} />
      </div>
      
      <div className="input-wrapper" style={{ position: 'sticky', bottom: 0 }}>
        <div className="input-container">
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask a question..."
            disabled={isLoading}
          />
          <button onClick={handleSend} disabled={isLoading || !input.trim()}>
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
