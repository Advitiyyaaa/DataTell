import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, AlertCircle } from 'lucide-react';
import ResultRenderer from './ResultRenderer';

const API_BASE = 'http://localhost:8000';

function ChatInterface({ onTraceUpdate }) {
  const [messages, setMessages] = useState([]);
  // conversationHistory tracks prior turns to give the agent multi-turn context
  const [conversationHistory, setConversationHistory] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  // isThinking = backend is processing but no token has arrived yet
  const [isThinking, setIsThinking] = useState(false);
  // thinkingRoute = route shown in thinking bubble once meta event fires
  const [thinkingRoute, setThinkingRoute] = useState(null);
  const endOfMessagesRef = useRef(null);
  // Abort controller to cancel an in-flight stream
  const abortControllerRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Append a character / token to the last bot message incrementally
  const appendToLastBotMessage = useCallback((token) => {
    setMessages(prev => {
      const updated = [...prev];
      const last = updated[updated.length - 1];
      if (last && last.role === 'bot' && last.streaming) {
        updated[updated.length - 1] = { ...last, text: (last.text || '') + token };
      }
      return updated;
    });
  }, []);

  const handleSend = async () => {
    const userText = input.trim();
    if (!userText || isLoading) return;

    setInput('');

    // Add user message
    const userMessage = { role: 'user', text: userText };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setIsThinking(true);
    setThinkingRoute(null);
    onTraceUpdate(null);

    // Abort any previous stream
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    // Add an empty streaming bot placeholder (hidden while isThinking=true)
    const botPlaceholder = { role: 'bot', text: '', streaming: true, sqlResult: null, route: null };
    setMessages(prev => [...prev, botPlaceholder]);

    // Build history from prior messages (not including the current question yet)
    const historyPayload = conversationHistory;

    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userText,
          conversation_history: historyPayload,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finalSqlResult = null;
      let finalRoute = null;
      let finalRouteReason = null;
      let finalAnswer = '';

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by double newlines
        const frames = buffer.split('\n\n');
        // The last element may be an incomplete frame — keep it in the buffer
        buffer = frames.pop() ?? '';

        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith('data: ')) continue;

          let event;
          try {
            event = JSON.parse(line.slice(6)); // strip "data: "
          } catch {
            continue;
          }

          if (event.type === 'meta') {
            // Routing info available immediately — update trace sidebar early
            onTraceUpdate({ route: event.route, route_reason: event.route_reason });
            finalRoute = event.route;
            finalRouteReason = event.route_reason;
            setThinkingRoute(event.route);
          } else if (event.type === 'token') {
            // First token: hide the thinking bubble
            setIsThinking(false);
            appendToLastBotMessage(event.text);
            finalAnswer += event.text;
          } else if (event.type === 'done') {
            finalSqlResult = event.sql_result;
            // Mark the bot message as finished streaming + attach metadata
            setMessages(prev => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last && last.role === 'bot') {
                updated[updated.length - 1] = {
                  ...last,
                  streaming: false,
                  sqlResult: event.sql_result,
                  route: finalRoute,
                };
              }
              return updated;
            });
            // Push full trace data to sidebar — merge with meta info already shown
            onTraceUpdate({
              route: finalRoute,
              route_reason: finalRouteReason,
              sql_result: event.sql_result,
              rag_result: event.rag_result,
              metadata: event.metadata,
              answer_quality: event.answer_quality,
              critique: event.critique,
              retry_count: event.retry_count,
              total_ms: event.total_ms,
            });
            // Update conversation history so the next question has context
            setConversationHistory(prev => [
              ...prev,
              { role: 'user', content: userText },
              { role: 'assistant', content: finalAnswer.trim() },
            ]);
          } else if (event.type === 'error') {
            throw new Error(event.message);
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') return;
      console.error('Stream error:', error);
      setIsThinking(false);
      // Replace the streaming placeholder with a styled error bubble
      setMessages(prev => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.role === 'bot' && last.streaming) {
          updated[updated.length - 1] = {
            ...last,
            text: null,
            streaming: false,
            error: error.message || 'An unexpected error occurred. Please try again.',
          };
        }
        return updated;
      });
    } finally {
      setIsLoading(false);
      setIsThinking(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // prevent any form submission or newline
      handleSend();
    }
  };

  return (
    <div className={`chat-container ${messages.length === 0 ? 'centered-layout' : ''}`}>
      {messages.length === 0 ? (
        <div className="hero-greeting">
          <h1>Hello! I am DataTell.</h1>
          <p>How can I help you with the Olist dataset today?</p>
        </div>
      ) : (
        <div className="messages-area">
          <div className="messages-content">
            {messages.map((msg, idx) => {
              // Hide the streaming placeholder bubble while thinking (before first token)
              const isHiddenPlaceholder =
                msg.role === 'bot' && msg.streaming && msg.text === '' && isThinking;
              if (isHiddenPlaceholder) return null;

              return (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-header">{msg.role === 'user' ? 'You' : 'DataTell'}</div>
                  <div className="message-bubble">
                    {msg.error ? (
                      <div className="error-bubble">
                        <AlertCircle size={16} style={{ flexShrink: 0 }} />
                        <span>{msg.error}</span>
                      </div>
                    ) : (
                      <>
                        <ResultRenderer
                          text={msg.text}
                          sqlResult={msg.sqlResult}
                          route={msg.route}
                          isStreaming={msg.streaming}
                        />
                        {/* Blinking cursor while streaming */}
                        {msg.streaming && <span className="streaming-cursor" aria-hidden="true" />}
                      </>
                    )}
                  </div>
                </div>
              );
            })}

            {/* Thinking animation — shown while backend processes, before first token */}
            {isThinking && (
              <div className="message bot">
                <div className="message-header">DataTell</div>
                <div className="message-bubble thinking-bubble">
                  {thinkingRoute ? (
                    <span className="thinking-label thinking-route">
                      {thinkingRoute === 'SQL' && '🔍 Querying database'}
                      {thinkingRoute === 'RAG' && '📄 Searching knowledge base'}
                      {thinkingRoute === 'BOTH' && '🔍📄 Querying database & knowledge base'}
                      {thinkingRoute === 'CHITCHAT' && '💬 Thinking'}
                      {!['SQL','RAG','BOTH','CHITCHAT'].includes(thinkingRoute) && '⚙️ Thinking'}
                    </span>
                  ) : (
                    <span className="thinking-label">Thinking</span>
                  )}
                  <span className="thinking-dots">
                    <span /><span /><span />
                  </span>
                </div>
              </div>
            )}
            <div ref={endOfMessagesRef} />
          </div>
        </div>
      )}

      <div className="input-wrapper">
        <div className="input-container">
          <input
            id="chat-input"
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question…"
            disabled={isLoading}
            autoComplete="off"
          />
          <button
            id="chat-send-btn"
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            aria-label="Send message"
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
