import React, { useState, useEffect, useRef } from 'react';
import { BrainRouter } from './services/BrainRouter';

declare global {
  interface Window {
    cortana: any;
  }
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

const brainRouter = new BrainRouter();

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [listening, setListening] = useState(false);
  const [haloEnabled, setHaloEnabled] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const history = await window.cortana.history.get();
        setMessages(history.slice(-20));
      } catch (error) {
        console.error('Failed to load history:', error);
      }
    };

    const loadConfig = async () => {
      try {
        const config = await window.cortana.config.get();
        setHaloEnabled(config?.ui?.animations?.halo ?? true);
      } catch (error) {
        console.error('Failed to load config:', error);
      }
    };

    loadHistory();
    loadConfig();

    const unsubConfigUpdate = window.cortana.config.onChange((config: any) => {
      setHaloEnabled(config?.ui?.animations?.halo ?? true);
    });

    const unsubWake = window.cortana.voice.onWake(() => {
      setListening(true);
    });

    const checkConnection = async () => {
      const isConnected = await brainRouter.testConnection();
      setConnected(isConnected);
    };
    checkConnection();

    return () => {
      unsubConfigUpdate();
      unsubWake();
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      window.cortana.history.add(userMessage);

      const response = await brainRouter.sendMessage(input);
      const assistantMessage: Message = {
        role: 'assistant',
        content: response,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      window.cortana.history.add(assistantMessage);
      setConnected(true);
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `Error: ${(error as Error).message}`,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMessage]);
      setConnected(false);
    } finally {
      setLoading(false);
    }
  };

  const handleVoiceToggle = async () => {
    try {
      const next = !listening;
      const enabled = await window.cortana.voice.toggle(next);
      setListening(enabled);
    } catch (error) {
      console.error('Voice toggle failed:', error);
    }
  };

  return (
    <div className="app-container">
      <div className="title-bar">
        <h1>Cortana Shell</h1>
        <div className="title-bar-controls">
          <button className="title-bar-button" onClick={() => window.cortana.windowControls.minimize()}>―</button>
          <button className="title-bar-button" onClick={() => window.cortana.windowControls.maximize()}>□</button>
          <button className="title-bar-button" onClick={() => window.cortana.windowControls.close()}>✕</button>
        </div>
      </div>

      <div className="main-content">
        {haloEnabled && (
          <div className="halo-animation">
            <div className="halo-circle"></div>
            <div className="halo-circle"></div>
            <div className="halo-circle"></div>
          </div>
        )}

        <div className="chat-container">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              {msg.content}
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="loading-indicator"></div>
            </div>
          )}
          <div ref={chatEndRef}></div>
        </div>
      </div>

      <div className="input-container">
        <button
          className={`voice-button ${listening ? 'active' : ''}`}
          onClick={handleVoiceToggle}
          title="Voice input (not yet fully implemented)"
        >
          🎤
        </button>
        <input
          type="text"
          className="input-field"
          placeholder="Ask Cortana anything..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          disabled={loading}
        />
        <button className="send-button" onClick={handleSend} disabled={loading}>
          ➤
        </button>
      </div>

      <div className="status-bar">
        <div className="status-indicator">
          <div className={`status-dot ${connected ? '' : 'disconnected'}`}></div>
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
        <span>Cortana Shell 2.0</span>
      </div>
    </div>
  );
};

export default App;
