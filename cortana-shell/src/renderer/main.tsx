import React, { useEffect, useState, useRef } from 'react';
import ReactDOM from 'react-dom/client';
import type { CortanaAPI } from '../preload/index.js';
import type { ChatMessage, CortanaState } from '../core/types/index.js';
import './styles.css';

declare global {
  interface Window {
    cortana: CortanaAPI;
  }
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [cortanaState, setCortanaState] = useState<CortanaState>('idle');
  const [jarvisStatus, setJarvisStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [mode, setMode] = useState<'jarvis' | 'local' | 'hybrid'>('jarvis');
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void loadInitialData();

    const handleCortanaStateChange = (newState: CortanaState): void => {
      setCortanaState(newState);
    };

    const handleBrainStatusChange = (status: { jarvis: 'online' | 'offline' | 'checking'; mode: 'jarvis' | 'local' | 'hybrid' }): void => {
      setJarvisStatus(status.jarvis);
      setMode(status.mode);
    };

    const handleSpeechRecognized = (text: string): void => {
      addUserMessage(text);
    };

    const handleBrainResponse = (response: { reply: string }): void => {
      if (response.reply) {
        addAssistantMessage(response.reply);
      }
    };

    window.cortana.on('cortana:stateChanged', handleCortanaStateChange);
    window.cortana.on('brain:statusChange', handleBrainStatusChange);
    window.cortana.on('voice:speechRecognized', handleSpeechRecognized);
    window.cortana.on('brain:response', handleBrainResponse);

    return () => {
      window.cortana.off('cortana:stateChanged', handleCortanaStateChange as (...args: unknown[]) => void);
      window.cortana.off('brain:statusChange', handleBrainStatusChange as (...args: unknown[]) => void);
      window.cortana.off('voice:speechRecognized', handleSpeechRecognized as (...args: unknown[]) => void);
      window.cortana.off('brain:response', handleBrainResponse as (...args: unknown[]) => void);
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadInitialData = async (): Promise<void> => {
    try {
      const [history, status, state] = await Promise.all([
        window.cortana.brain.getHistory(),
        window.cortana.brain.getStatus(),
        window.cortana.state.get()
      ]);

      setMessages(history);
      setJarvisStatus(status.jarvis);
      setMode(status.mode);
      setCortanaState(state);
    } catch (error) {
      console.error('Failed to load initial data:', error);
    }
  };

  const addUserMessage = (content: string): void => {
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: Date.now()
    };
    setMessages((prev) => [...prev, newMessage]);
  };

  const addAssistantMessage = (content: string): void => {
    const newMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content,
      timestamp: Date.now()
    };
    setMessages((prev) => [...prev, newMessage]);
  };

  const handleSendMessage = async (): Promise<void> => {
    if (!inputValue.trim() || isProcessing) return;

    const message = inputValue.trim();
    setInputValue('');
    setIsProcessing(true);

    addUserMessage(message);

    try {
      const response = await window.cortana.brain.sendMessage({
        prompt: message,
        allowTools: true
      });

      if (response.reply) {
        addAssistantMessage(response.reply);
      }

      if (response.error) {
        addAssistantMessage(`Error: ${response.error}`);
      }
    } catch (error) {
      addAssistantMessage(`Failed to send message: ${String(error)}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSendMessage();
    }
  };

  const handleMicClick = async (): Promise<void> => {
    try {
      if (cortanaState === 'listening') {
        await window.cortana.voice.stopListening();
      } else {
        await window.cortana.voice.startListening();
      }
    } catch (error) {
      console.error('Voice error:', error);
    }
  };

  const handleClearHistory = async (): Promise<void> => {
    await window.cortana.brain.clearHistory();
    setMessages([]);
  };

  const handleMinimize = (): void => {
    void window.cortana.window.minimize();
  };

  const handleClose = (): void => {
    void window.cortana.window.close();
  };

  const getStateIndicator = (): string => {
    switch (cortanaState) {
      case 'listening':
        return 'Listening...';
      case 'thinking':
        return 'Thinking...';
      case 'speaking':
        return 'Speaking...';
      case 'offline':
        return 'Offline';
      default:
        return 'Ready';
    }
  };

  const getStatusColor = (): string => {
    if (cortanaState === 'offline') return '#ff4444';
    if (jarvisStatus === 'offline') return '#ffa500';
    return '#0078d4';
  };

  return (
    <div className="app">
      <div className="title-bar">
        <div className="title-bar-left">
          <div className="cortana-icon" />
          <span className="title-text">Cortana 2.0</span>
        </div>
        <div className="title-bar-right">
          <button className="title-button" onClick={handleMinimize}>
            ─
          </button>
          <button className="title-button close" onClick={handleClose}>
            ✕
          </button>
        </div>
      </div>

      <div className="status-bar">
        <div className="status-indicator" style={{ backgroundColor: getStatusColor() }} />
        <span className="status-text">{getStateIndicator()}</span>
        <span className="status-mode">Mode: {mode}</span>
        {jarvisStatus === 'offline' && <span className="status-warning">⚠ Jarvis Offline</span>}
      </div>

      <div className="cortana-halo-container">
        <div className={`cortana-halo ${cortanaState}`}>
          <div className="halo-ring" />
          <div className="halo-ring ring-2" />
          <div className="halo-glow" />
        </div>
      </div>

      <div className="conversation-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h2>Hello! I'm Cortana 2.0</h2>
            <p>Powered by your personal AI infrastructure</p>
            <div className="suggestions">
              <button onClick={() => setInputValue('What can you do?')}>What can you do?</button>
              <button onClick={() => setInputValue('Open Chrome')}>Open Chrome</button>
              <button onClick={() => setInputValue('Set a reminder')}>Set a reminder</button>
            </div>
          </div>
        ) : (
          <div className="messages">
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.role}`}>
                <div className="message-avatar">{message.role === 'user' ? '👤' : '🔵'}</div>
                <div className="message-content">{message.content}</div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="input-container">
        <button className="mic-button" onClick={handleMicClick} disabled={isProcessing}>
          {cortanaState === 'listening' ? '🎤' : '🎙️'}
        </button>
        <input
          type="text"
          className="message-input"
          placeholder="Ask me anything..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isProcessing}
        />
        <button className="send-button" onClick={handleSendMessage} disabled={isProcessing || !inputValue.trim()}>
          ➤
        </button>
      </div>

      <div className="action-bar">
        <button className="action-button" onClick={handleClearHistory}>
          Clear History
        </button>
      </div>
    </div>
  );
};

const root = document.getElementById('root');
if (root) {
  ReactDOM.createRoot(root).render(<App />);
}
