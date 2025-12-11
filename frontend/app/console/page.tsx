"use client";

import { useState, useRef, useEffect } from "react";
import { Terminal, Send } from "lucide-react";
import apiClient from "@/lib/api-client";

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  adapters?: string[];
  processingTime?: number;
}

export default function ConsolePage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I'm J.A.R.V.I.S. 2v, your modular AI assistant. How can I help you today?",
      timestamp: Date.now(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: "user",
      content: input,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await apiClient.infer({
        query: input,
        context: {},
        features: [],
      });

      const assistantMessage: Message = {
        role: "assistant",
        content: response.response,
        timestamp: Date.now(),
        adapters: response.adapters_used,
        processingTime: response.processing_time,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      const errorMessage: Message = {
        role: "assistant",
        content: `Error: ${err.message}. Make sure the backend is running.`,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="space-y-6 h-[calc(100vh-12rem)]">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-jarvis mb-2">Console</h1>
        <p className="text-gray-400">Chat with J.A.R.V.I.S. through the adapter engine</p>
      </div>

      {/* Chat Container */}
      <div className="glass rounded-lg flex flex-col h-full">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((message, index) => (
            <MessageBubble key={index} message={message} />
          ))}
          {loading && (
            <div className="flex items-center space-x-2 text-gray-400">
              <div className="w-2 h-2 bg-jarvis-primary rounded-full animate-bounce"></div>
              <div
                className="w-2 h-2 bg-jarvis-primary rounded-full animate-bounce"
                style={{ animationDelay: "0.2s" }}
              ></div>
              <div
                className="w-2 h-2 bg-jarvis-primary rounded-full animate-bounce"
                style={{ animationDelay: "0.4s" }}
              ></div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-6 border-t border-jarvis/30">
          <div className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="flex-1 px-4 py-3 bg-jarvis-darker border border-jarvis/30 rounded-lg text-jarvis-primary focus:outline-none focus:border-jarvis"
              disabled={loading}
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="px-6 py-3 bg-jarvis-primary text-black rounded-lg hover:bg-jarvis-secondary transition disabled:opacity-50 flex items-center space-x-2"
            >
              <Send size={20} />
              <span>Send</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-lg p-4 ${
          isUser
            ? "bg-jarvis-primary text-black"
            : "bg-jarvis-darker border border-jarvis/30"
        }`}
      >
        <div className="flex items-center space-x-2 mb-2">
          <span className="font-bold text-sm">
            {isUser ? "You" : "J.A.R.V.I.S."}
          </span>
          <span className="text-xs opacity-60">
            {new Date(message.timestamp).toLocaleTimeString()}
          </span>
        </div>
        <p className={`whitespace-pre-wrap ${isUser ? "text-black" : "text-gray-200"}`}>
          {message.content}
        </p>
        {message.adapters && message.adapters.length > 0 && (
          <div className="mt-2 pt-2 border-t border-jarvis/20">
            <p className="text-xs opacity-75">
              Adapters: {message.adapters.join(", ")}
            </p>
            {message.processingTime && (
              <p className="text-xs opacity-75">
                Processing time: {(message.processingTime * 1000).toFixed(0)}ms
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
