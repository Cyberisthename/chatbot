import { contextBridge, ipcRenderer } from 'electron';
import type { AppConfig, BrainRouterRequest, BrainRouterResult, ChatMessage, CortanaState, Reminder, VoiceState } from '../core/types/index.js';

const listenerMap = new Map<string, Map<(...args: unknown[]) => void, (...args: unknown[]) => void>>();

const cortanaAPI = {
  config: {
    get: (): Promise<AppConfig> => ipcRenderer.invoke('getConfig'),
    update: (partial: Partial<AppConfig>): Promise<AppConfig> => ipcRenderer.invoke('updateConfig', partial)
  },

  brain: {
    sendMessage: (request: BrainRouterRequest): Promise<BrainRouterResult> => ipcRenderer.invoke('sendMessage', request),
    clearHistory: (): Promise<void> => ipcRenderer.invoke('clearHistory'),
    getHistory: (): Promise<ChatMessage[]> => ipcRenderer.invoke('getHistory'),
    getStatus: (): Promise<{ jarvis: 'online' | 'offline' | 'checking'; mode: 'jarvis' | 'local' | 'hybrid' }> =>
      ipcRenderer.invoke('getBrainStatus')
  },

  voice: {
    startListening: (): Promise<void> => ipcRenderer.invoke('startListening'),
    stopListening: (): Promise<void> => ipcRenderer.invoke('stopListening'),
    speak: (text: string): Promise<void> => ipcRenderer.invoke('speak', text),
    getState: (): Promise<VoiceState> => ipcRenderer.invoke('getVoiceState')
  },

  reminders: {
    getAll: (): Promise<Reminder[]> => ipcRenderer.invoke('getReminders'),
    cancel: (id: string): Promise<Reminder | undefined> => ipcRenderer.invoke('cancelReminder', id)
  },

  window: {
    minimize: (): Promise<void> => ipcRenderer.invoke('minimize'),
    close: (): Promise<void> => ipcRenderer.invoke('close')
  },

  state: {
    get: (): Promise<CortanaState> => ipcRenderer.invoke('getState')
  },

  on: (channel: string, callback: (...args: unknown[]) => void): void => {
    const validChannels = [
      'brain:response',
      'brain:statusChange',
      'voice:wakeWord',
      'voice:speechRecognized',
      'voice:stateChanged',
      'reminder:fired',
      'tool:executed',
      'config:updated',
      'cortana:stateChanged'
    ];

    if (validChannels.includes(channel)) {
      const wrapper = (_event: unknown, ...args: unknown[]): void => callback(...args);

      if (!listenerMap.has(channel)) {
        listenerMap.set(channel, new Map());
      }

      listenerMap.get(channel)!.set(callback, wrapper);
      ipcRenderer.on(channel, wrapper);
    }
  },

  off: (channel: string, callback: (...args: unknown[]) => void): void => {
    const channelListeners = listenerMap.get(channel);
    if (channelListeners) {
      const wrapper = channelListeners.get(callback);
      if (wrapper) {
        ipcRenderer.removeListener(channel, wrapper);
        channelListeners.delete(callback);
      }
    }
  }
};

contextBridge.exposeInMainWorld('cortana', cortanaAPI);

export type CortanaAPI = typeof cortanaAPI;
