import { contextBridge, ipcRenderer } from 'electron';
import type { CortanaAPI, CortanaChatEntry, CortanaConfig } from '../shared/types/cortana';

const cortanaApi: CortanaAPI = {
  config: {
    get: () => ipcRenderer.invoke('cortana/config:get') as Promise<CortanaConfig>,
    update: (path: string, value: unknown) =>
      ipcRenderer.send('cortana/config:update', { path, value }),
    onChange: (callback) => {
      const handler = (_: unknown, config: CortanaConfig) => callback(config);
      ipcRenderer.on('cortana/config:update', handler);
      return () => ipcRenderer.removeListener('cortana/config:update', handler);
    },
  },

  history: {
    get: () => ipcRenderer.invoke('cortana/history:get') as Promise<CortanaChatEntry[]>,
    add: (entry) => ipcRenderer.send('cortana/history:add', entry),
  },

  voice: {
    toggle: (enabled) => ipcRenderer.invoke('cortana/voice:toggle', enabled),
    onWake: (callback) => {
      const handler = () => callback();
      ipcRenderer.on('cortana/voice:wake', handler);
      return () => ipcRenderer.removeListener('cortana/voice:wake', handler);
    },
  },

  tools: {
    list: () => ipcRenderer.invoke('cortana/tools:list') as Promise<Array<{ name: string; description: string }>>,
    execute: (tool, args) => ipcRenderer.invoke('cortana/tools:execute', { tool, args }),
  },

  ui: {
    onUpdate: (callback) => {
      const handler = (_: unknown, ui: Record<string, unknown>) => callback(ui);
      ipcRenderer.on('cortana/ui-update', handler);
      return () => ipcRenderer.removeListener('cortana/ui-update', handler);
    },
  },

  external: {
    open: (url) => ipcRenderer.send('cortana/external-link', url),
  },

  windowControls: {
    minimize: () => ipcRenderer.send('cortana/window:minimize'),
    maximize: () => ipcRenderer.send('cortana/window:maximize'),
    close: () => ipcRenderer.send('cortana/window:close'),
    isMaximized: () => ipcRenderer.invoke('cortana/window:isMaximized'),
  },
};

contextBridge.exposeInMainWorld('cortana', cortanaApi);

export type { CortanaAPI };
