import { app, BrowserWindow, dialog, ipcMain, nativeTheme, shell, Tray, Menu, globalShortcut } from 'electron';
import path from 'node:path';
import { URL } from 'node:url';
import fs from 'node:fs';
import { ConfigManager } from './services/ConfigManager';
import { VoiceManager } from './services/VoiceManager';
import { ToolManager } from './services/ToolManager';
import { HistoryStore } from './services/HistoryStore';

const isDevelopment = !app.isPackaged;

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;

const configManager = new ConfigManager(path.resolve(__dirname, '../../config.yaml'));
const historyStore = new HistoryStore();
const voiceManager = new VoiceManager({
  modelDirectory: path.resolve(__dirname, '../../assets/wake-word-models'),
  defaultModel: 'heycortana_enUS.table',
  enabled: configManager.current.voice?.wakeWord?.enabled ?? false,
});
const toolManager = new ToolManager();

function createWindow(): void {
  const { app: appCfg, ui } = configManager.current;

  const window = new BrowserWindow({
    title: appCfg?.name ?? 'Cortana Shell',
    width: 1200,
    height: 720,
    minWidth: 960,
    minHeight: 600,
    vibrancy: 'dark',
    visualEffectState: 'active',
    frame: false,
    transparent: true,
    show: false,
    hasShadow: false,
    roundedCorners: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  window.once('ready-to-show', () => {
    window.show();
  });

  window.on('closed', () => {
    mainWindow = null;
  });

  const pageUrl = isDevelopment && process.env.VITE_DEV_SERVER_URL
    ? process.env.VITE_DEV_SERVER_URL
    : new URL(
        `file://${path.join(__dirname, '../renderer/index.html')}`,
      ).toString();

  window.loadURL(pageUrl).catch((error) => {
    dialog.showErrorBox('Failed to load UI', String(error));
  });

  if (isDevelopment) {
    window.webContents.openDevTools({ mode: 'detach' }).catch(() => undefined);
  }

  return window;
}

function registerShortcuts(): void {
  const hotkey = configManager.current.app?.hotkey ?? 'CommandOrControl+Alt+C';
  globalShortcut.unregisterAll();
  const registered = globalShortcut.register(hotkey, () => {
    if (!mainWindow) {
      mainWindow = createWindow();
      return;
    }

    if (mainWindow.isVisible()) {
      mainWindow.hide();
    } else {
      mainWindow.show();
      mainWindow.focus();
    }
  });

  if (!registered) {
    dialog.showErrorBox('Shortcut Registration Failed', `Unable to register global shortcut ${hotkey}.`);
  }
}

function createTray(): void {
  const iconPath = path.resolve(__dirname, '../../assets/icons/cortana.png');
  if (!fs.existsSync(iconPath)) {
    return;
  }

  tray = new Tray(iconPath);
  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Open Cortana',
      click: () => {
        if (!mainWindow) {
          mainWindow = createWindow();
        } else {
          mainWindow.show();
        }
      },
    },
    { type: 'separator' },
    {
      label: 'Toggle Halo Animation',
      type: 'checkbox',
      checked: configManager.current.ui?.animations?.halo ?? true,
      click: (menuItem) => {
        configManager.update('ui.animations.halo', menuItem.checked);
        mainWindow?.webContents.send('cortana/ui-update', {
          animations: { halo: menuItem.checked },
        });
      },
    },
    {
      label: 'Quit',
      click: () => {
        app.quit();
      },
    },
  ]);
  tray.setToolTip(configManager.current.app?.name ?? 'Cortana Shell');
  tray.setContextMenu(contextMenu);

  tray.on('double-click', () => {
    if (!mainWindow) {
      mainWindow = createWindow();
      return;
    }
    mainWindow.show();
    mainWindow.focus();
  });
}

function wireIpc(): void {
  ipcMain.handle('cortana/config:get', () => configManager.current);

  ipcMain.on('cortana/config:update', (_event, { path: cfgPath, value }) => {
    configManager.update(cfgPath, value);
  });

  ipcMain.handle('cortana/history:get', () => historyStore.getHistory());
  ipcMain.on('cortana/history:add', (_event, entry) => {
    historyStore.addEntry(entry);
  });

  ipcMain.handle('cortana/voice:toggle', (_event, enabled: boolean) => {
    voiceManager.setEnabled(enabled);
    return voiceManager.isEnabled();
  });

  voiceManager.onWake(() => {
    mainWindow?.webContents.send('cortana/voice:wake');
  });

  ipcMain.handle('cortana/tools:list', () => toolManager.listTools());
  ipcMain.handle('cortana/tools:execute', async (_event, payload) => {
    try {
      const result = await toolManager.execute(payload.tool, payload.args);
      return { ok: true, result };
    } catch (error) {
      return { ok: false, error: (error as Error).message };
    }
  });

  ipcMain.on('cortana/external-link', (_event, url: string) => {
    shell.openExternal(url).catch(() => undefined);
  });

  ipcMain.on('cortana/window:minimize', () => {
    mainWindow?.minimize();
  });

  ipcMain.on('cortana/window:maximize', () => {
    if (!mainWindow) {
      return;
    }
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow.maximize();
    }
  });

  ipcMain.on('cortana/window:close', () => {
    mainWindow?.close();
  });

  ipcMain.handle('cortana/window:isMaximized', () => {
    return mainWindow?.isMaximized() ?? false;
  });
}

function initialiseTheme(): void {
  const theme = configManager.current.ui?.theme ?? 'dark';
  nativeTheme.themeSource = theme === 'system' ? 'system' : theme === 'light' ? 'light' : 'dark';
}

function init(): void {
  app.whenReady().then(() => {
    initialiseTheme();
    mainWindow = createWindow();
    createTray();
    wireIpc();
    registerShortcuts();
    configManager.onChange((config) => {
      mainWindow?.webContents.send('cortana/config:update', config);
      registerShortcuts();
      initialiseTheme();
    });
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      mainWindow = createWindow();
    } else {
      mainWindow?.show();
    }
  });

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      app.quit();
    }
  });

  app.on('will-quit', () => {
    globalShortcut.unregisterAll();
    voiceManager.dispose();
  });
}

init();
