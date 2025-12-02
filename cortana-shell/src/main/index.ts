import { app, BrowserWindow, globalShortcut, ipcMain, Tray, Menu, nativeImage } from 'electron';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { ConfigManager } from '../core/config/ConfigManager.js';
import { BrainRouter } from '../core/brain/BrainRouter.js';
import { VoiceManager } from '../core/voice/VoiceManager.js';
import { ToolManager } from '../core/tools/ToolManager.js';
import type { AppConfig, BrainRouterRequest, BrainRouterResult, CortanaState, VoiceState } from '../core/types/index.js';

class CortanaApp {
  private mainWindow: BrowserWindow | null = null;
  private tray: Tray | null = null;
  private config!: ConfigManager;
  private brain!: BrainRouter;
  private voice!: VoiceManager;
  private tools!: ToolManager;
  private currentState: CortanaState = 'idle';

  constructor() {
    this.setupApp();
  }

  private setupApp(): void {
    app.whenReady().then(() => {
      this.initialize();
    });

    app.on('window-all-closed', () => {
      if (process.platform !== 'darwin') {
        app.quit();
      }
    });

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        this.createWindow();
      }
    });

    app.on('will-quit', () => {
      this.cleanup();
    });
  }

  private async initialize(): Promise<void> {
    const userDataPath = app.getPath('userData');
    let defaultConfigPath: string;

    if (app.isPackaged) {
      defaultConfigPath = path.join(process.resourcesPath, 'config', 'default-config.yaml');
    } else {
      defaultConfigPath = path.join(__dirname, '../../config', 'default-config.yaml');
    }

    if (!existsSync(defaultConfigPath)) {
      console.warn(`Default config not found at ${defaultConfigPath}, using fallback.`);
      defaultConfigPath = path.join(__dirname, '../../config', 'default-config.yaml');
    }

    this.config = new ConfigManager({
      defaultConfigPath,
      userConfigDir: path.join(userDataPath, 'config')
    });

    this.brain = new BrainRouter({
      config: this.config.config
    });

    this.voice = new VoiceManager({
      config: this.config.config.voice
    });

    this.tools = new ToolManager({
      config: this.config.config.tools,
      storageDir: userDataPath
    });

    this.setupEventHandlers();
    this.setupIPCHandlers();

    this.createWindow();
    this.createTray();
    this.registerHotkeys();

    if (this.config.config.voice.wake_word_enabled) {
      await this.voice.startWakeWordDetection();
    }
  }

  private createWindow(): void {
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.show();
      return;
    }

    this.mainWindow = new BrowserWindow({
      width: 500,
      height: 700,
      minWidth: 400,
      minHeight: 600,
      frame: false,
      transparent: true,
      resizable: true,
      alwaysOnTop: this.config.config.ui.always_on_top,
      webPreferences: {
        preload: path.join(__dirname, '../preload/index.js'),
        contextIsolation: true,
        nodeIntegration: false
      },
      show: !this.config.config.ui.start_minimized
    });

    const devServerUrl = process.env['ELECTRON_RENDERER_URL'] ?? process.env['MAIN_WINDOW_VITE_DEV_SERVER_URL'];

    if (!app.isPackaged && devServerUrl) {
      void this.mainWindow.loadURL(devServerUrl);
      this.mainWindow.webContents.openDevTools({ mode: 'detach' });
    } else {
      void this.mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
    }

    this.mainWindow.on('close', (event) => {
      if (this.config.config.ui.minimize_to_tray) {
        event.preventDefault();
        this.mainWindow?.hide();
      }
    });
  }

  private createTray(): void {
    const iconPath = app.isPackaged
      ? path.join(process.resourcesPath, 'assets', 'icons', 'tray-icon.png')
      : path.join(__dirname, '../../assets/icons/tray-icon.png');

    const icon = existsSync(iconPath) ? nativeImage.createFromPath(iconPath) : nativeImage.createEmpty();

    this.tray = new Tray(icon);

    const contextMenu = Menu.buildFromTemplate([
      {
        label: 'Show Cortana',
        click: () => {
          this.mainWindow?.show();
        }
      },
      {
        label: 'Hide Cortana',
        click: () => {
          this.mainWindow?.hide();
        }
      },
      { type: 'separator' },
      {
        label: `Mode: ${this.config.config.mode}`,
        enabled: false
      },
      {
        label: `Status: ${this.brain.getStatus().jarvis}`,
        enabled: false
      },
      { type: 'separator' },
      {
        label: 'Quit',
        click: () => {
          app.quit();
        }
      }
    ]);

    this.tray.setContextMenu(contextMenu);
    this.tray.setToolTip('Cortana 2.0');

    this.tray.on('click', () => {
      this.mainWindow?.show();
    });
  }

  private registerHotkeys(): void {
    const hotkey = this.config.config.ui.hotkey;

    globalShortcut.register(hotkey, () => {
      if (this.mainWindow?.isVisible()) {
        this.mainWindow.hide();
      } else {
        this.mainWindow?.show();
      }
    });
  }

  private setupEventHandlers(): void {
    this.brain.on('message', (response) => {
      this.sendToRenderer('brain:response', response);
    });

    this.brain.on('statusChange', (status) => {
      this.sendToRenderer('brain:statusChange', status);
    });

    this.voice.on('wakeWord', () => {
      this.mainWindow?.show();
      this.sendToRenderer('voice:wakeWord');
    });

    this.voice.on('speechRecognized', async (text: string) => {
      this.sendToRenderer('voice:speechRecognized', text);
      await this.processUserMessage(text);
    });

    this.voice.on('stateChanged', (state: VoiceState) => {
      this.updateCortanaState();
      this.sendToRenderer('voice:stateChanged', state);
    });

    this.tools.on('reminderFired', (reminder) => {
      this.sendToRenderer('reminder:fired', reminder);
    });

    this.config.on('updated', (newConfig: AppConfig) => {
      this.brain.updateConfig(newConfig);
      this.voice.updateConfig(newConfig.voice);
      this.sendToRenderer('config:updated', newConfig);
    });
  }

  private setupIPCHandlers(): void {
    ipcMain.handle('getConfig', () => this.config.config);

    ipcMain.handle('updateConfig', (_event, partial: Partial<AppConfig>) => this.config.update(partial));

    ipcMain.handle('sendMessage', async (_event, request: BrainRouterRequest): Promise<BrainRouterResult> => {
      return await this.processUserMessage(request.prompt, request.forceBackend, request.allowTools);
    });

    ipcMain.handle('clearHistory', () => {
      this.brain.clearHistory();
    });

    ipcMain.handle('getHistory', () => this.brain.getHistory());

    ipcMain.handle('getBrainStatus', () => this.brain.getStatus());

    ipcMain.handle('startListening', async () => {
      await this.voice.startListening();
    });

    ipcMain.handle('stopListening', () => {
      this.voice.stopListening();
    });

    ipcMain.handle('speak', async (_event, text: string) => {
      await this.voice.speak(text);
    });

    ipcMain.handle('getVoiceState', () => this.voice.getState());

    ipcMain.handle('getReminders', () => this.tools.getReminderService().list());

    ipcMain.handle('cancelReminder', (_event, id: string) => this.tools.getReminderService().cancelReminder(id));

    ipcMain.handle('getState', () => this.currentState);

    ipcMain.handle('minimize', () => {
      this.mainWindow?.minimize();
    });

    ipcMain.handle('close', () => {
      if (this.config.config.ui.minimize_to_tray) {
        this.mainWindow?.hide();
      } else {
        app.quit();
      }
    });
  }

  private async processUserMessage(
    text: string,
    forceBackend?: 'jarvis' | 'local' | 'infinite_capacity',
    allowTools = true
  ): Promise<BrainRouterResult> {
    this.voice.setThinking(true);
    this.updateCortanaState();

    const response = await this.brain.sendMessage(text, { forceBackend });

    const result: BrainRouterResult = {
      ...response,
      toolResults: []
    };

    if (response.tool_calls && allowTools && this.config.config.tools.enabled) {
      for (const toolCall of response.tool_calls) {
        const toolResult = await this.tools.execute(toolCall);
        result.toolResults?.push(toolResult);
        this.sendToRenderer('tool:executed', toolResult);
      }
    }

    this.voice.setThinking(false);
    this.updateCortanaState();

    if (result.reply) {
      await this.voice.speak(result.reply);
    }

    return result;
  }

  private updateCortanaState(): void {
    const voiceState = this.voice.getState();
    const brainStatus = this.brain.getStatus();

    if (brainStatus.jarvis === 'offline' && this.config.config.mode === 'jarvis') {
      this.currentState = 'offline';
    } else if (voiceState.speaking) {
      this.currentState = 'speaking';
    } else if (voiceState.thinking) {
      this.currentState = 'thinking';
    } else if (voiceState.listening) {
      this.currentState = 'listening';
    } else {
      this.currentState = 'idle';
    }

    this.sendToRenderer('cortana:stateChanged', this.currentState);
  }

  private sendToRenderer(channel: string, data?: unknown): void {
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send(channel, data);
    }
  }

  private cleanup(): void {
    globalShortcut.unregisterAll();
    this.voice.stopWakeWordDetection();
  }
}

new CortanaApp();
