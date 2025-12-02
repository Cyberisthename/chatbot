import { exec } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { promisify } from 'node:util';
import dayjs from 'dayjs';
import { EventEmitter } from 'eventemitter3';
import type { ToolCall, ToolExecutionResult, ToolsConfig } from '../types/index.js';
import { ReminderService } from './ReminderService.js';

const execAsync = promisify(exec);

export interface ToolManagerOptions {
  config: ToolsConfig;
  storageDir: string;
}

export class ToolManager extends EventEmitter {
  private readonly config: ToolsConfig;
  private readonly storageDir: string;
  private readonly notesDir: string;
  private readonly reminderService: ReminderService;

  constructor(options: ToolManagerOptions) {
    super();
    this.config = options.config;
    this.storageDir = options.storageDir;
    this.notesDir = path.join(this.storageDir, 'notes');

    if (!existsSync(this.notesDir)) {
      mkdirSync(this.notesDir, { recursive: true });
    }

    this.reminderService = new ReminderService({
      storageDir: this.storageDir,
      onReminderFired: (reminder) => {
        this.emit('reminderFired', reminder);
      }
    });
  }

  async execute(toolCall: ToolCall): Promise<ToolExecutionResult> {
    if (!this.config.enabled) {
      return {
        tool: toolCall.tool,
        success: false,
        error: 'System tools are disabled in configuration'
      };
    }

    if (!this.config.allowed_commands.includes(toolCall.tool)) {
      return {
        tool: toolCall.tool,
        success: false,
        error: `Tool "${toolCall.tool}" is not allowed in configuration`
      };
    }

    try {
      let result: ToolExecutionResult;

      switch (toolCall.tool) {
        case 'open_app': {
          result = await this.openApp(toolCall.args.name as string);
          break;
        }

        case 'open_url': {
          result = await this.openUrl(toolCall.args.url as string);
          break;
        }

        case 'control_volume': {
          result = await this.controlVolume(toolCall.args.action as string, toolCall.args.value as number);
          break;
        }

        case 'control_media': {
          result = await this.controlMedia(toolCall.args.action as string);
          break;
        }

        case 'create_note': {
          result = await this.createNote(toolCall.args.title as string, toolCall.args.content as string);
          break;
        }

        case 'set_reminder': {
          result = await this.setReminder(toolCall.args.text as string, toolCall.args.time as string);
          break;
        }

        default: {
          result = {
            tool: toolCall.tool,
            success: false,
            error: `Unknown tool: ${toolCall.tool}`
          };
        }
      }

      this.emit('executed', result);
      return result;
    } catch (error) {
      const errorResult = {
        tool: toolCall.tool,
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
      this.emit('executed', errorResult);
      return errorResult;
    }
  }

  private async openApp(name: string): Promise<ToolExecutionResult> {
    if (process.platform === 'win32') {
      await execAsync(`cmd /c start "" "${name}"`);
    } else if (process.platform === 'darwin') {
      await execAsync(`open -a "${name}"`);
    } else {
      await execAsync(`${name} &`);
    }

    return {
      tool: 'open_app',
      success: true,
      output: `Opening ${name}...`
    };
  }

  private async openUrl(url: string): Promise<ToolExecutionResult> {
    if (process.platform === 'win32') {
      await execAsync(`cmd /c start "" "${url}"`);
    } else if (process.platform === 'darwin') {
      await execAsync(`open "${url}"`);
    } else {
      await execAsync(`xdg-open "${url}"`);
    }

    return {
      tool: 'open_url',
      success: true,
      output: `Opening URL...`
    };
  }

  private async controlVolume(action: string, value?: number): Promise<ToolExecutionResult> {
    const loudness = await import('loudness');

    if (action === 'set' && typeof value === 'number') {
      await loudness.setVolume(Math.max(0, Math.min(100, value)));
      return {
        tool: 'control_volume',
        success: true,
        output: `Volume set to ${value}%`
      };
    }

    if (action === 'up') {
      const current = (await loudness.getVolume()) ?? 50;
      const newVolume = Math.min(100, current + 10);
      await loudness.setVolume(newVolume);
      return {
        tool: 'control_volume',
        success: true,
        output: `Volume increased to ${newVolume}%`
      };
    }

    if (action === 'down') {
      const current = (await loudness.getVolume()) ?? 50;
      const newVolume = Math.max(0, current - 10);
      await loudness.setVolume(newVolume);
      return {
        tool: 'control_volume',
        success: true,
        output: `Volume decreased to ${newVolume}%`
      };
    }

    if (action === 'mute') {
      await loudness.setMuted(true);
      return {
        tool: 'control_volume',
        success: true,
        output: 'Volume muted'
      };
    }

    if (action === 'unmute') {
      await loudness.setMuted(false);
      return {
        tool: 'control_volume',
        success: true,
        output: 'Volume unmuted'
      };
    }

    return {
      tool: 'control_volume',
      success: false,
      error: `Unknown volume action: ${action}`
    };
  }

  private async controlMedia(action: string): Promise<ToolExecutionResult> {
    if (process.platform === 'win32') {
      const keyMap: Record<string, string> = {
        play: '{MEDIA_PLAY_PAUSE}',
        pause: '{MEDIA_PLAY_PAUSE}',
        toggle: '{MEDIA_PLAY_PAUSE}',
        next: '{MEDIA_NEXT}',
        previous: '{MEDIA_PREV}',
        stop: '{MEDIA_STOP}'
      };

      const key = keyMap[action];
      if (key) {
        await execAsync(
          `powershell -Command "$shell = New-Object -ComObject WScript.Shell; $shell.SendKeys('${key}')"`
        );
        return {
          tool: 'control_media',
          success: true,
          output: `Media ${action} command sent`
        };
      }
    }

    return {
      tool: 'control_media',
      success: false,
      error: 'Media control not yet implemented for this platform'
    };
  }

  private async createNote(title: string, content: string): Promise<ToolExecutionResult> {
    const filename = `${dayjs().format('YYYY-MM-DD_HH-mm-ss')}_${title.replace(/[^\w\s-]/g, '').replace(/\s+/g, '_')}.md`;
    const filePath = path.join(this.notesDir, filename);

    const noteContent = `# ${title}\n\n${content}\n\n---\nCreated: ${dayjs().format('YYYY-MM-DD HH:mm:ss')}\n`;

    writeFileSync(filePath, noteContent, 'utf-8');

    return {
      tool: 'create_note',
      success: true,
      output: `Note saved: ${filename}`
    };
  }

  private async setReminder(text: string, time: string): Promise<ToolExecutionResult> {
    const reminder = this.reminderService.addReminder(text, time);

    return {
      tool: 'set_reminder',
      success: true,
      output: `Reminder set for ${dayjs(reminder.time).format('YYYY-MM-DD HH:mm')}`
    };
  }

  getReminderService(): ReminderService {
    return this.reminderService;
  }
}
