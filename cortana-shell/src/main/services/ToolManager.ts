import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import os from 'node:os';
import { shell } from 'electron';

const execAsync = promisify(exec);

export type ToolResult = {
  status: 'ok' | 'error';
  message?: string;
  data?: unknown;
};

type ToolHandler = (args: Record<string, unknown>) => Promise<ToolResult>;

type ToolDefinition = {
  description: string;
  handler: ToolHandler;
};

export class ToolManager {
  private readonly tools: Map<string, ToolDefinition> = new Map();

  constructor() {
    this.registerDefaultTools();
  }

  listTools(): Array<{ name: string; description: string }> {
    return Array.from(this.tools.entries()).map(([name, { description }]) => ({
      name,
      description,
    }));
  }

  async execute(tool: string, args: Record<string, unknown>): Promise<ToolResult> {
    const definition = this.tools.get(tool);
    if (!definition) {
      throw new Error(`Tool "${tool}" not found`);
    }

    return definition.handler(args);
  }

  private registerDefaultTools(): void {
    this.register('open_url', 'Open a URL in the default browser', async (args) => {
      const url = String(args.url ?? '');
      if (!url.startsWith('http')) {
        return { status: 'error', message: 'Invalid URL' };
      }
      await shell.openExternal(url);
      return { status: 'ok', message: `Opening ${url}` };
    });

    this.register('system_info', 'Retrieve basic system information', async () => {
      return {
        status: 'ok',
        data: {
          platform: os.platform(),
          release: os.release(),
          memory: os.totalmem(),
          cpus: os.cpus().map((cpu) => cpu.model),
        },
      };
    });

    this.register('control_volume', 'Control system volume (placeholder)', async (args) => {
      const { level } = args;
      return { status: 'ok', message: `Pretending to set volume to ${level}` };
    });

    this.register('open_app', 'Launch a desktop application', async (args) => {
      const command = String(args.command ?? '');
      if (!command) {
        return { status: 'error', message: 'No command provided' };
      }
      try {
        await execAsync(command, { shell: true });
        return { status: 'ok', message: `Launching ${command}` };
      } catch (error) {
        return { status: 'error', message: (error as Error).message };
      }
    });

    this.register('create_note', 'Persist a note to disk', async (args) => {
      const fs = await import('node:fs/promises');
      const content = String(args.content ?? '');
      const timestamp = Date.now();
      const filePath = `notes-${timestamp}.md`;
      await fs.writeFile(filePath, `# Cortana Note\n\n${content}\n`);
      return { status: 'ok', message: `Note created at ${filePath}` };
    });
  }

  private register(name: string, description: string, handler: ToolHandler): void {
    this.tools.set(name, { description, handler });
  }
}
