import fs from 'node:fs';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import yaml from 'js-yaml';

type ConfigObject = Record<string, any>;

type ConfigListener = (config: ConfigObject) => void;

function deepMerge<T extends ConfigObject>(target: T, updates: ConfigObject): T {
  const output = { ...target } as ConfigObject;
  for (const [key, value] of Object.entries(updates)) {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      output[key] = deepMerge(output[key] ?? {}, value);
    } else {
      output[key] = value;
    }
  }
  return output as T;
}

function setByPath(target: ConfigObject, pathString: string, value: unknown): ConfigObject {
  const segments = pathString.split('.');
  const root = { ...target };
  let pointer: ConfigObject = root;
  while (segments.length > 1) {
    const segment = segments.shift();
    if (!segment) continue;
    if (typeof pointer[segment] !== 'object' || pointer[segment] === null) {
      pointer[segment] = {};
    }
    pointer = pointer[segment];
  }
  const last = segments.shift();
  if (last) {
    pointer[last] = value;
  }
  return root;
}

export class ConfigManager {
  private readonly configPath: string;
  private config: ConfigObject;
  private readonly emitter = new EventEmitter();
  private watcher?: fs.FSWatcher;

  constructor(configPath: string) {
    this.configPath = configPath;
    this.config = this.readConfig();
    this.watch();
  }

  private readConfig(): ConfigObject {
    try {
      const content = fs.readFileSync(this.configPath, 'utf-8');
      const document = yaml.load(content);
      return (document as ConfigObject) ?? {};
    } catch (error) {
      console.warn('Unable to read config.yaml', error);
      return {};
    }
  }

  private watch() {
    try {
      if (!fs.existsSync(this.configPath)) {
        return;
      }
      this.watcher = fs.watch(this.configPath, { persistent: false }, () => {
        const parsed = this.readConfig();
        this.config = parsed;
        this.emitter.emit('change', this.current);
      });
    } catch (error) {
      console.warn('Config watcher failed', error);
    }
  }

  get current(): ConfigObject {
    return this.config;
  }

  onChange(listener: ConfigListener): void {
    this.emitter.on('change', listener);
  }

  update(pathString: string, value: unknown): void {
    this.config = setByPath(this.config, pathString, value);
    const serialised = yaml.dump(this.config, { lineWidth: 1000 });
    fs.writeFileSync(this.configPath, serialised, 'utf-8');
    this.emitter.emit('change', this.current);
  }
}
