import Store from 'electron-store';

export interface ChatEntry {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export class HistoryStore {
  private store: Store<{ history: ChatEntry[] }>;

  constructor() {
    this.store = new Store<{ history: ChatEntry[] }>({
      name: 'cortana-history',
      defaults: {
        history: [],
      },
    });
  }

  getHistory(): ChatEntry[] {
    return this.store.get('history', []);
  }

  addEntry(entry: ChatEntry): void {
    const history = this.getHistory();
    history.push(entry);
    const limit = 1000;
    if (history.length > limit) {
      history.splice(0, history.length - limit);
    }
    this.store.set('history', history);
  }

  clear(): void {
    this.store.set('history', []);
  }
}
