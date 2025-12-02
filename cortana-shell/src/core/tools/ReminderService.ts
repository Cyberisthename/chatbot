import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import schedule, { Job } from 'node-schedule';
import { EventEmitter } from 'eventemitter3';
import dayjs from 'dayjs';
import type { Reminder } from '../types/index.js';

interface ReminderServiceOptions {
  storageDir: string;
  onReminderFired?: (reminder: Reminder) => void;
}

export class ReminderService extends EventEmitter {
  private readonly storageDir: string;
  private readonly storagePath: string;
  private reminders: Reminder[] = [];
  private scheduled: Map<string, Job> = new Map();
  private readonly fireCallback?: (reminder: Reminder) => void;

  constructor(options: ReminderServiceOptions) {
    super();
    this.storageDir = options.storageDir;
    this.storagePath = path.join(this.storageDir, 'reminders.json');
    this.fireCallback = options.onReminderFired;

    if (!existsSync(this.storageDir)) {
      mkdirSync(this.storageDir, { recursive: true });
    }

    this.loadReminders();
    this.restoreSchedules();
  }

  private loadReminders(): void {
    if (!existsSync(this.storagePath)) {
      this.reminders = [];
      return;
    }

    const raw = readFileSync(this.storagePath, 'utf-8');
    if (!raw.trim()) {
      this.reminders = [];
      return;
    }

    try {
      const parsed = JSON.parse(raw) as Reminder[];
      this.reminders = parsed.filter((reminder) => reminder.status === 'pending');
    } catch {
      this.reminders = [];
    }
  }

  private persist(): void {
    writeFileSync(this.storagePath, JSON.stringify(this.reminders, null, 2), 'utf-8');
  }

  private restoreSchedules(): void {
    for (const reminder of this.reminders) {
      this.scheduleReminder(reminder);
    }
  }

  list(): Reminder[] {
    return [...this.reminders];
  }

  addReminder(text: string, time: string | Date): Reminder {
    const targetDate = dayjs(time);
    if (!targetDate.isValid()) {
      throw new Error('Invalid reminder time');
    }

    const reminder: Reminder = {
      id: `rem-${Date.now()}`,
      text,
      time: targetDate.toISOString(),
      createdAt: Date.now(),
      status: 'pending'
    };

    this.reminders.push(reminder);
    this.scheduleReminder(reminder);
    this.persist();
    this.emit('scheduled', reminder);
    return reminder;
  }

  cancelReminder(id: string): Reminder | undefined {
    const reminder = this.reminders.find((item) => item.id === id);
    if (!reminder) {
      return undefined;
    }

    reminder.status = 'cancelled';
    const job = this.scheduled.get(id);
    if (job) {
      job.cancel();
      this.scheduled.delete(id);
    }

    this.reminders = this.reminders.filter((item) => item.status === 'pending');
    this.persist();
    this.emit('cancelled', reminder);
    return reminder;
  }

  private scheduleReminder(reminder: Reminder): void {
    const fireTime = dayjs(reminder.time);
    if (!fireTime.isValid()) {
      return;
    }

    if (fireTime.isBefore(dayjs())) {
      reminder.status = 'completed';
      this.persist();
      return;
    }

    const job = schedule.scheduleJob(fireTime.toDate(), () => {
      reminder.status = 'completed';
      this.persist();
      this.scheduled.delete(reminder.id);
      this.emit('fired', reminder);
      this.fireCallback?.(reminder);
    });

    this.scheduled.set(reminder.id, job);
  }
}
