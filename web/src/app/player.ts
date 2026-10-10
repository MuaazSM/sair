// J1: the mission plays twice with a 2 s pause. Labels are the design's (Before the Walk, frame 1B):
// "Playing · 1 of 2", "Playing again in N second(s)", "Playing · 2 of 2", "That's today's walk."
// With no audio (TTS not rendered, or playback refused) the same timeline runs silently so the words still appear.

export type PlayState = { label: string; progress: number; revealed: number; paused: boolean };

export const GAP_MS = 2000;
const SILENT_MS_PER_WORD = 450;

export class MissionPlayer {
  private audio: HTMLAudioElement | null = null;
  private phase: 'play1' | 'gap' | 'play2' | 'end' = 'play1';
  private phaseStart = 0;
  private waited = 0;
  private last = 0;
  private ended = false;
  private silent: boolean;
  private paused = false;
  private timer: ReturnType<typeof setInterval> | undefined;
  private resolve!: () => void;
  private finished = false;
  private cur: PlayState = { label: '', progress: 0, revealed: 0, paused: false };

  constructor(private src: string | null, private words: number, private on: (s: PlayState) => void) {
    this.silent = src === null;
  }

  run(): Promise<void> {
    return new Promise((resolve) => {
      this.resolve = resolve;
      if (this.src) {
        this.audio = new Audio(this.src);
        this.audio.addEventListener('ended', () => {
          this.ended = true;
        });
      }
      this.startPlay('play1');
      this.last = performance.now();
      this.timer = setInterval(() => this.tick(), 100);
    });
  }

  private startPlay(phase: 'play1' | 'play2'): void {
    this.phase = phase;
    this.ended = false;
    this.waited = 0;
    if (this.audio && !this.silent) {
      try {
        this.audio.currentTime = 0;
      } catch {
        /* not seekable yet */
      }
      this.audio.play().catch(() => {
        this.silent = true;
      });
    }
    this.emit(phase === 'play1' ? 'Playing · 1 of 2' : 'Playing · 2 of 2', 0, phase === 'play1' ? 0 : this.words);
  }

  private emit(label: string, progress: number, revealed: number): void {
    this.cur = { label, progress, revealed, paused: this.paused };
    this.on(this.cur);
  }

  private tick(): void {
    const now = performance.now();
    const dt = now - this.last;
    this.last = now;
    if (this.paused || this.finished) return;
    this.waited += dt;
    if (this.phase === 'play1' || this.phase === 'play2') {
      const dur = this.audio && Number.isFinite(this.audio.duration) && this.audio.duration > 0 ? this.audio.duration : 0;
      const silentTotal = Math.max(2000, this.words * SILENT_MS_PER_WORD);
      const progress = this.silent || !dur ? Math.min(1, this.waited / silentTotal) : Math.min(1, this.audio!.currentTime / dur);
      const done = this.silent ? this.waited >= silentTotal : this.ended;
      const revealed = this.phase === 'play2' ? this.words : Math.ceil(Math.min(1, progress / 0.85) * this.words);
      const label = this.phase === 'play1' ? 'Playing · 1 of 2' : 'Playing · 2 of 2';
      if (!done) return this.emit(label, progress, revealed);
      if (this.phase === 'play1') {
        this.phase = 'gap';
        this.waited = 0;
        return this.emit(this.gapLabel(), 1, this.words);
      }
      this.phase = 'end';
      this.waited = 0;
      return this.emit("That's today's walk.", 1, this.words);
    }
    if (this.phase === 'gap') {
      if (this.waited >= GAP_MS) return this.startPlay('play2');
      return this.emit(this.gapLabel(), 1, this.words);
    }
    if (this.phase === 'end' && this.waited >= 1500) this.finish();
  }

  private gapLabel(): string {
    const n = Math.max(1, Math.ceil((GAP_MS - this.waited) / 1000));
    return `Playing again in ${n} second${n === 1 ? '' : 's'}`;
  }

  /** NOT IN DESIGN: the design has no paused state (REVIEW M8). */
  togglePause(): void {
    this.paused = !this.paused;
    this.last = performance.now();
    if (this.audio && !this.silent && (this.phase === 'play1' || this.phase === 'play2')) {
      if (this.paused) this.audio.pause();
      else this.audio.play().catch(() => undefined);
    }
    this.on({ ...this.cur, paused: this.paused });
  }

  private finish(): void {
    if (this.finished) return;
    this.finished = true;
    clearInterval(this.timer);
    this.audio?.pause();
    this.resolve();
  }

  stop(): void {
    this.finish();
  }
}
