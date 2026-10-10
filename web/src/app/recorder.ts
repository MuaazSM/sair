// Hold-to-record via MediaRecorder (WebM/Opus where supported). The stream is released as soon as recording ends.
export const MAX_SECONDS = 60;
export const WARN_AT_SECONDS = 50;

export class Recorder {
  private chunks: Blob[] = [];
  private startedAt = performance.now();

  private constructor(private rec: MediaRecorder, private stream: MediaStream) {
    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) this.chunks.push(e.data);
    };
  }

  static async start(): Promise<Recorder> {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = typeof MediaRecorder.isTypeSupported === 'function' && MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : undefined;
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const r = new Recorder(rec, stream);
    rec.start();
    r.startedAt = performance.now();
    return r;
  }

  elapsedSeconds(): number {
    return (performance.now() - this.startedAt) / 1000;
  }

  private release(): void {
    this.stream.getTracks().forEach((t) => t.stop());
  }

  cancel(): void {
    try {
      if (this.rec.state !== 'inactive') this.rec.stop();
    } finally {
      this.release();
    }
  }

  stop(): Promise<{ blob: Blob; seconds: number }> {
    const seconds = this.elapsedSeconds();
    return new Promise((resolve) => {
      const type = this.rec.mimeType || 'audio/webm';
      this.rec.onstop = () => {
        this.release();
        resolve({ blob: new Blob(this.chunks, { type }), seconds });
      };
      if (this.rec.state === 'inactive') this.rec.onstop?.(new Event('stop'));
      else this.rec.stop();
    });
  }
}
