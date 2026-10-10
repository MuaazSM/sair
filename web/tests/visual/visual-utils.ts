import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';

/** Mock OS status-bar band (D-005): the design draws a clock/battery in the first 1px border + 32px of every
 *  mobile frame; the build leaves that band empty. Both images get the same solid rectangle painted over it
 *  before the diff. x 0..width, y 0..34 (1px border + 32px bar + 1px slack for antialiasing). */
export const STATUS_BAR_MASK = { x: 0, y: 0, height: 34 } as const;

export function paintMask(png: PNG): void {
  for (let y = STATUS_BAR_MASK.y; y < STATUS_BAR_MASK.height; y++)
    for (let x = 0; x < png.width; x++) {
      const i = (png.width * y + x) << 2;
      png.data[i] = 255; png.data[i + 1] = 0; png.data[i + 2] = 255; png.data[i + 3] = 255;
    }
}

export type DiffResult = { diffPixels: number; ratio: number; diff: PNG };

export function diffImages(ref: PNG, build: PNG, threshold = 0.1): DiffResult {
  const { width, height } = ref;
  const diff = new PNG({ width, height });
  const diffPixels = pixelmatch(ref.data, build.data, diff.data, width, height, { threshold });
  return { diffPixels, ratio: diffPixels / (width * height), diff };
}

export const normLines = (text: string): string[] =>
  text.split('\n').map((s) => s.replace(/\s+/g, ' ').trim()).filter(Boolean);
