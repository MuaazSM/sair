// Per built frame: render the fixture page, compare to reference/<id>.png (pixelmatch), compare visible text, run axe.
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { PNG } from 'pngjs';
import fs from 'node:fs';
import path from 'node:path';
import { BUILT_FRAMES } from './built-frames.ts';
import { diffImages, normLines, paintMask } from './visual-utils.ts';

const HERE = import.meta.dirname;
const inventory: any[] = JSON.parse(fs.readFileSync(path.join(HERE, 'inventory.json'), 'utf8'));
const MAX_RATIO = 0.02;
fs.mkdirSync(path.join(HERE, 'diff'), { recursive: true });

for (const built of BUILT_FRAMES) {
  const frame = inventory.find((f) => f.id === built.id);
  test.describe(built.id, () => {
    test.use({ viewport: frame.viewport });

    test.beforeEach(async ({ page }) => {
      await page.route('**/*', (r) => {
        const u = new URL(r.request().url());
        return ['localhost', '127.0.0.1'].includes(u.hostname) || u.protocol === 'data:' || u.protocol === 'blob:' ? r.continue() : r.abort();
      });
      await page.goto(`/?frame=${built.id}`);
      const root = page.locator(`[data-frame-root="${built.id}"]`);
      await root.waitFor();
      await page.evaluate(async () => {
        await Promise.all(['700 20px Nunito', '800 20px Nunito', '400 20px "Atkinson Hyperlegible"', '700 20px "Atkinson Hyperlegible"'].map((f) => document.fonts.load(f)));
        await document.fonts.ready;
      });
      await page.evaluate(() => { for (const a of document.getAnimations()) a.finish(); });
    });

    test('pixels', async ({ page }) => {
      const root = page.locator(`[data-frame-root="${built.id}"]`);
      const buf = await root.screenshot();
      const build = PNG.sync.read(buf);
      const ref = PNG.sync.read(fs.readFileSync(path.join(HERE, 'reference', `${built.id}.png`)));
      expect([build.width, build.height]).toEqual([ref.width, ref.height]);
      paintMask(ref); paintMask(build);
      const { diffPixels, ratio, diff } = diffImages(ref, build, 0.1);
      fs.writeFileSync(path.join(HERE, 'diff', `${built.id}.build.png`), PNG.sync.write(build));
      fs.writeFileSync(path.join(HERE, 'diff', `${built.id}.diff.png`), PNG.sync.write(diff));
      console.log(`DIFF ${built.id}: ${diffPixels} px = ${(ratio * 100).toFixed(3)}% (limit ${MAX_RATIO * 100}%)`);
      expect(ratio).toBeLessThanOrEqual(MAX_RATIO);
    });

    test('text', async ({ page }) => {
      const got = normLines(await page.locator(`[data-frame-root="${built.id}"]`).evaluate((el) => (el as HTMLElement).innerText));
      const want = (frame.strings as string[]).map((s) => s.replace(/\s+/g, ' ').trim()).filter((s) => !built.textExclude.includes(s));
      expect(got).toEqual(want);
    });

    test('axe', async ({ page }) => {
      const results = await new AxeBuilder({ page }).analyze();
      const bad = results.violations.filter((v) => ['serious', 'critical'].includes(v.impact ?? '') && !(built.axeAllow ?? []).includes(v.id));
      if (bad.length) console.log('AXE ' + built.id + ': ' + JSON.stringify(bad.map((v) => ({ id: v.id, impact: v.impact, nodes: v.nodes.map((n) => n.target) }))));
      expect(bad.map((v) => v.id)).toEqual([]);
    });
  });
}
