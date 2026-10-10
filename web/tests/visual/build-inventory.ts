// Regenerates inventory.json from the live design export. Run: npx tsx tests/visual/build-inventory.ts
import { chromium } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { openDesign, frameStrings, WEB_ROOT, type FrameDef } from './design-page.ts';
import { META } from './frames-meta.ts';

const browser = await chromium.launch();
const out: any[] = [];
for (const file of [...new Set(META.map((m) => m.file))]) {
  const page = await openDesign(browser, file);
  for (const m of META.filter((x) => x.file === file)) {
    const selector = m.selector ?? `[id="${m.wrapper}"] > div:nth-child(2)`;
    const loc = page.locator(selector).first();
    const box = await loc.boundingBox();
    const strings = await frameStrings(loc);
    const notes = await page.evaluate((w) => (document.querySelector(`[id="${w}"] > div:nth-child(3)`) as HTMLElement | null)?.innerText ?? '', m.wrapper);
    const aria = await loc.evaluate((el) => [...el.querySelectorAll('[aria-label],img[alt]')].map((e) => (e.getAttribute('aria-label') ?? e.getAttribute('alt') ?? '')).filter(Boolean));
    const dims = { width: Math.round(box!.width), height: Math.round(box!.height) };
    if (dims.width !== m.viewport!.width || dims.height !== m.viewport!.height) console.log('SIZE MISMATCH', m.id, dims);
    out.push({ id: m.id, file: m.file, selector_or_index: selector, viewport: dims, tier: m.tier, theme: m.theme, strings,
      aria_labels: aria, component: m.component, route: m.route, state: m.state, visual_test: `visual/${m.id}`, design_note: notes.replace(/\s+/g, ' ').trim() });
  }
  await page.close();
}
await browser.close();
fs.writeFileSync(path.join(WEB_ROOT, 'tests/visual/inventory.json'), JSON.stringify(out, null, 2) + '\n');
console.log('frames', out.length);
