// Captures one reference PNG per inventory frame from the Claude Design export, offline, with local fonts.
// Run: npx tsx tests/visual/capture-design.ts [frame-id ...]
// Network is blocked except file:// (see design-page.ts); fonts come from src/styles/fonts.css via file:// URLs.
import { chromium } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { openDesign, WEB_ROOT, DESIGN_ROOT, frameStrings, type FrameDef } from './design-page.ts';

export { DESIGN_ROOT };
const inventory: any[] = JSON.parse(fs.readFileSync(path.join(WEB_ROOT, 'tests/visual/inventory.json'), 'utf8'));
const only = process.argv.slice(2);
const outDir = path.join(WEB_ROOT, 'tests/visual/reference');
fs.mkdirSync(outDir, { recursive: true });
const norm = (a: string[]) => a.join(' ').replace(/\s+/g, ' ').trim();

const browser = await chromium.launch();
let written = 0, mismatches = 0;
for (const file of [...new Set(inventory.map((f) => f.file))]) {
  const page = await openDesign(browser, file);
  for (const f of inventory.filter((x) => x.file === file && (!only.length || only.includes(x.id)))) {
    const loc = page.locator(f.selector_or_index).first();
    // Capture normalisation: the design draws a rounded "device" with a 1px border and shadow on the review canvas.
    // Flatten radius/shadow and paint the border in the frame's own background so the PNG is the bare screen.
    await loc.evaluate((el) => {
      const e = el as HTMLElement; const cs = getComputedStyle(e);
      if (parseFloat(cs.borderTopWidth) > 1) return; // iPhone mocks: the thick dark bezel is part of the illustration
      const bg = cs.backgroundColor === 'rgba(0, 0, 0, 0)' ? '#FFFFFF' : cs.backgroundColor;
      e.style.borderRadius = '0'; e.style.boxShadow = 'none'; e.style.borderColor = bg;
    });
    await page.evaluate(() => document.fonts.ready);
    const box = (await loc.boundingBox())!;
    if (Math.round(box.width) !== f.viewport.width || Math.round(box.height) !== f.viewport.height)
      console.log(`SIZE ${f.id}: got ${box.width}x${box.height}, want ${f.viewport.width}x${f.viewport.height}`);
    await loc.screenshot({ path: path.join(outDir, `${f.id}.png`) });
    written++;
    const got = await frameStrings(loc);
    if (norm(got) !== norm(f.strings)) { mismatches++; console.log(`TEXT MISMATCH ${f.id}\n  inventory: ${norm(f.strings)}\n  rendered:  ${norm(got)}`); }
  }
  await page.close();
}
await browser.close();
console.log(`PNGs written: ${written}; text mismatches: ${mismatches}`);
