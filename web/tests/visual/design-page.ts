// Shared helpers to open the Claude Design export offline and freeze it deterministically.
import { chromium, type Browser, type Page, type Locator } from '@playwright/test';
import { pathToFileURL, fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const WEB_ROOT = path.resolve(HERE, '../..');
/** The ONE place the design location lives. Repoint to docs/design/ once the export is moved there. */
export const DESIGN_ROOT = path.resolve(WEB_ROOT, '../brand/Sair Design System') + '/';

export type FrameDef = {
  id: string; file: string; wrapper: string; selector: string; viewport: { width: number; height: number };
  tier: 'Must' | 'Should' | 'Could'; theme: 'light' | 'dark'; strings: string[]; selector_or_index?: string;
};

export function fontsCssForFile(): string {
  const css = fs.readFileSync(path.join(WEB_ROOT, 'src/styles/fonts.css'), 'utf8');
  return css.replace(/url\(\/fonts\//g, `url(${pathToFileURL(path.join(WEB_ROOT, 'public/fonts')).href}/`);
}

/** Open one .dc.html with: all non-file network blocked, React UMD served from node_modules (the canvas runtime
 *  normally pulls it from unpkg), local fonts injected, animations frozen. */
export async function openDesign(browser: Browser, file: string): Promise<Page> {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
  await page.route('**/*', (r) => {
    const u = r.request().url();
    if (u.startsWith('file:') || u.startsWith('data:') || u.startsWith('blob:')) return r.continue();
    if (u.includes('react@18.3.1/umd/react.production'))
      return r.fulfill({ path: path.join(WEB_ROOT, 'node_modules/react/umd/react.production.min.js'), contentType: 'text/javascript' });
    if (u.includes('react-dom@18.3.1/umd/react-dom.production'))
      return r.fulfill({ path: path.join(WEB_ROOT, 'node_modules/react-dom/umd/react-dom.production.min.js'), contentType: 'text/javascript' });
    return r.abort();
  });
  // Freeze "Before the Walk" 1B's 150 ms playback ticker so it stays in its initial state (t = 7.6 s).
  await page.addInitScript(() => {
    const si = window.setInterval.bind(window);
    (window as any).setInterval = (fn: any, ms?: number, ...a: any[]) => (ms === 150 ? 0 : si(fn, ms, ...a));
  });
  // Chromium refuses fetch() on file://, but "After the Walk" fetches assets/sair-icon.svg for the logo-trace animation.
  // Serve that one file to fetch() from disk so the trace renders as it does over http.
  const assets: Record<string, string> = {};
  for (const n of fs.readdirSync(path.join(DESIGN_ROOT, 'assets'))) if (n.endsWith('.svg')) assets['assets/' + n] = fs.readFileSync(path.join(DESIGN_ROOT, 'assets', n), 'utf8');
  await page.addInitScript((a) => {
    const orig = window.fetch.bind(window);
    window.fetch = ((input: any, init?: any) => {
      const k = String(input).replace(/^\.\//, '');
      return k in a ? Promise.resolve(new Response(a[k], { headers: { 'content-type': 'image/svg+xml' } })) : orig(input, init);
    }) as typeof window.fetch;
  }, assets);
  await page.goto(pathToFileURL(DESIGN_ROOT + file).href);
  await page.waitForSelector('[id]:has(> div)', { timeout: 15000 });
  await page.addStyleTag({ content: fontsCssForFile() });
  await page.evaluate(async () => { await Promise.all(['700 20px Nunito', '800 20px Nunito', '400 20px "Atkinson Hyperlegible"', '700 20px "Atkinson Hyperlegible"', '500 20px Mukta'].map((f) => document.fonts.load(f))); await document.fonts.ready; });
  await page.waitForTimeout(500);
  await freezeAnimations(page);
  return page;
}

/** Pause every CSS animation at a fixed point; finish transitions. sair-trace* (5 s) paused at 3.5 s = ribbon fully drawn, not yet faded. */
export async function freezeAnimations(page: Page) {
  await page.evaluate(() => {
    for (const a of document.getAnimations()) {
      const name = (a as any).animationName as string | undefined;
      a.pause();
      if (name) a.currentTime = name.startsWith('sair-trace') ? 3500 : 0; else a.finish();
    }
  });
}

export function frameLocator(page: Page, f: FrameDef): Locator { return page.locator(f.selector).first(); }

/** Visible text as rendered, one string per line, whitespace-normalised. */
export async function frameStrings(loc: Locator): Promise<string[]> {
  const t = await loc.evaluate((el) => (el as HTMLElement).innerText);
  return t.split('\n').map((s) => s.replace(/\s+/g, ' ').trim()).filter(Boolean);
}
