// Privacy gate (NFR-P3, NFR-P5, NFR-P7): build the PWA and scan dist/ for outbound URLs, geolocation, beacons and analytics.
import { expect, test } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const WEB = path.resolve(import.meta.dirname, '..');
const DIST = path.join(WEB, 'dist');

// Explicit allowlist of absolute URLs that may appear as inert strings (XML namespaces, React's error-decoder link text).
// None of these is ever requested.
const ALLOWED_URLS = new Set([
  'http://www.w3.org/2000/svg',
  'http://www.w3.org/1999/xhtml',
  'http://www.w3.org/1998/Math/MathML',
  'http://www.w3.org/1999/xlink',
  'http://www.w3.org/XML/1998/namespace',
  'http://www.w3.org/2000/xmlns/',
  'https://react.dev/errors/',
  'https://reactjs.org/docs/error-decoder.html?invariant=', // react-dom production error text
  'http://c2pa.org/manifest', // content-credentials label inside the logo SVG's metadata; inert
]);
const ALLOWED_PREFIXES = ['https://react.dev/errors/'];
const FORBIDDEN = ['navigator.geolocation', 'geolocation', 'sendBeacon', 'gtag', 'analytics', 'googletagmanager', 'sentry'];

function walk(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)]));
}

test('production build has no outbound URLs, geolocation, beacons or analytics', async () => {
  fs.rmSync(DIST, { recursive: true, force: true });
  execFileSync('npm', ['run', 'build'], { cwd: WEB, stdio: 'pipe' });
  const files = walk(DIST);
  expect(files.length).toBeGreaterThan(0);

  const textFiles = files.filter((f) => /\.(html|js|css|svg|json|webmanifest|txt|map)$/.test(f));
  const urlRe = /\b(?:https?|wss?):\/\/[^\s"'`)<>\\]+/g;
  const offenders: string[] = [];
  const forbidden: string[] = [];
  for (const f of textFiles) {
    const rel = path.relative(DIST, f);
    const text = fs.readFileSync(f, 'utf8');
    for (const m of text.matchAll(urlRe)) {
      const u = m[0];
      if (ALLOWED_URLS.has(u) || ALLOWED_PREFIXES.some((p) => u.startsWith(p))) continue;
      offenders.push(`${rel}: ${u}`);
    }
    const lower = text.toLowerCase();
    for (const word of FORBIDDEN) if (lower.includes(word.toLowerCase())) forbidden.push(`${rel}: ${word}`);
  }
  expect(offenders).toEqual([]);
  expect(forbidden).toEqual([]);

  // fixture frame host must not ship
  const js = files.filter((f) => f.endsWith('.js')).map((f) => fs.readFileSync(f, 'utf8')).join('\n');
  expect(js).not.toContain('Unknown frame');

  // CSP-friendly: default-src 'self' + style-src 'unsafe-inline' => no inline scripts, no inline event handlers, same-origin refs only.
  const html = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8');
  const inlineScripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)].filter((m) => !/\bsrc=/.test(m[1]!) || m[2]!.trim() !== '');
  expect(inlineScripts.map((m) => m[0])).toEqual([]);
  expect(html).not.toMatch(/\son[a-z]+\s*=/i);
  for (const ref of html.matchAll(/\b(?:src|href)="([^"]+)"/g)) expect(ref[1]!.startsWith('/') || ref[1]!.startsWith('#')).toBe(true);
  // no asset is inlined as a data: URI in css/html
  const css = files.filter((f) => f.endsWith('.css')).map((f) => fs.readFileSync(f, 'utf8')).join('\n');
  expect(css).not.toContain('url(data:');
  // manifest present and same-origin
  const manifest = JSON.parse(fs.readFileSync(path.join(DIST, 'manifest.webmanifest'), 'utf8'));
  expect(manifest.display).toBe('standalone');
  expect(JSON.stringify(manifest)).not.toMatch(/https?:/);
});
