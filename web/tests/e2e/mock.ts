// Mock hub for the e2e specs: page.route handlers + browser-side fakes for audio playback and MediaRecorder.
import AxeBuilder from '@axe-core/playwright';
import { expect, type Page, type Route } from '@playwright/test';

export const TOKEN = 'test-token';

export type Seen = { path: string; auth: string | null; method: string; contentType?: string; body?: Buffer | null };

export type Hub = {
  seen: Seen[];
  /** Respond to POST /walk. Default: 202 with walk id w1. */
  walkPost: (route: Route) => Promise<void> | void;
  /** Respond to GET /walk/w1. Default: done with an entry. */
  walkGet: (route: Route) => Promise<void> | void;
  missionAudioPath: string | null;
};

const json = (route: Route, body: unknown, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

export const DONE_WALK = {
  status: 'done',
  date: '2026-10-10',
  entry: 'Mom and Dad walked down the lane. The gulmohar had a few red flowers again.',
  facts: [
    { subject: 'gulmohar', kind: 'plant', certainty: 'sure', place: 'red flowers' },
    { subject: 'koel', kind: 'animal', certainty: 'unsure' },
  ],
  mission_result: 'done',
};

export function newHub(): Hub {
  return {
    seen: [],
    missionAudioPath: '/audio/m1',
    walkPost: (route) => json(route, { walk_id: 'w1', status: 'received' }, 202),
    walkGet: (route) => json(route, DONE_WALK),
  };
}

/** Browser-side fakes, installed before the app loads. */
export async function installBrowserFakes(page: Page, opts: { token?: string | null } = {}) {
  await page.addInitScript(
    ({ token }) => {
      const w = window as any;
      w.__log = [];
      if (token) localStorage.setItem('sair.token', token);
      const proto = HTMLMediaElement.prototype as any;
      proto.play = function () {
        w.__log.push({ e: 'play', t: performance.now() });
        setTimeout(() => {
          w.__log.push({ e: 'ended', t: performance.now() });
          this.dispatchEvent(new Event('ended'));
        }, 1500);
        return Promise.resolve();
      };
      proto.pause = function () {};
      Object.defineProperty(proto, 'duration', { get: () => 1.5 });
      class FakeMediaRecorder {
        state = 'inactive';
        mimeType = 'audio/webm;codecs=opus';
        ondataavailable: ((e: any) => void) | null = null;
        onstop: ((e: any) => void) | null = null;
        constructor(public stream: unknown, _o?: unknown) {}
        static isTypeSupported() { return true; }
        start() { this.state = 'recording'; }
        stop() {
          this.state = 'inactive';
          this.ondataavailable?.({ data: new Blob(['fake-opus-bytes'], { type: 'audio/webm' }) });
          Promise.resolve().then(() => this.onstop?.(new Event('stop')));
        }
      }
      w.MediaRecorder = FakeMediaRecorder;
      const md = navigator.mediaDevices ?? ((navigator as any).mediaDevices = {});
      md.getUserMedia = (async () => ({ getTracks: () => [{ stop() {} }] })) as any;
    },
    { token: opts.token === undefined ? TOKEN : opts.token },
  );
}

export async function installHub(page: Page, hub: Hub) {
  const note = (route: Route, path: string) => {
    const r = route.request();
    hub.seen.push({ path, auth: r.headers()['authorization'] ?? null, method: r.method(), contentType: r.headers()['content-type'], body: r.postDataBuffer() });
  };
  await page.route((u) => u.pathname === '/mission/today', (route) => {
    note(route, '/mission/today');
    return json(route, { mission_id: 'm1', text: 'Look at the gulmohar on your lane. What colour is it right now?', type: 'return', audio_path: hub.missionAudioPath });
  });
  await page.route((u) => u.pathname === '/audio/m1', (route) => {
    note(route, '/audio/m1');
    return route.fulfill({ status: 200, contentType: 'audio/mpeg', body: Buffer.from('ID3fake') });
  });
  await page.route((u) => u.pathname === '/walk', (route) => {
    note(route, '/walk');
    return hub.walkPost(route);
  });
  await page.route((u) => u.pathname === '/walk/w1', (route) => {
    note(route, '/walk/w1');
    return hub.walkGet(route);
  });
  await page.route((u) => u.pathname === '/diary', (route) => {
    const u = new URL(route.request().url());
    note(route, u.pathname + u.search);
    const group = u.searchParams.get('group');
    const all = [
      { walk_id: 'w1', date: '2026-10-10', group: 'Mom and Dad', mission_text: 'Look at the gulmohar again.', entry: 'Mom and Dad walked down the lane.', facts: [{ subject: 'gulmohar', kind: 'plant', certainty: 'sure', place: null, asked_again_on: ['2026-10-12'] }], mission_result: 'done' },
      { walk_id: 'w0', date: '2026-10-09', group: 'Dad', mission_text: null, entry: 'Dad walked alone.', facts: [], mission_result: 'not_mentioned' },
    ];
    const entries = group ? all.filter((e) => e.group === group) : all;
    return json(route, { summary: { walks_this_month: 2, things_noticed: 1 }, entries });
  });
}

/** serious/critical axe violations on the current page (NFR-A1), same bar as the fixture frames. */
export async function expectNoSeriousAxe(page: Page) {
  const results = await new AxeBuilder({ page }).analyze();
  const bad = results.violations.filter((v) => ['serious', 'critical'].includes(v.impact ?? ''));
  expect(bad.map((v) => v.id)).toEqual([]);
}

/** Press and hold the big round button, then release. */
export async function holdToTalk(page: Page, ms = 150) {
  const btn = page.getByRole('button', { name: 'Hold to talk' });
  const box = (await btn.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await expect(page.getByText('Listening')).toBeVisible();
  await page.waitForTimeout(ms);
  await page.mouse.up();
}
