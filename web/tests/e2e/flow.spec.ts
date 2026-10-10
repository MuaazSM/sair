// J1 (hear the mission twice), J3 (record, upload, Writing -> Saved), the 45 s gap state, setup, diary. Mock hub, no network.
import { expect, test } from '@playwright/test';
import { DONE_WALK, expectNoSeriousAxe, holdToTalk, installBrowserFakes, installHub, newHub, TOKEN } from './mock.ts';

test.describe('J1: hear the mission', () => {
  test('fetches with the Authorization header, plays twice with a 2 s gap', async ({ page }) => {
    const hub = newHub();
    await installBrowserFakes(page);
    await installHub(page, hub);
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Ready for your sair?' })).toBeVisible();
    await expect(page.getByText('Your first walk')).toBeVisible(); // first-walk variant: this device has never walked
    await expectNoSeriousAxe(page);

    await page.getByRole('button', { name: "Hear today's walk" }).click();
    await expect(page.getByText('Playing · 1 of 2')).toBeVisible();
    // the design renders each word as its own span (spacing by CSS gap), so compare without whitespace
    await expect(page.locator('main')).toContainText(/Lookatthegulmoharonyourlane\./);
    await expectNoSeriousAxe(page);
    await expect(page.getByText(/Playing again in \d seconds?/)).toBeVisible();
    await expect(page.getByText('Playing · 2 of 2')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Enjoy your walk.' })).toBeVisible({ timeout: 15_000 });
    await expectNoSeriousAxe(page);

    const mission = hub.seen.find((s) => s.path === '/mission/today')!;
    const audio = hub.seen.find((s) => s.path === '/audio/m1')!;
    expect(mission.auth).toBe(`Bearer ${TOKEN}`);
    expect(audio.auth).toBe(`Bearer ${TOKEN}`);

    const log: { e: string; t: number }[] = await page.evaluate(() => (window as any).__log);
    expect(log.map((l) => l.e)).toEqual(['play', 'ended', 'play', 'ended']);
    expect(log[2]!.t - log[1]!.t).toBeGreaterThanOrEqual(2000);
    expect(log[2]!.t - log[1]!.t).toBeLessThan(2600);

    // Okay dims the screen; the dimmed screen carries "Tap to wake".
    await page.getByRole('button', { name: 'Okay' }).click();
    await expect(page.getByText('Tap to wake')).toBeVisible();
  });
});

test('NFR-L1: text on screen at once; audio plays twice when it arrives', async ({ page }) => {
  const hub = newHub();
  hub.missionAudioPath = null;
  await installBrowserFakes(page);
  await installHub(page, hub);
  await page.goto('/');
  setTimeout(() => (hub.missionAudioPath = '/audio/m1'), 3000);
  await page.getByRole('button', { name: "Hear today's walk" }).click();
  await expect(page.locator('main')).toContainText(/Lookatthegulmoharonyourlane\./, { timeout: 500 });
  expect(await page.evaluate(() => (window as any).__log.length)).toBe(0); // nothing plays before audio exists
  await expect(page.getByRole('heading', { name: 'Enjoy your walk.' })).toBeVisible({ timeout: 20_000 });
  const log: { e: string; t: number }[] = await page.evaluate(() => (window as any).__log);
  expect(log.map((l) => l.e)).toEqual(['play', 'ended', 'play', 'ended']);
  expect(log[2]!.t - log[1]!.t).toBeGreaterThanOrEqual(2000);
});

test.describe('J3: tell Sair', () => {
  test('record, upload with the header, Writing (only after the 202), then Saved', async ({ page }) => {
    const hub = newHub();
    const ok = hub.walkPost;
    hub.walkPost = async (route) => {
      await new Promise((r) => setTimeout(r, 1000));
      await ok(route);
    };
    await installBrowserFakes(page);
    await installHub(page, hub);
    await page.goto('/');
    await page.getByRole('link', { name: 'Back from your walk? Tell Sair' }).click();
    await expect(page.getByRole('heading', { name: 'How was your walk?' })).toBeVisible();
    await expect(page.getByText('Look at the gulmohar on your lane.')).toBeVisible(); // mission reminder from /mission/today
    await expectNoSeriousAxe(page);

    await holdToTalk(page);
    // upload in flight (mock holds the 202 for 1 s): still the recording screen, not Writing
    await expect(page.getByText('Listening')).toBeVisible();
    await expect(page.getByText('Writing it down…')).toHaveCount(0);
    await page.waitForTimeout(500);
    await expect(page.getByText('Writing it down…')).toHaveCount(0);
    await expect(page.getByText('Writing it down…')).toBeVisible();
    await expectNoSeriousAxe(page);

    await expect(page.getByRole('heading', { name: 'Got it. Saved.' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(DONE_WALK.entry)).toBeVisible();
    await expect(page.getByText('gulmohar · red flowers')).toBeVisible();
    await expect(page.getByText('koel')).toBeVisible();
    await expect(page.getByText('Mission done')).toBeVisible();
    await expect(page.getByText('Mom and Dad', { exact: true })).toBeVisible(); // group from /diary
    await expect(page.getByText('Read it to me')).toHaveCount(0);
    await expectNoSeriousAxe(page);

    const post = hub.seen.find((s) => s.path === '/walk' && s.method === 'POST')!;
    expect(post.auth).toBe(`Bearer ${TOKEN}`);
    expect(post.contentType).toMatch(/^multipart\/form-data/);
    expect(post.body!.toString('latin1')).toContain('name="audio"');

    await page.getByRole('button', { name: 'Done' }).click();
    await expect(page.getByRole('heading', { name: 'Ready for your sair?' })).toBeVisible();
    await expect(page.getByText('Your first walk')).toHaveCount(0); // a walk has been saved: normal door now
    await expect(page.getByText("Today's walk", { exact: true })).toBeVisible();
  });

  test('a silent note ends on the couldn\'t-hear state', async ({ page }) => {
    const hub = newHub();
    hub.walkGet = (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'lost', date: '2026-10-10', reason: 'audio_silent' }) });
    await installBrowserFakes(page);
    await installHub(page, hub);
    await page.goto('/');
    await page.getByRole('link', { name: 'Back from your walk? Tell Sair' }).click();
    await holdToTalk(page);
    await expect(page.getByRole('heading', { name: "I couldn't hear that clearly." })).toBeVisible({ timeout: 15_000 });
    await expectNoSeriousAxe(page);
    await page.getByRole('button', { name: 'Try again' }).click();
    await expect(page.getByRole('heading', { name: 'How was your walk?' })).toBeVisible();
  });

  test('45 s without the entry shows the gap state, then the entry arrives', async ({ page }) => {
    const hub = newHub();
    hub.walkGet = (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'received', date: '2026-10-10' }) });
    await installBrowserFakes(page);
    await installHub(page, hub);
    await page.clock.install();
    await page.goto('/');
    await page.getByRole('link', { name: 'Back from your walk? Tell Sair' }).click();
    await holdToTalk(page);
    await expect(page.getByText('Writing it down…')).toBeVisible();
    await page.clock.runFor(44_000);
    await expect(page.getByText('Writing it down…')).toBeVisible();
    await page.clock.runFor(2_000);
    await expect(page.getByText('Your diary entry will be ready soon')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Got it. Saved.' })).toBeVisible();
    await expectNoSeriousAxe(page);

    hub.walkGet = (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(DONE_WALK) });
    await expect(async () => {
      await page.clock.runFor(2_600);
      await expect(page.getByText(DONE_WALK.entry)).toBeVisible({ timeout: 300 });
    }).toPass({ timeout: 15_000 });
  });
});

test.describe('setup and diary', () => {
  test('no token: setup message; the setup link stores the token and clears the address bar', async ({ page }) => {
    await installBrowserFakes(page, { token: null });
    await installHub(page, newHub());
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Sair needs to be set up on this phone.' })).toBeVisible();
    await page.goto('/#setup?token=abc123');
    await expect(page.getByRole('heading', { name: 'Ready for your sair?' })).toBeVisible();
    expect(await page.evaluate(() => localStorage.getItem('sair.token'))).toBe('abc123');
    expect(page.url()).not.toContain('abc123');
    expect(page.url()).not.toContain('#');
  });

  test('diary loads with the header and filters by group', async ({ page }) => {
    const hub = newHub();
    await installBrowserFakes(page);
    await installHub(page, hub);
    await page.goto('/#diary');
    await expect(page.getByRole('heading', { name: 'Our walks' })).toBeVisible();
    await expect(page.getByText('2 walks this month · 1 thing noticed')).toBeVisible();
    await expect(page.getByText('Sat 10 Oct')).toBeVisible();
    await expect(page.getByText('Asked about again on Mon 12 Oct')).toBeVisible();
    await expectNoSeriousAxe(page);
    await page.getByRole('button', { name: 'Dad', exact: true }).click();
    await expect(page.getByText('Dad walked alone.')).toBeVisible();
    await expect(page.getByText('Mom and Dad walked down the lane.')).toHaveCount(0);
    const filtered = hub.seen.find((s) => s.path === '/diary?group=Dad')!;
    expect(filtered.auth).toBe(`Bearer ${TOKEN}`);
  });
});
