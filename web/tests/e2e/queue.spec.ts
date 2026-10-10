// FR-J9: upload fails -> blob in IndexedDB -> hub-sleeping state -> hub recovers -> upload succeeds -> IndexedDB empty.
import { expect, test, type Page } from '@playwright/test';
import { expectNoSeriousAxe, holdToTalk, installBrowserFakes, installHub, newHub } from './mock.ts';

const pendingCount = (page: Page) =>
  page.evaluate(
    () =>
      new Promise<number>((resolve, reject) => {
        const open = indexedDB.open('sair-queue', 1);
        open.onsuccess = () => {
          const db = open.result;
          if (!db.objectStoreNames.contains('pending')) return resolve(0);
          const req = db.transaction('pending').objectStore('pending').count();
          req.onsuccess = () => {
            db.close();
            resolve(req.result);
          };
          req.onerror = () => reject(new Error('count'));
        };
        open.onerror = () => reject(new Error('open'));
      }),
  );

test('FR-J9: queued while the hub sleeps, sent on recovery, local copy deleted at the ack', async ({ page }) => {
  const hub = newHub();
  let up = false;
  const ok = hub.walkPost;
  hub.walkPost = (route) => (up ? ok(route) : route.abort('connectionrefused'));
  await installBrowserFakes(page);
  await installHub(page, hub);
  await page.goto('/');
  await page.getByRole('link', { name: 'Back from your walk? Tell Sair' }).click();
  await holdToTalk(page);

  await expect(page.getByRole('heading', { name: "Sair's home computer is sleeping." })).toBeVisible();
  await expect(page.getByText('Your note is safe on this phone')).toBeVisible();
  await expect(page.getByText(/^0:0\d · it will send by itself$/)).toBeVisible();
  await expectNoSeriousAxe(page);
  expect(await pendingCount(page)).toBe(1);
  // audio never goes to localStorage
  const ls = await page.evaluate(() => JSON.stringify({ ...localStorage }));
  expect(ls).not.toContain('fake-opus');

  // still down on a retry (the `online` event): stays queued
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await expect(async () => expect(hub.seen.filter((s) => s.path === '/walk' && s.method === 'POST').length).toBeGreaterThanOrEqual(2)).toPass();
  expect(await pendingCount(page)).toBe(1);

  // hub wakes
  up = true;
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await expect(page.getByText('Writing it down…').or(page.getByRole('heading', { name: 'Got it. Saved.' }))).toBeVisible();
  await expect.poll(() => pendingCount(page)).toBe(0);
  await expect(page.getByRole('heading', { name: 'Got it. Saved.' })).toBeVisible({ timeout: 15_000 });
  const posts = hub.seen.filter((s) => s.path === '/walk' && s.method === 'POST');
  expect(posts.every((p) => p.auth === 'Bearer test-token')).toBe(true);
});

test('a 5xx is queued too, and the queue survives a reload', async ({ page }) => {
  const hub = newHub();
  hub.walkPost = (route) => route.fulfill({ status: 503, contentType: 'application/json', body: '{}' });
  await installBrowserFakes(page);
  await installHub(page, hub);
  await page.goto('/');
  await page.getByRole('link', { name: 'Back from your walk? Tell Sair' }).click();
  await holdToTalk(page);
  await expect(page.getByRole('heading', { name: "Sair's home computer is sleeping." })).toBeVisible();
  expect(await pendingCount(page)).toBe(1);
  await page.reload();
  await expect(page.getByRole('heading', { name: "Sair's home computer is sleeping." })).toBeVisible();
  expect(await pendingCount(page)).toBe(1);
});
