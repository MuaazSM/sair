import { defineConfig } from '@playwright/test';

// Fixture pages and the mocked-API e2e specs are served by the vite dev server on loopback (no hub needed).
export default defineConfig({
  testDir: 'tests',
  testMatch: /.*\.spec\.ts/,
  use: { deviceScaleFactor: 1, baseURL: 'http://127.0.0.1:5179' },
  projects: [
    { name: 'visual', testDir: 'tests/visual' },
    { name: 'e2e', testDir: 'tests/e2e', use: { viewport: { width: 390, height: 844 } } },
    { name: 'build-scan', testMatch: /build-scan\.spec\.ts/, timeout: 180_000 },
  ],
  webServer: {
    command: 'npx vite --host 127.0.0.1 --port 5179 --strictPort',
    url: 'http://127.0.0.1:5179',
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
