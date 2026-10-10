import { defineConfig } from '@playwright/test';

// Reference capture is a plain script (npm run capture). Screens project: fixture pages served by the vite dev server on loopback.
export default defineConfig({
  testDir: 'tests',
  testMatch: /.*\.spec\.ts/,
  use: { deviceScaleFactor: 1, baseURL: 'http://127.0.0.1:5179' },
  webServer: {
    command: 'npx vite --host 127.0.0.1 --port 5179 --strictPort',
    url: 'http://127.0.0.1:5179',
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
