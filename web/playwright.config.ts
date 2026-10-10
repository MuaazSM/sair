import { defineConfig } from '@playwright/test';

// Visual/axe tests are added in the screens wave. Reference capture is a plain script: npm run capture.
export default defineConfig({ testDir: 'tests', testMatch: /.*\.spec\.ts/, use: { deviceScaleFactor: 1 } });
