import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// assetsInlineLimit 0: no data: URIs for assets, so the build needs only `img-src 'self'` style access and stays CSP-simple.
export default defineConfig({ plugins: [react()], server: { host: '127.0.0.1' }, build: { assetsInlineLimit: 0 } });
