import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

const apiProxy = {
  // A cold range chunk runs up to 24 forecasts. Keep the proxy alive beyond
  // fetchRangeChunk's 180-second client deadline instead of cutting it at 10s.
  '/api/v1/predictions/range': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    proxyTimeout: 190_000,
  },
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    proxyTimeout: 10_000,
  },
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: apiProxy,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
    proxy: apiProxy,
  },
});
