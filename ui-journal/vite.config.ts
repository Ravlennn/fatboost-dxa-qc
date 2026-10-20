import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The «Журнал» UI. The build lands inside the Python package next to the classic one, and `dxaqc serve`
// (and the Docker image) serve it offline at /journal/. `bun run dev` proxies the API to `dxaqc serve --port 8766`.
export default defineConfig({
  plugins: [react()],
  base: '/journal/',
  build: {
    outDir: '../src/dxaqc/webui_journal_dist',
    emptyOutDir: true,
    assetsDir: 'assets',
    chunkSizeWarningLimit: 600,
  },
  server: { proxy: { '/api': 'http://127.0.0.1:8766' } },
})
