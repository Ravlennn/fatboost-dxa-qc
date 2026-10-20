import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The build lands inside the Python package, so `dxaqc serve` (and the Docker image) serve it
// offline with no Node at runtime. `bun run dev` proxies the API to a local `dxaqc serve --port 8766`.
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../src/dxaqc/webui_dist',
    emptyOutDir: true,
    assetsDir: 'assets',
    chunkSizeWarningLimit: 600,
  },
  server: { proxy: { '/api': 'http://127.0.0.1:8766' } },
})
