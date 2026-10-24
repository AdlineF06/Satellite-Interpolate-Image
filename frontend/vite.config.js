import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  plugins: [react(), viteStaticCopy({ targets: [{ src: 'node_modules/cesium/Build/Cesium/*', dest: 'cesium' }] })],
  define: { CESIUM_BASE_URL: JSON.stringify('/cesium/') },
  server: { proxy: { '/api': 'http://127.0.0.1:8001' } },
})
