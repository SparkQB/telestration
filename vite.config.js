import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

// Serve / emit MediaPipe tasks-vision WASM files at ./mediapipe-wasm/ so the
// pose engine never depends on a third-party CDN at runtime.
const WASM_DIR = path.resolve('node_modules/@mediapipe/tasks-vision/wasm')
function mediapipeWasm() {
  return {
    name: 'mediapipe-wasm',
    configureServer(server) {
      server.middlewares.use('/mediapipe-wasm', (req, res, next) => {
        const f = path.join(WASM_DIR, path.basename(req.url.split('?')[0]))
        if (!fs.existsSync(f)) return next()
        res.setHeader('Content-Type', f.endsWith('.wasm') ? 'application/wasm' : 'text/javascript')
        fs.createReadStream(f).pipe(res)
      })
    },
    generateBundle() {
      for (const name of fs.readdirSync(WASM_DIR)) {
        this.emitFile({ type: 'asset', fileName: `mediapipe-wasm/${name}`, source: fs.readFileSync(path.join(WASM_DIR, name)) })
      }
    },
  }
}

export default defineConfig({
  plugins: [react(), mediapipeWasm()],
  base: './'
})
