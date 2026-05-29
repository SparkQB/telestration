#!/usr/bin/env node
// Run once from project root: node download-model.js
// Downloads face-api.js model weights to public/faceapi/
// Models: SSD MobileNet (detection) + Face Landmark 68 (landmarks)

const https = require('https')
const http  = require('http')
const fs    = require('fs')
const path  = require('path')

const OUTDIR = path.join(__dirname, 'public', 'faceapi')
if (!fs.existsSync(OUTDIR)) fs.mkdirSync(OUTDIR, { recursive: true })

// face-api.js models hosted on GitHub
const BASE = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights'

const FILES = [
  // SSD MobileNet — face detection
  'ssd_mobilenetv1_model-weights_manifest.json',
  'ssd_mobilenetv1_model-shard1',
  'ssd_mobilenetv1_model-shard2',
  // Face Landmark 68 — facial landmarks
  'face_landmark_68_model-weights_manifest.json',
  'face_landmark_68_model-shard1',
]

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest)
    const protocol = url.startsWith('https') ? https : http
    protocol.get(url, res => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        file.close(); fs.unlinkSync(dest)
        return download(res.headers.location, dest).then(resolve).catch(reject)
      }
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode} for ${url}`)); return
      }
      res.pipe(file)
      file.on('finish', () => { file.close(); resolve() })
    }).on('error', err => {
      try { fs.unlinkSync(dest) } catch(e) {}
      reject(err)
    })
  })
}

async function run() {
  console.log('Downloading face-api.js models to public/faceapi/...\n')
  for (const fname of FILES) {
    const url  = `${BASE}/${fname}`
    const dest = path.join(OUTDIR, fname)
    process.stdout.write(`  ${fname}... `)
    try {
      await download(url, dest)
      const size = (fs.statSync(dest).size / 1024).toFixed(0)
      console.log(`✓ ${size}kb`)
    } catch(e) {
      console.log(`✗ ${e.message}`)
      process.exit(1)
    }
  }
  console.log('\nDone! Commit public/faceapi/ to GitHub and redeploy.')
}

run()
