#!/usr/bin/env node
// Run this once from your project root: node download-model.js
// It downloads the BlazeFace model weights and saves them to public/blazeface/
// After running, commit the files to GitHub and Vercel will serve them.

const https = require('https')
const fs    = require('fs')
const path  = require('path')

const BASE   = 'https://tfhub.dev/tensorflow/tfjs-model/blazeface/1/default/1'
const OUTDIR = path.join(__dirname, 'public', 'blazeface')

if (!fs.existsSync(OUTDIR)) fs.mkdirSync(OUTDIR, { recursive: true })

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest)
    https.get(url, res => {
      // Follow redirects
      if (res.statusCode === 301 || res.statusCode === 302) {
        file.close()
        fs.unlinkSync(dest)
        return download(res.headers.location, dest).then(resolve).catch(reject)
      }
      if (res.statusCode !== 200) {
        reject(new Error(`Failed: ${res.statusCode} for ${url}`))
        return
      }
      res.pipe(file)
      file.on('finish', () => { file.close(); resolve() })
    }).on('error', err => {
      fs.unlinkSync(dest)
      reject(err)
    })
  })
}

async function run() {
  console.log('Downloading BlazeFace model...')

  // 1. Download model.json
  const modelJsonUrl  = `${BASE}/model.json?tfjs-format=file`
  const modelJsonPath = path.join(OUTDIR, 'model.json')
  console.log('  model.json...')
  await download(modelJsonUrl, modelJsonPath)

  // 2. Parse model.json to find weight shard filenames
  const modelJson   = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'))
  const weightFiles = modelJson.weightsManifest
    .flatMap(group => group.paths)

  // 3. Download each weight shard
  for (const fname of weightFiles) {
    const url  = `${BASE}/${fname}?tfjs-format=file`
    const dest = path.join(OUTDIR, fname)
    console.log(`  ${fname}...`)
    await download(url, dest)
  }

  console.log(`\nDone! ${weightFiles.length + 1} files saved to public/blazeface/`)
  console.log('Now commit these files to GitHub and redeploy.')
}

run().catch(err => {
  console.error('Error:', err.message)
  process.exit(1)
})
