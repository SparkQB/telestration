// ── Pose detection using MediaPipe PoseLandmarker ──────────────────────────────────────
// Uses @mediapipe/tasks-vision PoseLandmarker (heavy model)
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

const POSE_LANDMARKS = {
  LEFT_SHOULDER:  11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW:     13,
  RIGHT_ELBOW:    14,
  LEFT_WRIST:     15,
  RIGHT_WRIST:    16,
  LEFT_HIP:       23,
  RIGHT_HIP:      24,
  LEFT_KNEE:      25,
  RIGHT_KNEE:     26,
  LEFT_ANKLE:     27,
  RIGHT_ANKLE:    28,
}

// Skeleton connections — draw lines but only show angles for shoulder/elbow/knee
export const SKELETON_CONNECTIONS = [
  ['LEFT_SHOULDER',  'RIGHT_SHOULDER'],
  ['LEFT_SHOULDER',  'LEFT_HIP'],
  ['RIGHT_SHOULDER', 'RIGHT_HIP'],
  ['LEFT_HIP',       'RIGHT_HIP'],
  ['LEFT_SHOULDER',  'LEFT_ELBOW'],
  ['LEFT_ELBOW',     'LEFT_WRIST'],
  ['RIGHT_SHOULDER', 'RIGHT_ELBOW'],
  ['RIGHT_ELBOW',    'RIGHT_WRIST'],
  ['LEFT_HIP',       'LEFT_KNEE'],
  ['LEFT_KNEE',      'LEFT_ANKLE'],
  ['RIGHT_HIP',      'RIGHT_KNEE'],
  ['RIGHT_KNEE',     'RIGHT_ANKLE'],
]

// Only these 6 joints get angle calculations
export const ANGLE_JOINTS = [
  { name: 'L.Shoulder', type: 'shoulder', side: 'left' },
  { name: 'R.Shoulder', type: 'shoulder', side: 'right' },
  { name: 'L.Elbow',    type: 'elbow',    side: 'left' },
  { name: 'R.Elbow',    type: 'elbow',    side: 'right' },
  { name: 'L.Knee',     type: 'knee',     side: 'left' },
  { name: 'R.Knee',     type: 'knee',     side: 'right' },
]

export const LANDMARK_KEYS = POSE_LANDMARKS

// ── 3D vector math ────────────────────────────────────────────────────────────
const MIN_VIS = 0.5
function vis(p) { return p && (p.visibility ?? 1) >= MIN_VIS ? p : null }
function vec3(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z }
}
function dot(a, b) { return a.x*b.x + a.y*b.y + a.z*b.z }
function mag(v) { return Math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) }
function norm(v) { const m = mag(v) || 1; return { x:v.x/m, y:v.y/m, z:v.z/m } }
function angleBetween(a, b) {
  const cosA = Math.max(-1, Math.min(1, dot(norm(a), norm(b))))
  return Math.acos(cosA) * 180 / Math.PI
}

// Angle at vertex B formed by A-B-C, from 3D world landmarks, visibility-gated
function angle3(wl, ia, ib, ic) {
  const A = vis(wl[ia]), B = vis(wl[ib]), C = vis(wl[ic])
  if (!A || !B || !C) return null
  const ba = vec3(B, A), bc = vec3(B, C)
  if (mag(ba) < 1e-4 || mag(bc) < 1e-4) return null
  return angleBetween(ba, bc)
}

// ── Angle calculations using 3D world landmarks ───────────────────────────────
// Elbow / knee flexion: 0° = straight, increasing = more bent
// Shoulder abduction: angle between torso (shoulder→hip) and upper arm
//   0° = arm at side, 90° = arm horizontal, 180° = straight overhead
export function calcJointAngle(joint, worldLandmarks) {
  if (!worldLandmarks) return null
  const L = joint.side === 'left'
  let a = null
  switch (joint.type) {
    case 'elbow':    a = angle3(worldLandmarks, L?11:12, L?13:14, L?15:16); return a == null ? null : Math.round(180 - a)
    case 'knee':     a = angle3(worldLandmarks, L?23:24, L?25:26, L?27:28); return a == null ? null : Math.round(180 - a)
    case 'shoulder': a = angle3(worldLandmarks, L?13:14, L?11:12, L?23:24); return a == null ? null : Math.round(a)
    default: return null
  }
}

// Draw angle arc only (no label — label goes in floating table)
function drawAngleArc(ctx, lm2d, joint, canvasW, canvasH) {
  let vidx
  if (joint.type === 'elbow')    vidx = joint.side === 'left' ? 13 : 14
  else if (joint.type === 'knee') vidx = joint.side === 'left' ? 25 : 26
  else vidx = joint.side === 'left' ? 11 : 12

  const vpt = vis(lm2d[vidx]); if (!vpt) return
  const vx = vpt.x * canvasW, vy = vpt.y * canvasH
  let ax, ay, bx, by

  if (joint.type === 'elbow') {
    const s = lm2d[joint.side === 'left' ? 11 : 12]
    const w = lm2d[joint.side === 'left' ? 15 : 16]
    if (s && w) { ax=s.x*canvasW; ay=s.y*canvasH; bx=w.x*canvasW; by=w.y*canvasH }
  } else if (joint.type === 'knee') {
    const h = lm2d[joint.side === 'left' ? 23 : 24]
    const a = lm2d[joint.side === 'left' ? 27 : 28]
    if (h && a) { ax=h.x*canvasW; ay=h.y*canvasH; bx=a.x*canvasW; by=a.y*canvasH }
  } else {
    const e = lm2d[joint.side === 'left' ? 13 : 14]
    const h = lm2d[joint.side === 'left' ? 23 : 24]
    if (e && h) { ax=e.x*canvasW; ay=e.y*canvasH; bx=h.x*canvasW; by=h.y*canvasH }
  }

  if (ax === undefined) return
  const radius = Math.min(canvasW, canvasH) * 0.028
  const angleA = Math.atan2(ay - vy, ax - vx)
  const angleB = Math.atan2(by - vy, bx - vx)

  ctx.save()
  ctx.strokeStyle = '#00E5FF'
  ctx.lineWidth = 1.5
  ctx.globalAlpha = 0.8
  ctx.beginPath()
  ctx.arc(vx, vy, radius, Math.min(angleA, angleB), Math.max(angleA, angleB))
  ctx.stroke()
  ctx.restore()
}

// Draw full skeleton + angles
export function drawPoseOverlay(ctx, landmarks, worldLandmarks, enabledAngles, canvasW, canvasH) {
  if (!landmarks || landmarks.length === 0) return
  const CYAN = '#00E5FF'

  const lm = (idx) => {
    const pt = landmarks[idx]
    return vis(pt)
  }

  // Skeleton lines
  ctx.save()
  ctx.strokeStyle = CYAN
  ctx.lineWidth   = Math.max(1.5, canvasW * 0.002)
  ctx.globalAlpha = 0.7
  ctx.lineCap     = 'round'
  SKELETON_CONNECTIONS.forEach(([a, b]) => {
    const ia = POSE_LANDMARKS[a], ib = POSE_LANDMARKS[b]
    const pa = lm(ia), pb = lm(ib)
    if (!pa || !pb) return
    ctx.beginPath()
    ctx.moveTo(pa.x * canvasW, pa.y * canvasH)
    ctx.lineTo(pb.x * canvasW, pb.y * canvasH)
    ctx.stroke()
  })

  // Joint dots
  ctx.globalAlpha = 1
  Object.values(POSE_LANDMARKS).forEach(idx => {
    const pt = lm(idx)
    if (!pt) return
    const r = Math.max(3, canvasW * 0.0035)
    ctx.beginPath()
    ctx.arc(pt.x * canvasW, pt.y * canvasH, r, 0, Math.PI * 2)
    ctx.fillStyle = CYAN; ctx.fill()
    ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = 1; ctx.stroke()
  })
  ctx.restore()

  // Angle arcs only (no labels — table handles display)
  ANGLE_JOINTS.forEach(joint => {
    if (enabledAngles && !enabledAngles[joint.name]) return
    const angle = calcJointAngle(joint, worldLandmarks)
    if (angle === null) return
    drawAngleArc(ctx, landmarks, joint, canvasW, canvasH)
  })
}

// ── MediaPipe PoseLandmarker (tasks-vision) ───────────────────────────────────
// Heavy model for accuracy. IMAGE mode for the live/scrub view (no tracking state
// carried across seeks), VIDEO mode for the frame-by-frame throw analysis.

// Bundled by the mediapipe-wasm plugin in vite.config.js
const WASM_URL  = new URL('mediapipe-wasm', document.baseURI).href.replace(/\/$/, '')
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'

let filesetP = null
const instances = {}   // mode -> Promise<PoseLandmarker>
let videoTs = 0        // VIDEO mode needs strictly increasing timestamps

async function create(mode, delegate) {
  filesetP = filesetP || FilesetResolver.forVisionTasks(WASM_URL)
  const fileset = await filesetP
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: mode,
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  })
}

function getLandmarker(mode) {
  if (!instances[mode]) {
    instances[mode] = create(mode, 'GPU').catch(e => {
      console.warn('[SparkQB] GPU pose failed, falling back to CPU:', e)
      return create(mode, 'CPU')
    })
    instances[mode].catch(() => { delete instances[mode] })
  }
  return instances[mode]
}

export function loadPose() { return getLandmarker('IMAGE') }

function toResult(r, crop) {
  const lm = r?.landmarks?.[0] || null
  const world = r?.worldLandmarks?.[0] || null
  if (!lm) return { landmarks: null, worldLandmarks: null }
  // landmarks come back normalized to the crop; map back to the full frame
  const mapped = crop
    ? lm.map(p => ({ ...p, x: crop.x + p.x * crop.w, y: crop.y + p.y * crop.h }))
    : lm
  // tasks-vision world landmarks can lack visibility; copy it from the 2D set
  const w = world ? world.map((p, i) => ({ ...p, visibility: p.visibility ?? lm[i]?.visibility ?? 1 })) : null
  return { landmarks: mapped, worldLandmarks: w }
}

// Freeze the current video frame into a canvas (capped resolution) so the
// skeleton is computed on — and drawn over — exactly the same picture.
export function snapshotFrame(video, canvas, maxSide = 1280) {
  const vw = video.videoWidth, vh = video.videoHeight
  if (!vw || !vh) return null
  const k = Math.min(1, maxSide / Math.max(vw, vh))
  const w = Math.round(vw * k), h = Math.round(vh * k)
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h }
  canvas.getContext('2d').drawImage(video, 0, 0, w, h)
  return canvas
}

// crop: optional {x,y,w,h} in normalized coords — detect only inside it
// (used when zoomed in, so a small QB on game film gets more pixels).
let cropCanvas = null
function sourceFor(src, crop) {
  if (!crop) return src
  const vw = src.videoWidth || src.width, vh = src.videoHeight || src.height
  const sw = Math.max(16, Math.round(crop.w * vw)), sh = Math.max(16, Math.round(crop.h * vh))
  cropCanvas = cropCanvas || document.createElement('canvas')
  if (cropCanvas.width !== sw || cropCanvas.height !== sh) { cropCanvas.width = sw; cropCanvas.height = sh }
  cropCanvas.getContext('2d').drawImage(src, crop.x * vw, crop.y * vh, sw, sh, 0, 0, sw, sh)
  return cropCanvas
}

export async function detectFrame(src, crop = null) {
  const lmk = await getLandmarker('IMAGE')
  return toResult(lmk.detect(sourceFor(src, crop)), crop)
}

// Start a new VIDEO-mode run; returns a per-frame detector (src, frameTimeSec, crop) -> result
export async function startVideoRun() {
  const lmk = await getLandmarker('VIDEO')
  const base = videoTs + 10000   // gap so tracking doesn't bridge two runs
  return (src, frameTimeSec, crop = null) => {
    const ts = Math.max(videoTs + 1, Math.round(base + frameTimeSec * 1000))
    videoTs = ts
    return toResult(lmk.detectForVideo(sourceFor(src, crop), ts), crop)
  }
}

export { POSE_LANDMARKS }
