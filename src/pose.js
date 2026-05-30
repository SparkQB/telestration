// ── Pose detection using MediaPipe Pose ──────────────────────────────────────
// Self-hosted assets in /public/pose/

const POSE_LANDMARKS = {
  // Upper body
  LEFT_SHOULDER:  11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW:     13,
  RIGHT_ELBOW:    14,
  LEFT_WRIST:     15,
  RIGHT_WRIST:    16,
  // Lower body
  LEFT_HIP:       23,
  RIGHT_HIP:      24,
  LEFT_KNEE:      25,
  RIGHT_KNEE:     26,
  LEFT_ANKLE:     27,
  RIGHT_ANKLE:    28,
}

// Joint connections to draw skeleton lines
export const SKELETON_CONNECTIONS = [
  // Torso
  ['LEFT_SHOULDER',  'RIGHT_SHOULDER'],
  ['LEFT_SHOULDER',  'LEFT_HIP'],
  ['RIGHT_SHOULDER', 'RIGHT_HIP'],
  ['LEFT_HIP',       'RIGHT_HIP'],
  // Left arm
  ['LEFT_SHOULDER',  'LEFT_ELBOW'],
  ['LEFT_ELBOW',     'LEFT_WRIST'],
  // Right arm
  ['RIGHT_SHOULDER', 'RIGHT_ELBOW'],
  ['RIGHT_ELBOW',    'RIGHT_WRIST'],
  // Left leg
  ['LEFT_HIP',       'LEFT_KNEE'],
  ['LEFT_KNEE',      'LEFT_ANKLE'],
  // Right leg
  ['RIGHT_HIP',      'RIGHT_KNEE'],
  ['RIGHT_KNEE',     'RIGHT_ANKLE'],
]

// Angle joints: [pointA, vertex, pointB] — angle calculated at vertex
export const ANGLE_JOINTS = [
  { name: 'L.Shoulder', points: ['LEFT_ELBOW',    'LEFT_SHOULDER',  'LEFT_HIP'] },
  { name: 'R.Shoulder', points: ['RIGHT_ELBOW',   'RIGHT_SHOULDER', 'RIGHT_HIP'] },
  { name: 'L.Elbow',    points: ['LEFT_SHOULDER', 'LEFT_ELBOW',     'LEFT_WRIST'] },
  { name: 'R.Elbow',    points: ['RIGHT_SHOULDER','RIGHT_ELBOW',    'RIGHT_WRIST'] },
  { name: 'L.Wrist',    points: ['LEFT_ELBOW',    'LEFT_WRIST',     'LEFT_ELBOW'] }, // wrist flex approximation
  { name: 'R.Wrist',    points: ['RIGHT_ELBOW',   'RIGHT_WRIST',    'RIGHT_ELBOW'] },
  { name: 'L.Hip',      points: ['LEFT_SHOULDER', 'LEFT_HIP',       'LEFT_KNEE'] },
  { name: 'R.Hip',      points: ['RIGHT_SHOULDER','RIGHT_HIP',      'RIGHT_KNEE'] },
  { name: 'L.Knee',     points: ['LEFT_HIP',      'LEFT_KNEE',      'LEFT_ANKLE'] },
  { name: 'R.Knee',     points: ['RIGHT_HIP',     'RIGHT_KNEE',     'RIGHT_ANKLE'] },
  { name: 'L.Ankle',    points: ['LEFT_KNEE',     'LEFT_ANKLE',     'LEFT_KNEE'] },
  { name: 'R.Ankle',    points: ['RIGHT_KNEE',    'RIGHT_ANKLE',    'RIGHT_KNEE'] },
]

export const LANDMARK_KEYS = POSE_LANDMARKS

// Calculate angle at vertex point B given three points
export function calcAngle(a, b, c) {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x)
  let angle = Math.abs(radians * 180 / Math.PI)
  if (angle > 180) angle = 360 - angle
  return Math.round(angle)
}

// Draw angle arc at joint
export function drawAngleArc(ctx, a, b, c, angle, color, canvasW, canvasH) {
  const bx = b.x * canvasW
  const by = b.y * canvasH
  const ax = a.x * canvasW
  const ay = a.y * canvasH
  const cx = c.x * canvasW
  const cy = c.y * canvasH

  const angleA = Math.atan2(ay - by, ax - bx)
  const angleC = Math.atan2(cy - by, cx - bx)
  const radius = Math.min(canvasW, canvasH) * 0.032

  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  ctx.globalAlpha = 0.85
  ctx.beginPath()
  ctx.arc(bx, by, radius, angleA, angleC)
  ctx.stroke()

  // Angle label
  const midAngle = (angleA + angleC) / 2
  const labelR = radius + 14
  const lx = bx + Math.cos(midAngle) * labelR
  const ly = by + Math.sin(midAngle) * labelR

  ctx.globalAlpha = 1
  ctx.font = `bold ${Math.max(10, canvasW * 0.013)}px 'Roboto Mono', monospace`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  // Background pill for readability
  const text = `${angle}°`
  const tw = ctx.measureText(text).width
  ctx.fillStyle = 'rgba(0,0,0,0.55)'
  ctx.beginPath()
  ctx.roundRect(lx - tw/2 - 3, ly - 8, tw + 6, 16, 4)
  ctx.fill()

  ctx.fillStyle = color
  ctx.fillText(text, lx, ly)
  ctx.restore()
}

// Draw full skeleton + angles onto a canvas context
export function drawPoseOverlay(ctx, landmarks, enabledAngles, canvasW, canvasH) {
  if (!landmarks || landmarks.length === 0) return
  const CYAN = '#00E5FF'
  const MIN_VISIBILITY = 0.5

  // Helper to get landmark by name
  const lm = (name) => {
    const idx = POSE_LANDMARKS[name]
    const pt  = landmarks[idx]
    if (!pt || pt.visibility < MIN_VISIBILITY) return null
    return pt
  }

  // Draw skeleton lines
  ctx.save()
  ctx.strokeStyle = CYAN
  ctx.lineWidth   = Math.max(1.5, canvasW * 0.002)
  ctx.globalAlpha = 0.75
  ctx.lineCap     = 'round'

  SKELETON_CONNECTIONS.forEach(([a, b]) => {
    const pa = lm(a), pb = lm(b)
    if (!pa || !pb) return
    ctx.beginPath()
    ctx.moveTo(pa.x * canvasW, pa.y * canvasH)
    ctx.lineTo(pb.x * canvasW, pb.y * canvasH)
    ctx.stroke()
  })

  // Draw joint dots
  ctx.globalAlpha = 1
  Object.entries(POSE_LANDMARKS).forEach(([name, idx]) => {
    const pt = landmarks[idx]
    if (!pt || pt.visibility < MIN_VISIBILITY) return
    const r = Math.max(3, canvasW * 0.004)
    ctx.beginPath()
    ctx.arc(pt.x * canvasW, pt.y * canvasH, r, 0, Math.PI * 2)
    ctx.fillStyle = CYAN
    ctx.fill()
    ctx.strokeStyle = 'rgba(0,0,0,0.5)'
    ctx.lineWidth = 1
    ctx.stroke()
  })

  ctx.restore()

  // Draw angle arcs
  ANGLE_JOINTS.forEach(joint => {
    if (enabledAngles && !enabledAngles[joint.name]) return
    const [an, bn, cn] = joint.points
    const a = lm(an), b = lm(bn), c = lm(cn)
    if (!a || !b || !c) return
    if (an === cn) return // skip wrist/ankle (need better data)
    const angle = calcAngle(a, b, c)
    drawAngleArc(ctx, a, b, c, angle, CYAN, canvasW, canvasH)
  })
}

// ── MediaPipe Pose loader ────────────────────────────────────────────────────
let poseInstance = null
let poseReady    = false
let poseCallbacks = []

export function loadPose(onResults) {
  return new Promise((resolve, reject) => {
    if (poseInstance && poseReady) {
      poseInstance.onResults(onResults)
      resolve(poseInstance)
      return
    }

    // Load MediaPipe Pose via CDN — most reliable cross-browser approach
    if (!document.getElementById('mediapipe-pose-script')) {
      const script = document.createElement('script')
      script.id  = 'mediapipe-pose-script'
      script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/pose.js'
      script.crossOrigin = 'anonymous'
      document.head.appendChild(script)

      script.onload = () => initPose(onResults, resolve, reject)
      script.onerror = () => reject(new Error('Failed to load MediaPipe Pose script'))
    } else {
      initPose(onResults, resolve, reject)
    }
  })
}

function initPose(onResults, resolve, reject) {
  try {
    const pose = new window.Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`
    })

    pose.setOptions({
      modelComplexity: 1,        // 0=lite, 1=full, 2=heavy
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })

    pose.onResults(onResults)

    pose.initialize().then(() => {
      poseInstance = pose
      poseReady    = true
      resolve(pose)
    }).catch(reject)
  } catch(e) {
    reject(e)
  }
}

export function getPoseInstance() { return poseInstance }
export { POSE_LANDMARKS }
