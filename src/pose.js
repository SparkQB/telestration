// ── Pose detection using MediaPipe Pose ──────────────────────────────────────
// Self-hosted assets in /public/pose/

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
function vec3(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z }
}
function dot(a, b) { return a.x*b.x + a.y*b.y + a.z*b.z }
function cross(a, b) {
  return { x: a.y*b.z - a.z*b.y, y: a.z*b.x - a.x*b.z, z: a.x*b.y - a.y*b.x }
}
function mag(v) { return Math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) }
function norm(v) { const m = mag(v) || 1; return { x:v.x/m, y:v.y/m, z:v.z/m } }
function angleBetween(a, b) {
  const cosA = Math.max(-1, Math.min(1, dot(norm(a), norm(b))))
  return Math.acos(cosA) * 180 / Math.PI
}

// ── Angle calculations using 3D world landmarks ───────────────────────────────

// Elbow flexion: 0° = full extension, 180° = full flexion
// upper arm vector dotted with forearm vector
function calcElbow(wl, side) {
  const S = side === 'left' ? wl[11] : wl[12]
  const E = side === 'left' ? wl[13] : wl[14]
  const W = side === 'left' ? wl[15] : wl[16]
  if (!S || !E || !W) return null
  const upperArm = vec3(E, S)  // elbow to shoulder
  const forearm  = vec3(E, W)  // elbow to wrist
  const angle = angleBetween(upperArm, forearm)
  return Math.round(180 - angle) // 0 = straight, 180 = fully bent
}

// Knee flexion: 0° = full extension, increasing = flexion
function calcKnee(wl, side) {
  const H = side === 'left' ? wl[23] : wl[24]
  const K = side === 'left' ? wl[25] : wl[26]
  const A = side === 'left' ? wl[27] : wl[28]
  if (!H || !K || !A) return null
  const thigh  = vec3(K, H)  // knee to hip
  const shin   = vec3(K, A)  // knee to ankle
  const angle = angleBetween(thigh, shin)
  return Math.round(180 - angle) // 0 = straight, 180 = fully bent
}

// Shoulder abduction/adduction — 2D angle between torso and upper arm
// 0° = arm at side, 90° = arm horizontal, 180° = arm straight up
// Uses 2D image landmarks for reliability
function calcShoulderAbduction(wl2d, side) {
  // wl2d here is actually poseLandmarks (2D) stored on the model
  // We pass worldLandmarks but fall back gracefully
  if (!wl2d) return null
  const S = side === 'left' ? wl2d[11] : wl2d[12]
  const E = side === 'left' ? wl2d[13] : wl2d[14]
  const H = side === 'left' ? wl2d[23] : wl2d[24]
  if (!S || !E || !H) return null

  // Vector from shoulder down to hip (torso reference = "straight down")
  const torso = { x: H.x - S.x, y: H.y - S.y }
  // Vector from shoulder to elbow (upper arm)
  const arm   = { x: E.x - S.x, y: E.y - S.y }

  const dot2d = torso.x * arm.x + torso.y * arm.y
  const magT  = Math.sqrt(torso.x**2 + torso.y**2)
  const magA  = Math.sqrt(arm.x**2 + arm.y**2)
  if (magT < 0.001 || magA < 0.001) return null

  const cosA = Math.max(-1, Math.min(1, dot2d / (magT * magA)))
  return Math.round(Math.acos(cosA) * 180 / Math.PI)
}

export function calcJointAngle(joint, worldLandmarks, landmarks2d) {
  if (!worldLandmarks) return null
  switch(joint.type) {
    case 'elbow':    return calcElbow(worldLandmarks, joint.side)
    case 'knee':     return calcKnee(worldLandmarks, joint.side)
    case 'shoulder': return calcShoulderAbduction(worldLandmarks, joint.side)
    default: return null
  }
}

// ── Draw angle label + arc ────────────────────────────────────────────────────
function drawAngleLabel(ctx, lm2d, joint, angle, canvasW, canvasH) {
  // Get vertex landmark in 2D for positioning
  let vidx
  if (joint.type === 'elbow')    vidx = joint.side === 'left' ? 13 : 14
  else if (joint.type === 'knee') vidx = joint.side === 'left' ? 25 : 26
  else vidx = joint.side === 'left' ? 11 : 12  // shoulder

  const vpt = lm2d[vidx]
  if (!vpt) return

  const vx = vpt.x * canvasW
  const vy = vpt.y * canvasH

  // Arc points for shoulder/elbow/knee
  let ax, ay, bx2, by2
  if (joint.type === 'elbow') {
    const s = lm2d[joint.side === 'left' ? 11 : 12]
    const w = lm2d[joint.side === 'left' ? 15 : 16]
    if (s && w) {
      ax = s.x * canvasW; ay = s.y * canvasH
      bx2 = w.x * canvasW; by2 = w.y * canvasH
    }
  } else if (joint.type === 'knee') {
    const h = lm2d[joint.side === 'left' ? 23 : 24]
    const a = lm2d[joint.side === 'left' ? 27 : 28]
    if (h && a) {
      ax = h.x * canvasW; ay = h.y * canvasH
      bx2 = a.x * canvasW; by2 = a.y * canvasH
    }
  } else {
    // Shoulder — use elbow and hip
    const e = lm2d[joint.side === 'left' ? 13 : 14]
    const h = lm2d[joint.side === 'left' ? 23 : 24]
    if (e && h) {
      ax = e.x * canvasW; ay = e.y * canvasH
      bx2 = h.x * canvasW; by2 = h.y * canvasH
    }
  }

  const CYAN = '#00E5FF'
  const radius = Math.min(canvasW, canvasH) * 0.028

  ctx.save()

  // Draw arc if we have both reference points
  if (ax !== undefined) {
    const angleA = Math.atan2(ay - vy, ax - vx)
    const angleB = Math.atan2(by2 - vy, bx2 - vx)
    ctx.strokeStyle = CYAN
    ctx.lineWidth = 1.5
    ctx.globalAlpha = 0.8
    ctx.beginPath()
    ctx.arc(vx, vy, radius, Math.min(angleA, angleB), Math.max(angleA, angleB))
    ctx.stroke()
  }

  // Label — offset away from body
  const offsetX = joint.side === 'left' ? -radius - 28 : radius + 10
  const offsetY = joint.type === 'knee' ? -10 : -radius - 10
  const lx = vx + offsetX
  const ly = vy + offsetY

  const label = joint.type === 'shoulder'
    ? `${angle > 0 ? '+' : ''}${angle}°`
    : `${angle}°`
  const text = `${joint.name} ${label}`

  ctx.globalAlpha = 1
  ctx.font = `bold ${Math.max(10, canvasW * 0.012)}px 'Roboto Mono', monospace`
  ctx.textAlign = joint.side === 'left' ? 'right' : 'left'
  ctx.textBaseline = 'middle'

  const tw = ctx.measureText(text).width
  const px = joint.side === 'left' ? lx - tw - 2 : lx - 2
  ctx.fillStyle = 'rgba(0,0,0,0.6)'
  ctx.beginPath()
  ctx.roundRect(px, ly - 9, tw + 8, 18, 4)
  ctx.fill()

  ctx.fillStyle = CYAN
  ctx.fillText(text, lx, ly)
  ctx.restore()
}

// Draw angle arc only (no label — label goes in floating table)
function drawAngleArc(ctx, lm2d, joint, canvasW, canvasH) {
  let vidx
  if (joint.type === 'elbow')    vidx = joint.side === 'left' ? 13 : 14
  else if (joint.type === 'knee') vidx = joint.side === 'left' ? 25 : 26
  else vidx = joint.side === 'left' ? 11 : 12

  const vpt = lm2d[vidx]; if (!vpt) return
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
  const MIN_VIS = 0.4

  const lm = (idx) => {
    const pt = landmarks[idx]
    return (pt && pt.visibility >= MIN_VIS) ? pt : null
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
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
      enableWorldLandmarks: true,  // required for 3D world coordinates
    })

    pose.onResults((results) => {
      console.log('[SparkQB] Pose results:', results.poseLandmarks ? `${results.poseLandmarks.length} landmarks` : 'no landmarks')
      onResults(results)
    })

    pose.initialize().then(() => {
      console.log('[SparkQB] Pose initialized OK')
      poseInstance = pose
      poseReady    = true
      resolve(pose)
    }).catch(e => {
      console.error('[SparkQB] Pose initialize failed:', e)
      reject(e)
    })
  } catch(e) {
    reject(e)
  }
}

export function getPoseInstance() { return poseInstance }
export { POSE_LANDMARKS }
