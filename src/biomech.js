// ── QB throw biomechanics from MediaPipe world landmarks ─────────────────────
// Pure functions — no React, no DOM. Input is a per-frame series:
//   frames: [{ t, world: [{x,y,z,visibility}] | null, lm2d: [...] | null }]
// World landmarks are metric-ish (meters), hip-centred, y pointing down.

export const L = {
  NOSE: 0,
  L_SH: 11, R_SH: 12, L_EL: 13, R_EL: 14, L_WR: 15, R_WR: 16,
  L_HIP: 23, R_HIP: 24, L_KN: 25, R_KN: 26, L_AN: 27, R_AN: 28,
  L_HEEL: 29, R_HEEL: 30, L_TOE: 31, R_TOE: 32,
}
const USED = Object.values(L)
const MIN_VIS = 0.5
const DEG = 180 / Math.PI

// ── vector helpers ───────────────────────────────────────────────────────────
const sub   = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z })
const add   = (a, b) => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z })
const scale = (a, k) => ({ x: a.x * k, y: a.y * k, z: a.z * k })
const mid   = (a, b) => scale(add(a, b), 0.5)
const dot   = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z
const cross = (a, b) => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x })
const mag   = a => Math.hypot(a.x, a.y, a.z)
const unit  = a => { const m = mag(a); return m > 1e-9 ? scale(a, 1 / m) : null }
const isNum = p => p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z)
const rejectFrom = (v, n) => sub(v, scale(n, dot(v, n)))   // remove component along unit n

// ── filtering ────────────────────────────────────────────────────────────────

// Linear-interpolate NaNs; leading/trailing NaNs take nearest value.
export function fillGaps(arr) {
  const out = arr.slice()
  const idx = []
  out.forEach((v, i) => { if (Number.isFinite(v)) idx.push(i) })
  if (idx.length === 0) return out
  for (let i = 0; i < out.length; i++) {
    if (Number.isFinite(out[i])) continue
    let lo = -1, hi = -1
    for (let j = i - 1; j >= 0; j--) if (Number.isFinite(arr[j])) { lo = j; break }
    for (let j = i + 1; j < arr.length; j++) if (Number.isFinite(arr[j])) { hi = j; break }
    if (lo < 0) out[i] = arr[hi]
    else if (hi < 0) out[i] = arr[lo]
    else out[i] = arr[lo] + (arr[hi] - arr[lo]) * (i - lo) / (hi - lo)
  }
  return out
}

// 2nd-order Butterworth low-pass, run forward + backward (zero phase).
export function lowpass(arr, fs, fc) {
  const n = arr.length
  if (n < 7 || !(fc > 0) || fc >= fs / 2) return arr.slice()
  const K = Math.tan(Math.PI * fc / fs)
  const norm = 1 / (1 + Math.SQRT2 * K + K * K)
  const b0 = K * K * norm, b1 = 2 * b0, b2 = b0
  const a1 = 2 * (K * K - 1) * norm, a2 = (1 - Math.SQRT2 * K + K * K) * norm
  const pad = Math.min(n - 1, 12)
  // odd reflection padding to suppress edge transients
  const ext = []
  for (let i = pad; i > 0; i--) ext.push(2 * arr[0] - arr[i])
  ext.push(...arr)
  for (let i = n - 2; i >= n - 1 - pad; i--) ext.push(2 * arr[n - 1] - arr[i])
  const run = x => {
    const y = new Array(x.length)
    y[0] = x[0]; y[1] = x[1]
    // warm start at steady state of first sample
    let x1 = x[0], x2 = x[0], y1 = x[0], y2 = x[0]
    for (let i = 0; i < x.length; i++) {
      const v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
      x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v
    }
    return y
  }
  let y = run(ext)
  y = run(y.reverse()).reverse()
  return y.slice(pad, pad + n)
}

// Central-difference derivative (per second).
export function derivative(arr, t) {
  const n = arr.length, d = new Array(n).fill(NaN)
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1)
    const dt = t[b] - t[a]
    d[i] = dt > 0 ? (arr[b] - arr[a]) / dt : 0
  }
  return d
}

export function unwrapDeg(arr) {
  const out = arr.slice()
  for (let i = 1; i < out.length; i++) {
    let d = out[i] - out[i - 1]
    while (d > 180)  { out[i] -= 360; d -= 360 }
    while (d < -180) { out[i] += 360; d += 360 }
  }
  return out
}

const wrap180 = a => { let v = ((a + 180) % 360 + 360) % 360 - 180; return v }

// Clean + smooth every used landmark series. Returns frames with smoothed world pts.
export function smoothSeries(frames, fps, fcPos = 10) {
  const n = frames.length
  const fc = Math.min(fcPos, fps * 0.3)
  const t = frames.map(f => f.t)
  const out = frames.map(f => ({ t: f.t, world: [] }))
  USED.forEach(li => {
    ;['x', 'y', 'z'].forEach(k => {
      const raw = frames.map(f => {
        const p = f.world?.[li]
        return p && (p.visibility ?? 1) >= MIN_VIS ? p[k] : NaN
      })
      const valid = raw.filter(Number.isFinite).length
      if (valid < Math.max(3, n * 0.3)) {
        out.forEach(o => { o.world[li] = o.world[li] || {}; o.world[li][k] = NaN })
        return
      }
      const sm = lowpass(fillGaps(raw), fps, fc)
      out.forEach((o, i) => { o.world[li] = o.world[li] || {}; o.world[li][k] = sm[i] })
    })
  })
  return { frames: out, t }
}

// Variable-frame-rate clips (e.g. iPhone 240fps with dropped frames): drop
// duplicate timestamps and resample every landmark onto an even grid at fps.
export function resampleUniform(frames, fps) {
  const src = frames.slice().sort((a, b) => a.t - b.t)
    .filter((f, i, arr) => i === 0 || f.t - arr[i - 1].t > 1e-6)
  if (src.length < 2) return src
  const t0 = src[0].t, t1 = src[src.length - 1].t
  const n = Math.floor((t1 - t0) * fps + 1e-6) + 1
  const grid = Array.from({ length: n }, (_, i) => t0 + i / fps)
  const out = grid.map(t => ({ t, world: null, lm2d: null }))
  const ts = src.map(f => f.t)
  for (const key of ['world', 'lm2d']) USED.forEach(li => {
    ;['x', 'y', 'z'].forEach(k => {
      // visibility always comes from the 2D set
      const raw = src.map(f => { const p = f[key]?.[li], q = f.lm2d?.[li] ?? p; return p && (q?.visibility ?? 1) >= MIN_VIS ? p[k] : NaN })
      if (raw.filter(Number.isFinite).length < 2) return
      let j = 0
      grid.forEach((t, gi) => {
        while (j < ts.length - 2 && ts[j + 1] < t) j++
        let a = j, b = j + 1
        while (a > 0 && !Number.isFinite(raw[a])) a--
        while (b < ts.length - 1 && !Number.isFinite(raw[b])) b++
        let v = NaN
        // interpolate across short gaps only (≤ 4 grid steps)
        if (Number.isFinite(raw[a]) && Number.isFinite(raw[b]) && ts[b] - ts[a] <= 4 / fps + 1e-6) {
          v = ts[b] === ts[a] ? raw[a] : raw[a] + (raw[b] - raw[a]) * Math.max(0, Math.min(1, (t - ts[a]) / (ts[b] - ts[a])))
        } else if (Number.isFinite(raw[a]) && Math.abs(ts[a] - t) < 0.6 / fps) v = raw[a]
        else if (Number.isFinite(raw[b]) && Math.abs(ts[b] - t) < 0.6 / fps) v = raw[b]
        if (!Number.isFinite(v)) return
        const o = out[gi]
        o[key] = o[key] || []
        o[key][li] = o[key][li] || { x: NaN, y: NaN, z: NaN, visibility: 1 }
        o[key][li][k] = v
      })
    })
  })
  return out
}

// ── per-frame kinematics ─────────────────────────────────────────────────────

// Trunk frame: up (hips→shoulders), lateral (L→R shoulder), forward (toward nose).
export function trunkFrame(w) {
  const lsh = w[L.L_SH], rsh = w[L.R_SH], lhp = w[L.L_HIP], rhp = w[L.R_HIP]
  if (![lsh, rsh, lhp, rhp].every(isNum)) return null
  const msh = mid(lsh, rsh), mhp = mid(lhp, rhp)
  const up = unit(sub(msh, mhp)); if (!up) return null
  const lat0 = sub(rsh, lsh)
  const lat = unit(rejectFrom(lat0, up)); if (!lat) return null
  let fwd = unit(cross(lat, up)); if (!fwd) return null
  const nose = w[L.NOSE]
  if (isNum(nose) && dot(sub(nose, msh), fwd) < 0) fwd = scale(fwd, -1)
  return { up, lat, fwd, msh, mhp }
}

// Shoulder external rotation (degrees) of the throwing arm.
// 0° = forearm points forward, 90° = forearm vertical, ~180° = forearm laid back.
export function shoulderER(w, side, tf = trunkFrame(w)) {
  if (!tf) return NaN
  const S = w[side === 'left' ? L.L_SH : L.R_SH]
  const E = w[side === 'left' ? L.L_EL : L.R_EL]
  const W = w[side === 'left' ? L.L_WR : L.R_WR]
  if (![S, E, W].every(isNum)) return NaN
  const h = unit(sub(E, S)); if (!h) return NaN
  const f  = rejectFrom(sub(W, E), h)
  const e0 = unit(rejectFrom(tf.fwd, h))
  if (!e0 || mag(f) < 1e-4) return NaN
  const upP = rejectFrom(rejectFrom(tf.up, h), e0)
  const e1 = unit(upP); if (!e1) return NaN
  // Degenerate when humerus nearly parallels the trunk axes
  if (Math.abs(dot(h, tf.up)) > 0.9) return NaN
  let a = Math.atan2(dot(f, e1), dot(f, e0)) * DEG
  if (a < -90) a += 360
  return a
}

export function elbowFlexion(w, side) {
  const S = w[side === 'left' ? L.L_SH : L.R_SH]
  const E = w[side === 'left' ? L.L_EL : L.R_EL]
  const W = w[side === 'left' ? L.L_WR : L.R_WR]
  if (![S, E, W].every(isNum)) return NaN
  const a = unit(sub(S, E)), b = unit(sub(W, E)); if (!a || !b) return NaN
  return 180 - Math.acos(Math.max(-1, Math.min(1, dot(a, b)))) * DEG
}

// Knee flexion: 0° = straight
export function kneeFlexion(w, side) {
  const H = w[side === 'left' ? L.L_HIP : L.R_HIP], K = w[side === 'left' ? L.L_KN : L.R_KN], A = w[side === 'left' ? L.L_AN : L.R_AN]
  if (![H, K, A].every(isNum)) return NaN
  const a = unit(sub(H, K)), b = unit(sub(A, K)); if (!a || !b) return NaN
  return 180 - Math.acos(Math.max(-1, Math.min(1, dot(a, b)))) * DEG
}

// Shoulder abduction: humerus vs trunk-down. 90° = arm level with shoulders.
export function shoulderAbduction(w, side, tf = trunkFrame(w)) {
  if (!tf) return NaN
  const S = w[side === 'left' ? L.L_SH : L.R_SH], E = w[side === 'left' ? L.L_EL : L.R_EL]
  if (![S, E].every(isNum)) return NaN
  const h = unit(sub(E, S)); if (!h) return NaN
  return Math.acos(Math.max(-1, Math.min(1, dot(h, scale(tf.up, -1))))) * DEG
}

// Trunk tilt from vertical (camera y is down): forward (+) / toward glove side (+)
export function trunkTilt(tf, side) {
  if (!tf) return { forward: NaN, lateral: NaN }
  const vert = { x: 0, y: -1, z: 0 }
  const horiz = v => unit({ x: v.x, y: 0, z: v.z })
  const fH = horiz(tf.fwd), lH = horiz(tf.lat)
  if (!fH || !lH) return { forward: NaN, lateral: NaN }
  const forward = Math.atan2(dot(tf.up, fH), dot(tf.up, vert)) * DEG
  // lat points L→R; for a right-hander the glove side is left
  const lat = Math.atan2(dot(tf.up, lH), dot(tf.up, vert)) * DEG
  return { forward, lateral: side === 'right' ? -lat : lat }
}

// Yaw of a left→right segment about the camera-vertical axis (y), degrees.
// Single-camera depth (z) is compressed and jittery, while x/y are reliable.
// So depth is rebuilt from the segment's true length (Taylor method):
//   |dz| = sqrt(L² − dx² − dy²), sign taken from the model's depth estimate.
const yawOf = (lp, rp, L) => {
  if (!isNum(lp) || !isNum(rp)) return NaN
  const dx = rp.x - lp.x, dy = rp.y - lp.y, dzRaw = rp.z - lp.z
  let dz = dzRaw
  if (L > 0) dz = Math.sign(dzRaw || 1) * Math.sqrt(Math.max(0, L * L - dx * dx - dy * dy))
  return Math.atan2(dz, dx) * DEG
}
function segLength(frames, a, b) {
  const v = frames.map(({ world: w }) => isNum(w[a]) && isNum(w[b]) ? mag(sub(w[a], w[b])) : NaN)
    .filter(Number.isFinite).sort((x, y) => x - y)
  return v.length ? v[Math.floor(v.length * 0.9)] : NaN   // 90th pct ≈ true length
}

// ── anthropometric scale ─────────────────────────────────────────────────────
// Winter segment ratios: upper arm .186H, forearm .146H, thigh .245H, shank .246H
const SEG_RATIO = 0.186 + 0.146 + 0.245 + 0.246
function median(a) { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN }

export function bodyScale(frames, heightM) {
  const segs = frames.map(({ world: w }) => {
    const seg = (a, b) => isNum(w[a]) && isNum(w[b]) ? mag(sub(w[a], w[b])) : NaN
    const ua = (seg(L.L_SH, L.L_EL) + seg(L.R_SH, L.R_EL)) / 2
    const fa = (seg(L.L_EL, L.L_WR) + seg(L.R_EL, L.R_WR)) / 2
    const th = (seg(L.L_HIP, L.L_KN) + seg(L.R_HIP, L.R_KN)) / 2
    const sh = (seg(L.L_KN, L.L_AN) + seg(L.R_KN, L.R_AN)) / 2
    return ua + fa + th + sh
  })
  const measured = median(segs)
  if (!(measured > 0) || !(heightM > 0)) return 1
  return (SEG_RATIO * heightM) / measured
}

// ── full throw analysis ──────────────────────────────────────────────────────
/**
 * @param rawFrames [{t, world}] one per video frame in order
 * @param opts { fps, side: 'right'|'left', heightM, separationIdx?, releaseIdx? }
 */
export function analyzeThrow(rawFrames, opts) {
  const { fps, side = 'right', heightM = 1.854, aspect = 16 / 9 } = opts
  const lead = side === 'right' ? 'left' : 'right'
  rawFrames = resampleUniform(rawFrames, fps)
  const n = rawFrames.length
  if (n < 5) return { error: 'Not enough frames analyzed.' }
  const detected = rawFrames.filter(f => f.world).length
  if (detected < n * 0.5) return { error: 'Player not detected in most frames. Try zooming the clip or a clearer angle.' }

  // 10 Hz validated against a SpinLab report (240fps): MER, release time and
  // hand speed all within ~5%
  const fcPos = opts.fcPos ?? 10, fcRot = opts.fcRot ?? 10
  const { frames, t } = smoothSeries(rawFrames, fps, fcPos)
  const k = bodyScale(frames, heightM)

  const WR  = side === 'left' ? L.L_WR : L.R_WR
  const OWR = side === 'left' ? L.R_WR : L.L_WR

  const tfs    = frames.map(f => trunkFrame(f.world))
  // ER: rebuild the throwing arm's depth from its known segment lengths. At
  // layback the forearm points away from the camera, and single-camera depth
  // is compressed, so raw landmarks undercount ER.
  const SH = side === 'left' ? L.L_SH : L.R_SH, EL = side === 'left' ? L.L_EL : L.R_EL
  const uaL = segLength(frames, SH, EL), faL = segLength(frames, EL, WR)
  const rebuild = (a, b, Ls) => {
    if (!isNum(a) || !isNum(b) || !(Ls > 0)) return b
    const dx = b.x - a.x, dy = b.y - a.y, dzRaw = b.z - a.z
    const dz = Math.sign(dzRaw || 1) * Math.sqrt(Math.max(dzRaw * dzRaw, Ls * Ls - dx * dx - dy * dy))
    return { ...b, x: a.x + dx, y: a.y + dy, z: a.z + dz }
  }
  // Off by default: across 4 reference throws it helped one and hurt others
  const armFix = opts.armDepth ?? false
  const er     = frames.map((f, i) => {
    if (!armFix) return shoulderER(f.world, side, tfs[i])
    const w = f.world.slice()
    w[EL] = rebuild(w[SH], w[EL], uaL)
    w[WR] = rebuild(w[EL], w[WR], faL)
    return shoulderER(w, side, tfs[i])
  })
  const elbow  = frames.map(f => elbowFlexion(f.world, side))
  // Segment orientation is noisier than position (depth jitter), so it gets
  // an extra, lower-cutoff pass before differentiating.
  const rotFc = Math.min(fcRot, fps * 0.3)
  const useTaylor = opts.taylor ?? true
  const hipL = useTaylor ? segLength(frames, L.L_HIP, L.R_HIP) : 0
  const shL  = useTaylor ? segLength(frames, L.L_SH,  L.R_SH)  : 0
  const pelvisYaw = lowpass(unwrapDeg(fillGaps(frames.map(f => yawOf(f.world[L.L_HIP], f.world[L.R_HIP], hipL)))), fps, rotFc)
  const torsoYaw  = lowpass(unwrapDeg(fillGaps(frames.map(f => yawOf(f.world[L.L_SH],  f.world[L.R_SH],  shL)))), fps, rotFc)
  const pelvisVel = derivative(pelvisYaw, t)
  const torsoVel  = derivative(torsoYaw, t)
  const separation = pelvisYaw.map((p, i) => wrap180(torsoYaw[i] - p))

  // Throwing-wrist speed (m/s, scaled to player height)
  const wx = frames.map(f => f.world[WR]?.x), wy = frames.map(f => f.world[WR]?.y), wz = frames.map(f => f.world[WR]?.z)
  const vx = derivative(wx, t), vy = derivative(wy, t), vz = derivative(wz, t)
  const handSpeed = vx.map((_, i) => Math.hypot(vx[i], vy[i], vz[i]) * k)
  const wristGap = frames.map(f => {
    const a = f.world[WR], b = f.world[OWR]
    return isNum(a) && isNum(b) ? mag(sub(a, b)) * k : NaN
  })

  // ── Events ──
  const edge = Math.min(2, Math.floor(n / 10))
  const argmax = (arr, lo = 0, hi = arr.length - 1, f = v => v) => {
    let bi = -1, bv = -Infinity
    for (let i = Math.max(0, lo); i <= Math.min(arr.length - 1, hi); i++) {
      const v = f(arr[i]); if (Number.isFinite(v) && v > bv) { bv = v; bi = i }
    }
    return bi
  }

  // Events, anchored on the throwing hand's peak speed (the one robust spike):
  //   MER     = max shoulder ER from 150ms before to 50ms after that peak
  //   Release = max elbow extension within 80ms after MER
  const ms = x => Math.round(fps * x / 1000)
  const handPk = argmax(handSpeed, edge, n - 1 - edge)
  const merAuto = handPk >= 0 ? argmax(er, handPk - ms(150), handPk + ms(50)) : -1
  let releaseAuto = -1
  if (merAuto >= 0) releaseAuto = argmax(elbow, merAuto + 1, merAuto + ms(80), v => -v)
  if (releaseAuto < 0) releaseAuto = handPk
  const releaseIdx = Number.isInteger(opts.releaseIdx) ? opts.releaseIdx : releaseAuto

  // Hand separation: last frame (before release) where hands were together
  let separationAuto = -1
  if (releaseIdx > 0) {
    const pre = wristGap.slice(0, releaseIdx + 1).filter(Number.isFinite)
    const minGap = pre.length ? Math.min(...pre) : NaN
    const thresh = Math.max(0.15, (minGap || 0) + 0.08)
    let c = -1
    for (let i = releaseIdx; i >= 0; i--) {
      if (Number.isFinite(wristGap[i]) && wristGap[i] <= thresh) { c = i; break }
    }
    if (c >= 0) {
      // Walk back from the threshold crossing to where the gap left its resting level
      const lo = Math.max(0, c - Math.round(fps * 0.5))
      const rest = wristGap.slice(lo, c + 1).filter(Number.isFinite)
      const base = rest.length ? Math.min(...rest) : wristGap[c]
      let s = c
      while (s > lo && Number.isFinite(wristGap[s - 1]) && wristGap[s - 1] > base + 0.02 && wristGap[s - 1] < wristGap[s]) s--
      separationAuto = Math.min(releaseIdx, s)
    }
  }
  const separationIdx = Number.isInteger(opts.separationIdx) ? opts.separationIdx : (separationAuto >= 0 ? separationAuto : null)

  // Foot strike: measured in body coordinates (immune to camera pans/zooms).
  // The stride is the biggest burst of ankle-to-ankle separation speed before
  // MER; lift / strike are where that burst rises above / falls below 25% of
  // its peak. (The planted foot still pivots, so it never goes fully still.)
  const ankleGap = frames.map(({ world: w }) => isNum(w[L.L_AN]) && isNum(w[L.R_AN]) ? mag(sub(w[L.L_AN], w[L.R_AN])) * k : NaN)
  const footSpeed = derivative(lowpass(fillGaps(ankleGap), fps, Math.min(6, fps * 0.3)), t)   // m/s, + = widening
  let footStrikeAuto = -1, footLiftAuto = -1
  const mer0 = merAuto >= 0 ? merAuto : releaseIdx
  if (mer0 > 0 && footSpeed.some(Number.isFinite)) {
    const lo = Math.max(0, releaseIdx - ms(900))
    const pk = argmax(footSpeed, lo, mer0 - ms(30))
    if (pk >= 0 && footSpeed[pk] > 0.3) {
      const thr = footSpeed[pk] * 0.25
      let i = pk; while (i < mer0 && footSpeed[i] >= thr) i++
      footStrikeAuto = i
      let j = pk; while (j > lo && footSpeed[j] >= thr) j--
      footLiftAuto = j > lo ? j : -1
    }
  }
  const footStrikeIdx = Number.isInteger(opts.footStrikeIdx) ? opts.footStrikeIdx : (footStrikeAuto >= 0 ? footStrikeAuto : null)

  // Joint angles at events
  const leadKnee = frames.map(f => kneeFlexion(f.world, lead))
  const abd = frames.map((f, i) => shoulderAbduction(f.world, side, tfs[i]))
  const tilt = tfs.map(tf => trunkTilt(tf, side))
  const snap = i => (i == null || i < 0) ? null : {
    leadKnee: leadKnee[i], elbow: elbow[i], abduction: abd[i], er: er[i],
    trunkForward: tilt[i].forward, trunkLateral: tilt[i].lateral, separation: separation[i],
  }
  // Stride: ankle-to-ankle distance at foot strike, as % of height
  let strideLen = null
  if (footStrikeIdx != null) {
    const w = frames[footStrikeIdx].world, a = w[L.L_AN], b = w[L.R_AN]
    // ×1.12: ankle-to-ankle reads a consistent ~11% short of SpinLab's stride
    // across 4 reference throws (it still tracks rep-to-rep differences)
    if (isNum(a) && isNum(b)) strideLen = mag(sub(a, b)) * k / heightM * 100 * 1.12
  }

  // ── Metrics ──
  const winLo = separationIdx ?? Math.max(0, releaseIdx - Math.round(fps * 0.6))
  const winHi = releaseIdx
  // MER: auto unless release was moved by hand, then search the 150ms before it
  const merIdx = Number.isInteger(opts.releaseIdx)
    ? argmax(er, releaseIdx - ms(150), releaseIdx)
    : merAuto

  // Peak rotation speeds: magnitude in the window, signed with the throw direction
  // Trunk rotation peaks fall in the ~200ms before release
  const rotLo = Math.max(0, releaseIdx - ms(200)), rotHi = Math.min(n - 1, releaseIdx + ms(20))
  const pelvisPeakIdx = argmax(pelvisVel, rotLo, rotHi, Math.abs)
  const torsoPeakIdx  = argmax(torsoVel,  rotLo, rotHi, Math.abs)
  const handPeakIdx   = argmax(handSpeed, rotLo, Math.min(n - 1, winHi + 2))
  const sepPeakIdx    = argmax(separation, winLo, winHi, Math.abs)

  const at = (arr, i) => (i >= 0 && Number.isFinite(arr[i])) ? arr[i] : null
  const sequenceOk = pelvisPeakIdx >= 0 && torsoPeakIdx >= 0 ? t[pelvisPeakIdx] <= t[torsoPeakIdx] : null

  const notes = []
  if (fps < 120) notes.push(`Video is ${Math.round(fps)}fps — speeds will read low. Film at 120–240fps for accurate speed numbers.`)
  notes.push('Rotation and MER use estimated depth from a single camera — best for comparing reps, not exact lab values.')
  notes.push('Pelvis speed and hip-shoulder separation are experimental: hip points are the hardest to track, and pelvis speed reads well under lab values.')
  if (k > 1.6 || k < 0.6) notes.push('Body scale looks off — check the player height setting.')

  return {
    n, t, fps, side, scale: k,
    series: { footSpeed, leadKnee, er, elbow, pelvisYaw, torsoYaw, pelvisVel, torsoVel, separation, handSpeed, wristGap },
    events: {
      separation: separationIdx, separationAuto: separationAuto >= 0 ? separationAuto : null,
      release: releaseIdx >= 0 ? releaseIdx : null, releaseAuto: releaseAuto >= 0 ? releaseAuto : null,
      mer: merIdx >= 0 ? merIdx : null,
      footStrike: footStrikeIdx, footStrikeAuto: footStrikeAuto >= 0 ? footStrikeAuto : null,
      footLift: footLiftAuto >= 0 ? footLiftAuto : null,
      pelvisPeak: pelvisPeakIdx, torsoPeak: torsoPeakIdx, handPeak: handPeakIdx, sepPeak: sepPeakIdx,
    },
    metrics: {
      mer:          at(er, merIdx),
      releaseTime:  separationIdx != null && releaseIdx >= 0 ? t[releaseIdx] - t[separationIdx] : null,
      // ×0.93: wrist speed read ~7% high vs SpinLab arm speed on all 4 reference
      // throws (players' heights known); held-out error 7% → 4% with it
      handSpeedMph: at(handSpeed, handPeakIdx) != null ? handSpeed[handPeakIdx] * 2.23694 * 0.93 : null,
      pelvisPeak:   at(pelvisVel, pelvisPeakIdx) != null ? Math.abs(pelvisVel[pelvisPeakIdx]) : null,
      torsoPeak:    at(torsoVel, torsoPeakIdx) != null ? Math.abs(torsoVel[torsoPeakIdx]) : null,
      maxSeparation: at(separation, sepPeakIdx) != null ? Math.abs(separation[sepPeakIdx]) : null,
      sequenceOk,
      hipTorsoGapMs: pelvisPeakIdx >= 0 && torsoPeakIdx >= 0 ? (t[torsoPeakIdx] - t[pelvisPeakIdx]) * 1000 : null,
      strideLenPct: strideLen,
      strideTimeMs: footLiftAuto >= 0 && footStrikeIdx != null ? (t[footStrikeIdx] - t[footLiftAuto]) * 1000 : null,
      plantToReleaseMs: footStrikeIdx != null && releaseIdx >= 0 ? (t[releaseIdx] - t[footStrikeIdx]) * 1000 : null,
      merToReleaseMs: merIdx >= 0 && releaseIdx >= 0 ? (t[releaseIdx] - t[merIdx]) * 1000 : null,
      atFootStrike: snap(footStrikeIdx),
      atMer: snap(merIdx),
      atRelease: snap(releaseIdx),
    },
    notes,
  }
}

// Live, single-frame readouts (no smoothing) for the joint table
export function liveER(world, side) {
  if (!world) return null
  const w = world.map(p => (p && (p.visibility ?? 1) >= MIN_VIS) ? p : null)
  const v = shoulderER(w, side)
  return Number.isFinite(v) ? Math.round(v) : null
}
