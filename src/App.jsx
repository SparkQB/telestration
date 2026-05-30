import React, { useRef, useEffect, useState, useCallback } from 'react'
import sparkqbLogo from './assets/sparkqb-logo.svg'
import sparkqbMark from './assets/sparkqb-logomark.svg'
import LeadGate, { isLeadSubmitted } from './LeadGate.jsx'
import { loadPose, drawPoseOverlay, ANGLE_JOINTS, calcJointAngle } from './pose.js'
import './App.css'

// ── Face detection — face-api.js (Safari compatible, self-hosted) ────────────
let faceApiPromise = null

function getFaceApi() {
  if (!faceApiPromise) {
    faceApiPromise = import('face-api.js').then(async (faceapi) => {
      const base = window.location.origin + '/faceapi'
      console.log('[SparkQB] Loading face-api.js models from:', base)

      // Load detection + landmark models in parallel
      await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri(base),
        faceapi.nets.faceLandmark68Net.loadFromUri(base),
      ])

      console.log('[SparkQB] face-api.js models ready')
      return faceapi
    })
  }
  return faceApiPromise
}

async function detectFacesInFrame(videoEl) {
  const faceapi = await getFaceApi()
  try {
    // Detect all faces with landmarks
    const detections = await faceapi
      .detectAllFaces(videoEl, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.35 }))
      .withFaceLandmarks()

    return detections.map(d => ({
      topLeft:     [d.detection.box.x,                      d.detection.box.y],
      bottomRight: [d.detection.box.x + d.detection.box.width,
                    d.detection.box.y + d.detection.box.height],
      landmarks:   d.landmarks.positions.map(p => ({ x: p.x, y: p.y })),
    }))
  } catch(e) {
    console.warn('[SparkQB] Detection error:', e)
    return []
  }
}

// ── Icons ─────────────────────────────────────────────────────────────────────
const I = ({ d, size = 20, sw = 1.8, fill = 'none' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill}
    stroke="currentColor" strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
    {Array.isArray(d) ? d.map((p,i)=><path key={i} d={p}/>) : <path d={d}/>}
  </svg>
)

// ── Tools ─────────────────────────────────────────────────────────────────────
const TOOLS = [
  { id:'pen',    label:'Pen',   icon:'M12 20h9M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z' },
  { id:'arrow',  label:'Arrow', icon:'M5 12h14M12 5l7 7-7 7' },
  { id:'route',  label:'Route', icon:'M3 17c3-3 6-5 9-5s6 2 9-2' },
  { id:'circle', label:'Circle',icon:'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z' },
  { id:'rect',   label:'Box',   icon:'M3 3h18v18H3z' },
  { id:'text',   label:'Text',  icon:'M4 7V4h16v3M9 20h6M12 4v16' },
  { id:'blur',   label:'Blur',  icon:null },  // text button, no icon
  { id:'gonio',  label:'Angle', icon:'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zM12 12l-4-4M12 12l4-4M12 12v5' },
  { id:'select', label:'Move',  icon:'M5 3l14 9-7 1-4 7z' },
  { id:'eraser', label:'Erase', icon:'M20 20H7L3 16l10-10 7 7-1.5 1.5M6.5 17.5l10-10' },
]

const PALETTE = ['#B2FF00','#004FFF','#FF0400','#F8F8F8','#FFD600','#FF6B00','#00E5FF','#101214']
const SPEEDS  = [0.1, 0.25, 0.5, 1, 2]

const defaultStyle = () => ({ color: '#B2FF00', lw: 3, opacity: 1 })
const initToolStyles = () => Object.fromEntries(TOOLS.map(t => [t.id, defaultStyle()]))

let _uid = 1
const uid = () => `s${_uid++}`

// ── History ───────────────────────────────────────────────────────────────────
function useHistory(init) {
  const [stack, setStack] = useState([init])
  const [idx, setIdx]     = useState(0)
  const push = useCallback(s => {
    setStack(p => [...p.slice(0, idx + 1), s])
    setIdx(i => i + 1)
  }, [idx])
  const undo = useCallback(() => idx > 0 && setIdx(i => i - 1), [idx])
  const redo = useCallback(() => idx < stack.length - 1 && setIdx(i => i + 1), [idx, stack.length])
  return { shapes: stack[idx], push, undo, redo, canUndo: idx > 0, canRedo: idx < stack.length - 1 }
}

// ── Canvas helpers ────────────────────────────────────────────────────────────
function getPos(e, canvas) {
  const r  = canvas.getBoundingClientRect()
  const sx = canvas.width / r.width
  const sy = canvas.height / r.height
  const s  = e.touches ? e.touches[0] : e
  return { x: (s.clientX - r.left) * sx, y: (s.clientY - r.top) * sy }
}
// getPos already works correctly because getBoundingClientRect reflects
// the CSS transform, so coordinates map directly to canvas space

function drawArrow(ctx, x1, y1, x2, y2, color, lw, opacity) {
  const a  = Math.atan2(y2 - y1, x2 - x1)
  const hl = Math.max(14, lw * 5)
  ctx.save()
  ctx.globalAlpha = opacity ?? 1
  ctx.strokeStyle = color; ctx.lineWidth = lw
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke()
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.moveTo(x2, y2)
  ctx.lineTo(x2 - hl * Math.cos(a - Math.PI/6), y2 - hl * Math.sin(a - Math.PI/6))
  ctx.lineTo(x2 - hl * Math.cos(a + Math.PI/6), y2 - hl * Math.sin(a + Math.PI/6))
  ctx.closePath(); ctx.fill()
  ctx.restore()
}

function renderShape(ctx, s, selected = false) {
  ctx.save()
  ctx.globalAlpha = s.opacity ?? 1
  ctx.strokeStyle = s.color; ctx.fillStyle = s.color
  ctx.lineWidth   = s.lw ?? 2
  ctx.lineCap = 'round'; ctx.lineJoin = 'round'
  if (selected) { ctx.shadowColor = '#B2FF00'; ctx.shadowBlur = 14 }

  switch (s.type) {
    case 'pen':
      if (s.pts.length < 2) break
      ctx.beginPath(); ctx.moveTo(s.pts[0].x, s.pts[0].y)
      s.pts.slice(1).forEach(p => ctx.lineTo(p.x, p.y))
      ctx.stroke(); break

    case 'route':
      if (s.pts.length < 2) break
      ctx.setLineDash([9, 5])
      ctx.beginPath(); ctx.moveTo(s.pts[0].x, s.pts[0].y)
      s.pts.slice(1).forEach(p => ctx.lineTo(p.x, p.y))
      ctx.stroke(); ctx.setLineDash([])
      if (s.pts.length >= 2) {
        const n = s.pts.length
        drawArrow(ctx, s.pts[n-2].x, s.pts[n-2].y, s.pts[n-1].x, s.pts[n-1].y, s.color, s.lw, s.opacity)
      }
      break

    case 'arrow':
      ctx.restore()
      drawArrow(ctx, s.x1, s.y1, s.x2, s.y2, s.color, s.lw, s.opacity)
      if (selected) {
        ctx.save()
        ctx.setLineDash([4,4]); ctx.strokeStyle='#B2FF00'; ctx.lineWidth=1.5; ctx.shadowBlur=0
        const b = bounds(s); if(b) ctx.strokeRect(b.x-6,b.y-6,b.w+12,b.h+12)
        ctx.setLineDash([]); ctx.restore()
      }
      return

    case 'circle':
      ctx.beginPath(); ctx.arc(s.cx, s.cy, s.r, 0, Math.PI*2); ctx.stroke(); break

    case 'rect':
      ctx.strokeRect(s.x, s.y, s.w, s.h); break

    case 'text':
      ctx.font = `700 ${s.fs||28}px Anton, sans-serif`
      ctx.fillText(s.text, s.x, s.y); break

    case 'player-o':
      ctx.lineWidth = 3
      ctx.beginPath(); ctx.arc(s.x, s.y, s.r||20, 0, Math.PI*2); ctx.stroke()
      if (s.label) {
        ctx.font = `700 ${(s.r||20)*1.1}px Anton, sans-serif`
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
        ctx.fillText(s.label, s.x, s.y)
      }
      break

    case 'player-x': {
      const r = s.r||20; ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(s.x-r, s.y-r); ctx.lineTo(s.x+r, s.y+r)
      ctx.moveTo(s.x+r, s.y-r); ctx.lineTo(s.x-r, s.y+r)
      ctx.stroke(); break
    }

    case 'gonio': {
      // Need at least 2 points to draw anything
      if (!s.pts || s.pts.length < 2) break
      const CYAN = '#00E5FF'
      ctx.strokeStyle = CYAN; ctx.fillStyle = CYAN; ctx.lineWidth = 2
      ctx.globalAlpha = 1

      const [pA, pV, pB] = s.pts  // A = first arm, V = vertex, B = second arm

      // Draw lines from vertex to each point
      if (pV) {
        ctx.beginPath(); ctx.moveTo(pA.x, pA.y); ctx.lineTo(pV.x, pV.y); ctx.stroke()
        if (pB) {
          ctx.beginPath(); ctx.moveTo(pV.x, pV.y); ctx.lineTo(pB.x, pB.y); ctx.stroke()
        }
      }

      // Draw angle arc and label at vertex
      if (pV && pB) {
        const angleA = Math.atan2(pA.y - pV.y, pA.x - pV.x)
        const angleB = Math.atan2(pB.y - pV.y, pB.x - pV.x)
        const radius = Math.min(
          Math.hypot(pA.x-pV.x, pA.y-pV.y),
          Math.hypot(pB.x-pV.x, pB.y-pV.y)
        ) * 0.4

        ctx.beginPath()
        ctx.arc(pV.x, pV.y, radius, Math.min(angleA,angleB), Math.max(angleA,angleB))
        ctx.stroke()

        // Calculate angle
        const vA = { x: pA.x-pV.x, y: pA.y-pV.y }
        const vB = { x: pB.x-pV.x, y: pB.y-pV.y }
        const dot = vA.x*vB.x + vA.y*vB.y
        const magA = Math.hypot(vA.x, vA.y), magB = Math.hypot(vB.x, vB.y)
        const angle = Math.round(Math.acos(Math.max(-1,Math.min(1,dot/(magA*magB)))) * 180/Math.PI)

        // Label
        const midAngle = (angleA + angleB) / 2
        const lx = pV.x + Math.cos(midAngle) * (radius + 16)
        const ly = pV.y + Math.sin(midAngle) * (radius + 16)
        ctx.font = `bold 14px 'Roboto Mono', monospace`
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
        const tw = ctx.measureText(`${angle}°`).width
        ctx.fillStyle = 'rgba(0,0,0,0.65)'
        ctx.beginPath(); ctx.roundRect(lx-tw/2-4, ly-10, tw+8, 20, 4); ctx.fill()
        ctx.fillStyle = CYAN
        ctx.fillText(`${angle}°`, lx, ly)
      }

      // Draw draggable control points
      s.pts.forEach((p, i) => {
        const r2 = 6
        ctx.beginPath(); ctx.arc(p.x, p.y, r2, 0, Math.PI*2)
        ctx.fillStyle = i === 1 ? CYAN : 'rgba(0,229,255,0.4)'  // vertex brighter
        ctx.fill()
        ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = 1; ctx.stroke()
      })
      break
    }
  }

  if (selected) {
    const b = bounds(s)
    if (b) {
      ctx.setLineDash([4,4]); ctx.strokeStyle='#B2FF00'
      ctx.lineWidth=1.5; ctx.shadowBlur=0
      ctx.strokeRect(b.x-6, b.y-6, b.w+12, b.h+12)
      ctx.setLineDash([])
    }
  }
  ctx.restore()
}

// ── Pixelate blur region ──────────────────────────────────────────────────────
function renderPixelate(drCtx, bgCanvas, x, y, w, h, selected, tracking) {
  if (w < 4 || h < 4) return
  const blockSize = 8
  const tmp = document.createElement('canvas')
  tmp.width = w; tmp.height = h
  const tctx = tmp.getContext('2d')
  tctx.drawImage(bgCanvas, x, y, w, h, 0, 0, w, h)

  const small = document.createElement('canvas')
  small.width  = Math.max(1, Math.round(w / blockSize))
  small.height = Math.max(1, Math.round(h / blockSize))
  const sctx = small.getContext('2d')
  sctx.imageSmoothingEnabled = false
  sctx.drawImage(tmp, 0, 0, small.width, small.height)

  drCtx.save()
  drCtx.imageSmoothingEnabled = false
  drCtx.drawImage(small, x, y, w, h)
  drCtx.fillStyle = 'rgba(16,18,20,0.28)'
  drCtx.fillRect(x, y, w, h)

  // only show border when selected, nothing when tracking
  if (selected) {
    drCtx.setLineDash([4,4])
    drCtx.strokeStyle = '#B2FF00'
    drCtx.lineWidth = 1.5
    drCtx.shadowColor = '#B2FF00'
    drCtx.shadowBlur = 8
    drCtx.strokeRect(x, y, w, h)
    drCtx.setLineDash([])
  }

  drCtx.restore()
}

// ── Shape bounds + hit ────────────────────────────────────────────────────────
function bounds(s) {
  switch (s.type) {
    case 'player-o': case 'player-x': return { x:s.x-(s.r||20), y:s.y-(s.r||20), w:(s.r||20)*2, h:(s.r||20)*2 }
    case 'circle': return { x:s.cx-s.r, y:s.cy-s.r, w:s.r*2, h:s.r*2 }
    case 'rect': case 'blur': return { x:Math.min(s.x,s.x+s.w), y:Math.min(s.y,s.y+s.h), w:Math.abs(s.w), h:Math.abs(s.h) }
    case 'arrow': return { x:Math.min(s.x1,s.x2), y:Math.min(s.y1,s.y2), w:Math.abs(s.x2-s.x1), h:Math.abs(s.y2-s.y1) }
    case 'text':  return { x:s.x, y:s.y-(s.fs||28), w:120, h:(s.fs||28)+8 }
    case 'gonio': {
      if (!s.pts?.length) return null
      const xs = s.pts.map(p=>p.x), ys = s.pts.map(p=>p.y)
      return { x:Math.min(...xs), y:Math.min(...ys), w:Math.max(...xs)-Math.min(...xs), h:Math.max(...ys)-Math.min(...ys) }
    }
    default: return null
  }
}

function hitTest(s, x, y) {
  if (s.type === 'gonio') return (s.pts||[]).some(p => Math.hypot(p.x-x, p.y-y) < 18)
  const b = bounds(s)
  if (!b) return (s.pts||[]).some(p => Math.hypot(p.x-x,p.y-y) < 16)
  return x >= b.x-8 && x <= b.x+b.w+8 && y >= b.y-8 && y <= b.y+b.h+8
}

// Find which goniometer point was hit (for per-point dragging)
function gonioHitPoint(s, x, y) {
  if (s.type !== 'gonio') return -1
  return (s.pts||[]).findIndex(p => Math.hypot(p.x-x, p.y-y) < 18)
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const bgRef   = useRef(null)
  const drRef   = useRef(null)
  const ovRef   = useRef(null)
  const vidRef  = useRef(null)
  const fileRef = useRef(null)
  const txtRef  = useRef(null)
  const wrapRef = useRef(null)
  const rafRef  = useRef(null)

  // Face tracking refs
  const trackingRef    = useRef({})    // shapeId -> bool
  const lastDetectRef  = useRef(0)     // timestamp of last detection
  const DETECT_INTERVAL = 50           // ms between detections (~20fps)
  const apiAvailable   = useRef(null)  // null=unknown, true/false
  const lastKnownPos   = useRef({})    // shapeId -> {x,y,w,h} last detected position

  const hist = useHistory([])
  const { shapes, push, undo, redo, canUndo, canRedo } = hist

  const [tool,        setTool]        = useState('pen')
  const [toolStyles,  setToolStyles]  = useState(initToolStyles)
  const [popover,     setPopover]     = useState(null)
  const [videoMeta,   setVideoMeta]   = useState(null)
  const [playing,     setPlaying]     = useState(false)
  const [currentT,    setCurrentT]    = useState(0)
  const [duration,    setDuration]    = useState(0)
  const [speedIdx,    setSpeedIdx]    = useState(3)
  const [selId,       setSelId]       = useState(null)
  const [pendingTxt,  setPendingTxt]  = useState(null)
  const [txtVal,      setTxtVal]      = useState('')
  const [lblModal,    setLblModal]    = useState(null)
  const [lblVal,      setLblVal]      = useState('')
  const [plays,       setPlays]       = useState([])
  const [showPlays,   setShowPlays]   = useState(false)
  const [playName,    setPlayName]    = useState('')
  const [animating,   setAnimating]   = useState(false)
  const [tfStatus,    setTfStatus]    = useState('idle') // idle|loading|ready|error
  const [tracking,    setTracking]    = useState({})     // shapeId -> bool (reactive mirror of trackingRef)

  const [unlocked, setUnlocked] = useState(isLeadSubmitted())

  const drawing = useRef(false)
  const stroke  = useRef(null)
  const dragSt  = useRef(null)
  const csz     = useRef({ w: 1280, h: 720 })
  const shapesRef = useRef(shapes)

  // ── Pose state ───────────────────────────────────────────────────────────────
  const [poseEnabled,   setPoseEnabled]   = useState(false)
  const poseEnabledRef = useRef(false)
  const [poseStatus,    setPoseStatus]    = useState('idle') // idle|loading|ready|error
  const [showAnglePanel,setShowAnglePanel]= useState(false)
  const [enabledAngles, setEnabledAngles] = useState(
    Object.fromEntries(ANGLE_JOINTS.map(j => [j.name, true]))
  )
  const enabledAnglesRef = useRef(Object.fromEntries(ANGLE_JOINTS.map(j => [j.name, true])))
  const [anglesVisible,    setAnglesVisible]    = useState(true)
  const anglesVisibleRef   = useRef(true)
  const poseCanvasRef       = useRef(null)
  const poseLandmarks       = useRef(null)
  const poseWorldLandmarks  = useRef(null)
  const poseDetecting       = useRef(false)
  const POSE_INTERVAL  = 60  // ms between pose detections

  // ── Zoom / pan state ─────────────────────────────────────────────────────────
  const [zoom,    setZoom]    = useState(1)
  const [pan,     setPan]     = useState({ x: 0, y: 0 })
  const zoomRef   = useRef(1)
  const panRef    = useRef({ x: 0, y: 0 })
  const isPanning = useRef(false)
  const panStart  = useRef({ x: 0, y: 0, px: 0, py: 0 })
  const pinchRef  = useRef(null) // { dist, zoom, px, py }

  function resetView() {
    zoomRef.current = 1; panRef.current = { x: 0, y: 0 }
    setZoom(1); setPan({ x: 0, y: 0 })
  }

  // ── Goniometer state ─────────────────────────────────────────────────────────
  const gonioRef = useRef(null)  // in-progress goniometer { id, pts[] }
  const gonioDragPoint = useRef(-1) // which point is being dragged in select mode

  // ── Table drag state ─────────────────────────────────────────────────────────
  const [tablePos,     setTablePos]     = useState({ x: null, y: null })
  const tableDragRef = useRef(null)

  function clampPan(px, py, z) {
    const cw = csz.current.w, ch = csz.current.h
    const maxX = (cw * z - cw) / 2
    const maxY = (ch * z - ch) / 2
    return { x: Math.max(-maxX, Math.min(maxX, px)), y: Math.max(-maxY, Math.min(maxY, py)) }
  }

  useEffect(() => { shapesRef.current = shapes }, [shapes])

  const ts = toolStyles[tool]

  // ── Layout ────────────────────────────────────────────────────────────────────
  const layout = useCallback(() => {
    if (!wrapRef.current) return
    const { clientWidth: aw, clientHeight: ah } = wrapRef.current
    const ratio = videoMeta ? videoMeta.w / videoMeta.h : 16/9
    let w = aw, h = aw / ratio
    if (h > ah) { h = ah; w = ah * ratio }
    w = Math.floor(w); h = Math.floor(h)
    csz.current = { w, h }
    ;[bgRef, drRef, poseCanvasRef, ovRef].forEach(r => {
      if (!r.current) return
      r.current.width  = w; r.current.height = h
      r.current.style.width  = w + 'px'; r.current.style.height = h + 'px'
    })
    renderBg()
    renderShapes(shapes)
  }, [videoMeta, shapes])

  useEffect(() => {
    const ro = new ResizeObserver(layout)
    if (wrapRef.current) ro.observe(wrapRef.current)
    layout()
    return () => ro.disconnect()
  }, [layout])

  // ── Background ────────────────────────────────────────────────────────────────
  const renderBg = useCallback(() => {
    const c = bgRef.current; if (!c) return
    const ctx = c.getContext('2d')
    if (videoMeta && vidRef.current) {
      ctx.drawImage(vidRef.current, 0, 0, c.width, c.height)
    } else {
      ctx.fillStyle = '#0a0c0e'
      ctx.fillRect(0, 0, c.width, c.height)
      ctx.fillStyle = 'rgba(248,248,248,0.18)'
      ctx.font = "500 15px Inter, sans-serif"
      ctx.textAlign = 'center'
      ctx.fillText('Tap LOAD FILM to get started', c.width/2, c.height/2)
    }
  }, [videoMeta])

  // ── Shapes render ─────────────────────────────────────────────────────────────
  const renderShapes = useCallback((list, active = null, sel = null, trackMap = null) => {
    const c  = drRef.current; if (!c) return
    const bg = bgRef.current
    const ctx = c.getContext('2d')
    ctx.clearRect(0, 0, c.width, c.height)
    const tm = trackMap ?? trackingRef.current
    list.forEach(s => {
      if (s.type === 'blur') {
        const b = bounds(s)
        if (b) renderPixelate(ctx, bg, b.x, b.y, b.w, b.h, s.id === sel, !!tm[s.id])
      } else renderShape(ctx, s, s.id === sel)
    })
    if (active) {
      if (active.type === 'blur') {
        const b = bounds(active)
        if (b) renderPixelate(ctx, bg, b.x, b.y, b.w, b.h, false, false)
      } else renderShape(ctx, active, false)
    }
  }, [])

  useEffect(() => { renderShapes(shapes, null, selId) }, [shapes, selId, renderShapes])
  useEffect(() => { renderPoseOverlay() }, [enabledAngles])
  useEffect(() => { poseEnabledRef.current = poseEnabled }, [poseEnabled])
  useEffect(() => { enabledAnglesRef.current = enabledAngles; renderPoseOverlay() }, [enabledAngles])
  useEffect(() => { anglesVisibleRef.current = anglesVisible; renderPoseOverlay() }, [anglesVisible])

  // ── Video frame loop ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!videoMeta) return
    const loop = (ts) => {
      renderBg()
      if (vidRef.current) setCurrentT(vidRef.current.currentTime)

      // Always render shapes every frame — never block on detection
      renderShapes(shapesRef.current, null, null, trackingRef.current)

      // Fire face detection in background
      const hasTracked = Object.values(trackingRef.current).some(Boolean)
      if (hasTracked && ts - lastDetectRef.current > DETECT_INTERVAL) {
        lastDetectRef.current = ts
        detectAndUpdateBlurs() // intentionally NOT awaited
      }

      // Fire pose detection in background
      if (poseEnabledRef.current && !poseDetecting.current && ts - (loop._lastPose||0) > POSE_INTERVAL) {
        loop._lastPose = ts
        runPoseDetection()
      }

      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(rafRef.current)
  }, [videoMeta, renderBg])

  // ── Face detection ────────────────────────────────────────────────────────────
  async function detectAndUpdateBlurs() {
    if (!vidRef.current) return
    const cv = bgRef.current; if (!cv) return

    const scaleX = cv.width  / (vidRef.current.videoWidth  || cv.width)
    const scaleY = cv.height / (vidRef.current.videoHeight || cv.height)

    let preds = []
    try {
      preds = await detectFacesInFrame(vidRef.current)
      if (apiAvailable.current === null) {
        apiAvailable.current = true
        setTfStatus('ready')
      }
    } catch(e) {
      apiAvailable.current = false
      setTfStatus('error')
      return
    }

    const currentShapes = shapesRef.current
    let updated = false

    const newShapes = currentShapes.map(s => {
      if (s.type !== 'blur' || !trackingRef.current[s.id]) return s

      const cx = Math.min(s.x, s.x + s.w) + Math.abs(s.w) / 2
      const cy = Math.min(s.y, s.y + s.h) + Math.abs(s.h) / 2

      // Find closest detected face to current box center
      let closest = null, minDist = Infinity
      preds.forEach(p => {
        const [fx, fy]   = p.topLeft
        const [fx2, fy2] = p.bottomRight
        const fcx = (fx + fx2) / 2 * scaleX
        const fcy = (fy + fy2) / 2 * scaleY
        const dist = Math.hypot(fcx - cx, fcy - cy)
        if (dist < minDist) { minDist = dist; closest = p }
      })

      // No face detected — hide box (return unchanged, won't render differently)
      if (!closest) return s

      const [tx, ty]   = closest.topLeft
      const [tx2, ty2] = closest.bottomRight
      const pad = 0.1
      const fw  = (tx2 - tx) * scaleX
      const fh  = (ty2 - ty) * scaleY
      const nx  = tx * scaleX - fw * pad
      const ny  = ty * scaleY - fh * pad
      const nw  = fw * (1 + pad * 2)
      const nh  = fh * (1 + pad * 2)

      updated = true
      return { ...s, x: nx, y: ny, w: nw, h: nh }
    })

    if (updated) {
      shapesRef.current = newShapes
      renderShapes(newShapes, null, null, trackingRef.current)
    }
  }

  // ── Pose detection ───────────────────────────────────────────────────────────
  async function enablePose() {
    if (poseStatus === 'loading') return
    setPoseStatus('loading')
    try {
      await loadPose((results) => {
        poseLandmarks.current = results.poseLandmarks || null
        poseWorldLandmarks.current = results.poseWorldLandmarks || null
        poseDetecting.current = false
        renderPoseOverlay()
      })
      console.log('[SparkQB] Pose ready, enabled')
      setPoseStatus('ready')
      setPoseEnabled(true)
    } catch(e) {
      console.error('[SparkQB] Pose load error:', e)
      setPoseStatus('error')
    }
  }

  

  function renderPoseOverlay() {
    const c = poseCanvasRef.current; if (!c) return
    const ctx = c.getContext('2d')
    ctx.clearRect(0, 0, c.width, c.height)
    if (!poseLandmarks.current) return
    const anglesEnabled = anglesVisibleRef.current ? enabledAnglesRef.current : {}
    drawPoseOverlay(ctx, poseLandmarks.current, poseWorldLandmarks.current, anglesEnabled, c.width, c.height)
  }

  async function runPoseDetection() {
    const { getPoseInstance } = await import('./pose.js')
    const pose = getPoseInstance()
    if (!pose) return
    if (!vidRef.current) return
    poseDetecting.current = true
    try {
      await pose.send({ image: vidRef.current })
    } catch(e) {
      console.error('[SparkQB] pose.send error:', e)
      poseDetecting.current = false
    }
  }

  function togglePose() {
    if (!poseEnabled) {
      enablePose()
    } else {
      setPoseEnabled(false)
      poseLandmarks.current = null
      const c = poseCanvasRef.current
      if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height)
    }
  }

  // Toggle tracking on a blur shape
  async function toggleTracking(shapeId) {
    const already = trackingRef.current[shapeId]
    if (already) {
      trackingRef.current = { ...trackingRef.current, [shapeId]: false }
      delete lastKnownPos.current[shapeId]
      setTracking({ ...trackingRef.current })
      return
    }
    setTfStatus('loading')
    try {
      await getFaceApi()  // preload — shows spinner while loading
      setTfStatus('ready')
    } catch(e) {
      console.error('[SparkQB] Model load failed:', e)
      setTfStatus('error')
      alert('Could not load face tracking model. Make sure you have run download-model.js and committed the files to GitHub.')
      return
    }
    trackingRef.current = { ...trackingRef.current, [shapeId]: true }
    setTracking({ ...trackingRef.current })
    await detectAndUpdateBlurs()
  }

  // ── Pointer events ────────────────────────────────────────────────────────────
  function onDown(e) {
    e.preventDefault()
    setPopover(null)
    const pos = getPos(e, ovRef.current)

    if (tool === 'select') {
      const hit = [...shapes].reverse().find(s => hitTest(s, pos.x, pos.y))
      setSelId(hit?.id || null)
      if (hit) {
        // Check if clicking a goniometer control point
        const ptIdx = gonioHitPoint(hit, pos.x, pos.y)
        if (ptIdx >= 0) {
          gonioDragPoint.current = ptIdx
          dragSt.current = { id: hit.id, sx: pos.x, sy: pos.y, orig: JSON.parse(JSON.stringify(hit)), gonioPt: ptIdx }
        } else {
          gonioDragPoint.current = -1
          dragSt.current = { id: hit.id, sx: pos.x, sy: pos.y, orig: JSON.parse(JSON.stringify(hit)) }
        }
      }
      return
    }
    if (tool === 'eraser') {
      const hit = [...shapes].reverse().find(s => hitTest(s, pos.x, pos.y))
      if (hit) {
        // Clean up tracking
        delete trackingRef.current[hit.id]
        setTracking({ ...trackingRef.current })
        push(shapes.filter(s => s.id !== hit.id))
      }
      return
    }
    if (tool === 'text') { setPendingTxt(pos); setTxtVal(''); setTimeout(() => txtRef.current?.focus(), 60); return }
    if (tool === 'player-o') { setLblModal({ ...pos }); setLblVal(''); return }
    if (tool === 'player-x') { push([...shapes, { id: uid(), type: 'player-x', x: pos.x, y: pos.y, r: 20, ...ts }]); return }

    // Goniometer — tap to place points one at a time
    if (tool === 'gonio') {
      if (!gonioRef.current) {
        // First tap — create with first point
        gonioRef.current = { id: uid(), type: 'gonio', pts: [pos], color: '#00E5FF', lw: 2, opacity: 1 }
      } else {
        gonioRef.current.pts.push(pos)
        if (gonioRef.current.pts.length === 3) {
          // Third tap — commit
          push([...shapes, gonioRef.current])
          gonioRef.current = null
        }
      }
      // Show preview
      const oc = ovRef.current; if (oc) {
        const octx = oc.getContext('2d')
        octx.clearRect(0, 0, oc.width, oc.height)
        if (gonioRef.current) renderShape(octx, gonioRef.current, false)
      }
      return
    }

    drawing.current = true
    if (tool==='pen'||tool==='route') stroke.current = { id:uid(), type:tool, pts:[pos], ...ts }
    else if (tool==='arrow')  stroke.current = { id:uid(), type:'arrow',  x1:pos.x, y1:pos.y, x2:pos.x, y2:pos.y, ...ts }
    else if (tool==='circle') stroke.current = { id:uid(), type:'circle', cx:pos.x, cy:pos.y, r:0, ...ts }
    else if (tool==='rect')   stroke.current = { id:uid(), type:'rect',   x:pos.x, y:pos.y, w:0, h:0, ...ts }
    else if (tool==='blur')   stroke.current = { id:uid(), type:'blur',   x:pos.x, y:pos.y, w:0, h:0 }
  }

  function onMove(e) {
    e.preventDefault()
    const pos = getPos(e, ovRef.current)

    if (dragSt.current) {
      const dx = pos.x - dragSt.current.sx, dy = pos.y - dragSt.current.sy
      const o = dragSt.current.orig; let m
      if (o.type === 'gonio' && dragSt.current.gonioPt >= 0) {
        // Drag individual goniometer point
        const newPts = o.pts.map((p, i) =>
          i === dragSt.current.gonioPt ? { x: pos.x, y: pos.y } : p
        )
        m = { ...o, pts: newPts }
      } else if (o.type==='pen'||o.type==='route') m = { ...o, pts: o.pts.map(p=>({x:p.x+dx,y:p.y+dy})) }
      else if (o.type==='arrow')  m = { ...o, x1:o.x1+dx, y1:o.y1+dy, x2:o.x2+dx, y2:o.y2+dy }
      else if (o.type==='circle') m = { ...o, cx:o.cx+dx, cy:o.cy+dy }
      else m = { ...o, x:o.x+dx, y:o.y+dy }
      if (m) renderShapes(shapes.map(s=>s.id===m.id?m:s), null, m.id, trackingRef.current)
      return
    }

    // Goniometer in-progress preview — show last placed point to cursor
    if (tool === 'gonio' && gonioRef.current) {
      const oc = ovRef.current; if (!oc) return
      const octx = oc.getContext('2d')
      octx.clearRect(0, 0, oc.width, oc.height)
      const preview = { ...gonioRef.current, pts: [...gonioRef.current.pts, pos] }
      renderShape(octx, preview, false)
      return
    }

    if (!drawing.current || !stroke.current) return
    const s = stroke.current
    if (s.type==='pen'||s.type==='route') s.pts.push(pos)
    else if (s.type==='arrow')  { s.x2=pos.x; s.y2=pos.y }
    else if (s.type==='circle') s.r = Math.hypot(pos.x-s.cx, pos.y-s.cy)
    else if (s.type==='rect'||s.type==='blur') { s.w=pos.x-s.x; s.h=pos.y-s.y }

    // Draw live preview on overlay canvas so it's visible while drawing
    const oc = ovRef.current; if (!oc) return
    const octx = oc.getContext('2d')
    octx.clearRect(0, 0, oc.width, oc.height)
    renderShape(octx, s, false)
  }

  function onUp(e) {
    e.preventDefault()
    if (dragSt.current) {
      const oc = ovRef.current
      const r  = oc.getBoundingClientRect()
      const src = e.changedTouches ? e.changedTouches[0] : e
      const pos = { x:(src.clientX-r.left)*(oc.width/r.width), y:(src.clientY-r.top)*(oc.height/r.height) }
      const dx = pos.x-dragSt.current.sx, dy = pos.y-dragSt.current.sy
      const o = dragSt.current.orig; let m
      if (o.type === 'gonio' && dragSt.current.gonioPt >= 0) {
        const newPts = o.pts.map((p, i) =>
          i === dragSt.current.gonioPt ? { x: pos.x, y: pos.y } : p
        )
        m = { ...o, pts: newPts }
      } else if (o.type==='pen'||o.type==='route') m = { ...o, pts: o.pts.map(p=>({x:p.x+dx,y:p.y+dy})) }
      else if (o.type==='arrow')  m = { ...o, x1:o.x1+dx,y1:o.y1+dy,x2:o.x2+dx,y2:o.y2+dy }
      else if (o.type==='circle') m = { ...o, cx:o.cx+dx,cy:o.cy+dy }
      else m = { ...o, x:o.x+dx, y:o.y+dy }
      if (m) push(shapes.map(s=>s.id===m.id?m:s))
      dragSt.current = null; gonioDragPoint.current = -1; return
    }

    if (!drawing.current || !stroke.current) return
    drawing.current = false
    const s = stroke.current; stroke.current = null

    // Clear the live preview from overlay canvas
    const oc = ovRef.current
    if (oc) oc.getContext('2d').clearRect(0, 0, oc.width, oc.height)
    const valid = (s.type==='pen'||s.type==='route') ? s.pts.length>2
      : s.type==='arrow'  ? Math.hypot(s.x2-s.x1,s.y2-s.y1)>8
      : s.type==='circle' ? s.r>5
      : (s.type==='rect'||s.type==='blur') ? Math.abs(s.w)>10&&Math.abs(s.h)>10 : false
    if (valid) push([...shapes, s])
    else renderShapes(shapes, null, selId, trackingRef.current)
  }

  // ── Text / label ──────────────────────────────────────────────────────────────
  function commitTxt() {
    if (!pendingTxt || !txtVal.trim()) { setPendingTxt(null); return }
    push([...shapes, { id:uid(), type:'text', x:pendingTxt.x, y:pendingTxt.y, text:txtVal, fs:30, ...ts }])
    setPendingTxt(null); setTxtVal('')
  }

  function commitLabel() {
    if (!lblModal) { setLblModal(null); return }
    push([...shapes, { id:uid(), type:'player-o', x:lblModal.x, y:lblModal.y, r:20, label:lblVal, ...ts }])
    setLblModal(null); setLblVal('')
  }

  // ── Tool style ────────────────────────────────────────────────────────────────
  function updateStyle(key, val) {
    setToolStyles(prev => ({ ...prev, [tool]: { ...prev[tool], [key]: val } }))
  }

  // ── Video ─────────────────────────────────────────────────────────────────────
  function onFileLoad(e) {
    const file = e.target.files?.[0]; if (!file) return
    // Revoke previous URL to avoid memory leak
    if (vidRef.current.src) URL.revokeObjectURL(vidRef.current.src)
    const url = URL.createObjectURL(file)
    const v   = vidRef.current
    v.muted   = true
    v.src     = url
    v.load()  // required on Safari to trigger metadata load
    v.onloadedmetadata = () => {
      let fps = 30
      try {
        if (v.videoTracks?.[0]) fps = v.videoTracks[0].getSettings().frameRate || 30
      } catch(e) {}
      setVideoMeta({ w: v.videoWidth, h: v.videoHeight, fps })
      setDuration(v.duration)
      setCurrentT(0)
      resetView()
    }
    v.onerror = (err) => console.error('Video load error:', err)
  }

  function togglePlay() {
    const v = vidRef.current; if (!v) return
    v.playbackRate = SPEEDS[speedIdx]
    v.muted = true  // required for autoplay on Safari
    if (v.paused) {
      v.play().then(() => setPlaying(true)).catch(err => {
        console.warn('Play failed:', err)
        setPlaying(false)
      })
    } else {
      v.pause()
      setPlaying(false)
    }
  }

  function seek(val) {
    const v = vidRef.current; if (!v) return
    v.currentTime = parseFloat(val); setCurrentT(parseFloat(val))
  }

  function stepFrame(dir) {
    const v = vidRef.current; if (!v) return
    v.pause(); setPlaying(false)
    v.currentTime = Math.max(0, Math.min(duration, v.currentTime + dir / (videoMeta?.fps || 30)))
  }

  function cycleSpeed() {
    const next = (speedIdx + 1) % SPEEDS.length
    setSpeedIdx(next)
    if (vidRef.current) vidRef.current.playbackRate = SPEEDS[next]
  }

  function fmt(t) {
    const m  = Math.floor(t / 60)
    const s  = Math.floor(t % 60)
    const ms = Math.floor((t % 1) * 100)
    return `${m}:${String(s).padStart(2,'0')}.${String(ms).padStart(2,'0')}`
  }

  // ── Animate routes ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!animating) return
    const animated = shapes.filter(s => s.type==='route'||s.type==='arrow')
    if (!animated.length) { setAnimating(false); return }
    let t0 = null; const dur = 2400
    const tick = ts => {
      if (!t0) t0 = ts
      const p = Math.min((ts - t0) / dur, 1)
      const c = drRef.current; if (!c) return
      const ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height)
      shapes.forEach(s => {
        if (s.type === 'route') renderShape(ctx, { ...s, pts: s.pts.slice(0, Math.max(2, Math.floor(s.pts.length * p))) })
        else if (s.type === 'arrow') renderShape(ctx, { ...s, x2: s.x1+(s.x2-s.x1)*p, y2: s.y1+(s.y2-s.y1)*p })
        else renderShape(ctx, s)
      })
      if (p < 1) rafRef.current = requestAnimationFrame(tick)
      else { setAnimating(false); renderShapes(shapes, null, selId, trackingRef.current) }
    }
    rafRef.current = requestAnimationFrame(tick)
  }, [animating])

  // ── Export ────────────────────────────────────────────────────────────────────
  function exportImg() {
    const tmp = document.createElement('canvas')
    tmp.width = csz.current.w; tmp.height = csz.current.h
    const ctx = tmp.getContext('2d')
    ctx.drawImage(bgRef.current, 0, 0)
    ctx.drawImage(drRef.current, 0, 0)
    const a = document.createElement('a')
    a.download = 'sparkqb-film.png'; a.href = tmp.toDataURL('image/png'); a.click()
  }

  function savePlay() {
    const name = playName.trim() || `Play ${plays.length+1}`
    const tmp  = document.createElement('canvas')
    const th   = Math.round(200 * csz.current.h / csz.current.w)
    tmp.width = 200; tmp.height = th
    const ctx = tmp.getContext('2d')
    ctx.drawImage(bgRef.current, 0,0,200,th); ctx.drawImage(drRef.current, 0,0,200,th)
    setPlays(p => [...p, { id:uid(), name, shapes:[...shapes], thumb:tmp.toDataURL('image/jpeg',0.6) }])
    setPlayName('')
  }

  function handleToolClick(id) {
    if (tool === id) setPopover(popover === id ? null : id)
    else { setTool(id); setSelId(null); setPopover(null) }
  }

  const isPortrait = videoMeta ? videoMeta.h > videoMeta.w : false
  const selectedShape = shapes.find(s => s.id === selId)
  const selectedIsBlur = selectedShape?.type === 'blur'

  // Also show track UI when blur tool is active and hovering over a blur shape
  // Find the most recently placed blur shape to show controls on
  const activeBlurId = selectedIsBlur ? selId :
    (tool === 'blur' ? shapes.filter(s => s.type === 'blur').slice(-1)[0]?.id : null)
  const activeBlurShape = shapes.find(s => s.id === activeBlurId)

  return (
    <div className={`app ${isPortrait ? 'portrait' : 'landscape'}`} onClick={() => setPopover(null)}>

      {/* Top bar */}
      <header className="topbar">
        <img src={sparkqbLogo} alt="SparkQB" className="sl-logo sl-logo-full" />
        <img src={sparkqbMark} alt="SparkQB" className="sl-logo sl-logo-mark" />
        <div className="topbar-actions">
          <button className="tb-btn" disabled={!canUndo} onClick={undo} title="Undo">
            <I d="M9 14L4 9l5-5M4 9h11a5 5 0 010 10h-3" size={18}/>
          </button>
          <button className="tb-btn" disabled={!canRedo} onClick={redo} title="Redo">
            <I d="M15 14l5-5-5-5M19 9H8a5 5 0 000 10h3" size={18}/>
          </button>


          <button className="tb-btn" onClick={exportImg} title="Export PNG">
            <I d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" size={18}/>
          </button>
          <button className="tb-btn red" onClick={() => { push([]); setSelId(null); trackingRef.current = {}; setTracking({}) }} title="Clear all">
            <I d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" size={18}/>
          </button>
          <button className={`pose-btn ${poseEnabled ? 'active' : ''}`}
            onClick={togglePose} title="Toggle pose detection"
            disabled={poseStatus === 'loading'}>
            {poseStatus === 'loading'
              ? <><span className="spin">⟳</span> POSE…</>
              : <><I d="M12 2a5 5 0 015 5 5 5 0 01-5 5 5 5 0 01-5-5 5 5 0 015-5M5 20a7 7 0 0114 0" size={16}/> JOINTS</>
            }
          </button>

          <button className="load-btn" onClick={() => fileRef.current?.click()}>
            <I d="M15 10l4.553-2.069A1 1 0 0121 8.82v6.36a1 1 0 01-1.447.889L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" size={16}/>
            LOAD FILM
          </button>
        </div>
      </header>

      {/* Stage */}
      <div className="stage" ref={wrapRef}
        onWheel={e => {
          e.preventDefault()
          const delta = e.deltaY > 0 ? 0.9 : 1.1
          const newZoom = Math.max(1, Math.min(8, zoomRef.current * delta))
          const newPan = clampPan(panRef.current.x, panRef.current.y, newZoom)
          zoomRef.current = newZoom; panRef.current = newPan
          setZoom(newZoom); setPan(newPan)
        }}
      >
        <div className="canvas-transform" style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transformOrigin: 'center center',
        }}>
        <canvas ref={bgRef} className="layer layer-bg"/>
        <canvas ref={drRef} className="layer layer-dr"/>
        <canvas ref={poseCanvasRef} className="layer layer-pose"/>
        <canvas ref={ovRef} className="layer layer-ov"
          style={{ cursor: zoom > 1 && tool==='select' ? 'grab' : tool==='select'?'grab':tool==='eraser'?'cell':'crosshair' }}
          onMouseDown={e => {
            // Two-finger pan on desktop (middle mouse or space+drag)
            if (e.button === 1 || e.altKey) {
              e.preventDefault()
              isPanning.current = true
              panStart.current = { x: e.clientX, y: e.clientY, px: panRef.current.x, py: panRef.current.y }
              return
            }
            onDown(e)
          }}
          onMouseMove={e => {
            if (isPanning.current) {
              const dx = e.clientX - panStart.current.x
              const dy = e.clientY - panStart.current.y
              const newPan = clampPan(panStart.current.px + dx, panStart.current.py + dy, zoomRef.current)
              panRef.current = newPan; setPan(newPan)
              return
            }
            onMove(e)
          }}
          onMouseUp={e => { isPanning.current = false; onUp(e) }}
          onMouseLeave={e => { isPanning.current = false; onUp(e) }}
          onTouchStart={e => {
            if (e.touches.length === 2) {
              e.preventDefault()
              const dx = e.touches[0].clientX - e.touches[1].clientX
              const dy = e.touches[0].clientY - e.touches[1].clientY
              const dist = Math.hypot(dx, dy)
              const mx = (e.touches[0].clientX + e.touches[1].clientX) / 2
              const my = (e.touches[0].clientY + e.touches[1].clientY) / 2
              pinchRef.current = { dist, zoom: zoomRef.current, px: panRef.current.x, py: panRef.current.y, mx, my }
              return
            }
            onDown(e)
          }}
          onTouchMove={e => {
            if (e.touches.length === 2 && pinchRef.current) {
              e.preventDefault()
              const dx = e.touches[0].clientX - e.touches[1].clientX
              const dy = e.touches[0].clientY - e.touches[1].clientY
              const dist = Math.hypot(dx, dy)
              const scale = dist / pinchRef.current.dist
              const newZoom = Math.max(1, Math.min(8, pinchRef.current.zoom * scale))
              // Pan toward pinch center
              const mx = (e.touches[0].clientX + e.touches[1].clientX) / 2
              const my = (e.touches[0].clientY + e.touches[1].clientY) / 2
              const dpx = (mx - pinchRef.current.mx) 
              const dpy = (my - pinchRef.current.my)
              const newPan = clampPan(pinchRef.current.px + dpx, pinchRef.current.py + dpy, newZoom)
              zoomRef.current = newZoom; panRef.current = newPan
              setZoom(newZoom); setPan(newPan)
              return
            }
            onMove(e)
          }}
          onTouchEnd={e => {
            if (pinchRef.current && e.touches.length < 2) { pinchRef.current = null; return }
            onUp(e)
          }}
        />

        </div>{/* end canvas-transform */}

        {/* Zoom reset button — only shows when zoomed */}
        {zoom > 1.05 && (
          <button className="zoom-reset" onClick={resetView} title="Reset zoom">
            <I d="M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0zM11 8v3M11 11h3" size={16}/>
            {Math.round(zoom * 10) / 10}×
          </button>
        )}

        {/* Tool palette */}
        <div className={`palette ${isPortrait ? 'palette-right' : 'palette-bottom'}`}
          onClick={e => e.stopPropagation()}>
          {TOOLS.map(t => (
            <div key={t.id} className="pal-item">
              <button className={`pal-btn ${tool===t.id?'active':''} ${t.id==='blur'?'pal-btn-text':''}`}
                title={t.label} onClick={() => handleToolClick(t.id)}>
                {t.icon ? <I d={t.icon} size={20}/> : <span className="pal-text-label">{t.label}</span>}
              </button>
              {popover === t.id && tool === t.id && (
                <div className={`tool-popover ${isPortrait ? 'pop-left' : 'pop-top'}`}
                  onClick={e => e.stopPropagation()}>
                  <div className="pop-head">{t.label.toUpperCase()}</div>
                  {t.id !== 'blur' && t.id !== 'eraser' && t.id !== 'select' && (
                    <>
                      <div className="pop-label">COLOR</div>
                      <div className="pop-swatches">
                        {PALETTE.map(c => (
                          <button key={c} className={`pop-sw ${ts.color===c?'sel':''}`}
                            style={{background:c, border: c==='#101214'?'1px solid rgba(255,255,255,0.2)':'none'}}
                            onClick={() => updateStyle('color', c)}/>
                        ))}
                        <input type="color" value={ts.color} onChange={e => updateStyle('color', e.target.value)} title="Custom"/>
                      </div>
                      <div className="pop-label">THICKNESS</div>
                      <div className="pop-slider-wrap">
                        <input type="range" min={1} max={14} step={1} value={ts.lw}
                          onChange={e => updateStyle('lw', +e.target.value)} className="pop-slider"/>
                        <span className="pop-val">{ts.lw}px</span>
                      </div>
                      <div className="pop-label">OPACITY</div>
                      <div className="pop-slider-wrap">
                        <input type="range" min={0.1} max={1} step={0.05} value={ts.opacity}
                          onChange={e => updateStyle('opacity', +e.target.value)} className="pop-slider"/>
                        <span className="pop-val">{Math.round(ts.opacity*100)}%</span>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Track face overlay — sits directly on the blur box */}
        {activeBlurShape && (() => {
          const b = bounds(activeBlurShape)
          if (!b || !bgRef.current) return null
          const cw = bgRef.current.width
          const ch = bgRef.current.height
          // Center of the blur box in % of canvas
          const cx = ((b.x + b.w / 2) / cw * 100)
          const cy = ((b.y + b.h / 2) / ch * 100)
          return (
            <div className="blur-overlay"
              style={{ left: cx + '%', top: cy + '%' }}
              onClick={e => e.stopPropagation()}>
              <button
                className={`track-btn ${tracking[activeBlurId] ? 'tracking' : ''}`}
                onClick={() => toggleTracking(activeBlurId)}
                disabled={tfStatus === 'loading'}
              >
                {tfStatus === 'loading' ? (
                  <><span className="spin">⟳</span> LOADING…</>
                ) : tracking[selId] ? (
                  <><I d="M6 18L18 6M6 6l12 12" size={13}/> STOP</>
                ) : (
                  <><I d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zM12 8v4l3 3" size={13}/> TRACK FACE</>
                )}
              </button>
              <button className="del-btn-inline"
                onClick={() => { push(shapes.filter(s => s.id !== activeBlurId)); delete trackingRef.current[activeBlurId]; setTracking({...trackingRef.current}); setSelId(null) }}>
                <I d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" size={13}/>
              </button>
            </div>
          )
        })()}

        {/* Inline text input */}
        {pendingTxt && (() => {
          const c = bgRef.current
          return (
            <div className="txt-overlay"
              style={{ left:(pendingTxt.x/(c?.width||1)*100)+'%', top:(pendingTxt.y/(c?.height||1)*100)+'%' }}>
              <input ref={txtRef} value={txtVal} style={{ color: ts.color }}
                onChange={e => setTxtVal(e.target.value)}
                onKeyDown={e => { if(e.key==='Enter') commitTxt(); if(e.key==='Escape') setPendingTxt(null) }}
                placeholder="Type…" className="txt-input" autoFocus/>
              <button className="txt-ok" onClick={commitTxt}>OK</button>
            </div>
          )
        })()}

        {/* Generic delete for non-blur selected */}
        {selId && !selectedIsBlur && (
          <button className="del-btn"
            onClick={() => { push(shapes.filter(s => s.id !== selId)); setSelId(null) }}>
            <I d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" size={14}/> Delete
          </button>
        )}
      </div>

      {/* Video controls */}
      {videoMeta && (
        <div className="vbar">
          <button className="vb" onClick={() => stepFrame(-10)} title="-10 frames"><I d="M19 20L9 12l10-8v16zM5 4v16" size={15}/></button>
          <button className="vb" onClick={() => stepFrame(-1)}  title="-1 frame"><I d="M15 19l-7-7 7-7" size={15}/></button>
          <button className="vb play" onClick={togglePlay}>
            {playing ? <I d="M6 4h4v16H6zM14 4h4v16h-4" size={17}/> : <I d="M5 3l14 9-14 9V3z" size={17} fill="currentColor" sw={0}/>}
          </button>
          <button className="vb" onClick={() => stepFrame(1)}   title="+1 frame"><I d="M9 5l7 7-7 7" size={15}/></button>
          <button className="vb" onClick={() => stepFrame(10)}  title="+10 frames"><I d="M5 4l10 8-10 8V4zM19 4v16" size={15}/></button>
          <div className="scrub-wrap">
            <input type="range" className="scrub" min={0} max={duration} step={1/(videoMeta.fps||30)/2}
              value={currentT} onChange={e => seek(e.target.value)}/>
          </div>
          <button className="speed-btn" onClick={cycleSpeed}>{SPEEDS[speedIdx]}×</button>
          <span className="tc">{fmt(currentT)}</span>
          {videoMeta.fps && <span className="fps-badge">{Math.round(videoMeta.fps)}fps</span>}

        </div>
      )}

      <video ref={vidRef} style={{display:'none'}} playsInline loop muted onEnded={() => setPlaying(false)}/>
      <input ref={fileRef} type="file" accept="video/*" style={{display:'none'}} onChange={onFileLoad}/>

      {/* Floating joint angle table — draggable, inline toggles */}
      {poseEnabled && (() => {
        const defaultX = window.innerWidth - 220
        const defaultY = window.innerHeight - 320
        const tx = tablePos.x ?? defaultX
        const ty = tablePos.y ?? defaultY
        return (
          <div className="joint-table"
            style={{ left: tx, top: ty }}
            onMouseDown={e => {
              if (e.target.closest('.jt-toggle')) return
              e.stopPropagation()
              const startX = e.clientX - tx, startY = e.clientY - ty
              tableDragRef.current = { startX, startY }
              const onMv = ev => setTablePos({ x: ev.clientX - tableDragRef.current.startX, y: ev.clientY - tableDragRef.current.startY })
              const onUp = () => { tableDragRef.current = null; window.removeEventListener('mousemove', onMv); window.removeEventListener('mouseup', onUp) }
              window.addEventListener('mousemove', onMv); window.addEventListener('mouseup', onUp)
            }}
            onTouchStart={e => {
              if (e.target.closest('.jt-toggle')) return
              e.stopPropagation()
              const t0 = e.touches[0]
              const startX = t0.clientX - tx, startY = t0.clientY - ty
              tableDragRef.current = { startX, startY }
              const onMv = ev => { const t1=ev.touches[0]; setTablePos({ x: t1.clientX-tableDragRef.current.startX, y: t1.clientY-tableDragRef.current.startY }) }
              const onEnd = () => { tableDragRef.current = null; window.removeEventListener('touchmove', onMv); window.removeEventListener('touchend', onEnd) }
              window.addEventListener('touchmove', onMv); window.addEventListener('touchend', onEnd)
            }}
          >
            <div className="joint-table-head">
              <span>JOINT ANGLES</span>
              <button className={`jt-master-toggle ${anglesVisible ? 'on' : 'off'}`}
                onClick={() => setAnglesVisible(v => !v)}>
                <span className={`jt-dot ${anglesVisible ? 'on' : 'off'}`}/>
              </button>
            </div>
            <table className="joint-table-body">
              <tbody>
                {ANGLE_JOINTS.map(j => {
                  const val = poseLandmarks.current ? calcJointAngle(j, poseWorldLandmarks.current) : null
                  const on = enabledAngles[j.name]
                  return (
                    <tr key={j.name} className={on ? '' : 'jt-off'}>
                      <td className="jt-name">{j.name}</td>
                      <td className="jt-val">{on && anglesVisible && val !== null ? `${val}°` : '–'}</td>
                      <td className="jt-toggle" onClick={() => setEnabledAngles(prev => ({ ...prev, [j.name]: !prev[j.name] }))}>
                        <span className={`jt-dot ${on ? 'on' : 'off'}`}/>
                      </td>
                    </tr>
                  )
                })}

              </tbody>
            </table>
          </div>
        )
      })()}



      {/* Player label modal */}
      {lblModal && (
        <div className="modal-bg" onClick={() => setLblModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h3>PLAYER LABEL</h3>
            <input value={lblVal} onChange={e => setLblVal(e.target.value)}
              onKeyDown={e => e.key==='Enter' && commitLabel()}
              placeholder="QB, WR, 12…" className="modal-input" autoFocus/>
            <div className="modal-row">
              <button className="btn-sec" onClick={() => setLblModal(null)}>Cancel</button>
              <button className="btn-pri" onClick={commitLabel}>Place →</button>
            </div>
          </div>
        </div>
      )}

      {/* Playbook */}
      {showPlays && (
        <div className="modal-bg" onClick={() => setShowPlays(false)}>
          <div className="modal modal-wide" onClick={e => e.stopPropagation()}>
            <h3>PLAYBOOK</h3>
            <div className="save-row">
              <input value={playName} onChange={e => setPlayName(e.target.value)}
                placeholder="Name this play…" className="modal-input flex1"/>
              <button className="btn-pri" onClick={savePlay}>Save →</button>
            </div>
            <div className="plays-grid">
              {plays.length === 0 && <p className="empty">No plays saved yet.</p>}
              {plays.map(p => (
                <button key={p.id} className="play-card"
                  onClick={() => { push(p.shapes); setShowPlays(false) }}>
                  {p.thumb && <img src={p.thumb} alt={p.name}/>}
                  <span>{p.name}</span>
                </button>
              ))}
            </div>
            <button className="btn-sec full mt" onClick={() => setShowPlays(false)}>Close</button>
          </div>
        </div>
      )}
    </div>
  )
}
