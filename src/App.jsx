import React, { useRef, useEffect, useState, useCallback } from 'react'
import sparkqbLogo from './assets/sparkqb-logo.svg'
import sparkqbMark from './assets/sparkqb-logomark.svg'
import LeadGate, { isLeadSubmitted } from './LeadGate.jsx'
import { loadPose, drawPoseOverlay, ANGLE_JOINTS, calcJointAngle, detectFrame, startVideoRun, snapshotFrame } from './pose.js'
import { analyzeThrow, liveER } from './biomech.js'
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
  { id:'line',   label:'Line',  icon:'M5 19L19 5' },
  { id:'arrow',  label:'Arrow', icon:'M5 12h14M12 5l7 7-7 7' },
  { id:'circle', label:'Circle',icon:'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z' },
  { id:'rect',   label:'Box',   icon:'M3 3h18v18H3z' },
  { id:'text',   label:'Text',  icon:'M4 7V4h16v3M9 20h6M12 4v16' },
  { id:'blur',   label:'Blur',  icon:null },
  { id:'gonio',  label:'Angle', icon:'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zM12 12l-4-4M12 12l4-4M12 12v5' },
  { id:'select', label:'Move',  icon:'M5 3l14 9-7 1-4 7z' },
  { id:'eraser', label:'Erase', icon:'M20 20H7L3 16l10-10 7 7-1.5 1.5M6.5 17.5l10-10' },
]

const PALETTE = ['#B2FF00','#004FFF','#FF0400','#F8F8F8','#FFD600','#FF6B00','#00E5FF','#101214']
const SPEEDS  = [0.1, 0.25, 0.5, 1, 2]

const defaultStyle = () => ({ color: '#B2FF00', lw: 3, opacity: 1, dotted: false, textBg: true, textColor: '#101214' })
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

    case 'line': {
      ctx.restore()
      ctx.save()
      ctx.globalAlpha = s.opacity ?? 1
      ctx.strokeStyle = s.color; ctx.lineWidth = s.lw ?? 2
      ctx.lineCap = 'round'
      if (s.dotted) ctx.setLineDash([8, 6])
      ctx.beginPath(); ctx.moveTo(s.x1, s.y1); ctx.lineTo(s.x2, s.y2); ctx.stroke()
      ctx.setLineDash([])
      ctx.restore()
      return
    }

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

    case 'text': {
      const fontSize = s.fs || 20
      ctx.font = `700 ${fontSize}px 'Roboto Mono', monospace`
      const tw = ctx.measureText(s.text).width
      const pad = 6
      if (s.textBg !== false) {
        ctx.save()
        ctx.fillStyle = 'rgba(255,255,255,0.92)'
        ctx.fillRect(s.x - pad, s.y - fontSize, tw + pad*2, fontSize + 8)
        ctx.restore()
      }
      ctx.fillStyle = s.textColor || '#101214'
      ctx.textBaseline = 'alphabetic'
      ctx.fillText(s.text, s.x, s.y)
      break
    }

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
      if (!s.pts || s.pts.length < 2) break
      const CYAN = '#00E5FF'
      ctx.strokeStyle = CYAN; ctx.lineWidth = 2; ctx.globalAlpha = 1

      const [p0, p1, p2] = s.pts

      // Draw lines
      ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke()
      if (p2) {
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke()

        // Angle calc at p1 (vertex)
        const vA = { x: p0.x-p1.x, y: p0.y-p1.y }
        const vB = { x: p2.x-p1.x, y: p2.y-p1.y }
        const magA = Math.hypot(vA.x, vA.y)
        const magB = Math.hypot(vB.x, vB.y)
        if (magA > 0.01 && magB > 0.01) {
          const cosA = Math.max(-1, Math.min(1, (vA.x*vB.x + vA.y*vB.y) / (magA*magB)))
          const angle = Math.round(Math.acos(cosA) * 180 / Math.PI)
          if (!isNaN(angle)) {
            // Arc — always draw the shorter arc between the two arms
            const radius = Math.min(magA, magB) * 0.35
            const a1 = Math.atan2(vA.y, vA.x)
            const a2 = Math.atan2(vB.y, vB.x)
            // Calculate angular difference to pick correct arc direction
            let diff = a2 - a1
            while (diff > Math.PI) diff -= 2*Math.PI
            while (diff < -Math.PI) diff += 2*Math.PI
            const counterClockwise = diff < 0
            ctx.beginPath()
            ctx.arc(p1.x, p1.y, radius, a1, a2, counterClockwise)
            ctx.stroke()
            // Label at bisector
            const bx = vA.x/magA + vB.x/magB
            const by = vA.y/magA + vB.y/magB
            const bm = Math.hypot(bx, by) || 1
            const lx = p1.x + (bx/bm) * (radius + 18)
            const ly = p1.y + (by/bm) * (radius + 18)
            ctx.font = `600 13px 'Roboto Mono', monospace`
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
            const tw = ctx.measureText(`${angle}°`).width
            ctx.fillStyle = 'rgba(0,0,0,0.7)'; ctx.fillRect(lx-tw/2-4, ly-9, tw+8, 18)
            ctx.fillStyle = CYAN; ctx.fillText(`${angle}°`, lx, ly)
          }
        }
      }

      // Control points
      s.pts.forEach((p, i) => {
        ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI*2)
        ctx.fillStyle = i === 1 ? CYAN : 'rgba(0,229,255,0.5)'; ctx.fill()
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

// ── Circle pixelate blur ─────────────────────────────────────────────────────
function renderCircleBlur(drCtx, bgCanvas, cx, cy, r, selected) {
  if (r < 4) return
  const x = Math.round(cx - r), y = Math.round(cy - r)
  const w = Math.round(r * 2),  h = Math.round(r * 2)
  if (w < 4 || h < 4) return

  const blockSize = 12
  const tmp = document.createElement('canvas')
  tmp.width = w; tmp.height = h
  tmp.getContext('2d').drawImage(bgCanvas, x, y, w, h, 0, 0, w, h)

  const small = document.createElement('canvas')
  small.width  = Math.max(1, Math.round(w / blockSize))
  small.height = Math.max(1, Math.round(h / blockSize))
  const sctx = small.getContext('2d')
  sctx.imageSmoothingEnabled = false
  sctx.drawImage(tmp, 0, 0, small.width, small.height)

  drCtx.save()
  // Clip to circle
  drCtx.beginPath()
  drCtx.arc(cx, cy, r, 0, Math.PI * 2)
  drCtx.clip()

  drCtx.imageSmoothingEnabled = false
  drCtx.drawImage(small, x, y, w, h)
  drCtx.imageSmoothingEnabled = true

  drCtx.fillStyle = 'rgba(16,18,20,0.2)'
  drCtx.fill()

  if (selected) {
    drCtx.setLineDash([4,4])
    drCtx.strokeStyle = '#B2FF00'
    drCtx.lineWidth = 1.5
    drCtx.shadowColor = '#B2FF00'
    drCtx.shadowBlur = 8
    drCtx.beginPath()
    drCtx.arc(cx, cy, r, 0, Math.PI * 2)
    drCtx.stroke()
    drCtx.setLineDash([])
  }
  drCtx.restore()
}

// ── Shape bounds + hit ────────────────────────────────────────────────────────
function bounds(s) {
  switch (s.type) {
    case 'player-o': case 'player-x': return { x:s.x-(s.r||20), y:s.y-(s.r||20), w:(s.r||20)*2, h:(s.r||20)*2 }
    case 'circle': return { x:s.cx-s.r, y:s.cy-s.r, w:s.r*2, h:s.r*2 }
    case 'rect': return { x:Math.min(s.x,s.x+s.w), y:Math.min(s.y,s.y+s.h), w:Math.abs(s.w), h:Math.abs(s.h) }
    case 'blur': return { x: s.cx-s.r, y: s.cy-s.r, w: s.r*2, h: s.r*2 }
    case 'arrow': return { x:Math.min(s.x1,s.x2), y:Math.min(s.y1,s.y2), w:Math.abs(s.x2-s.x1), h:Math.abs(s.y2-s.y1) }
    case 'text':  return { x:s.x, y:s.y-(s.fs||28), w:120, h:(s.fs||28)+8 }
    case 'line': return { x:Math.min(s.x1,s.x2), y:Math.min(s.y1,s.y2), w:Math.abs(s.x2-s.x1), h:Math.abs(s.y2-s.y1) }
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
  const [selId,         setSelId]         = useState(null)
  const [editingTextId, setEditingTextId] = useState(null)
  const [editingText,   setEditingText]   = useState('')
  const [pendingTxt,    setPendingTxt]    = useState(null)
  const [txtVal,      setTxtVal]      = useState('')
  const [lblModal,    setLblModal]    = useState(null)
  const [lblVal,      setLblVal]      = useState('')


  const [tfStatus,    setTfStatus]    = useState('idle') // idle|loading|ready|error
  const [tracking,    setTracking]    = useState({})     // shapeId -> bool (reactive mirror of trackingRef)

  const [unlocked, setUnlocked] = useState(isLeadSubmitted())

  const drawing      = useRef(false)
  const scrubbing    = useRef(false)
  const scrubTrack   = useRef(null)
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
  const poseFrameT          = useRef(-1)     // video time of the frame the skeleton belongs to
  const poseWantT           = useRef(-1)     // video time we last requested
  const poseSnapRef         = useRef(null)   // canvas holding the frame pose ran on
  const [poseTick, setPoseTick] = useState(0) // re-render joint table on new results

  // ── Throw analysis state ─────────────────────────────────────────────────────
  const [showThrow,   setShowThrow]   = useState(false)
  const [throwSide,   setThrowSide]   = useState('right')
  const [heightIn,    setHeightIn]    = useState(73)      // player height, inches
  const [fpsOverride, setFpsOverride] = useState(null)    // file fps if auto-detect is wrong
  const [slowmo,      setSlowmo]      = useState(1)       // file plays N× slower than real time
  const [anaProgress, setAnaProgress] = useState(null)    // null | 0..1
  const [analysis,    setAnalysis]    = useState(null)    // { raw, vts, result, fps, overrides }
  const analysisRef = useRef(null)
  const anaCancel   = useRef(false)
  const anaRunning  = useRef(false)
  const [throwPos,  setThrowPos]  = useState({ x: null, y: null })

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
  const lastTapRef = useRef({ id: null, time: 0 }) // for double-tap detection on touch

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
        renderCircleBlur(ctx, bg, s.cx, s.cy, s.r, s.id === sel)
      } else renderShape(ctx, s, s.id === sel)
    })
    if (active) {
      if (active.type === 'blur') {
        renderCircleBlur(ctx, bg, active.cx, active.cy, active.r, false)
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
      const v = vidRef.current
      // When live pose is on, the background is painted from the exact frame the
      // skeleton was computed on (see runPoseDetection) so the two never drift.
      const livePose = poseEnabledRef.current && !cachedPoseAt(v?.currentTime)
      if (!livePose || !poseSnapRef.current?.width) renderBg()
      if (v && !scrubbing.current) setCurrentT(v.currentTime)

      // Always render shapes every frame — never block on detection
      renderShapes(shapesRef.current, null, null, trackingRef.current)

      // Fire face detection in background
      const hasTracked = Object.values(trackingRef.current).some(Boolean)
      if (hasTracked && ts - lastDetectRef.current > DETECT_INTERVAL) {
        lastDetectRef.current = ts
        detectAndUpdateBlurs() // intentionally NOT awaited
      }

      // Pose: show analyzed frame if cached, otherwise detect each new frame
      if (poseEnabledRef.current && v && !anaRunning.current) {
        if (v.currentTime !== poseWantT.current) {
          poseWantT.current = v.currentTime
          if (cachedPoseAt(v.currentTime)) showCachedPose(v.currentTime)
          else if (!poseDetecting.current) runPoseDetection()
        } else if (!poseDetecting.current && poseFrameT.current !== v.currentTime && !cachedPoseAt(v.currentTime)) {
          runPoseDetection()
        }
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

      // Find closest detected face to current circle center
      let closest = null, minDist = Infinity
      preds.forEach(p => {
        const [fx, fy]   = p.topLeft
        const [fx2, fy2] = p.bottomRight
        const fcx = (fx + fx2) / 2 * scaleX
        const fcy = (fy + fy2) / 2 * scaleY
        const dist = Math.hypot(fcx - s.cx, fcy - s.cy)
        if (dist < minDist) { minDist = dist; closest = p }
      })

      // No face detected — hold last known position
      if (!closest) return s

      const [tx, ty]   = closest.topLeft
      const [tx2, ty2] = closest.bottomRight
      const fw  = (tx2 - tx) * scaleX
      const fh  = (ty2 - ty) * scaleY
      const ncx = (tx + (tx2-tx)/2) * scaleX
      const ncy = (ty + (ty2-ty)/2) * scaleY
      const nr  = Math.max(fw, fh) / 2 * 1.25  // keep 25% padding

      updated = true
      return { ...s, cx: ncx, cy: ncy, r: nr }
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
      await loadPose()
      setPoseStatus('ready')
      setPoseEnabled(true)
      poseEnabledRef.current = true
      poseWantT.current = -1
    } catch(e) {
      console.error('[SparkQB] Pose load error:', e)
      setPoseStatus('error')
      alert('Could not load the pose model. Check your connection and try again.')
    }
  }

  // Visible part of the video (normalized) when zoomed — detect only there
  function cropRect() {
    const z = zoomRef.current
    if (z <= 1.05) return null
    const { w, h } = csz.current
    const cx = w / 2 - panRef.current.x / z, cy = h / 2 - panRef.current.y / z
    const vw = w / z, vh = h / z
    const x = Math.max(0, (cx - vw / 2) / w), y = Math.max(0, (cy - vh / 2) / h)
    return { x, y, w: Math.min(1 - x, vw / w), h: Math.min(1 - y, vh / h) }
  }

  function renderPoseOverlay() {
    const c = poseCanvasRef.current; if (!c) return
    const ctx = c.getContext('2d')
    ctx.clearRect(0, 0, c.width, c.height)
    if (!poseEnabledRef.current || !poseLandmarks.current) return
    const anglesEnabled = anglesVisibleRef.current ? enabledAnglesRef.current : {}
    drawPoseOverlay(ctx, poseLandmarks.current, poseWorldLandmarks.current, anglesEnabled, c.width, c.height)
  }

  function paintSnapshot() {
    const c = bgRef.current, snap = poseSnapRef.current
    if (!c || !snap?.width) return
    c.getContext('2d').drawImage(snap, 0, 0, c.width, c.height)
  }

  async function runPoseDetection() {
    const v = vidRef.current
    if (!v || !v.videoWidth || poseDetecting.current) return
    poseDetecting.current = true
    const t = v.currentTime
    try {
      poseSnapRef.current = poseSnapRef.current || document.createElement('canvas')
      const snap = snapshotFrame(v, poseSnapRef.current)
      if (!snap) return
      const r = await detectFrame(snap, cropRect())
      if (!poseEnabledRef.current) return
      poseLandmarks.current = r.landmarks
      poseWorldLandmarks.current = r.worldLandmarks
      poseFrameT.current = t
      paintSnapshot()
      renderShapes(shapesRef.current, null, null, trackingRef.current)
      renderPoseOverlay()
      setPoseTick(x => x + 1)
    } catch(e) {
      console.error('[SparkQB] pose detect error:', e)
    } finally {
      poseDetecting.current = false
    }
  }

  // ── Analyzed-frame cache ─────────────────────────────────────────────────────
  function cachedIndexAt(vt) {
    const a = analysisRef.current
    if (!a || vt == null) return -1
    const { vts, fileFps } = a
    if (vt < vts[0] - 1 / fileFps || vt > vts[vts.length - 1] + 1 / fileFps) return -1
    let lo = 0, hi = vts.length - 1
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (vts[m] <= vt) lo = m; else hi = m }
    return Math.abs(vts[hi] - vt) < Math.abs(vts[lo] - vt) ? hi : lo
  }
  function cachedPoseAt(vt) { return cachedIndexAt(vt) >= 0 }
  function showCachedPose(vt) {
    const i = cachedIndexAt(vt); if (i < 0) return
    const f = analysisRef.current.raw[i]
    poseLandmarks.current = f.lm2d
    poseWorldLandmarks.current = f.world
    poseFrameT.current = vt
    renderPoseOverlay()
    setPoseTick(x => x + 1)
  }

  function togglePose() {
    if (!poseEnabled) {
      enablePose()
    } else {
      setPoseEnabled(false)
      poseEnabledRef.current = false
      poseLandmarks.current = null
      const c = poseCanvasRef.current
      if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height)
      renderBg()
    }
  }

  // ── Throw analysis ───────────────────────────────────────────────────────────
  const fileFps = fpsOverride || videoMeta?.fps || 30
  const realFps = fileFps * slowmo

  // Seek and resolve with the media time of the frame actually presented
  // (variable-frame-rate clips can land on a neighbouring frame).
  function waitSeek(v, t) {
    return new Promise(resolve => {
      let done = false
      const finish = (mt) => { if (done) return; done = true; v.removeEventListener('seeked', onSeeked); resolve(Number.isFinite(mt) ? mt : v.currentTime) }
      const onSeeked = () => {
        if (v.requestVideoFrameCallback) { v.requestVideoFrameCallback((_, meta) => finish(meta?.mediaTime)); setTimeout(finish, 150) }
        else finish()
      }
      v.addEventListener('seeked', onSeeked)
      v.currentTime = t
      setTimeout(finish, 1500)
    })
  }

  function runMetrics(raw, overrides = {}) {
    return analyzeThrow(raw, {
      fps: realFps, side: throwSide, heightM: heightIn * 0.0254,
      separationIdx: overrides.separation, releaseIdx: overrides.release,
    })
  }

  async function analyzeThrowClip() {
    const v = vidRef.current; if (!v || !videoMeta || anaProgress != null) return
    v.pause(); setPlaying(false)
    const t0 = v.currentTime
    // Default window: 1.2s before → 0.4s after the current frame (real time)
    const before = 1.2 * slowmo, after = 0.4 * slowmo
    const start = Math.max(0, t0 - before), end = Math.min(duration, t0 + after)
    const fstart = Math.floor(start * fileFps), fend = Math.floor(end * fileFps)
    const nFrames = Math.max(1, fend - fstart)
    anaCancel.current = false
    anaRunning.current = true
    setAnaProgress(0)
    try {
      if (poseStatus !== 'ready') { await loadPose(); setPoseStatus('ready') }
      const detect = await startVideoRun()
      const crop = cropRect()
      const snap = document.createElement('canvas')
      const raw = [], vts = []
      for (let fi = fstart; fi < fend; fi++) {
        if (anaCancel.current) throw new Error('cancelled')
        const want = (fi + 0.5) / fileFps
        const mt = await waitSeek(v, want)
        if (vts.length && Math.abs(mt - vts[vts.length - 1]) < 1e-5) continue   // duplicate frame
        snapshotFrame(v, snap)
        const tReal = (mt - start) / slowmo
        const r = detect(snap, tReal, crop)
        raw.push({ t: tReal, world: r.worldLandmarks, lm2d: r.landmarks })
        vts.push(mt)
        if ((fi - fstart) % 4 === 0) setAnaProgress((fi - fstart) / nFrames)
      }
      const result = runMetrics(raw)
      const a = { raw, vts, fileFps, vtStart: start, result, overrides: {} }
      analysisRef.current = a
      setAnalysis(a)
      if (!poseEnabledRef.current) { setPoseEnabled(true); poseEnabledRef.current = true }
      const jump = result.events?.release != null ? idxToVt(a, result.events.release) : t0
      await waitSeek(v, jump)
      poseWantT.current = -1
    } catch (e) {
      if (e.message !== 'cancelled') { console.error('[SparkQB] analysis error:', e); alert('Analysis failed: ' + e.message) }
      await waitSeek(v, t0)
    } finally {
      anaRunning.current = false
      setAnaProgress(null)
    }
  }

  // Re-run metrics when settings or event overrides change
  function recompute(overrides) {
    const a = analysisRef.current; if (!a) return
    const ov = overrides ?? a.overrides
    const next = { ...a, overrides: ov, result: runMetrics(a.raw, ov) }
    analysisRef.current = next
    setAnalysis(next)
  }
  useEffect(() => { recompute() }, [throwSide, heightIn, slowmo, fpsOverride])

  // analysis result indices live on an even time grid (see resampleUniform)
  function idxToVt(a, i) { return a.vtStart + a.result.t[i] * slowmo }
  function vtToIdx(a, vt) {
    const t = a?.result?.t; if (!t?.length || vt == null) return -1
    const i = Math.round(((vt - a.vtStart) / slowmo - t[0]) * realFps)
    return i >= 0 && i < t.length ? i : -1
  }
  function setEventHere(name) {
    const a = analysisRef.current; if (!a) return
    const i = vtToIdx(a, vidRef.current?.currentTime)
    if (i < 0) { alert('Move the playhead inside the analyzed section first.'); return }
    recompute({ ...a.overrides, [name]: i })
  }
  function resetEvent(name) {
    const a = analysisRef.current; if (!a) return
    const ov = { ...a.overrides }; delete ov[name]; recompute(ov)
  }
  function jumpToIdx(i) {
    const a = analysisRef.current; if (!a || i == null || i < 0) return
    const v = vidRef.current; if (!v) return
    v.pause(); setPlaying(false)
    const vt = idxToVt(a, i)
    v.currentTime = vt; setCurrentT(vt)
  }
  function clearAnalysis() { analysisRef.current = null; setAnalysis(null); poseWantT.current = -1 }

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
    if (tool === 'text') {
      // Double-tap existing text to edit it
      const now = Date.now()
      const hitText = [...shapes].reverse().find(s => s.type === 'text' && hitTest(s, pos.x, pos.y))
      const isDouble = hitText && (
        e.detail === 2 ||
        (lastTapRef.current.id === hitText.id && now - lastTapRef.current.time < 400)
      )
      if (hitText) lastTapRef.current = { id: hitText.id, time: now }
      if (isDouble) {
        setEditingTextId(hitText.id); setEditingText(hitText.text); return
      }
      // Single tap on empty area — create new text
      if (!hitText) { setPendingTxt(pos); setTxtVal(''); setTimeout(() => txtRef.current?.focus(), 60) }
      return
    }
    if (tool === 'player-o') { setLblModal({ ...pos }); setLblVal(''); return }
    if (tool === 'player-x') { push([...shapes, { id: uid(), type: 'player-x', x: pos.x, y: pos.y, r: 20, ...ts }]); return }

    // Goniometer — tap to place points one at a time
    if (tool === 'gonio') {
      if (!gonioRef.current) {
        gonioRef.current = { id: uid(), type: 'gonio', pts: [pos], color: '#00E5FF', lw: 2, opacity: 1 }
      } else {
        gonioRef.current.pts.push(pos)
        if (gonioRef.current.pts.length === 3) {
          push([...shapes, gonioRef.current])
          gonioRef.current = null
        }
      }
      const oc = ovRef.current; if (oc) {
        const octx = oc.getContext('2d')
        octx.clearRect(0, 0, oc.width, oc.height)
        if (gonioRef.current) renderShape(octx, gonioRef.current, false)
      }
      return
    }

    drawing.current = true
    if (tool==='pen') stroke.current = { id:uid(), type:'pen', pts:[pos], ...ts }
    else if (tool==='line')   stroke.current = { id:uid(), type:'line',   x1:pos.x, y1:pos.y, x2:pos.x, y2:pos.y, ...ts }
    else if (tool==='arrow')  stroke.current = { id:uid(), type:'arrow',  x1:pos.x, y1:pos.y, x2:pos.x, y2:pos.y, ...ts }
    else if (tool==='circle') stroke.current = { id:uid(), type:'circle', cx:pos.x, cy:pos.y, r:0, ...ts }
    else if (tool==='rect')   stroke.current = { id:uid(), type:'rect',   x:pos.x, y:pos.y, w:0, h:0, ...ts }
    else if (tool==='blur')   stroke.current = { id:uid(), type:'blur',   cx:pos.x, cy:pos.y, r:0 }
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
    if (s.type==='pen') s.pts.push(pos)
    else if (s.type==='line')   { s.x2=pos.x; s.y2=pos.y }
    else if (s.type==='arrow')  { s.x2=pos.x; s.y2=pos.y }
    else if (s.type==='circle') s.r = Math.hypot(pos.x-s.cx, pos.y-s.cy)
    else if (s.type==='rect') { s.w=pos.x-s.x; s.h=pos.y-s.y }
    else if (s.type==='blur') { s.r = Math.hypot(pos.x-s.cx, pos.y-s.cy) }

    // Draw live preview on overlay canvas
    const oc = ovRef.current; if (!oc) return
    const octx = oc.getContext('2d')
    octx.clearRect(0, 0, oc.width, oc.height)
    if (s.type === 'blur') {
      octx.save()
      octx.strokeStyle = '#B2FF00'; octx.lineWidth = 2
      octx.setLineDash([6, 4])
      octx.beginPath(); octx.arc(s.cx, s.cy, Math.max(1, s.r), 0, Math.PI*2); octx.stroke()
      octx.setLineDash([])
      octx.restore()
    } else {
      renderShape(octx, s, false)
    }
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
      } else if (o.type==='pen') m = { ...o, pts: o.pts.map(p=>({x:p.x+dx,y:p.y+dy})) }
      else if (o.type==='line')   m = { ...o, x1:o.x1+dx,y1:o.y1+dy,x2:o.x2+dx,y2:o.y2+dy }
      else if (o.type==='arrow')  m = { ...o, x1:o.x1+dx,y1:o.y1+dy,x2:o.x2+dx,y2:o.y2+dy }
      else if (o.type==='circle') m = { ...o, cx:o.cx+dx,cy:o.cy+dy }
      else m = { ...o, x:o.x+dx, y:o.y+dy }
      if (m) push(shapes.map(s=>s.id===m.id?m:s))
      dragSt.current = null; gonioDragPoint.current = -1; return
    }

    // Goniometer — place point on finger lift
    if (tool === 'gonio') {
      const oc = ovRef.current
      const r  = oc.getBoundingClientRect()
      const src = e.changedTouches ? e.changedTouches[0] : e
      const pos = { x:(src.clientX-r.left)*(oc.width/r.width), y:(src.clientY-r.top)*(oc.height/r.height) }
      if (!gonioRef.current) {
        gonioRef.current = { id: uid(), type: 'gonio', pts: [pos], color: '#00E5FF', lw: 2, opacity: 1 }
      } else {
        gonioRef.current.pts.push(pos)
        if (gonioRef.current.pts.length === 3) {
          push([...shapes, gonioRef.current])
          gonioRef.current = null
          const octx = oc.getContext('2d')
          octx.clearRect(0, 0, oc.width, oc.height)
        }
      }
      if (gonioRef.current) {
        const octx = oc.getContext('2d')
        octx.clearRect(0, 0, oc.width, oc.height)
        renderShape(octx, gonioRef.current, false)
      }
      return
    }

    if (!drawing.current || !stroke.current) return
    drawing.current = false
    const s = stroke.current; stroke.current = null

    // Clear the live preview from overlay canvas
    const oc = ovRef.current
    if (oc) oc.getContext('2d').clearRect(0, 0, oc.width, oc.height)
    const valid = s.type==='pen' ? s.pts.length>2
      : s.type==='line'   ? Math.hypot(s.x2-s.x1,s.y2-s.y1)>8
      : s.type==='arrow'  ? Math.hypot(s.x2-s.x1,s.y2-s.y1)>8
      : s.type==='circle' ? s.r>5
      : s.type==='rect' ? Math.abs(s.w)>10&&Math.abs(s.h)>10
      : s.type==='blur' ? s.r > 10 : false
    if (valid) {
      const committed = s.type === 'blur'
        ? { ...s, r: s.r * 1.25 }
        : s
      push([...shapes, committed])
    }
    else renderShapes(shapes, null, selId, trackingRef.current)
  }

  // ── Text / label ──────────────────────────────────────────────────────────────
  function commitTxt() {
    if (!pendingTxt || !txtVal.trim()) { setPendingTxt(null); return }
    push([...shapes, { id:uid(), type:'text', x:pendingTxt.x, y:pendingTxt.y, text:txtVal, fs:22,
      textBg: ts.textBg !== false, textColor: ts.textColor || '#101214', ...ts }])
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
      let fps = 60
      try {
        if (v.videoTracks?.[0]) {
          const fr = v.videoTracks[0].getSettings().frameRate
          if (fr && fr > 0) fps = Math.round(fr)
        }
      } catch(e) {}
      setVideoMeta({ w: v.videoWidth, h: v.videoHeight, fps })
      analysisRef.current = null; setAnalysis(null); setFpsOverride(null)
      setDuration(v.duration)
      setCurrentT(0)
      resetView()
      // Show first frame immediately
      v.currentTime = 0.001
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

  function scrubFromPointer(e) {
    const track = scrubTrack.current; if (!track || !duration) return
    const r = track.getBoundingClientRect()
    const clientX = e.touches ? e.touches[0].clientX : (e.clientX ?? 0)
    const ratio = Math.max(0, Math.min(1, (clientX - r.left) / r.width))
    seek(ratio * duration)
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

  if (!unlocked) return <LeadGate onUnlock={() => setUnlocked(true)} />

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

          <button className={`pose-btn ${showThrow ? 'active' : ''}`}
            onClick={() => setShowThrow(v => !v)} title="QB throw analysis" disabled={!videoMeta}>
            <I d="M4 20L20 4M14 4h6v6" size={16}/> THROW
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
                  {(t.id === 'line' || t.id === 'arrow') && (
                  <div className="pop-dotted-row">
                    <span className="pop-label">STYLE</span>
                    <div className="pop-dotted-btns">
                      <button className={`pop-dot-btn ${!ts.dotted ? 'active' : ''}`}
                        onClick={() => updateStyle('dotted', false)}>──</button>
                      <button className={`pop-dot-btn ${ts.dotted ? 'active' : ''}`}
                        onClick={() => updateStyle('dotted', true)}>- -</button>
                    </div>
                  </div>
                )}
                {t.id === 'text' && (
                  <div className="pop-text-options">
                    <div className="pop-label">TEXT COLOR</div>
                    <div className="pop-swatches">
                      {PALETTE.map(c => (
                        <button key={c} className={`pop-sw ${ts.textColor===c?'sel':''}`}
                          style={{ background:c, border: c==='#101214'?'1px solid rgba(255,255,255,0.2)':'none' }}
                          onClick={() => updateStyle('textColor', c)}/>
                      ))}
                    </div>
                    <div className="pop-label">BACKGROUND</div>
                    <div className="pop-dotted-btns">
                      <button className={`pop-dot-btn ${ts.textBg!==false?'active':''}`}
                        onClick={() => updateStyle('textBg', true)}>ON</button>
                      <button className={`pop-dot-btn ${ts.textBg===false?'active':''}`}
                        onClick={() => updateStyle('textBg', false)}>OFF</button>
                    </div>
                  </div>
                )}
                {t.id !== 'blur' && t.id !== 'eraser' && t.id !== 'select' && t.id !== 'text' && (
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

        {/* Floating blur controls — bottom left when a blur is selected */}
        {activeBlurShape && (
          <div className="blur-controls" onClick={e => e.stopPropagation()}>
            <span className="blur-controls-label">BLUR</span>
            <button
              className={`track-btn ${tracking[activeBlurId] ? 'tracking' : ''}`}
              onClick={() => toggleTracking(activeBlurId)}
              disabled={tfStatus === 'loading'}
            >
              {tfStatus === 'loading' ? (
                <><span className="spin">⟳</span> LOADING…</>
              ) : tracking[activeBlurId] ? (
                <>● TRACKING</>
              ) : (
                <>TRACK FACE</>
              )}
            </button>
            <button className="blur-del-btn"
              onPointerDown={e => e.stopPropagation()}
              onClick={e => { e.stopPropagation(); push(shapes.filter(s => s.id !== activeBlurId)); delete trackingRef.current[activeBlurId]; setTracking({...trackingRef.current}); setSelId(null) }}>
              <I d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" size={13}/> DELETE
            </button>
          </div>
        )}

        {/* Inline text input */}
        {pendingTxt && (() => {
          const c = bgRef.current
          return (
            <div className="txt-overlay"
              style={{ left:(pendingTxt.x/(c?.width||1)*100)+'%', top:(pendingTxt.y/(c?.height||1)*100)+'%' }}>
              <input ref={txtRef} value={txtVal} style={{ color: '#101214' }}
                onChange={e => setTxtVal(e.target.value)}
                onKeyDown={e => { if(e.key==='Enter') commitTxt(); if(e.key==='Escape') setPendingTxt(null) }}
                placeholder="Type…" className="txt-input" autoFocus/>
              <button className="txt-ok" onClick={commitTxt}>OK</button>
            </div>
          )
        })()}

        {editingTextId && (() => {
          const s = shapes.find(sh => sh.id === editingTextId)
          if (!s) return null
          const c = bgRef.current
          const commitEdit = () => {
            if (editingText.trim()) push(shapes.map(sh => sh.id === editingTextId ? { ...sh, text: editingText } : sh))
            setEditingTextId(null); setEditingText('')
          }
          return (
            <div className="txt-overlay"
              style={{ left:(s.x/(c?.width||1)*100)+'%', top:(s.y/(c?.height||1)*100)+'%' }}>
              <input value={editingText} style={{ color: s.textColor || '#101214' }}
                onChange={e => setEditingText(e.target.value)}
                onKeyDown={e => { if(e.key==='Enter') commitEdit(); if(e.key==='Escape') { setEditingTextId(null); setEditingText('') } }}
                className="txt-input" autoFocus/>
              <button className="txt-ok" onClick={commitEdit}>OK</button>
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
          <div className="scrub-wrap" ref={scrubTrack}
            onPointerDown={e => {
              e.currentTarget.setPointerCapture(e.pointerId)
              scrubbing.current = true
              scrubFromPointer(e)
            }}
            onPointerMove={e => { if (scrubbing.current) scrubFromPointer(e) }}
            onPointerUp={e => { scrubbing.current = false; scrubFromPointer(e) }}
            onPointerCancel={() => { scrubbing.current = false }}
          >
            <div className="scrub-track">
              <div className="scrub-fill" style={{ width: `${duration ? (currentT/duration)*100 : 0}%` }}/>
              <div className="scrub-thumb" style={{ left: `${duration ? (currentT/duration)*100 : 0}%` }}/>
            </div>
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
                  const val = poseLandmarks.current ? calcJointAngle(j, poseWorldLandmarks.current) : null // poseTick re-renders
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
                {(() => {
                  const er = poseLandmarks.current ? liveER(poseWorldLandmarks.current, throwSide) : null
                  return (
                    <tr key="er" data-tick={poseTick}>
                      <td className="jt-name">{throwSide === 'right' ? 'R' : 'L'}.Shoulder ER</td>
                      <td className="jt-val">{anglesVisible && er !== null ? `${er}°` : '–'}</td>
                      <td/>
                    </tr>
                  )
                })()}
              </tbody>
            </table>
          </div>
        )
      })()}



      {/* QB throw analysis panel */}
      {showThrow && videoMeta && (() => {
        const tx = throwPos.x ?? 12
        const ty = throwPos.y ?? 70
        const res = analysis?.result
        const m = res?.metrics, ev = res?.events
        const fmtT = i => (i == null || !analysis?.result?.t) ? '–' : `${idxToVt(analysis, i).toFixed(3)}s`
        const startDrag = (cx, cy, isTouch) => {
          const sx = cx - tx, sy = cy - ty
          const mv = ev2 => { const p = isTouch ? ev2.touches[0] : ev2; setThrowPos({ x: p.clientX - sx, y: p.clientY - sy }) }
          const up = () => { window.removeEventListener(isTouch?'touchmove':'mousemove', mv); window.removeEventListener(isTouch?'touchend':'mouseup', up) }
          window.addEventListener(isTouch?'touchmove':'mousemove', mv); window.addEventListener(isTouch?'touchend':'mouseup', up)
        }
        const Metric = ({ label, value, unit, idx, hint }) => (
          <button className="tp-metric" onClick={() => jumpToIdx(idx)} title={hint || 'Jump to frame'} disabled={idx == null || idx < 0}>
            <span className="tp-mlabel">{label}</span>
            <span className="tp-mval">{value == null ? '–' : value}<small>{value == null ? '' : unit}</small></span>
          </button>
        )
        // velocity graph
        let graph = null
        if (res && !res.error) {
          const W = 260, H = 70, t = res.t, n = t.length
          const pv = res.series.pelvisVel, tv = res.series.torsoVel
          const maxV = Math.max(200, ...pv.map(Math.abs).filter(Number.isFinite), ...tv.map(Math.abs).filter(Number.isFinite))
          const X = i => (i / Math.max(1, n - 1)) * W
          const Y = v => H / 2 - (v / maxV) * (H / 2 - 4)
          const line = arr => arr.map((v, i) => Number.isFinite(v) ? `${X(i).toFixed(1)},${Y(v).toFixed(1)}` : null).filter(Boolean).join(' ')
          const ci = vtToIdx(analysis, currentT)
          const mark = (i, c) => i != null && i >= 0 ? <line x1={X(i)} x2={X(i)} y1={0} y2={H} stroke={c} strokeDasharray="3 3" strokeWidth="1"/> : null
          graph = (
            <div className="tp-graph">
              <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}
                onClick={e => { const r = e.currentTarget.getBoundingClientRect(); jumpToIdx(Math.round((e.clientX - r.left) / r.width * (n - 1))) }}>
                <line x1="0" x2={W} y1={H/2} y2={H/2} stroke="rgba(255,255,255,0.15)"/>
                {mark(ev.separation, '#FFD54F')}
                {mark(ev.mer, '#FF5252')}
                {mark(ev.release, '#69F0AE')}
                <polyline points={line(pv)} fill="none" stroke="#FF9100" strokeWidth="1.6"/>
                <polyline points={line(tv)} fill="none" stroke="#00E5FF" strokeWidth="1.6"/>
                {ci >= 0 && <line x1={X(ci)} x2={X(ci)} y1={0} y2={H} stroke="#fff" strokeWidth="1.2"/>}
              </svg>
              <div className="tp-legend">
                <span style={{color:'#FF9100'}}>● pelvis °/s</span>
                <span style={{color:'#00E5FF'}}>● torso °/s</span>
                <span style={{color:'#FFD54F'}}>┆sep</span>
                <span style={{color:'#FF5252'}}>┆MER</span>
                <span style={{color:'#69F0AE'}}>┆rel</span>
              </div>
            </div>
          )
        }
        return (
          <div className="throw-panel" style={{ left: tx, top: ty }} onClick={e => e.stopPropagation()}>
            <div className="joint-table-head"
              onMouseDown={e => startDrag(e.clientX, e.clientY, false)}
              onTouchStart={e => startDrag(e.touches[0].clientX, e.touches[0].clientY, true)}>
              <span>THROW ANALYSIS</span>
              <button className="tp-x" onClick={() => setShowThrow(false)}>×</button>
            </div>

            <div className="tp-row">
              <span className="tp-lbl">Arm</span>
              <div className="tp-seg">
                {['right','left'].map(sd => <button key={sd} className={throwSide===sd?'on':''} onClick={() => setThrowSide(sd)}>{sd==='right'?'RH':'LH'}</button>)}
              </div>
              <span className="tp-lbl">Ht</span>
              <select className="tp-sel" value={heightIn} onChange={e => setHeightIn(+e.target.value)}>
                {Array.from({length: 25}, (_, i) => 62 + i).map(h => <option key={h} value={h}>{Math.floor(h/12)}'{h%12}"</option>)}
              </select>
            </div>
            <div className="tp-row">
              <span className="tp-lbl">File fps</span>
              <select className="tp-sel" value={fileFps} onChange={e => setFpsOverride(+e.target.value)}>
                {[24,25,30,50,60,120,240].map(f => <option key={f} value={f}>{f}</option>)}
                {![24,25,30,50,60,120,240].includes(fileFps) && <option value={fileFps}>{fileFps}</option>}
              </select>
              <span className="tp-lbl">Slo-mo</span>
              <select className="tp-sel" value={slowmo} onChange={e => setSlowmo(+e.target.value)}>
                {[1,2,4,8].map(f => <option key={f} value={f}>{f === 1 ? 'off' : `${f}×`}</option>)}
              </select>
            </div>
            <div className="tp-hint">Real capture rate: {Math.round(realFps)}fps. If the clip is a slow-motion export that plays slowed down, set Slo-mo.</div>

            {anaProgress != null ? (
              <div className="tp-progress">
                <div className="tp-bar"><div style={{ width: `${Math.round(anaProgress*100)}%` }}/></div>
                <button className="tp-btn sec" onClick={() => { anaCancel.current = true }}>Cancel</button>
              </div>
            ) : (
              <div className="tp-row">
                <button className="tp-btn" onClick={analyzeThrowClip}>{analysis ? 'RE-ANALYZE' : 'ANALYZE THROW'}</button>
                {analysis && <button className="tp-btn sec" onClick={clearAnalysis}>Clear</button>}
              </div>
            )}
            {!analysis && anaProgress == null && <div className="tp-hint">Scrub to about the release frame, then tap Analyze. It reads 1.2s before → 0.4s after. Zoom in on the QB first if they're small in the frame.</div>}

            {res?.error && <div className="tp-warn">{res.error}</div>}
            {res && !res.error && <>
              <div className="tp-events">
                {[['separation','Hand separation','#FFD54F'],['release','Release','#69F0AE']].map(([k, label, c]) => (
                  <div key={k} className="tp-event">
                    <span className="tp-dot" style={{ background: c }}/>
                    <button className="tp-elabel" onClick={() => jumpToIdx(ev[k])}>{label} <b>{fmtT(ev[k])}</b>{analysis.overrides[k] != null ? ' (set)' : ' (auto)'}</button>
                    <button className="tp-mini" onClick={() => setEventHere(k)} title="Set to current frame">set here</button>
                    {analysis.overrides[k] != null && <button className="tp-mini" onClick={() => resetEvent(k)} title="Back to auto">auto</button>}
                  </div>
                ))}
              </div>
              <div className="tp-grid">
                <Metric label="MER (est.)" value={m.mer != null ? Math.round(m.mer) : null} unit="°" idx={ev.mer}/>
                <Metric label="Release time" value={m.releaseTime != null ? m.releaseTime.toFixed(2) : null} unit="s" idx={ev.separation} hint="Hand separation → release"/>
                <Metric label="Hand speed" value={m.handSpeedMph != null ? m.handSpeedMph.toFixed(1) : null} unit="mph" idx={ev.handPeak}/>
                <Metric label="Peak pelvis" value={m.pelvisPeak != null ? Math.round(m.pelvisPeak) : null} unit="°/s" idx={ev.pelvisPeak}/>
                <Metric label="Peak torso" value={m.torsoPeak != null ? Math.round(m.torsoPeak) : null} unit="°/s" idx={ev.torsoPeak}/>
                <Metric label="Hip-shoulder sep" value={m.maxSeparation != null ? Math.round(m.maxSeparation) : null} unit="°" idx={ev.sepPeak}/>
              </div>
              {m.sequenceOk != null && (
                <div className={`tp-seq ${m.sequenceOk ? 'ok' : 'bad'}`}>
                  {m.sequenceOk ? '✓ Sequence: pelvis peaks before torso' : '✗ Sequence: torso peaks before pelvis'}
                </div>
              )}
              {graph}
              {res.notes.map((nt, i) => <div key={i} className="tp-note">{nt}</div>)}
            </>}
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


    </div>
  )
}
