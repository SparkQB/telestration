import React, { useRef, useEffect, useState, useCallback } from 'react'
import { drawFootballField } from './field'
import sparkqbLogo from './assets/sparkqb-logo.svg'
import './App.css'

// ── tiny icon ─────────────────────────────────────────────────────────────────
const I = ({ d, size = 20, sw = 1.8, fill = 'none' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill}
    stroke="currentColor" strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
    {Array.isArray(d) ? d.map((p,i)=><path key={i} d={p}/>) : <path d={d}/>}
  </svg>
)

// ── tool definitions ──────────────────────────────────────────────────────────
const TOOLS = [
  { id:'pen',      label:'Pen',    icon:'M12 20h9M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z' },
  { id:'arrow',    label:'Arrow',  icon:'M5 12h14M12 5l7 7-7 7' },
  { id:'route',    label:'Route',  icon:'M3 17c3-3 6-5 9-5s6 2 9-2' },
  { id:'circle',   label:'Circle', icon:'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z' },
  { id:'rect',     label:'Box',    icon:'M3 3h18v18H3z' },
  { id:'text',     label:'Text',   icon:'M4 7V4h16v3M9 20h6M12 4v16' },
  { id:'player-o', label:'Off',    icon:'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z' },
  { id:'player-x', label:'Def',    icon:'M18 6L6 18M6 6l12 12' },
  { id:'select',   label:'Move',   icon:'M5 3l14 9-7 1-4 7z' },
  { id:'eraser',   label:'Erase',  icon:'M20 20H7L3 16l10-10 7 7-1.5 1.5M6.5 17.5l10-10' },
  { id:'blur',     label:'Blur',   icon:'M3 12a9 9 0 1018 0 9 9 0 00-18 0zM3 12h3M18 12h3M12 3v3M12 18v3' },
]

const PALETTE = ['#B2FF00','#004FFF','#FF0400','#F8F8F8','#FFD600','#FF6B00','#00E5FF','#101214']
const SPEEDS  = [0.1, 0.25, 0.5, 1, 2]

// per-tool default styles
const defaultStyle = () => ({ color: '#B2FF00', lw: 3, opacity: 1 })
const initToolStyles = () => Object.fromEntries(TOOLS.map(t => [t.id, defaultStyle()]))

let _uid = 1
const uid = () => `s${_uid++}`

// ── history hook ──────────────────────────────────────────────────────────────
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

// ── canvas helpers ────────────────────────────────────────────────────────────
function getPos(e, canvas) {
  const r  = canvas.getBoundingClientRect()
  const sx = canvas.width  / r.width
  const sy = canvas.height / r.height
  const s  = e.touches ? e.touches[0] : e
  return { x: (s.clientX - r.left) * sx, y: (s.clientY - r.top) * sy }
}

function drawArrow(ctx, x1, y1, x2, y2, color, lw, opacity) {
  const a = Math.atan2(y2 - y1, x2 - x1)
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

function bounds(s) {
  switch (s.type) {
    case 'player-o': case 'player-x': return { x:s.x-(s.r||20), y:s.y-(s.r||20), w:(s.r||20)*2, h:(s.r||20)*2 }
    case 'circle': return { x:s.cx-s.r, y:s.cy-s.r, w:s.r*2, h:s.r*2 }
    case 'rect':   return { x:Math.min(s.x,s.x+s.w), y:Math.min(s.y,s.y+s.h), w:Math.abs(s.w), h:Math.abs(s.h) }
    case 'blur':   return { x:Math.min(s.x,s.x+s.w), y:Math.min(s.y,s.y+s.h), w:Math.abs(s.w), h:Math.abs(s.h) }
    case 'arrow':  return { x:Math.min(s.x1,s.x2), y:Math.min(s.y1,s.y2), w:Math.abs(s.x2-s.x1), h:Math.abs(s.y2-s.y1) }
    case 'text':   return { x:s.x, y:s.y-(s.fs||28), w:120, h:(s.fs||28)+8 }
    default: return null
  }
}

// pixelate a region from bgCanvas onto drCanvas
function renderPixelate(drCtx, bgCanvas, s, selected) {
  const x = Math.min(s.x, s.x + s.w)
  const y = Math.min(s.y, s.y + s.h)
  const w = Math.abs(s.w)
  const h = Math.abs(s.h)
  if (w < 4 || h < 4) return

  const blockSize = 12

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
  drCtx.fillStyle = 'rgba(16,18,20,0.3)'
  drCtx.fillRect(x, y, w, h)

  if (selected) {
    drCtx.setLineDash([4,4])
    drCtx.strokeStyle = '#B2FF00'; drCtx.lineWidth = 1.5
    drCtx.shadowColor = '#B2FF00'; drCtx.shadowBlur = 10
    drCtx.strokeRect(x-4, y-4, w+8, h+8)
    drCtx.setLineDash([]); drCtx.shadowBlur = 0
  } else {
    drCtx.strokeStyle = 'rgba(178,255,0,0.35)'; drCtx.lineWidth = 1
    drCtx.strokeRect(x, y, w, h)
  }
  drCtx.restore()
}

function hitTest(s, x, y) {
  const b = bounds(s)
  if (!b) return (s.pts||[]).some(p => Math.hypot(p.x-x,p.y-y) < 16)
  return x >= b.x-8 && x <= b.x+b.w+8 && y >= b.y-8 && y <= b.y+b.h+8
}

// ── component ─────────────────────────────────────────────────────────────────
export default function App() {
  const bgRef  = useRef(null)
  const drRef  = useRef(null)
  const ovRef  = useRef(null)
  const vidRef = useRef(null)
  const fileRef= useRef(null)
  const txtRef = useRef(null)
  const wrapRef= useRef(null)
  const rafRef = useRef(null)

  const hist = useHistory([])
  const { shapes, push, undo, redo, canUndo, canRedo } = hist

  const [tool,       setTool]       = useState('pen')
  const [toolStyles, setToolStyles] = useState(initToolStyles)
  const [popover,    setPopover]    = useState(null)   // tool id or null
  const [mode,       setMode]       = useState('field') // 'field'|'video'
  const [videoMeta,  setVideoMeta]  = useState(null)   // {w,h,fps}
  const [playing,    setPlaying]    = useState(false)
  const [currentT,   setCurrentT]   = useState(0)
  const [duration,   setDuration]   = useState(0)
  const [speedIdx,   setSpeedIdx]   = useState(3)      // index into SPEEDS
  const [selId,      setSelId]      = useState(null)
  const [pendingTxt, setPendingTxt] = useState(null)
  const [txtVal,     setTxtVal]     = useState('')
  const [lblModal,   setLblModal]   = useState(null)
  const [lblVal,     setLblVal]     = useState('')
  const [plays,      setPlays]      = useState([])
  const [showPlays,  setShowPlays]  = useState(false)
  const [playName,   setPlayName]   = useState('')
  const [animating,  setAnimating]  = useState(false)

  const drawing = useRef(false)
  const stroke  = useRef(null)
  const dragSt  = useRef(null)
  const csz     = useRef({ w: 1280, h: 720 })

  // current tool style
  const ts = toolStyles[tool]

  // ── layout / resize ──────────────────────────────────────────────────────────
  const layout = useCallback(() => {
    if (!wrapRef.current) return
    const { clientWidth: aw, clientHeight: ah } = wrapRef.current
    let ratio = videoMeta ? videoMeta.w / videoMeta.h : (mode === 'field' ? 2.4 : 16/9)
    let w = aw, h = aw / ratio
    if (h > ah) { h = ah; w = ah * ratio }
    w = Math.floor(w); h = Math.floor(h)
    csz.current = { w, h }
    ;[bgRef, drRef, ovRef].forEach(r => {
      if (!r.current) return
      r.current.width  = w; r.current.height = h
      r.current.style.width  = w + 'px'
      r.current.style.height = h + 'px'
    })
    renderBg()
    renderShapes(shapes)
  }, [videoMeta, mode, shapes])

  useEffect(() => {
    const ro = new ResizeObserver(layout)
    if (wrapRef.current) ro.observe(wrapRef.current)
    layout()
    return () => ro.disconnect()
  }, [layout])

  // ── background render ─────────────────────────────────────────────────────────
  const renderBg = useCallback(() => {
    const c = bgRef.current; if (!c) return
    const ctx = c.getContext('2d')
    if (mode === 'field') {
      drawFootballField(ctx, c.width, c.height)
    } else if (videoMeta && vidRef.current) {
      ctx.drawImage(vidRef.current, 0, 0, c.width, c.height)
    } else {
      ctx.fillStyle = '#0a0c0e'
      ctx.fillRect(0, 0, c.width, c.height)
      ctx.fillStyle = 'rgba(248,248,248,0.2)'
      ctx.font = "500 16px Inter, sans-serif"
      ctx.textAlign = 'center'
      ctx.fillText('Tap FILM to load video', c.width/2, c.height/2)
    }
  }, [mode, videoMeta])

  // ── video frame loop ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (mode !== 'video' || !videoMeta) return
    const loop = () => {
      renderBg()
      if (vidRef.current) setCurrentT(vidRef.current.currentTime)
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(rafRef.current)
  }, [mode, videoMeta, renderBg])

  // ── shapes render ─────────────────────────────────────────────────────────────
  const renderShapes = useCallback((list, active = null, sel = null) => {
    const c = drRef.current; if (!c) return
    const bg = bgRef.current
    const ctx = c.getContext('2d')
    ctx.clearRect(0, 0, c.width, c.height)
    list.forEach(s => {
      if (s.type === 'blur') renderPixelate(ctx, bg, s, s.id === sel)
      else renderShape(ctx, s, s.id === sel)
    })
    if (active) {
      if (active.type === 'blur') renderPixelate(ctx, bg, active, false)
      else renderShape(ctx, active, false)
    }
  }, [])

  useEffect(() => { renderShapes(shapes, null, selId) }, [shapes, selId, renderShapes])

  // ── animate routes ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!animating) return
    const animated = shapes.filter(s => s.type==='route'||s.type==='arrow')
    if (!animated.length) { setAnimating(false); return }
    let t0 = null; const dur = 2400
    const tick = ts => {
      if (!t0) t0 = ts
      const p = Math.min((ts - t0) / dur, 1)
      const c = drRef.current; if (!c) return
      const ctx = c.getContext('2d')
      ctx.clearRect(0, 0, c.width, c.height)
      shapes.forEach(s => {
        if (s.type === 'route') {
          const vis = Math.max(2, Math.floor(s.pts.length * p))
          renderShape(ctx, { ...s, pts: s.pts.slice(0, vis) })
        } else if (s.type === 'arrow') {
          renderShape(ctx, { ...s, x2: s.x1+(s.x2-s.x1)*p, y2: s.y1+(s.y2-s.y1)*p })
        } else renderShape(ctx, s)
      })
      if (p < 1) rafRef.current = requestAnimationFrame(tick)
      else { setAnimating(false); renderShapes(shapes, null, selId) }
    }
    rafRef.current = requestAnimationFrame(tick)
  }, [animating])

  // ── pointer events ────────────────────────────────────────────────────────────
  function onDown(e) {
    e.preventDefault()
    setPopover(null)
    const pos = getPos(e, ovRef.current)

    if (tool === 'select') {
      const hit = [...shapes].reverse().find(s => hitTest(s, pos.x, pos.y))
      setSelId(hit?.id || null)
      if (hit) dragSt.current = { id: hit.id, sx: pos.x, sy: pos.y, orig: JSON.parse(JSON.stringify(hit)) }
      return
    }
    if (tool === 'eraser') {
      const hit = [...shapes].reverse().find(s => hitTest(s, pos.x, pos.y))
      if (hit) push(shapes.filter(s => s.id !== hit.id))
      return
    }
    if (tool === 'text') {
      setPendingTxt(pos); setTxtVal('')
      setTimeout(() => txtRef.current?.focus(), 60)
      return
    }
    if (tool === 'player-o') {
      setLblModal({ ...pos }); setLblVal('')
      return
    }
    if (tool === 'player-x') {
      push([...shapes, { id: uid(), type: 'player-x', x: pos.x, y: pos.y, r: 20, ...ts }])
      return
    }

    drawing.current = true
    if (tool==='pen'||tool==='route') stroke.current = { id:uid(), type:tool, pts:[pos], ...ts }
    else if (tool==='arrow')  stroke.current = { id:uid(), type:'arrow',  x1:pos.x,y1:pos.y,x2:pos.x,y2:pos.y, ...ts }
    else if (tool==='circle') stroke.current = { id:uid(), type:'circle', cx:pos.x,cy:pos.y,r:0, ...ts }
    else if (tool==='rect')   stroke.current = { id:uid(), type:'rect',   x:pos.x,y:pos.y,w:0,h:0, ...ts }
    else if (tool==='blur')   stroke.current = { id:uid(), type:'blur',   x:pos.x,y:pos.y,w:0,h:0 }
  }

  function onMove(e) {
    e.preventDefault()
    const pos = getPos(e, ovRef.current)

    if (dragSt.current) {
      const dx = pos.x - dragSt.current.sx, dy = pos.y - dragSt.current.sy
      const o = dragSt.current.orig; let m
      if (o.type==='pen'||o.type==='route') m = { ...o, pts: o.pts.map(p=>({x:p.x+dx,y:p.y+dy})) }
      else if (o.type==='arrow')  m = { ...o, x1:o.x1+dx,y1:o.y1+dy,x2:o.x2+dx,y2:o.y2+dy }
      else if (o.type==='circle') m = { ...o, cx:o.cx+dx,cy:o.cy+dy }
      else m = { ...o, x:o.x+dx, y:o.y+dy }
      if (m) renderShapes(shapes.map(s=>s.id===m.id?m:s), null, m.id)
      return
    }

    if (!drawing.current || !stroke.current) return
    const s = stroke.current
    if (s.type==='pen'||s.type==='route') s.pts.push(pos)
    else if (s.type==='arrow')  { s.x2=pos.x; s.y2=pos.y }
    else if (s.type==='circle') s.r = Math.hypot(pos.x-s.cx, pos.y-s.cy)
    else if (s.type==='rect')   { s.w=pos.x-s.x; s.h=pos.y-s.y }
    else if (s.type==='blur')   { s.w=pos.x-s.x; s.h=pos.y-s.y }
    renderShapes(shapes, s, selId)
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
      if (o.type==='pen'||o.type==='route') m = { ...o, pts: o.pts.map(p=>({x:p.x+dx,y:p.y+dy})) }
      else if (o.type==='arrow')  m = { ...o, x1:o.x1+dx,y1:o.y1+dy,x2:o.x2+dx,y2:o.y2+dy }
      else if (o.type==='circle') m = { ...o, cx:o.cx+dx,cy:o.cy+dy }
      else m = { ...o, x:o.x+dx, y:o.y+dy }
      if (m) push(shapes.map(s=>s.id===m.id?m:s))
      dragSt.current = null; return
    }

    if (!drawing.current || !stroke.current) return
    drawing.current = false
    const s = stroke.current; stroke.current = null
    const valid = (s.type==='pen'||s.type==='route') ? s.pts.length>2
      : s.type==='arrow'  ? Math.hypot(s.x2-s.x1,s.y2-s.y1)>8
      : s.type==='circle' ? s.r>5
      : s.type==='rect'   ? Math.abs(s.w)>5&&Math.abs(s.h)>5
      : s.type==='blur'   ? Math.abs(s.w)>10&&Math.abs(s.h)>10 : false
    if (valid) push([...shapes, s])
    else renderShapes(shapes, null, selId)
  }

  // ── text / player label ───────────────────────────────────────────────────────
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

  // ── tool style update ─────────────────────────────────────────────────────────
  function updateStyle(key, val) {
    setToolStyles(prev => ({ ...prev, [tool]: { ...prev[tool], [key]: val } }))
  }

  // ── video ─────────────────────────────────────────────────────────────────────
  function onFileLoad(e) {
    const file = e.target.files?.[0]; if (!file) return
    const url  = URL.createObjectURL(file)
    const v    = vidRef.current
    v.src = url
    v.onloadedmetadata = () => {
      // Try to detect frame rate
      let fps = 30
      if (v.videoTracks && v.videoTracks[0]) {
        fps = v.videoTracks[0].getSettings().frameRate || 30
      }
      setVideoMeta({ w: v.videoWidth, h: v.videoHeight, fps })
      setDuration(v.duration)
      setMode('video')
    }
  }

  function togglePlay() {
    const v = vidRef.current; if (!v) return
    v.playbackRate = SPEEDS[speedIdx]
    if (v.paused) { v.play(); setPlaying(true) }
    else { v.pause(); setPlaying(false) }
  }

  function seek(val) {
    const v = vidRef.current; if (!v) return
    v.currentTime = parseFloat(val)
    setCurrentT(parseFloat(val))
  }

  function stepFrame(dir) {
    const v = vidRef.current; if (!v) return
    v.pause(); setPlaying(false)
    const fps = videoMeta?.fps || 30
    v.currentTime = Math.max(0, Math.min(duration, v.currentTime + dir / fps))
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

  // ── export ────────────────────────────────────────────────────────────────────
  function exportImg() {
    const tmp = document.createElement('canvas')
    tmp.width = csz.current.w; tmp.height = csz.current.h
    const ctx = tmp.getContext('2d')
    ctx.drawImage(bgRef.current, 0, 0)
    ctx.drawImage(drRef.current, 0, 0)
    const a = document.createElement('a')
    a.download = 'spinlab-film.png'
    a.href = tmp.toDataURL('image/png')
    a.click()
  }

  function savePlay() {
    const name = playName.trim() || `Play ${plays.length+1}`
    const tmp  = document.createElement('canvas')
    const th   = Math.round(200 * csz.current.h / csz.current.w)
    tmp.width = 200; tmp.height = th
    const ctx = tmp.getContext('2d')
    ctx.drawImage(bgRef.current, 0,0,200,th)
    ctx.drawImage(drRef.current, 0,0,200,th)
    setPlays(p => [...p, { id:uid(), name, shapes:[...shapes], thumb:tmp.toDataURL('image/jpeg',0.6) }])
    setPlayName('')
  }

  // ── tool palette click: select or open popover ────────────────────────────────
  function handleToolClick(id) {
    if (tool === id) {
      setPopover(popover === id ? null : id)
    } else {
      setTool(id)
      setSelId(null)
      setPopover(null)
    }
  }

  const isPortrait = videoMeta ? videoMeta.h > videoMeta.w : false

  // ── render ────────────────────────────────────────────────────────────────────
  return (
    <div className={`app ${isPortrait ? 'portrait' : 'landscape'}`} onClick={() => setPopover(null)}>

      {/* ── Top bar ── */}
      <header className="topbar">
        <img src={sparkqbLogo} alt="SparkQB" className="sl-logo" />
        <div className="mode-tabs">
          <button className={mode==='field'?'active':''} onClick={()=>setMode('field')}>FIELD</button>
          <button className={mode==='video'?'active':''} onClick={()=>fileRef.current?.click()}>FILM</button>
        </div>
        <div className="topbar-actions">
          <button className="tb-btn" disabled={!canUndo} onClick={undo} title="Undo">
            <I d="M9 14L4 9l5-5M4 9h11a5 5 0 010 10h-3" size={18}/>
          </button>
          <button className="tb-btn" disabled={!canRedo} onClick={redo} title="Redo">
            <I d="M15 14l5-5-5-5M19 9H8a5 5 0 000 10h3" size={18}/>
          </button>
          <button className="tb-btn green" onClick={()=>setAnimating(true)} title="Animate routes">
            <I d="M5 3l14 9-14 9V3z" size={18} fill="currentColor" sw={0}/>
          </button>
          <button className="tb-btn" onClick={()=>setShowPlays(true)} title="Playbook">
            <I d="M4 19.5A2.5 2.5 0 016.5 17H20M4 19.5A2.5 2.5 0 016.5 22H20V2H6.5A2.5 2.5 0 004 4.5v15z" size={18}/>
          </button>
          <button className="tb-btn" onClick={exportImg} title="Export PNG">
            <I d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" size={18}/>
          </button>
          <button className="tb-btn red" onClick={()=>{push([]);setSelId(null)}} title="Clear all">
            <I d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" size={18}/>
          </button>
        </div>
      </header>

      {/* ── Stage ── */}
      <div className="stage" ref={wrapRef}>
        <canvas ref={bgRef} className="layer layer-bg"/>
        <canvas ref={drRef} className="layer layer-dr"/>
        <canvas ref={ovRef} className="layer layer-ov"
          style={{ cursor: tool==='select'?'grab':tool==='eraser'?'cell':'crosshair' }}
          onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
          onTouchStart={onDown} onTouchMove={onMove} onTouchEnd={onUp}
        />

        {/* ── Tool palette ── */}
        <div className={`palette ${isPortrait?'palette-right':'palette-bottom'}`}
          onClick={e => e.stopPropagation()}>
          {TOOLS.map(t => (
            <div key={t.id} className="pal-item">
              <button
                className={`pal-btn ${tool===t.id?'active':''}`}
                title={t.label}
                onClick={()=>handleToolClick(t.id)}
              >
                <I d={t.icon} size={20}/>
              </button>
              {/* Per-tool popover */}
              {popover === t.id && tool === t.id && (
                <div className={`tool-popover ${isPortrait?'pop-left':'pop-top'}`}
                  onClick={e=>e.stopPropagation()}>
                  <div className="pop-head">{t.label.toUpperCase()}</div>

                  <div className="pop-label">COLOR</div>
                  <div className="pop-swatches">
                    {PALETTE.map(c => (
                      <button key={c} className={`pop-sw ${ts.color===c?'sel':''}`}
                        style={{background:c, border: c==='#101214'?'1px solid rgba(255,255,255,0.2)':'none'}}
                        onClick={()=>updateStyle('color',c)}/>
                    ))}
                    <input type="color" value={ts.color}
                      onChange={e=>updateStyle('color',e.target.value)} title="Custom"/>
                  </div>

                  <div className="pop-label">THICKNESS</div>
                  <div className="pop-slider-wrap">
                    <input type="range" min={1} max={14} step={1} value={ts.lw}
                      onChange={e=>updateStyle('lw',+e.target.value)} className="pop-slider"/>
                    <span className="pop-val">{ts.lw}px</span>
                  </div>

                  <div className="pop-label">OPACITY</div>
                  <div className="pop-slider-wrap">
                    <input type="range" min={0.1} max={1} step={0.05} value={ts.opacity}
                      onChange={e=>updateStyle('opacity',+e.target.value)} className="pop-slider"/>
                    <span className="pop-val">{Math.round(ts.opacity*100)}%</span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* ── Inline text input ── */}
        {pendingTxt && (() => {
          const c = bgRef.current
          return (
            <div className="txt-overlay"
              style={{ left:(pendingTxt.x/(c?.width||1)*100)+'%', top:(pendingTxt.y/(c?.height||1)*100)+'%' }}>
              <input ref={txtRef} value={txtVal} style={{ color: ts.color }}
                onChange={e=>setTxtVal(e.target.value)}
                onKeyDown={e=>{ if(e.key==='Enter') commitTxt(); if(e.key==='Escape') setPendingTxt(null) }}
                placeholder="Type…" className="txt-input" autoFocus/>
              <button className="txt-ok" onClick={commitTxt}>OK</button>
            </div>
          )
        })()}

        {/* ── Delete selected ── */}
        {selId && (
          <button className="del-btn"
            onClick={()=>{ push(shapes.filter(s=>s.id!==selId)); setSelId(null) }}>
            <I d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" size={14}/> Delete
          </button>
        )}
      </div>

      {/* ── Video controls ── */}
      {mode === 'video' && videoMeta && (
        <div className="vbar">
          <button className="vb" onClick={()=>stepFrame(-10)} title="-10 frames">
            <I d="M19 20L9 12l10-8v16zM5 4v16" size={16}/>
          </button>
          <button className="vb" onClick={()=>stepFrame(-1)} title="-1 frame">
            <I d="M15 19l-7-7 7-7" size={16}/>
          </button>
          <button className="vb play" onClick={togglePlay}>
            {playing
              ? <I d="M6 4h4v16H6zM14 4h4v16h-4" size={18}/>
              : <I d="M5 3l14 9-14 9V3z" size={18} fill="currentColor" sw={0}/>
            }
          </button>
          <button className="vb" onClick={()=>stepFrame(1)} title="+1 frame">
            <I d="M9 5l7 7-7 7" size={16}/>
          </button>
          <button className="vb" onClick={()=>stepFrame(10)} title="+10 frames">
            <I d="M5 4l10 8-10 8V4zM19 4v16" size={16}/>
          </button>
          <div className="scrub-wrap">
            <input type="range" className="scrub" min={0} max={duration} step={1/(videoMeta.fps||30)/2}
              value={currentT} onChange={e=>seek(e.target.value)}/>
          </div>
          <button className="speed-btn" onClick={cycleSpeed}>{SPEEDS[speedIdx]}×</button>
          <span className="tc">{fmt(currentT)}</span>
          {videoMeta.fps && <span className="fps-badge">{Math.round(videoMeta.fps)}fps</span>}
        </div>
      )}

      {/* ── Hidden elements ── */}
      <video ref={vidRef} style={{display:'none'}} playsInline loop
        onEnded={()=>setPlaying(false)}/>
      <input ref={fileRef} type="file" accept="video/*" style={{display:'none'}} onChange={onFileLoad}/>

      {/* ── Player label modal ── */}
      {lblModal && (
        <div className="modal-bg" onClick={()=>setLblModal(null)}>
          <div className="modal" onClick={e=>e.stopPropagation()}>
            <h3>PLAYER LABEL</h3>
            <input value={lblVal} onChange={e=>setLblVal(e.target.value)}
              onKeyDown={e=>e.key==='Enter'&&commitLabel()}
              placeholder="QB, WR, 12…" className="modal-input" autoFocus/>
            <div className="modal-row">
              <button className="btn-sec" onClick={()=>setLblModal(null)}>Cancel</button>
              <button className="btn-pri" onClick={commitLabel}>Place →</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Playbook ── */}
      {showPlays && (
        <div className="modal-bg" onClick={()=>setShowPlays(false)}>
          <div className="modal modal-wide" onClick={e=>e.stopPropagation()}>
            <h3>PLAYBOOK</h3>
            <div className="save-row">
              <input value={playName} onChange={e=>setPlayName(e.target.value)}
                placeholder="Name this play…" className="modal-input flex1"/>
              <button className="btn-pri" onClick={savePlay}>Save →</button>
            </div>
            <div className="plays-grid">
              {plays.length===0 && <p className="empty">No plays saved yet.</p>}
              {plays.map(p=>(
                <button key={p.id} className="play-card"
                  onClick={()=>{ push(p.shapes); setShowPlays(false) }}>
                  {p.thumb && <img src={p.thumb} alt={p.name}/>}
                  <span>{p.name}</span>
                </button>
              ))}
            </div>
            <button className="btn-sec full mt" onClick={()=>setShowPlays(false)}>Close</button>
          </div>
        </div>
      )}
    </div>
  )
}
