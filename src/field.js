export function drawFootballField(ctx, W, H) {
  // Base grass
  ctx.fillStyle = '#1a3a0f'
  ctx.fillRect(0, 0, W, H)

  const yardW = W / 120

  // Alternating stripes
  for (let i = 0; i < 60; i++) {
    if (i % 2 === 0) {
      ctx.fillStyle = 'rgba(0,0,0,0.06)'
      ctx.fillRect(i * 2 * yardW, 0, yardW, H)
    }
  }

  // End zones
  ctx.fillStyle = '#122a09'
  ctx.fillRect(0, 0, yardW * 10, H)
  ctx.fillRect(W - yardW * 10, 0, yardW * 10, H)

  // End zone diagonal hatch
  ctx.save()
  ctx.strokeStyle = 'rgba(255,255,255,0.06)'
  ctx.lineWidth = 1
  for (let i = -H; i < W; i += 18) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i + H, H); ctx.stroke()
  }
  ctx.restore()

  // 5-yard lines
  ctx.strokeStyle = 'rgba(255,255,255,0.8)'
  ctx.lineWidth = Math.max(1, W * 0.0018)
  for (let y = 0; y <= 120; y += 5) {
    const x = y * yardW
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
  }

  // Hash marks every yard
  const hashT = H * 0.3, hashB = H * 0.7, hashLen = H * 0.042
  ctx.lineWidth = Math.max(1, W * 0.0012)
  for (let y = 10; y <= 110; y++) {
    if (y % 5 === 0) continue
    const x = y * yardW
    ctx.beginPath(); ctx.moveTo(x, hashT); ctx.lineTo(x, hashT + hashLen); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(x, hashB - hashLen); ctx.lineTo(x, hashB); ctx.stroke()
  }

  // Sidelines
  ctx.strokeStyle = 'rgba(255,255,255,0.9)'
  ctx.lineWidth = Math.max(2, W * 0.003)
  ctx.strokeRect(1, 1, W - 2, H - 2)

  // Yard numbers
  const nums = [10,20,30,40,50,40,30,20,10]
  ctx.fillStyle = 'rgba(255,255,255,0.65)'
  const fs = Math.max(10, H * 0.13)
  ctx.font = `900 ${fs}px Anton, sans-serif`
  ctx.textAlign = 'center'
  nums.forEach((n, i) => {
    const x = (i + 1.5) * 10 * yardW
    ctx.fillText(n, x, H * 0.23)
    ctx.fillText(n, x, H * 0.89)
  })

  // Goal posts
  ;[yardW * 10, W - yardW * 10].forEach(x => {
    const pw = W * 0.022, ph = H * 0.18
    ctx.strokeStyle = '#f5c842'
    ctx.lineWidth = Math.max(2, W * 0.003)
    ctx.beginPath(); ctx.moveTo(x, H * 0.5); ctx.lineTo(x, H * 0.5 - ph * 0.6); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(x - pw, H * 0.5 - ph * 0.35); ctx.lineTo(x + pw, H * 0.5 - ph * 0.35); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(x - pw, H * 0.5 - ph * 0.35); ctx.lineTo(x - pw, H * 0.5 - ph * 0.6); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(x + pw, H * 0.5 - ph * 0.35); ctx.lineTo(x + pw, H * 0.5 - ph * 0.6); ctx.stroke()
  })
}
