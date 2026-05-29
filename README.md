# SpinLab Film

QB film annotation tool — built with React + Vite.

## Features (Phase 1)
- Load any video (9:16 or 16:9) — canvas auto-fits the aspect ratio
- Auto-detects frame rate (supports up to 240fps)
- Frame-by-frame stepping (±1 frame, ±10 frames)
- Precision scrubber
- Playback speeds: 0.1×, 0.25×, 0.5×, 1×, 2×
- Drawing tools: Pen, Arrow, Dashed Route, Circle, Box, Text, Offensive player (O), Defensive player (X), Move, Eraser
- Per-tool color, thickness, and opacity — tap an active tool to style it
- Animate routes
- Playbook — save & reload named plays with thumbnails
- Export as PNG
- Undo / Redo
- Football field mode for play drawing
- SparkQB brand — SpinLab logo, Anton / Roboto Mono / Inter typography

---

## Local Development

```bash
npm install
npm run dev
```

Open http://localhost:5173

## Build

```bash
npm run build
```

Output goes to `/dist`

---

## Deploy to GitHub + Vercel

### 1. Create a GitHub repo

1. Go to https://github.com/new
2. Name it `spinlab-film` (or whatever you like)
3. Set to Private
4. Don't add a README — you already have one
5. Click **Create repository**

### 2. Push the code

In your terminal, from this project folder:

```bash
git init
git add .
git commit -m "Initial build — SpinLab Film Phase 1"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/spinlab-film.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### 3. Deploy on Vercel

1. Go to https://vercel.com and sign in with GitHub
2. Click **Add New → Project**
3. Import your `spinlab-film` repo
4. Vercel auto-detects Vite — no config needed
5. Click **Deploy**

You'll get a live URL like `spinlab-film.vercel.app` in ~30 seconds.

### 4. Auto-deploy on updates

Every time you push to `main`, Vercel automatically redeploys. So the workflow going forward is:

```bash
git add .
git commit -m "describe your change"
git push
```

Done — live in ~20 seconds.

---

## Custom Domain (optional)

In Vercel → your project → Settings → Domains — add any custom domain you own.
