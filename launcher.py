#!/usr/bin/env python3
"""
LMDIS Application Launcher
Performs all pre-flight checks, opens a loading screen, and starts the server.
"""
import os
import sys
import socket
import time
import tempfile
import subprocess
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8000

# ── Suppress OpenMP duplicate-lib warning (common with PyTorch on Windows) ────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── Force UTF-8 output on Windows (box-drawing chars require UTF-8) ───────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── ANSI colors (Windows 10+ compatible) ──────────────────────────────────────
os.system("")  # Enable ANSI on Windows
G   = "\033[92m"
Y   = "\033[93m"
R   = "\033[91m"
C   = "\033[96m"
W   = "\033[97m"
DIM = "\033[2m"
RST = "\033[0m"


def banner():
    print(f"""
{C}  ╔══════════════════════════════════════════════════════════╗
  ║                                                          ║
  ║{W}        LMDIS — Document Intelligence System{C}              ║
  ║{DIM}        Lossless Multimodal Pipeline  v1.0{C}                ║
  ║                                                          ║
  ╚══════════════════════════════════════════════════════════╝{RST}
""")


def check(label, ok=True, detail=""):
    mark  = f"{G}✓{RST}" if ok else f"{R}✗{RST}"
    extra = f"  {DIM}{detail}{RST}" if detail else ""
    print(f"  {mark}  {label}{extra}")


def step(n, total, label):
    print(f"\n  {C}[{n}/{total}]{RST}  {W}{label}{RST}")


def err(msg):
    print(f"\n  {R}ERROR:{RST} {msg}\n")
    input("  Press Enter to exit...")
    sys.exit(1)


def kill_port(port):
    try:
        r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
                time.sleep(1.5)
                break
    except Exception:
        pass


# ── Loading page HTML (Python string — zero escaping issues) ──────────────────
LOADING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LMDIS — Starting</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --accent: #6366f1; --accent-secondary: #06b6d4; --accent-tertiary: #8b5cf6;
  --accent-light: #a5b4fc;
  --bg-primary: #06080f; --bg-secondary: #0c1021; --bg-tertiary: #111827;
  --bg-glass: rgba(255,255,255,0.03);
  --border-glass: rgba(255,255,255,0.06);
  --text-primary: #f1f5f9; --text-secondary: #cbd5e1; --text-muted: #64748b;
  --green: #10b981; --amber: #f59e0b; --red: #ef4444;
  --radius-lg: 20px; --radius-md: 14px; --radius-sm: 10px;
}
body {
  min-height: 100vh; display: flex; align-items: center;
  justify-content: center; background: var(--bg-primary);
  font-family: "Inter", system-ui, sans-serif;
  color: var(--text-primary); overflow: hidden;
}
body::before {
  content: ""; position: fixed; inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 20% 20%, rgba(99,102,241,.18) 0%, transparent 70%),
    radial-gradient(ellipse 60% 80% at 80% 80%, rgba(139,92,246,.12) 0%, transparent 70%),
    radial-gradient(ellipse 50% 50% at 60% 10%, rgba(6,182,212,.08) 0%, transparent 70%);
  pointer-events: none;
  animation: drift 14s ease-in-out infinite alternate;
}
@keyframes drift { to { transform: translate(24px, 18px) scale(1.05); } }
body::after {
  content: ""; position: fixed; inset: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(99,102,241,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(99,102,241,.03) 1px, transparent 1px);
  background-size: 44px 44px;
}
.card {
  position: relative; z-index: 1;
  width: 500px; padding: 52px 48px;
  background: var(--bg-glass);
  backdrop-filter: blur(32px) saturate(1.5);
  -webkit-backdrop-filter: blur(32px) saturate(1.5);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-lg);
  box-shadow: 0 32px 96px rgba(0,0,0,.55), 0 0 0 1px rgba(255,255,255,.04), inset 0 1px 0 rgba(255,255,255,.06);
  text-align: center;
}
.ring-wrap {
  width: 96px; height: 96px; margin: 0 auto 28px;
  position: relative;
}
.ring-wrap svg { width: 96px; height: 96px; }
.ring-track { opacity: .10; }
.ring-arc {
  stroke-dasharray: 70 230;
  animation: spin 1.6s cubic-bezier(.4,0,.2,1) infinite;
  transform-origin: center;
}
@keyframes spin { to { stroke-dashoffset: -300; } }
.ring-icon {
  position: absolute; top: 50%; left: 50%;
  transform: translate(-50%,-50%);
  font-size: 32px; line-height: 1;
  animation: breathe 2.4s ease-in-out infinite;
}
@keyframes breathe {
  0%,100% { opacity:1; transform:translate(-50%,-50%) scale(1); }
  50%      { opacity:.7; transform:translate(-50%,-50%) scale(.9); }
}
h1 {
  font-size: 28px; font-weight: 700; letter-spacing: -.5px;
  background: linear-gradient(135deg, var(--accent-light), var(--accent));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 6px;
}
.tagline {
  font-size: 11px; color: var(--text-muted);
  letter-spacing: 2.5px; text-transform: uppercase;
  margin-bottom: 10px; font-weight: 500;
}
.arch-note {
  font-size: 10.5px; color: var(--accent-light); opacity: 0.7;
  letter-spacing: 0.5px; margin-bottom: 32px;
}
.progress-wrap {
  background: rgba(255,255,255,.06);
  border-radius: 99px; height: 3px;
  overflow: hidden; margin-bottom: 12px;
}
.progress-fill {
  height: 100%; border-radius: 99px; width: 5%;
  background: linear-gradient(90deg, var(--accent), var(--accent-tertiary), var(--accent-secondary));
  transition: width .9s cubic-bezier(.4,0,.2,1);
  background-size: 200% 100%;
  animation: shimmer 2.4s linear infinite;
}
@keyframes shimmer {
  0%   { background-position: 200% center; }
  100% { background-position: -200% center; }
}
.status {
  font-size: 12px; color: var(--text-muted);
  min-height: 20px; margin-bottom: 24px;
  transition: opacity .4s; font-weight: 400;
}
.steps { display: flex; flex-direction: column; gap: 8px; text-align: left; }
.step {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 14px; border-radius: var(--radius-sm);
  font-size: 12.5px; color: var(--text-muted);
  border: 1px solid transparent;
  transition: all .4s; font-weight: 400;
}
.step.active {
  color: var(--text-primary); background: rgba(99,102,241,.08);
  border-color: rgba(99,102,241,.18);
}
.step.done { color: var(--text-muted); }
.step-dot {
  width: 22px; height: 22px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; flex-shrink: 0;
  border: 1.5px solid rgba(255,255,255,.08);
  transition: all .4s;
}
.step.active .step-dot {
  border-color: var(--accent);
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  0%,100% { box-shadow: 0 0 0 3px rgba(99,102,241,.12); }
  50%      { box-shadow: 0 0 0 7px rgba(99,102,241,.04); }
}
.step.done .step-dot {
  background: rgba(16,185,129,.12); border-color: var(--green); color: var(--green);
}
.ready {
  margin-top: 22px; font-size: 13px; color: var(--green);
  opacity: 0; transition: opacity .6s;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  font-weight: 500;
}
.ready.show { opacity: 1; }
.ready-dot {
  width: 8px; height: 8px; background: var(--green); border-radius: 50%;
  animation: readypulse 1s ease-in-out infinite;
}
@keyframes readypulse {
  0%,100% { transform: scale(1); opacity: 1; }
  50%      { transform: scale(1.4); opacity: .6; }
}
</style>
</head>
<body>
<div class="card">
  <div class="ring-wrap">
    <svg viewBox="0 0 96 96">
      <circle class="ring-track" cx="48" cy="48" r="40"
        fill="none" stroke="#6366f1" stroke-width="3"/>
      <circle class="ring-arc" cx="48" cy="48" r="40"
        fill="none" stroke="url(#rg)" stroke-width="3"
        stroke-linecap="round" stroke-dashoffset="0"/>
      <defs>
        <linearGradient id="rg" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#6366f1"/>
          <stop offset="100%" stop-color="#06b6d4"/>
        </linearGradient>
      </defs>
    </svg>
    <div class="ring-icon">&#128196;</div>
  </div>
  <h1>LMDIS</h1>
  <div class="tagline">Document Intelligence System</div>
  <div class="arch-note">Evidence Graph Architecture &mdash; LLMs as conversational layer only</div>
  <div class="progress-wrap"><div class="progress-fill" id="pf"></div></div>
  <div class="status" id="st">Booting up...</div>
  <div class="steps">
    <div class="step active" id="s0">
      <div class="step-dot" id="d0"></div>
      <span>Loading Python environment</span>
    </div>
    <div class="step" id="s1">
      <div class="step-dot" id="d1"></div>
      <span>Initializing AI models (BGE + Cross-Encoder)</span>
    </div>
    <div class="step" id="s2">
      <div class="step-dot" id="d2"></div>
      <span>Starting FastAPI server on port 8000</span>
    </div>
    <div class="step" id="s3">
      <div class="step-dot" id="d3"></div>
      <span>Connecting application</span>
    </div>
  </div>
  <div class="ready" id="ready">
    <div class="ready-dot"></div>
    <span>System ready &mdash; launching now</span>
  </div>
</div>
<script>
var pf = document.getElementById('pf');
var st = document.getElementById('st');
var rc = document.getElementById('ready');

function prog(pct, msg) {
  pf.style.width = pct + '%';
  if (msg) st.textContent = msg;
}
function setStep(i, state) {
  var s = document.getElementById('s' + i);
  var d = document.getElementById('d' + i);
  s.className = 'step ' + state;
  if (state === 'done')   d.textContent = '\u2713';
  if (state === 'active') d.textContent = '';
}

prog(10, 'Python environment loaded');
setTimeout(function() {
  setStep(0, 'done'); setStep(1, 'active');
  prog(22, 'Loading embedding model (BAAI/bge-large-en-v1.5)\u2026');
}, 1200);
setTimeout(function() {
  prog(34, 'Loading cross-encoder reranker\u2026');
}, 5500);
setTimeout(function() {
  setStep(1, 'done'); setStep(2, 'active');
  prog(50, 'Starting FastAPI server on port 8000\u2026');
}, 10000);

var attempts = 0;
function poll() {
  attempts++;
  var xhr = new XMLHttpRequest();
  xhr.timeout = 4000;
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4 && xhr.status === 200) {
      try {
        var d = JSON.parse(xhr.responseText);
        if (d && d.status) {
          setStep(1, 'done'); setStep(2, 'done'); setStep(3, 'active');
          prog(88, 'Backend online \u2014 connecting frontend\u2026');
          setTimeout(function() {
            setStep(3, 'done');
            prog(100, 'All systems ready');
            rc.className = 'ready show';
            setTimeout(function() {
              window.location.href = 'http://127.0.0.1:8000/app';
            }, 900);
          }, 600);
          return;
        }
      } catch(e) {}
      setTimeout(poll, 2000);
    }
  };
  xhr.ontimeout = xhr.onerror = function() {
    var p = Math.min(50 + attempts * 1.8, 82);
    var msgs = [
      'Initializing pipeline modules\u2026',
      'Building knowledge structures\u2026',
      'Server warming up\u2026',
      'Almost ready\u2026'
    ];
    prog(p, msgs[Math.min(Math.floor(attempts / 5), msgs.length - 1)]);
    setTimeout(poll, 2000);
  };
  xhr.open('GET', 'http://127.0.0.1:8000/');
  xhr.send();
}
setTimeout(poll, 3500);
</script>
</body>
</html>
"""


def write_loader():
    """Write loading HTML to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".html", prefix="lmdis_loader_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(LOADING_HTML)
    return path


def main():
    os.system("")  # Enable ANSI colors on Windows terminal
    banner()

    # ── Step 1: Python version ─────────────────────────────────────────
    step(1, 4, "Checking Python environment")
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    check(f"Python {ver}")
    if sys.version_info < (3, 10):
        err("Python 3.10+ is required. Please upgrade.")

    # ── Step 2: Dependencies ───────────────────────────────────────────
    step(2, 4, "Checking dependencies")
    missing = []
    for pkg, imp in [
        ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
        ("torch", "torch"), ("networkx", "networkx"),
        ("pymupdf", "fitz"), ("openai", "openai"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  {Y}!{RST}  Missing: {', '.join(missing)} — installing now…")
        req = os.path.join(ROOT, "backend", "requirements.txt")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", req, "-q"]
        )
        check("Dependencies installed")
    else:
        check("All dependencies present")

    # ── Step 3: Port check ─────────────────────────────────────────────
    step(3, 4, "Checking port 8000")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        in_use = s.connect_ex(("127.0.0.1", PORT)) == 0
    if in_use:
        print(f"  {Y}!{RST}  Port {PORT} busy — stopping existing process…")
        kill_port(PORT)
        time.sleep(1.5)
        check(f"Port {PORT} freed")
    else:
        check(f"Port {PORT} available")

    # ── Step 4: Loading screen ─────────────────────────────────────────
    step(4, 4, "Opening loading screen")
    loader_path = write_loader()

    # os.startfile is most reliable on Windows for local HTML files
    try:
        os.startfile(loader_path)
    except Exception:
        webbrowser.open(f"file:///{loader_path.replace(os.sep, '/')}")

    check("Loading screen opened in browser")
    time.sleep(0.6)

    print(f"""
  {C}══════════════════════════════════════════════════════════{RST}
  {W}  Server:{RST}    http://127.0.0.1:{PORT}
  {W}  App:{RST}       http://127.0.0.1:{PORT}/app
  {W}  API docs:{RST}  http://127.0.0.1:{PORT}/docs
  {C}══════════════════════════════════════════════════════════{RST}
  {DIM}  Press Ctrl+C to stop the server{RST}
""")

    # ── Start server (blocks until Ctrl+C) ────────────────────────────
    sys.path.insert(0, ROOT)
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
