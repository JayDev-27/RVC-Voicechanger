/**
 * main.js — Electron main process for RVC Voicechanger
 *
 * Architecture:
 *   Electron window ──IPC──► preload.js ──WS/HTTP──► Python FastAPI (child process)
 *
 * Startup sequence:
 *   1. Acquire single-instance lock
 *   2. If voice models are missing, show the first-run downloader
 *   3. Find a free port and spawn the Python backend with --port <port>
 *   4. Wait for "READY:{port}" on stdout
 *   5. Create BrowserWindow loading 127.0.0.1:{port}
 *   6. Set up system tray
 */

const {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  Tray,
  Menu,
  nativeImage,
  shell,
} = require('electron');

const path   = require('path');
const net    = require('net');
const { spawn, execFile, execFileSync } = require('child_process');
const fs     = require('fs');
const os     = require('os');
const https  = require('https');
const http   = require('http');

// ── Single instance (must run before anything else) ─────────────────────────
if (!app.requestSingleInstanceLock()) {
  app.quit();
  process.exit(0);
}

// ── Constants ─────────────────────────────────────────────────────────────────
const IS_WIN      = process.platform === 'win32';
const IS_MAC      = process.platform === 'darwin';
const APP_ROOT    = path.join(__dirname, '..');
const ASSETS_DIR  = path.join(APP_ROOT, 'assets');
const ICON_PATH   = path.join(ASSETS_DIR, 'icon.png');
const USER_ROOT   = path.join(os.homedir(), '.aivoicechanger');
const ASSET_CACHE = path.join(USER_ROOT, 'assets');

// Packaged builds keep user data (voice models, images, soundboard) outside the
// app bundle: the bundle may be read-only (macOS App Translocation, Program
// Files) and is replaced wholesale on every update.  Dev runs keep using the
// repository's ./models folder.
const MODELS_DIR = app.isPackaged
  ? path.join(USER_ROOT, 'models')
  : path.join(APP_ROOT, 'models');

const MODELS_ZIP_URL =
  'https://github.com/satiricalguru/RVC-Voicechanger/releases/download/v1.0.0/models.zip';
const HUBERT_URL =
  'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt';
const RMVPE_URL =
  'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx';

// Try to load electron-store for persistent window bounds
let store = null;
try {
  const Store = require('electron-store');
  store = new (Store.default || Store)();
} catch (_) { store = null; }

// ── State ─────────────────────────────────────────────────────────────────────
let mainWindow   = null;
let downloaderWindow = null;
let tray         = null;
let pythonProc   = null;
let backendPort  = null;
let isQuitting   = false;
let restartCount = 0;
const MAX_RESTARTS = 5;
let downloaderCompleted = false;
let downloadInFlight = false;

// ── Helpers ───────────────────────────────────────────────────────────────────
function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
    srv.on('error', reject);
  });
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

/**
 * Locate a Python interpreter, in priority order:
 *   1. The standalone runtime bundled with packaged builds (resources/python)
 *   2. A project virtualenv (.venv) for development
 *   3. A system interpreter
 */
function findPython() {
  const candidates = [];
  if (app.isPackaged) {
    const bundled = path.join(process.resourcesPath, 'python');
    candidates.push(IS_WIN
      ? path.join(bundled, 'python.exe')
      : path.join(bundled, 'bin', 'python3'));
  }
  candidates.push(IS_WIN
    ? path.join(APP_ROOT, '.venv', 'Scripts', 'python.exe')
    : path.join(APP_ROOT, '.venv', 'bin', 'python'));

  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }

  const system = IS_WIN ? ['python', 'py', 'python3'] : ['python3', 'python'];
  for (const p of system) {
    try {
      execFileSync(p, ['--version'], { stdio: 'ignore', windowsHide: true });
      return p;
    } catch (_) { /* continue */ }
  }
  return IS_WIN ? 'python' : 'python3';
}

/** Copy images that ship with the app into the user's models folder once. */
function seedBundledImages() {
  const src = path.join(APP_ROOT, 'models', 'images');
  const dst = ensureDir(path.join(MODELS_DIR, 'images'));
  if (src === dst || !fs.existsSync(src)) return;
  for (const f of fs.readdirSync(src)) {
    if (f.startsWith('.')) continue;
    const target = path.join(dst, f);
    if (!fs.existsSync(target)) {
      try { fs.copyFileSync(path.join(src, f), target); } catch (_) {}
    }
  }
}

// ── Python backend ────────────────────────────────────────────────────────────
async function startPythonBackend() {
  const port   = await getFreePort();
  const python = findPython();
  const script = path.join(APP_ROOT, 'main.py');

  console.log(`[electron] Spawning Python: ${python} ${script} --port ${port}`);

  const proc = spawn(python, [script, '--port', String(port)], {
    cwd:   APP_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
    env:   {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      // Windows pipes default to the ANSI code page; force UTF-8 so log lines
      // containing non-ASCII characters never raise UnicodeEncodeError.
      PYTHONIOENCODING: 'utf-8',
      PYTHONUTF8:       '1',
      PYTHONNOUSERSITE: '1',          // ignore ~/.local site-packages
      RVC_MODELS_DIR:   MODELS_DIR,
      // ── OpenMP conflict fix ──────────────────────────────────────────────
      // faiss-cpu, PyTorch, and scikit-learn each bundle their own OpenMP
      // runtime.  Loading several at once crashes inside FAISS parallel
      // search; tolerate duplicates and stay single-threaded (negligible cost
      // at real-time RVC chunk sizes).
      KMP_DUPLICATE_LIB_OK: 'TRUE',
      OMP_NUM_THREADS:      '1',
      MKL_NUM_THREADS:      '1',
    },
  });

  pythonProc = proc;

  return new Promise((resolve, reject) => {
    let resolved = false;
    let stdoutBuf = '';
    let stderrTail = '';
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        try { proc.kill(); } catch (_) {}
        reject(new Error('Python backend timed out (180s)'));
      }
    }, 180000);

    proc.stdout.on('data', (data) => {
      stdoutBuf += data.toString();
      const lines = stdoutBuf.split(/\r?\n/);
      stdoutBuf = lines.pop();
      for (const line of lines) {
        if (line.trim()) console.log(`[python] ${line}`);
        const match = line.match(/READY:(\d+)/);
        if (match && !resolved) {
          resolved = true;
          clearTimeout(timeout);
          backendPort = Number(match[1]);
          resolve(backendPort);
        }
      }
    });

    proc.stderr.on('data', (data) => {
      const text = data.toString();
      stderrTail = (stderrTail + text).slice(-2000);
      if (text.trim()) console.error(`[python-err] ${text.trim()}`);
    });

    proc.on('error', (err) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        reject(new Error(`Could not launch Python (${python}): ${err.message}`));
      }
    });

    proc.on('exit', (code, signal) => {
      console.warn(`[electron] Python exited (code=${code}, signal=${signal})`);
      if (pythonProc === proc) pythonProc = null;
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        const tail = stderrTail.trim().split('\n').slice(-6).join('\n');
        reject(new Error(`Python exited with code ${code}${tail ? `\n\n${tail}` : ''}`));
        return;
      }
      // Auto-restart after a crash, unless we are quitting
      if (!isQuitting && restartCount < MAX_RESTARTS) {
        restartCount++;
        const delay = Math.min(restartCount * 2000, 10000);
        console.log(`[electron] Restarting Python in ${delay}ms (attempt ${restartCount})`);
        setTimeout(() => {
          if (isQuitting) return;
          startPythonBackend()
            .then((p) => {
              if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.loadURL(`http://127.0.0.1:${p}`);
              }
            })
            .catch(console.error);
        }, delay);
      }
    });
  });
}

function stopPythonBackend() {
  const proc = pythonProc;
  pythonProc = null;
  if (!proc) return;
  try { proc.kill(IS_WIN ? undefined : 'SIGTERM'); } catch (_) {}
  // Escalate if it ignores SIGTERM (e.g. stuck in an audio driver call)
  setTimeout(() => {
    if (proc.exitCode === null && proc.signalCode === null) {
      try {
        if (IS_WIN) execFile('taskkill', ['/pid', String(proc.pid), '/T', '/F'], () => {});
        else proc.kill('SIGKILL');
      } catch (_) {}
    }
  }, 3000).unref();
}

// ── Window ────────────────────────────────────────────────────────────────────
function hardenWebContents(contents) {
  // Open external links in the user's browser; never navigate the app away.
  contents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  contents.on('will-navigate', (event, url) => {
    const allowed = url.startsWith(`http://127.0.0.1:${backendPort}`) || url.startsWith('file://');
    if (!allowed) {
      event.preventDefault();
      if (/^https?:\/\//.test(url)) shell.openExternal(url);
    }
  });
}

function createWindow(port) {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
    return mainWindow;
  }

  const bounds = store ? store.get('windowBounds', { width: 1280, height: 800 }) : { width: 1280, height: 800 };

  mainWindow = new BrowserWindow({
    width:           Math.max(bounds.width  || 1280, 900),
    height:          Math.max(bounds.height || 800, 640),
    minWidth:        900,
    minHeight:       640,
    frame:           false,        // Custom titlebar
    transparent:     false,
    backgroundColor: '#0a0a0f',
    titleBarStyle:   IS_MAC ? 'hiddenInset' : 'hidden',
    ...(IS_MAC ? { trafficLightPosition: { x: 16, y: 16 } } : {}),
    show:            false,
    icon:            fs.existsSync(ICON_PATH) ? nativeImage.createFromPath(ICON_PATH) : undefined,
    webPreferences: {
      preload:              path.join(__dirname, 'preload.js'),
      contextIsolation:     true,
      nodeIntegration:      false,
      sandbox:              true,
      webSecurity:          true,
      allowRunningInsecureContent: false,
      additionalArguments:  [`--app-version=${app.getVersion()}`],
    },
  });
  hardenWebContents(mainWindow.webContents);

  let loadRetries = 0;
  const MAX_LOAD_RETRIES = 10;

  function tryLoadURL() {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.loadURL(`http://127.0.0.1:${backendPort || port}`);
    }
  }

  // Small delay before first load — the READY signal means the health check
  // passed, but Chromium can still race with the server's accept queue.
  setTimeout(tryLoadURL, 300);

  // ── Renderer error logging ────────────────────────────────────────────────
  mainWindow.webContents.on('did-fail-load', (event, code, desc, url, isMainFrame) => {
    if (!isMainFrame) return;
    console.error(`[renderer] did-fail-load  code=${code}  desc=${desc}  url=${url}`);

    // Retry on connection-refused / connection-reset (backend still starting)
    if ((code === -102 || code === -101 || code === -6) && loadRetries < MAX_LOAD_RETRIES) {
      loadRetries++;
      const delay = Math.min(500 * loadRetries, 3000);
      console.log(`[electron] Retrying page load in ${delay}ms (attempt ${loadRetries}/${MAX_LOAD_RETRIES})`);
      setTimeout(tryLoadURL, delay);
    }
  });
  mainWindow.webContents.on('render-process-gone', (event, details) => {
    console.error('[renderer] render-process-gone:', JSON.stringify(details));
    if (!isQuitting && details.reason !== 'clean-exit') setTimeout(tryLoadURL, 1000);
  });

  if (process.env.NODE_ENV === 'development' || process.argv.includes('--devtools')) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }

  // ready-to-show may never fire if React throws during hydration; force-show
  // after 12 s so the user always sees something.
  let windowShown = false;
  const showFallback = setTimeout(() => {
    if (!windowShown && mainWindow && !mainWindow.isDestroyed()) {
      console.warn('[electron] ready-to-show never fired — forcing window visible');
      mainWindow.show();
      windowShown = true;
    }
  }, 12000);

  mainWindow.once('ready-to-show', () => {
    clearTimeout(showFallback);
    if (!windowShown) {
      mainWindow.show();
      windowShown = true;
    }
    restartCount = 0;
  });

  // Save window bounds (debounced)
  let boundsTimer = null;
  const saveBounds = () => {
    clearTimeout(boundsTimer);
    boundsTimer = setTimeout(() => {
      if (store && mainWindow && !mainWindow.isDestroyed() && !mainWindow.isMaximized()) {
        store.set('windowBounds', mainWindow.getBounds());
      }
    }, 400);
  };
  mainWindow.on('resize', saveBounds);
  mainWindow.on('move', saveBounds);

  // Minimize to tray instead of close (like Voicemod)
  let trayHintShown = false;
  mainWindow.on('close', (e) => {
    if (!isQuitting) {
      e.preventDefault();
      mainWindow.hide();
      if (tray && IS_WIN && !trayHintShown) {
        trayHintShown = true;
        tray.displayBalloon({
          title:   'RVC Voicechanger',
          content: 'Still running in the background. Right-click the tray icon to quit.',
        });
      }
    }
  });

  mainWindow.on('closed', () => { mainWindow = null; });

  mainWindow.webContents.on('did-finish-load', () => {
    loadRetries = 0;
    mainWindow.webContents.send('backend:ready', backendPort);
  });

  return mainWindow;
}

// ── System Tray ───────────────────────────────────────────────────────────────
function showMainWindow() {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  } else if (backendPort) {
    createWindow(backendPort);
  }
}

function createTray() {
  if (tray) return;
  const icon = fs.existsSync(ICON_PATH)
    ? nativeImage.createFromPath(ICON_PATH).resize({ width: 16, height: 16 })
    : nativeImage.createEmpty();

  tray = new Tray(icon);
  tray.setToolTip('RVC Voicechanger');
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: 'Show RVC Voicechanger', click: showMainWindow },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        isQuitting = true;
        stopPythonBackend();
        app.quit();
      },
    },
  ]));
  tray.on('click', showMainWindow);
}

// ── IPC Handlers ──────────────────────────────────────────────────────────────
function registerIPC() {
  ipcMain.handle('window:minimize', () => mainWindow?.minimize());
  ipcMain.handle('window:maximize', () => {
    if (!mainWindow) return;
    mainWindow.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize();
  });
  ipcMain.handle('window:close', (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (win && win === downloaderWindow) { win.close(); return; }
    mainWindow?.hide();
  });
  ipcMain.handle('window:isMaximized', () => mainWindow?.isMaximized() ?? false);

  ipcMain.handle('dialog:openFile', async (_event, options = {}) => {
    // Only pass through presentational options from the renderer.
    const { title, buttonLabel, defaultPath, filters } = options || {};
    return dialog.showOpenDialog(mainWindow, {
      title,
      buttonLabel,
      defaultPath,
      properties: ['openFile', 'multiSelections'],
      filters: Array.isArray(filters) && filters.length ? filters : [
        { name: 'RVC Model Files', extensions: ['pth', 'index'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
  });

  ipcMain.handle('app:getBackendPort', () => backendPort);
  ipcMain.handle('app:hideToTray', () => mainWindow?.hide());

  ipcMain.on('download:start', async () => {
    if (downloadInFlight) return;
    downloadInFlight = true;
    try {
      await runSetupDownloads();
      downloaderCompleted = true; // so the closed handler doesn't quit the app
      sendToDownloader('download:success');
      setTimeout(() => {
        launchAppFlow().catch(handleStartupError).finally(() => {
          if (downloaderWindow && !downloaderWindow.isDestroyed()) downloaderWindow.close();
        });
      }, 800);
    } catch (err) {
      console.error('[electron] Setup download failed:', err);
      sendToDownloader('download:error', err.message);
    } finally {
      downloadInFlight = false;
    }
  });
}

// ── First-run setup ───────────────────────────────────────────────────────────
function hasPth(dir) {
  try {
    for (const f of fs.readdirSync(dir)) {
      const full = path.join(dir, f);
      const stat = fs.statSync(full);
      if (stat.isDirectory() ? hasPth(full) : f.endsWith('.pth')) return true;
    }
  } catch (_) { /* ignore */ }
  return false;
}

function areModelsMissing() {
  const defaultDir = ensureDir(path.join(MODELS_DIR, 'default'));
  return !hasPth(defaultDir);
}

function sendToDownloader(channel, payload) {
  if (downloaderWindow && !downloaderWindow.isDestroyed()) {
    downloaderWindow.webContents.send(channel, payload);
  }
}

function showDownloaderWindow() {
  downloaderWindow = new BrowserWindow({
    width: 550,
    height: 380,
    resizable: false,
    frame: false,
    hasShadow: true,
    backgroundColor: '#0a0a0f',
    icon: fs.existsSync(ICON_PATH) ? nativeImage.createFromPath(ICON_PATH) : undefined,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  hardenWebContents(downloaderWindow.webContents);
  downloaderWindow.loadFile(path.join(__dirname, 'downloader.html'));

  downloaderWindow.on('closed', () => {
    downloaderWindow = null;
    if (!mainWindow && !downloaderCompleted && !isQuitting) {
      app.quit();
    }
  });
}

/**
 * Stream a URL to disk with progress reporting.
 *  - follows 301/302/303/307/308 (incl. relative Location headers)
 *  - writes to <dest>.part and renames on success, so an interrupted download
 *    is never mistaken for a complete file
 *  - respects stream back-pressure and resolves only after the file is flushed
 */
function downloadFileWithProgress(url, dest, { redirectsLeft = 10 } = {}) {
  return new Promise((resolve, reject) => {
    const partPath = `${dest}.part`;
    try { fs.unlinkSync(partPath); } catch (_) {}

    let settled = false;
    const fail = (err) => {
      if (settled) return;
      settled = true;
      try { fs.unlinkSync(partPath); } catch (_) {}
      reject(err);
    };

    function get(requestUrl, hopsLeft) {
      const lib = requestUrl.startsWith('https:') ? https : http;
      const req = lib.get(requestUrl, {
        headers: { 'User-Agent': `RVC-Voicechanger/${app.getVersion()}` },
        timeout: 30000,
      }, (res) => {
        if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
          res.resume();
          if (hopsLeft <= 0) return fail(new Error('Too many redirects'));
          return get(new URL(res.headers.location, requestUrl).toString(), hopsLeft - 1);
        }
        if (res.statusCode !== 200) {
          res.resume();
          return fail(new Error(`Server responded with status code ${res.statusCode}`));
        }

        const totalBytes = parseInt(res.headers['content-length'], 10) || 0;
        let downloadedBytes = 0;
        let lastTime = Date.now();
        let lastBytes = 0;
        const file = fs.createWriteStream(partPath);

        res.on('data', (chunk) => {
          downloadedBytes += chunk.length;
          const now = Date.now();
          if (now - lastTime > 200) {
            const speed = (downloadedBytes - lastBytes) / ((now - lastTime) / 1000);
            sendToDownloader('download:progress', {
              percent: totalBytes ? downloadedBytes / totalBytes : 0,
              downloaded: downloadedBytes,
              total: totalBytes,
              speed,
            });
            lastTime = now;
            lastBytes = downloadedBytes;
          }
        });

        res.on('aborted', () => { file.destroy(); fail(new Error('Download interrupted')); });
        res.on('error', (err) => { file.destroy(); fail(err); });
        file.on('error', (err) => { res.destroy(); fail(err); });
        file.on('finish', () => {
          if (settled) return;
          if (totalBytes && downloadedBytes !== totalBytes) {
            return fail(new Error('Download incomplete — please retry'));
          }
          try {
            fs.renameSync(partPath, dest);
            settled = true;
            resolve();
          } catch (err) { fail(err); }
        });
        res.pipe(file);
      });
      req.on('timeout', () => req.destroy(new Error('Connection timed out')));
      req.on('error', fail);
    }

    get(url, redirectsLeft);
  });
}

/** Cross-platform zip extraction without shell interpolation. */
function extractZip(zipPath, destDir) {
  ensureDir(destDir);
  const attempts = IS_WIN
    ? [
        // bsdtar ships with Windows 10 1803+ and handles .zip
        ['tar', ['-xf', zipPath, '-C', destDir]],
        ['powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
          'Expand-Archive -LiteralPath $env:RVC_ZIP -DestinationPath $env:RVC_DEST -Force']],
      ]
    : [
        ['/usr/bin/unzip', ['-o', '-q', zipPath, '-d', destDir]],
        ['/usr/bin/ditto', ['-x', '-k', zipPath, destDir]],
      ];

  return attempts.reduce((chain, [cmd, args]) => chain.catch(() => new Promise((resolve, reject) => {
    execFile(cmd, args, {
      windowsHide: true,
      maxBuffer: 16 * 1024 * 1024,
      env: { ...process.env, RVC_ZIP: zipPath, RVC_DEST: destDir },
    }, (err, _stdout, stderr) => {
      if (err) {
        console.error(`[electron] ${cmd} failed:`, stderr || err.message);
        reject(new Error(`Extraction failed: ${(stderr || err.message).toString().slice(0, 300)}`));
      } else resolve();
    });
  })), Promise.reject(new Error('start')));
}

async function runSetupDownloads() {
  ensureDir(ASSET_CACHE);
  ensureDir(MODELS_DIR);

  const hubertPath = path.join(ASSET_CACHE, 'hubert_base.pt');
  if (!fs.existsSync(hubertPath)) {
    sendToDownloader('download:status', 'Downloading base models (1/3)...');
    await downloadFileWithProgress(HUBERT_URL, hubertPath);
  }

  const rmvpePath = path.join(ASSET_CACHE, 'rmvpe.onnx');
  if (!fs.existsSync(rmvpePath)) {
    sendToDownloader('download:status', 'Downloading pitch models (2/3)...');
    await downloadFileWithProgress(RMVPE_URL, rmvpePath);
  }

  const zipDest = path.join(USER_ROOT, 'models.zip');
  sendToDownloader('download:status', 'Downloading voice models (3/3)...');
  await downloadFileWithProgress(MODELS_ZIP_URL, zipDest);

  sendToDownloader('download:status', 'Extracting voice models...');
  const staging = path.join(USER_ROOT, 'models-staging');
  fs.rmSync(staging, { recursive: true, force: true });
  try {
    await extractZip(zipDest, staging);

    // The archive contains models/applio/… — install it as models/default.
    const candidates = [
      path.join(staging, 'models', 'applio'),
      path.join(staging, 'applio'),
      path.join(staging, 'models', 'default'),
    ];
    const extracted = candidates.find((p) => fs.existsSync(p) && hasPth(p));
    if (!extracted) throw new Error('Voice model archive did not contain any .pth files');

    const defaultDir = path.join(MODELS_DIR, 'default');
    fs.rmSync(defaultDir, { recursive: true, force: true });
    try {
      fs.renameSync(extracted, defaultDir);
    } catch (_) {
      // rename across volumes is not possible; fall back to copy
      fs.cpSync(extracted, defaultDir, { recursive: true });
    }

    // Bring along any images shipped inside the archive
    const zipImages = path.join(staging, 'models', 'images');
    if (fs.existsSync(zipImages)) {
      fs.cpSync(zipImages, ensureDir(path.join(MODELS_DIR, 'images')), { recursive: true, force: false });
    }
  } finally {
    fs.rmSync(staging, { recursive: true, force: true });
    try { fs.unlinkSync(zipDest); } catch (_) {}
  }
}

async function launchAppFlow() {
  const port = await startPythonBackend();
  console.log(`[electron] Python backend ready on port ${port}`);
  createWindow(port);
  createTray();
}

function handleStartupError(err) {
  console.error('[electron] Failed to start backend:', err);
  const hint = app.isPackaged
    ? 'Try restarting the app. If the problem persists, reinstall RVC Voicechanger or open an issue on GitHub with the details above.'
    : 'Make sure Python 3.10–3.12 and all dependencies are installed:\n  pip install -r requirements.txt';
  dialog.showErrorBox(
    'RVC Voicechanger — Startup Error',
    `Could not start the audio backend:\n\n${err.message}\n\n${hint}`
  );
  isQuitting = true;
  app.quit();
}

// ── App lifecycle ─────────────────────────────────────────────────────────────
app.on('second-instance', showMainWindow);

app.whenReady().then(async () => {
  app.on('activate', showMainWindow);

  registerIPC();
  ensureDir(MODELS_DIR);
  seedBundledImages();

  if (IS_MAC && fs.existsSync(ICON_PATH)) {
    try { app.dock.setIcon(nativeImage.createFromPath(ICON_PATH)); }
    catch (err) { console.error('Failed to set dock icon:', err); }
  }

  if (areModelsMissing()) {
    showDownloaderWindow();
  } else {
    launchAppFlow().catch(handleStartupError);
  }
});

app.on('window-all-closed', () => {
  // The main window hides to the tray instead of closing, so this only fires
  // when the user really quits (or the downloader is dismissed).
  if (!IS_MAC || !mainWindow) {
    isQuitting = true;
    stopPythonBackend();
    app.quit();
  }
});

app.on('before-quit', () => {
  isQuitting = true;
  stopPythonBackend();
});
