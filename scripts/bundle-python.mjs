#!/usr/bin/env node
/**
 * bundle-python.mjs — build a self-contained, relocatable Python runtime with
 * all backend dependencies into ./build/python, which electron-builder ships
 * as <resources>/python (see package.json → build.extraResources).
 *
 * Why: a virtualenv's interpreter is a symlink to an absolute path on the
 * build machine, so shipping .venv only works on the developer's own computer.
 * python-build-standalone (installed via `uv`) is fully relocatable.
 *
 * Requires: uv  (https://docs.astral.sh/uv/)
 * Usage:    node scripts/bundle-python.mjs [--python 3.11]
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'build', 'python');
const STAGE = path.join(ROOT, 'build', 'python-install');
const IS_WIN = process.platform === 'win32';
const argIdx = process.argv.indexOf('--python');
const PY_VERSION = argIdx > 0 ? process.argv[argIdx + 1] : '3.11';

function run(cmd, args, opts = {}) {
  console.log(`$ ${cmd} ${args.join(' ')}`);
  execFileSync(cmd, args, { stdio: 'inherit', cwd: ROOT, ...opts });
}

fs.rmSync(OUT, { recursive: true, force: true });
fs.rmSync(STAGE, { recursive: true, force: true });
fs.mkdirSync(STAGE, { recursive: true });

// 1. Install a standalone CPython into a staging dir
run('uv', ['python', 'install', PY_VERSION, '--install-dir', STAGE, '--no-bin']);
// uv also creates minor-version alias symlinks (cpython-3.11-…); take the real dir.
const installed = fs.readdirSync(STAGE, { withFileTypes: true })
  .find((d) => d.name.startsWith('cpython-') && d.isDirectory() && !d.isSymbolicLink());
if (!installed) throw new Error('uv did not install a CPython build');
fs.renameSync(path.join(STAGE, installed.name), OUT);
fs.rmSync(STAGE, { recursive: true, force: true });

const python = IS_WIN ? path.join(OUT, 'python.exe') : path.join(OUT, 'bin', 'python3');
if (!fs.existsSync(python)) throw new Error(`Interpreter not found at ${python}`);

const stdlib = IS_WIN ? path.join(OUT, 'Lib') : path.join(OUT, 'lib', `python${PY_VERSION}`);

// Standalone builds are marked EXTERNALLY-MANAGED; we own this copy.
fs.rmSync(path.join(stdlib, 'EXTERNALLY-MANAGED'), { force: true });

// 2. Install backend dependencies into it
run('uv', ['pip', 'install', '--python', python, '--no-cache', '--compile-bytecode',
  '-r', path.join(ROOT, 'requirements.txt')]);

// 3. Trim stdlib parts the app never needs at runtime
for (const d of ['test', 'idlelib', 'tkinter', 'turtledemo', 'ensurepip', 'lib2to3']) {
  fs.rmSync(path.join(stdlib, d), { recursive: true, force: true });
}
fs.rmSync(path.join(OUT, 'include'), { recursive: true, force: true });
fs.rmSync(path.join(OUT, 'share'), { recursive: true, force: true });

// 4. Smoke-test the critical native imports from the final tree
run(python, ['-c', [
  'import torch, numpy, faiss, onnxruntime, sounddevice, pedalboard, librosa, torchfcpe, fastapi, uvicorn, multipart',
  'print("bundle OK — torch", torch.__version__, "numpy", numpy.__version__)',
].join('; ')]);
console.log(`\nPython runtime ready at ${OUT}`);
