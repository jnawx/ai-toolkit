import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

import { TOOLKIT_ROOT } from '@/paths';
import { getDatasetsRoot } from '@/server/settings';
import { resolvePythonPath } from '../../../../../cron/pythonPath';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 1200;

const SCRIPT_PATH = path.join(TOOLKIT_ROOT, 'ui_scripts', 'character_dop_annotator.py');
const TIMEOUT_MS = 20 * 60 * 1000;
const ACTIONS = new Set(['state', 'save-audio', 'track', 'preview']);

const isWithin = (root: string, target: string) => target === root || target.startsWith(root + path.sep);

async function resolveDatasetMedia(datasetName: unknown, rawMediaPath: unknown) {
  if (typeof datasetName !== 'string' || !datasetName.trim() || typeof rawMediaPath !== 'string') return null;
  const datasetsRoot = path.resolve(await getDatasetsRoot());
  const datasetDir = path.resolve(datasetsRoot, datasetName);
  const mediaPath = path.resolve(rawMediaPath);
  if (!isWithin(datasetsRoot, datasetDir) || !isWithin(datasetDir, mediaPath)) return null;
  const stat = await fs.promises.stat(mediaPath).catch(() => null);
  if (!stat?.isFile()) return null;
  return { datasetDir, mediaPath };
}

function parseLastJson(stdout: string): unknown {
  const lines = stdout.trimEnd().split(/\r?\n/);
  for (let index = lines.length - 1; index >= 0; index--) {
    const line = lines[index].trim();
    if (!line) continue;
    try {
      return JSON.parse(line);
    } catch {
      continue;
    }
  }
  return null;
}

function runAnnotator(args: string[], signal: AbortSignal): Promise<{ ok: boolean; result: unknown; error?: string }> {
  return new Promise(resolve => {
    const child = spawn(resolvePythonPath(), ['-u', SCRIPT_PATH, ...args], {
      cwd: TOOLKIT_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
    });
    let stdout = '';
    let stderr = '';
    let timedOut = false;
    let settled = false;
    const finish = (value: { ok: boolean; result: unknown; error?: string }) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
      resolve(value);
    };
    const onAbort = () => {
      if (!child.killed) child.kill('SIGKILL');
      finish({ ok: false, result: null, error: 'Annotation request was cancelled' });
    };
    const timer = setTimeout(() => {
      timedOut = true;
      if (!child.killed) child.kill('SIGKILL');
    }, TIMEOUT_MS);
    signal.addEventListener('abort', onAbort, { once: true });
    child.stdout.on('data', (chunk: Buffer) => (stdout += chunk.toString('utf-8')));
    child.stderr.on('data', (chunk: Buffer) => (stderr += chunk.toString('utf-8')));
    child.on('error', error => finish({ ok: false, result: null, error: error.message }));
    child.on('close', code => {
      if (timedOut) {
        finish({ ok: false, result: null, error: 'Character tracking timed out after 20 minutes' });
        return;
      }
      const result = parseLastJson(stdout);
      if (code !== 0 || result == null) {
        const message = stderr.trim().split(/\r?\n/).pop() || 'Character annotation failed';
        finish({ ok: false, result: null, error: message });
        return;
      }
      finish({ ok: true, result });
    });
  });
}

export async function POST(request: Request) {
  let body: any;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  if (!ACTIONS.has(body?.action)) {
    return NextResponse.json({ error: 'Invalid Character DOP annotation action' }, { status: 400 });
  }
  const resolved = await resolveDatasetMedia(body?.datasetName, body?.mediaPath);
  if (!resolved) {
    return NextResponse.json({ error: 'Media must be a file inside the selected dataset' }, { status: 403 });
  }
  const args = [
    body.action,
    '--dataset-dir',
    resolved.datasetDir,
    '--media-path',
    resolved.mediaPath,
  ];
  if (body.action === 'save-audio') {
    args.push('--intervals', JSON.stringify(body.intervals ?? []));
  } else if (body.action === 'track') {
    args.push('--prompts', JSON.stringify(body.prompts ?? []));
  } else if (body.action === 'preview') {
    args.push('--frame-index', String(body.frameIndex ?? 0));
  }
  const outcome = await runAnnotator(args, request.signal);
  if (!outcome.ok) {
    return NextResponse.json({ error: outcome.error }, { status: 500 });
  }
  return NextResponse.json(outcome.result);
}
