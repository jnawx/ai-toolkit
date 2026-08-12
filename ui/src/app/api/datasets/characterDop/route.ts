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
const ACTIONS = new Set(['models', 'state', 'save-audio', 'detect', 'track', 'preview']);
const MAX_ANNOTATOR_OUTPUT_BYTES = 64 * 1024 * 1024;
const MAX_ANNOTATOR_ERROR_BYTES = 1024 * 1024;
const MAX_ANNOTATOR_INPUT_BYTES = 32 * 1024 * 1024;

class RequestTooLargeError extends Error {}

const isWithin = (root: string, target: string) => target === root || target.startsWith(root + path.sep);

async function resolveDatasetMedia(datasetName: unknown, rawMediaPath: unknown) {
  if (typeof datasetName !== 'string' || !datasetName.trim() || typeof rawMediaPath !== 'string') return null;
  const lexicalRoot = path.resolve(await getDatasetsRoot());
  const lexicalDataset = path.resolve(lexicalRoot, datasetName);
  const lexicalMedia = path.resolve(rawMediaPath);
  if (!isWithin(lexicalRoot, lexicalDataset) || !isWithin(lexicalDataset, lexicalMedia)) return null;
  try {
    const [datasetsRoot, datasetDir, mediaPath] = await Promise.all([
      fs.promises.realpath(lexicalRoot),
      fs.promises.realpath(lexicalDataset),
      fs.promises.realpath(lexicalMedia),
    ]);
    if (!isWithin(datasetsRoot, datasetDir) || !isWithin(datasetDir, mediaPath)) return null;
    const [datasetStat, mediaStat] = await Promise.all([
      fs.promises.stat(datasetDir),
      fs.promises.stat(mediaPath),
    ]);
    if (!datasetStat.isDirectory() || !mediaStat.isFile()) return null;
    return { datasetDir, mediaPath };
  } catch {
    return null;
  }
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

async function readBoundedJson(request: Request): Promise<any> {
  const contentLength = Number(request.headers.get('content-length') || 0);
  if (contentLength > MAX_ANNOTATOR_INPUT_BYTES) throw new RequestTooLargeError();
  if (!request.body) throw new SyntaxError('Missing JSON body');
  const reader = request.body.getReader();
  const chunks: Buffer[] = [];
  let totalBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    if (totalBytes > MAX_ANNOTATOR_INPUT_BYTES) {
      await reader.cancel();
      throw new RequestTooLargeError();
    }
    chunks.push(Buffer.from(value));
  }
  return JSON.parse(Buffer.concat(chunks, totalBytes).toString('utf-8'));
}

function runAnnotator(
  args: string[],
  signal: AbortSignal,
  payload?: Record<string, unknown>,
): Promise<{ ok: boolean; result: unknown; error?: string }> {
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
    let streamLimitError: string | null = null;
    let stdoutBytes = 0;
    let stderrBytes = 0;
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
    child.stdout.on('data', (chunk: Buffer) => {
      stdoutBytes += chunk.length;
      if (stdoutBytes > MAX_ANNOTATOR_OUTPUT_BYTES) {
        streamLimitError = 'Character annotation output exceeded 64 MB';
        if (!child.killed) child.kill('SIGKILL');
        return;
      }
      stdout += chunk.toString('utf-8');
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderrBytes += chunk.length;
      if (stderrBytes > MAX_ANNOTATOR_ERROR_BYTES) {
        streamLimitError = 'Character annotation error output exceeded 1 MB';
        if (!child.killed) child.kill('SIGKILL');
        return;
      }
      stderr += chunk.toString('utf-8');
    });
    child.on('error', error => finish({ ok: false, result: null, error: error.message }));
    child.on('close', code => {
      if (timedOut) {
        finish({ ok: false, result: null, error: 'Character tracking timed out after 20 minutes' });
        return;
      }
      if (streamLimitError) {
        finish({ ok: false, result: null, error: streamLimitError });
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
    if (payload) child.stdin.end(JSON.stringify(payload));
    else child.stdin.end();
  });
}

export async function POST(request: Request) {
  let body: any;
  try {
    body = await readBoundedJson(request);
  } catch (error) {
    if (error instanceof RequestTooLargeError) {
      return NextResponse.json({ error: 'Character annotation request exceeds 32 MB' }, { status: 413 });
    }
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  if (!ACTIONS.has(body?.action)) {
    return NextResponse.json({ error: 'Invalid Character DOP annotation action' }, { status: 400 });
  }
  if (body.action === 'models') {
    const outcome = await runAnnotator(['models'], request.signal);
    return outcome.ok
      ? NextResponse.json(outcome.result)
      : NextResponse.json({ error: outcome.error }, { status: 500 });
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
  let payload: Record<string, unknown> | undefined;
  if (body.action === 'save-audio') {
    args.push('--payload-stdin');
    payload = { intervals: body.intervals ?? [] };
  } else if (body.action === 'detect') {
    args.push('--payload-stdin');
    payload = {
      concept: body.concept ?? 'person',
      time_seconds: body.timeSeconds ?? 0,
      model_id: body.modelId,
    };
  } else if (body.action === 'track') {
    args.push('--payload-stdin');
    payload = {
      prompts: body.prompts ?? [],
      initial_masks: body.initialMasks ?? [],
      initial_time_seconds: body.initialTimeSeconds,
      model_id: body.modelId,
    };
  } else if (body.action === 'preview') {
    args.push('--frame-index', String(body.frameIndex ?? 0));
  }
  const outcome = await runAnnotator(args, request.signal, payload);
  if (!outcome.ok) {
    return NextResponse.json({ error: outcome.error }, { status: 500 });
  }
  return NextResponse.json(outcome.result);
}
