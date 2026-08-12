'use client';

import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { AudioLines, Check, Loader2, MousePointer2, Pause, Play, RotateCcw, Save, ScanSearch, Trash2, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { apiClient } from '@/utils/api';
import { isAudio, isVideo } from '@/utils/basic';

type CharacterPoint = { x: number; y: number; label: 0 | 1 };
type CharacterPrompt = { time_seconds: number; points: CharacterPoint[] };
type SpeakingInterval = [number, number];
const MAX_WAVEFORM_SOURCE_BYTES = 32 * 1024 * 1024;

type AnnotationState = {
  root: string;
  visual: { exists: boolean; path: string; shape: number[] | null };
  audio: { exists: boolean; path: string; intervals: SpeakingInterval[] };
  prompts: CharacterPrompt[];
};

type Props = {
  open: boolean;
  datasetName: string;
  mediaPath: string;
  onClose: () => void;
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const formatTime = (seconds: number) => {
  if (!Number.isFinite(seconds)) return '0:00.000';
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds % 60).toFixed(3).padStart(6, '0')}`;
};

const mergeIntervals = (intervals: SpeakingInterval[]) => {
  const sorted = intervals
    .map(([start, end]) => [Math.min(start, end), Math.max(start, end)] as SpeakingInterval)
    .filter(([start, end]) => end - start > 0.001)
    .sort((a, b) => a[0] - b[0]);
  const merged: SpeakingInterval[] = [];
  for (const interval of sorted) {
    const previous = merged[merged.length - 1];
    if (previous && interval[0] <= previous[1] + 0.01) previous[1] = Math.max(previous[1], interval[1]);
    else merged.push([...interval]);
  }
  return merged;
};

function SpeakingTimeline({
  duration,
  currentTime,
  intervals,
  peaks,
  onSeek,
  onAdd,
}: {
  duration: number;
  currentTime: number;
  intervals: SpeakingInterval[];
  peaks: number[];
  onSeek: (time: number) => void;
  onAdd: (interval: SpeakingInterval) => void;
}) {
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const [dragStart, setDragStart] = useState<number | null>(null);
  const [dragEnd, setDragEnd] = useState<number | null>(null);
  const timeAt = (clientX: number) => {
    const rect = timelineRef.current?.getBoundingClientRect();
    if (!rect || duration <= 0) return 0;
    return clamp((clientX - rect.left) / rect.width, 0, 1) * duration;
  };
  return (
    <div>
      <div
        ref={timelineRef}
        className="relative h-28 overflow-hidden rounded border border-gray-700 bg-gray-900 cursor-crosshair touch-none select-none"
        onPointerDown={event => {
          event.currentTarget.setPointerCapture(event.pointerId);
          const time = timeAt(event.clientX);
          setDragStart(time);
          setDragEnd(time);
        }}
        onPointerMove={event => {
          if (dragStart != null) setDragEnd(timeAt(event.clientX));
        }}
        onPointerUp={event => {
          const end = timeAt(event.clientX);
          if (dragStart != null && Math.abs(end - dragStart) > 0.01) onAdd([dragStart, end]);
          else onSeek(end);
          setDragStart(null);
          setDragEnd(null);
        }}
      >
        <div className="absolute inset-0 flex items-center gap-px px-1 opacity-70 pointer-events-none">
          {(peaks.length ? peaks : new Array(120).fill(0.12)).map((peak, index) => (
            <div
              key={index}
              className="flex-1 rounded-sm bg-slate-400"
              style={{ height: `${Math.max(4, peak * 88)}%` }}
            />
          ))}
        </div>
        {duration > 0 &&
          intervals.map(([start, end], index) => (
            <div
              key={`${start}-${end}-${index}`}
              className="absolute inset-y-0 bg-emerald-400/35 border-x border-emerald-300 pointer-events-none"
              style={{ left: `${(start / duration) * 100}%`, width: `${((end - start) / duration) * 100}%` }}
            />
          ))}
        {duration > 0 && dragStart != null && dragEnd != null && (
          <div
            className="absolute inset-y-0 bg-blue-400/35 border-x border-blue-300 pointer-events-none"
            style={{
              left: `${(Math.min(dragStart, dragEnd) / duration) * 100}%`,
              width: `${(Math.abs(dragEnd - dragStart) / duration) * 100}%`,
            }}
          />
        )}
        {duration > 0 && (
          <div
            className="absolute inset-y-0 w-0.5 bg-amber-300 pointer-events-none"
            style={{ left: `${(currentTime / duration) * 100}%` }}
          />
        )}
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-gray-500">
        <span>0:00</span>
        <span>Drag across the target character&apos;s speech; click to seek</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}

export default function CharacterDOPAnnotator({ open, datasetName, mediaPath, onClose }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [state, setState] = useState<AnnotationState | null>(null);
  const [prompts, setPrompts] = useState<CharacterPrompt[]>([]);
  const [intervals, setIntervals] = useState<SpeakingInterval[]>([]);
  const [pointLabel, setPointLabel] = useState<0 | 1>(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [preview, setPreview] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(true);
  const [peaks, setPeaks] = useState<number[]>([]);
  const [markStart, setMarkStart] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [busy, setBusy] = useState<'load' | 'track' | 'audio' | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const videoItem = isVideo(mediaPath);
  const audioItem = isAudio(mediaPath);
  const hasVisual = !audioItem;
  const hasTimeline = videoItem || audioItem;
  const mediaSrc = `/api/img/${encodeURIComponent(mediaPath)}`;

  const request = useCallback(
    async (action: string, extra: Record<string, unknown> = {}) => {
      const response = await apiClient.post('/api/datasets/characterDop', {
        action,
        datasetName,
        mediaPath,
        ...extra,
      });
      return response.data;
    },
    [datasetName, mediaPath],
  );

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setBusy('load');
    setError(null);
    setMessage(null);
    setPreview(null);
    setCurrentTime(0);
    setDuration(0);
    setPeaks([]);
    request('state')
      .then((nextState: AnnotationState) => {
        if (cancelled) return;
        setState(nextState);
        setPrompts(nextState.prompts ?? []);
        setIntervals(nextState.audio?.intervals ?? []);
      })
      .catch(reason => !cancelled && setError(reason?.response?.data?.error || reason.message))
      .finally(() => !cancelled && setBusy(null));
    return () => {
      cancelled = true;
    };
  }, [open, request]);

  useEffect(() => {
    if (!open || !hasTimeline) return;
    let cancelled = false;
    let context: AudioContext | null = null;
    const controller = new AbortController();
    fetch(mediaSrc, { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error(`Waveform source returned ${response.status}`);
        const contentLength = Number(response.headers.get('content-length') || 0);
        if (!contentLength || contentLength > MAX_WAVEFORM_SOURCE_BYTES) {
          void response.body?.cancel();
          return null;
        }
        return response.arrayBuffer();
      })
      .then(async bytes => {
        if (bytes == null) return;
        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        if (!AudioContextClass) return;
        context = new AudioContextClass();
        const buffer = await context.decodeAudioData(bytes.slice(0));
        const channel = buffer.getChannelData(0);
        const bins = 180;
        const stride = Math.max(1, Math.floor(channel.length / bins));
        const values = new Array(bins).fill(0).map((_, index) => {
          let peak = 0;
          const end = Math.min(channel.length, (index + 1) * stride);
          for (let sample = index * stride; sample < end; sample++) peak = Math.max(peak, Math.abs(channel[sample]));
          return peak;
        });
        if (!cancelled) setPeaks(values);
      })
      .catch(() => undefined)
      .finally(() => context?.close());
    return () => {
      cancelled = true;
      controller.abort();
      void context?.close();
    };
  }, [open, hasTimeline, mediaSrc]);

  const frameIndex = useMemo(() => {
    const count = state?.visual?.shape?.[0] ?? 1;
    if (!videoItem || duration <= 0) return 0;
    return clamp(Math.round((currentTime / duration) * (count - 1)), 0, count - 1);
  }, [state?.visual?.shape, videoItem, duration, currentTime]);

  useEffect(() => {
    if (!open || !state?.visual?.exists || !showPreview) {
      setPreview(null);
      return;
    }
    const timer = setTimeout(() => {
      request('preview', { frameIndex })
        .then(data => setPreview(data.data_url))
        .catch(() => setPreview(null));
    }, 120);
    return () => clearTimeout(timer);
  }, [open, state?.visual?.exists, showPreview, frameIndex, request]);

  const activePromptIndex = useMemo(() => {
    if (!prompts.length) return -1;
    let best = -1;
    let distance = Number.POSITIVE_INFINITY;
    prompts.forEach((prompt, index) => {
      const nextDistance = Math.abs(prompt.time_seconds - currentTime);
      if (nextDistance < distance) {
        best = index;
        distance = nextDistance;
      }
    });
    return distance <= 0.08 || !videoItem ? best : -1;
  }, [prompts, currentTime, videoItem]);
  const activePoints = activePromptIndex >= 0 ? prompts[activePromptIndex].points : [];

  const addPoint = (event: React.PointerEvent<HTMLDivElement>) => {
    if (busy) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point: CharacterPoint = {
      x: clamp((event.clientX - rect.left) / rect.width, 0, 1),
      y: clamp((event.clientY - rect.top) / rect.height, 0, 1),
      label: pointLabel,
    };
    setPrompts(previous => {
      const next = previous.map(prompt => ({ ...prompt, points: [...prompt.points] }));
      const matching = next.findIndex(prompt => Math.abs(prompt.time_seconds - currentTime) <= 0.08);
      if (matching >= 0) next[matching].points.push(point);
      else next.push({ time_seconds: Number(currentTime.toFixed(4)), points: [point] });
      return next.sort((a, b) => a.time_seconds - b.time_seconds);
    });
  };

  const setMediaTime = (time: number) => {
    const next = clamp(time, 0, duration || 0);
    if (videoRef.current) videoRef.current.currentTime = next;
    if (audioRef.current) audioRef.current.currentTime = next;
    setCurrentTime(next);
  };

  const trackCharacter = async () => {
    setBusy('track');
    setError(null);
    setMessage('SAM2 is preparing and tracking the target character through the source media. The first run downloads the model.');
    try {
      const nextState: AnnotationState = await request('track', { prompts });
      setState(nextState);
      setPrompts(nextState.prompts);
      setMessage(`Visual character mask saved across ${nextState.visual.shape?.[0] ?? 1} frame(s).`);
    } catch (reason: any) {
      setError(reason?.response?.data?.error || reason.message || 'Character tracking failed');
    } finally {
      setBusy(null);
    }
  };

  const saveIntervals = async () => {
    setBusy('audio');
    setError(null);
    try {
      const normalized = mergeIntervals(intervals);
      const nextState: AnnotationState = await request('save-audio', { intervals: normalized });
      setState(nextState);
      setIntervals(nextState.audio.intervals);
      setMessage(`Saved ${nextState.audio.intervals.length} target-character speaking interval(s).`);
    } catch (reason: any) {
      setError(reason?.response?.data?.error || reason.message || 'Speaking intervals could not be saved');
    } finally {
      setBusy(null);
    }
  };

  return (
    <Dialog open={open} onClose={() => !busy && onClose()} className="relative z-[70]">
      <DialogBackdrop className="fixed inset-0 bg-black/80" />
      <div className="fixed inset-0 flex items-center justify-center p-2 sm:p-5">
        <DialogPanel className="flex max-h-[96vh] w-full max-w-7xl flex-col overflow-hidden rounded-xl border border-gray-700 bg-gray-950 shadow-2xl">
          <div className="flex items-center gap-3 border-b border-gray-800 px-4 py-3">
            <ScanSearch className="text-violet-400" />
            <div className="min-w-0 flex-1">
              <h2 className="font-semibold text-gray-100">Character DOP Annotator</h2>
              <p className="truncate text-xs text-gray-500">{mediaPath.split(/[\\/]/).pop()}</p>
            </div>
            <button className="rounded p-1 text-gray-400 hover:bg-gray-800 hover:text-white" onClick={onClose} disabled={Boolean(busy)}>
              <X />
            </button>
          </div>

          <div className="grid min-h-0 flex-1 grid-cols-1 overflow-y-auto lg:grid-cols-[minmax(0,1fr)_25rem]">
            <div className="flex min-h-[24rem] items-center justify-center bg-black p-3">
              {busy === 'load' ? (
                <Loader2 className="h-10 w-10 animate-spin text-violet-400" />
              ) : audioItem ? (
                <audio
                  ref={audioRef}
                  src={mediaSrc}
                  controls
                  className="w-full max-w-2xl"
                  onLoadedMetadata={event => setDuration(event.currentTarget.duration || 0)}
                  onTimeUpdate={event => setCurrentTime(event.currentTarget.currentTime)}
                />
              ) : (
                <div className="relative inline-block max-h-[72vh] max-w-full overflow-hidden">
                  {videoItem ? (
                    <video
                      ref={videoRef}
                      src={mediaSrc}
                      className="block max-h-[72vh] max-w-full"
                      playsInline
                      onLoadedMetadata={event => {
                        setDuration(event.currentTarget.duration || 0);
                        event.currentTarget.pause();
                      }}
                      onTimeUpdate={event => setCurrentTime(event.currentTarget.currentTime)}
                      onPlay={() => setIsPlaying(true)}
                      onPause={() => setIsPlaying(false)}
                    />
                  ) : (
                    <img src={mediaSrc} alt="Character annotation source" className="block max-h-[72vh] max-w-full" draggable={false} />
                  )}
                  {preview && showPreview && (
                    <img src={preview} alt="Character mask preview" className="pointer-events-none absolute inset-0 h-full w-full opacity-45" />
                  )}
                  <div className="absolute inset-0 cursor-crosshair" onPointerDown={addPoint}>
                    {activePoints.map((point, index) => (
                      <span
                        key={`${point.x}-${point.y}-${index}`}
                        className={`absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow ${
                          point.label ? 'bg-emerald-500' : 'bg-red-500'
                        }`}
                        style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>

            <aside className="space-y-5 overflow-y-auto border-l border-gray-800 p-4 text-sm">
              {error && <div className="rounded border border-red-700 bg-red-950/60 p-3 text-red-200">{error}</div>}
              {message && <div className="rounded border border-blue-800 bg-blue-950/40 p-3 text-blue-200">{message}</div>}

              {hasVisual && (
                <section className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="flex items-center gap-2 font-medium text-gray-100"><MousePointer2 size={17} /> Visual identity</h3>
                    {state?.visual.exists && <span className="flex items-center gap-1 text-xs text-emerald-400"><Check size={14} /> Mask ready</span>}
                  </div>
                  <p className="text-xs leading-relaxed text-gray-400">
                    Click the target character in green. Click other people or background in red. Add corrections on later frames, then run SAM2.
                  </p>
                  {videoItem && duration > 0 && (
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <button
                          className="rounded bg-gray-800 p-1.5 hover:bg-gray-700"
                          onClick={() => {
                            const video = videoRef.current;
                            if (!video) return;
                            if (video.paused) void video.play();
                            else video.pause();
                          }}
                        >
                          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
                        </button>
                        <input
                          className="w-full accent-violet-500"
                          type="range"
                          min={0}
                          max={duration}
                          step={0.001}
                          value={currentTime}
                          onChange={event => setMediaTime(Number(event.target.value))}
                        />
                      </div>
                      <div className="flex justify-between text-xs text-gray-500"><span>{formatTime(currentTime)}</span><span>{formatTime(duration)}</span></div>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => setPointLabel(1)}
                      className={`rounded border px-3 py-2 ${pointLabel === 1 ? 'border-emerald-400 bg-emerald-900/50 text-emerald-100' : 'border-gray-700 bg-gray-900 text-gray-400'}`}
                    >
                      Include target
                    </button>
                    <button
                      onClick={() => setPointLabel(0)}
                      className={`rounded border px-3 py-2 ${pointLabel === 0 ? 'border-red-400 bg-red-900/50 text-red-100' : 'border-gray-700 bg-gray-900 text-gray-400'}`}
                    >
                      Exclude others
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      className="flex items-center gap-1 rounded bg-gray-800 px-2 py-1.5 text-xs hover:bg-gray-700"
                      onClick={() => {
                        if (activePromptIndex < 0) return;
                        setPrompts(previous => previous.filter((_, index) => index !== activePromptIndex));
                      }}
                    ><RotateCcw size={14} /> Clear this frame</button>
                    {state?.visual.exists && (
                      <button className="rounded bg-gray-800 px-2 py-1.5 text-xs hover:bg-gray-700" onClick={() => setShowPreview(value => !value)}>
                        {showPreview ? 'Hide mask' : 'Show mask'}
                      </button>
                    )}
                  </div>
                  <button
                    className="flex w-full items-center justify-center gap-2 rounded bg-violet-600 px-3 py-2 font-medium text-white hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={Boolean(busy) || prompts.length === 0}
                    onClick={trackCharacter}
                  >
                    {busy === 'track' ? <Loader2 size={17} className="animate-spin" /> : <ScanSearch size={17} />}
                    {state?.visual.exists ? 'Regenerate mask with SAM2' : 'Generate mask with SAM2'}
                  </button>
                  <p className="text-[11px] text-gray-500">Green permits character learning. Everything outside the generated mask is protected from identity bleed.</p>
                </section>
              )}

              {hasTimeline && (
                <section className="space-y-3 border-t border-gray-800 pt-4">
                  <div className="flex items-center justify-between">
                    <h3 className="flex items-center gap-2 font-medium text-gray-100"><AudioLines size={17} /> Target character speaking</h3>
                    {state?.audio.exists && <span className="flex items-center gap-1 text-xs text-emerald-400"><Check size={14} /> Saved</span>}
                  </div>
                  <SpeakingTimeline
                    duration={duration}
                    currentTime={currentTime}
                    intervals={intervals}
                    peaks={peaks}
                    onSeek={setMediaTime}
                    onAdd={interval => setIntervals(previous => mergeIntervals([...previous, interval]))}
                  />
                  <div className="flex gap-2">
                    <button
                      className="flex-1 rounded bg-gray-800 px-2 py-1.5 text-xs hover:bg-gray-700"
                      onClick={() => setMarkStart(currentTime)}
                    >
                      [ Mark start {markStart == null ? '' : formatTime(markStart)}
                    </button>
                    <button
                      className="flex-1 rounded bg-gray-800 px-2 py-1.5 text-xs hover:bg-gray-700 disabled:opacity-40"
                      disabled={markStart == null}
                      onClick={() => {
                        if (markStart == null) return;
                        setIntervals(previous => mergeIntervals([...previous, [markStart, currentTime]]));
                        setMarkStart(null);
                      }}
                    >
                      ] Mark end
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      className="rounded border border-gray-700 px-2 py-1.5 text-xs text-gray-300 hover:bg-gray-800"
                      disabled={duration <= 0}
                      onClick={() => setIntervals([[0, duration]])}
                    >
                      Target speaks throughout
                    </button>
                    <button
                      className="rounded border border-gray-700 px-2 py-1.5 text-xs text-gray-300 hover:bg-gray-800"
                      onClick={() => setIntervals([])}
                    >
                      Target never speaks
                    </button>
                  </div>
                  <div className="max-h-40 space-y-1 overflow-y-auto">
                    {intervals.length === 0 && <p className="rounded bg-gray-900 p-2 text-xs text-gray-500">No target-character speech marked. Saving this explicitly protects the complete soundtrack.</p>}
                    {intervals.map(([start, end], index) => (
                      <div key={`${start}-${end}-${index}`} className="flex items-center gap-2 rounded bg-gray-900 p-2 text-xs">
                        <button className="text-left text-gray-300 hover:text-white" onClick={() => setMediaTime(start)}>
                          {formatTime(start)} to {formatTime(end)}
                        </button>
                        <span className="ml-auto text-gray-500">{(end - start).toFixed(2)}s</span>
                        <button className="text-gray-500 hover:text-red-400" onClick={() => setIntervals(previous => previous.filter((_, item) => item !== index))}><Trash2 size={14} /></button>
                      </div>
                    ))}
                  </div>
                  <button
                    className="flex w-full items-center justify-center gap-2 rounded bg-emerald-700 px-3 py-2 font-medium text-white hover:bg-emerald-600 disabled:opacity-50"
                    disabled={Boolean(busy) || duration <= 0}
                    onClick={saveIntervals}
                  >
                    {busy === 'audio' ? <Loader2 size={17} className="animate-spin" /> : <Save size={17} />} Save speaking intervals
                  </button>
                </section>
              )}
            </aside>
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  );
}
