'use client';

import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { AudioLines, Check, Loader2, MousePointer2, Pause, Play, RotateCcw, Save, ScanSearch, Trash2, UserRoundSearch, Users, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { apiClient } from '@/utils/api';
import { isAudio, isVideo } from '@/utils/basic';

type CharacterPoint = { x: number; y: number; label: 0 | 1 };
type CharacterPrompt = { time_seconds: number; points: CharacterPoint[] };
type SpeakingInterval = [number, number];
const MAX_WAVEFORM_SOURCE_BYTES = 32 * 1024 * 1024;

type CharacterIdentity = {
  id: string;
  display_name: string;
  trigger_word: string;
  class_prompt: string;
};

type AnnotationState = {
  root: string;
  identity: CharacterIdentity | null;
  identities: CharacterIdentity[];
  visual: { exists: boolean; path: string; shape: number[] | null };
  audio: { exists: boolean; path: string; intervals: SpeakingInterval[] };
  prompts: CharacterPrompt[];
};

type MaskModel = {
  id: string;
  label: string;
  description: string;
  gated?: boolean;
};

type MaskModelCatalog = {
  trackers: MaskModel[];
  detectors: MaskModel[];
  defaults: { tracker: string; detector: string; concept: string };
};

type MaskCandidate = {
  id: number;
  score: number;
  box: [number, number, number, number];
  area: number;
  mask_data_url: string;
};

type DetectionResult = {
  concept: string;
  model_id: string;
  time_seconds: number;
  width: number;
  height: number;
  candidates: MaskCandidate[];
};

const DEFAULT_TRACKER_MODEL = 'facebook/sam2.1-hiera-tiny';
const DEFAULT_DETECTOR_MODEL = 'facebook/sam3';
const CHARACTER_DOP_TRACKER_STORAGE_KEY = 'ai-toolkit.character-dop.sam2-tracker-model';
const CHARACTER_DOP_LEGACY_IDENTITY = '__legacy__';
const characterIdentityStorageKey = (datasetName: string) => `ai-toolkit.character-dop.identity.${datasetName}`;

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
  const identitySelectionInitializedRef = useRef(false);
  const [state, setState] = useState<AnnotationState | null>(null);
  const [activeIdentityId, setActiveIdentityId] = useState<string | null>(null);
  const [newIdentityName, setNewIdentityName] = useState('');
  const [newIdentityTrigger, setNewIdentityTrigger] = useState('');
  const [newIdentityClass, setNewIdentityClass] = useState('a person');
  const [editIdentityName, setEditIdentityName] = useState('');
  const [editIdentityTrigger, setEditIdentityTrigger] = useState('');
  const [editIdentityClass, setEditIdentityClass] = useState('');
  const [prompts, setPrompts] = useState<CharacterPrompt[]>([]);
  const [intervals, setIntervals] = useState<SpeakingInterval[]>([]);
  const [pointLabel, setPointLabel] = useState<0 | 1>(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [preview, setPreview] = useState<string | null>(null);
  const [previewRevision, setPreviewRevision] = useState(0);
  const [showPreview, setShowPreview] = useState(true);
  const [modelCatalog, setModelCatalog] = useState<MaskModelCatalog | null>(null);
  const [trackerModel, setTrackerModel] = useState(DEFAULT_TRACKER_MODEL);
  const [detectorModel, setDetectorModel] = useState(DEFAULT_DETECTOR_MODEL);
  const [concept, setConcept] = useState('person');
  const [detection, setDetection] = useState<DetectionResult | null>(null);
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<number[]>([]);
  const [peaks, setPeaks] = useState<number[]>([]);
  const [markStart, setMarkStart] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [busy, setBusy] = useState<'load' | 'identity' | 'detect' | 'track' | 'audio' | null>(null);
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
    identitySelectionInitializedRef.current = false;
    setActiveIdentityId(null);
    setCurrentTime(0);
    setDuration(0);
    setPeaks([]);
  }, [open, mediaPath]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setBusy('load');
    setError(null);
    setMessage(null);
    setPreview(null);
    setDetection(null);
    setSelectedCandidateIds([]);
    Promise.all([
      request('state', { identityId: activeIdentityId }),
      request('models'),
    ])
      .then(([nextState, catalog]: [AnnotationState, MaskModelCatalog]) => {
        if (cancelled) return;
        setState(nextState);
        setPrompts(nextState.prompts ?? []);
        setIntervals(nextState.audio?.intervals ?? []);
        if (!identitySelectionInitializedRef.current) {
          identitySelectionInitializedRef.current = true;
          if (activeIdentityId == null && nextState.identities.length) {
            let preferredIdentityId: string | null = nextState.identities[0].id;
            try {
              const storedIdentityId = window.localStorage.getItem(characterIdentityStorageKey(datasetName));
              if (storedIdentityId === CHARACTER_DOP_LEGACY_IDENTITY) {
                preferredIdentityId = null;
              } else if (storedIdentityId && nextState.identities.some(identity => identity.id === storedIdentityId)) {
                preferredIdentityId = storedIdentityId;
              }
            } catch {
              // Fall back to the first configured identity when storage is unavailable.
            }
            setActiveIdentityId(preferredIdentityId);
          }
        }
        setModelCatalog(catalog);
        let preferredTrackerModel = catalog.defaults.tracker;
        try {
          const storedTrackerModel = window.localStorage.getItem(CHARACTER_DOP_TRACKER_STORAGE_KEY);
          if (storedTrackerModel && catalog.trackers.some(model => model.id === storedTrackerModel)) {
            preferredTrackerModel = storedTrackerModel;
          }
        } catch {
          // Storage may be unavailable in privacy-restricted browser contexts.
        }
        setTrackerModel(preferredTrackerModel);
        setDetectorModel(catalog.defaults.detector);
        setConcept(catalog.defaults.concept);
      })
      .catch(reason => !cancelled && setError(reason?.response?.data?.error || reason.message))
      .finally(() => !cancelled && setBusy(null));
    return () => {
      cancelled = true;
    };
  }, [activeIdentityId, datasetName, open, request]);

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

  useEffect(() => {
    setEditIdentityName(state?.identity?.display_name ?? '');
    setEditIdentityTrigger(state?.identity?.trigger_word ?? '');
    setEditIdentityClass(state?.identity?.class_prompt ?? '');
  }, [
    state?.identity?.id,
    state?.identity?.display_name,
    state?.identity?.trigger_word,
    state?.identity?.class_prompt,
  ]);

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
    let cancelled = false;
    const timer = setTimeout(() => {
      request('preview', { identityId: activeIdentityId, frameIndex })
        .then(data => {
          if (!cancelled) setPreview(data.data_url);
        })
        .catch(() => {
          if (!cancelled) setPreview(null);
        });
    }, 120);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [activeIdentityId, open, state?.visual?.exists, showPreview, frameIndex, previewRevision, request]);

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
    if (detectionIsOnCurrentFrame) {
      setMessage('Use the numbered person toggles on the auto-mask frame. Add manual corrections on other video frames, or clear the auto-mask to use points here.');
      return;
    }
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

  const selectedCandidates = useMemo(
    () => detection?.candidates.filter(candidate => selectedCandidateIds.includes(candidate.id)) ?? [],
    [detection, selectedCandidateIds],
  );
  const selectedCandidateSet = useMemo(() => new Set(selectedCandidateIds), [selectedCandidateIds]);
  const selectedTracker = modelCatalog?.trackers.find(model => model.id === trackerModel);
  const detectionIsOnCurrentFrame = Boolean(
    detection && (!videoItem || Math.abs(detection.time_seconds - currentTime) <= 0.08),
  );

  const toggleCandidate = (candidateId: number) => {
    setSelectedCandidateIds(previous =>
      previous.includes(candidateId)
        ? previous.filter(id => id !== candidateId)
        : [...previous, candidateId],
    );
  };

  const autoMaskPeople = async () => {
    const seedTime = currentTime;
    const seedPointCount = prompts
      .filter(prompt => !videoItem || Math.abs(prompt.time_seconds - seedTime) <= 0.08)
      .reduce((total, prompt) => total + prompt.points.length, 0);
    setBusy('detect');
    setError(null);
    setMessage('SAM 3 is finding every matching person in this frame. Its first run downloads the gated model.');
    videoRef.current?.pause();
    try {
      const result: DetectionResult = await request('detect', {
        concept: concept.trim() || 'person',
        timeSeconds: seedTime,
        modelId: detectorModel,
      });
      setDetection(result);
      setSelectedCandidateIds(result.candidates.map(candidate => candidate.id));
      if (result.candidates.length) {
        setPrompts(previous =>
          previous.filter(prompt => videoItem && Math.abs(prompt.time_seconds - seedTime) > 0.08),
        );
      }
      setShowPreview(false);
      setMessage(
        result.candidates.length
          ? `Found ${result.candidates.length} ${result.concept} instance(s). They all start included; exclude everyone except the training character.${seedPointCount ? ` Cleared ${seedPointCount} seed-frame point(s) because the auto-mask replaces them.` : ''}`
          : `SAM 3 did not find any instances matching “${result.concept}” on this frame.`,
      );
    } catch (reason: any) {
      setError(reason?.response?.data?.error || reason.message || 'Automatic person masking failed');
    } finally {
      setBusy(null);
    }
  };

  const trackCharacter = async () => {
    setBusy('track');
    setError(null);
    setMessage(`${selectedTracker?.label ?? 'SAM 2'} is tracking the selected character through the source media. Its first run downloads the model.`);
    try {
      const nextState: AnnotationState = await request('track', {
        identityId: activeIdentityId,
        prompts,
        initialMasks: selectedCandidates.map(candidate => candidate.mask_data_url),
        initialTimeSeconds: selectedCandidates.length ? detection?.time_seconds : undefined,
        modelId: trackerModel,
      });
      setState(nextState);
      setPrompts(nextState.prompts);
      setDetection(null);
      setSelectedCandidateIds([]);
      setPreviewRevision(previous => previous + 1);
      setShowPreview(true);
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
      const nextState: AnnotationState = await request('save-audio', {
        identityId: activeIdentityId,
        intervals: normalized,
      });
      setState(nextState);
      setIntervals(nextState.audio.intervals);
      setMessage(`Saved ${nextState.audio.intervals.length} target-character speaking interval(s).`);
    } catch (reason: any) {
      setError(reason?.response?.data?.error || reason.message || 'Speaking intervals could not be saved');
    } finally {
      setBusy(null);
    }
  };

  const createIdentity = async () => {
    const triggerWord = newIdentityTrigger.trim();
    const identitySlug = triggerWord
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 48);
    const identitySuffix = globalThis.crypto?.randomUUID
      ? globalThis.crypto.randomUUID().slice(0, 8)
      : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`.slice(0, 8);
    const identityId = identitySlug ? `${identitySlug}-${identitySuffix}` : '';
    if (!identityId || !newIdentityName.trim() || !newIdentityClass.trim()) {
      setError('Display name, trigger word, and generic class prompt are required.');
      return;
    }
    setBusy('identity');
    setError(null);
    try {
      const nextState: AnnotationState = await request('save-identity', {
        identityId,
        displayName: newIdentityName.trim(),
        triggerWord,
        classPrompt: newIdentityClass.trim(),
      });
      setActiveIdentityId(identityId);
      identitySelectionInitializedRef.current = true;
      try {
        window.localStorage.setItem(characterIdentityStorageKey(datasetName), identityId);
      } catch {
        // The identity remains selected for this session when storage is unavailable.
      }
      setState(nextState);
      setPrompts(nextState.prompts ?? []);
      setIntervals(nextState.audio?.intervals ?? []);
      setNewIdentityName('');
      setNewIdentityTrigger('');
      setMessage(`${nextState.identity?.display_name ?? triggerWord} is ready to annotate on this media.`);
    } catch (reason: any) {
      setError(reason?.response?.data?.error || reason.message || 'Character identity could not be saved');
    } finally {
      setBusy(null);
    }
  };

  const updateIdentity = async () => {
    if (!state?.identity || !editIdentityName.trim() || !editIdentityTrigger.trim() || !editIdentityClass.trim()) {
      setError('Display name, trigger word, and generic class prompt are required.');
      return;
    }
    const previousTrigger = state.identity.trigger_word;
    setBusy('identity');
    setError(null);
    try {
      const nextState: AnnotationState = await request('update-identity', {
        identityId: state.identity.id,
        displayName: editIdentityName.trim(),
        triggerWord: editIdentityTrigger.trim(),
        classPrompt: editIdentityClass.trim(),
      });
      setState(nextState);
      setMessage(
        previousTrigger === nextState.identity?.trigger_word
          ? `Updated ${nextState.identity?.display_name}. Existing masks and speaking intervals were preserved.`
          : `Updated ${nextState.identity?.display_name}. Existing masks and speaking intervals were preserved. Update dataset captions that still use “${previousTrigger}”.`,
      );
    } catch (reason: any) {
      setError(reason?.response?.data?.error || reason.message || 'Character identity could not be updated');
    } finally {
      setBusy(null);
    }
  };

  const deleteIdentity = async () => {
    if (!state?.identity) return;
    const identity = state.identity;
    const confirmed = window.confirm(
      `Delete ${identity.display_name}? This permanently deletes its masks, speaking intervals, and saved prompts across the entire dataset. This cannot be undone.`,
    );
    if (!confirmed) return;
    setBusy('identity');
    setError(null);
    try {
      const nextState: AnnotationState & {
        deleted_identity: CharacterIdentity;
        cleanup_pending: boolean;
      } = await request('delete-identity', {
        identityId: identity.id,
      });
      const fallbackIdentityId = nextState.identity?.id ?? null;
      identitySelectionInitializedRef.current = true;
      setActiveIdentityId(fallbackIdentityId);
      try {
        window.localStorage.setItem(
          characterIdentityStorageKey(datasetName),
          fallbackIdentityId ?? CHARACTER_DOP_LEGACY_IDENTITY,
        );
      } catch {
        // The fallback remains selected for this session when storage is unavailable.
      }
      setState(nextState);
      setPrompts(nextState.prompts ?? []);
      setIntervals(nextState.audio?.intervals ?? []);
      setPreview(null);
      setDetection(null);
      setSelectedCandidateIds([]);
      setMessage(
        nextState.cleanup_pending
          ? `Deleted ${nextState.deleted_identity.display_name}. Its annotations are no longer usable, but some staged files could not be removed; restart the toolkit and remove the dataset's _character_dop/.deleted-identities folder if they remain.`
          : `Deleted ${nextState.deleted_identity.display_name} and its Character DOP annotations.`,
      );
    } catch (reason: any) {
      setError(reason?.response?.data?.error || reason.message || 'Character identity could not be deleted');
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
                  {detectionIsOnCurrentFrame && !showPreview && selectedCandidates.map(candidate => (
                    <img
                      key={`candidate-mask-${candidate.id}`}
                      src={candidate.mask_data_url}
                      alt=""
                      className="pointer-events-none absolute inset-0 h-full w-full opacity-40"
                      style={{ mixBlendMode: 'screen' }}
                    />
                  ))}
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
                    {detectionIsOnCurrentFrame && detection?.candidates.map(candidate => {
                      const centerX = detection.width ? ((candidate.box[0] + candidate.box[2]) / 2 / detection.width) * 100 : 50;
                      const centerY = detection.height ? ((candidate.box[1] + candidate.box[3]) / 2 / detection.height) * 100 : 50;
                      const selected = selectedCandidateSet.has(candidate.id);
                      return (
                        <button
                          key={`candidate-button-${candidate.id}`}
                          type="button"
                          title={selected ? `Person ${candidate.id}: included (click to exclude)` : `Person ${candidate.id}: excluded (click to include)`}
                          className={`absolute z-10 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-white text-xs font-bold text-white shadow-lg ${
                            selected ? 'bg-violet-600' : 'bg-red-600'
                          }`}
                          style={{ left: `${centerX}%`, top: `${centerY}%` }}
                          onPointerDown={event => {
                            event.stopPropagation();
                            toggleCandidate(candidate.id);
                          }}
                        >
                          {candidate.id}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            <aside className="space-y-5 overflow-y-auto border-l border-gray-800 p-4 text-sm">
              {error && <div className="rounded border border-red-700 bg-red-950/60 p-3 text-red-200">{error}</div>}
              {message && <div className="rounded border border-blue-800 bg-blue-950/40 p-3 text-blue-200">{message}</div>}

              <section className="space-y-3 rounded-lg border border-gray-800 bg-gray-900/50 p-3">
                <div>
                  <h3 className="font-medium text-gray-100">Character identities</h3>
                  <p className="mt-1 text-[11px] leading-relaxed text-gray-500">
                    Select which identity this mask and speaking track belong to. One shared file becomes a separate protected training view for every annotated identity.
                  </p>
                </div>
                <select
                  aria-label="Character identity"
                  value={activeIdentityId ?? ''}
                  disabled={Boolean(busy)}
                  onChange={event => {
                    const identityId = event.target.value || null;
                    identitySelectionInitializedRef.current = true;
                    setActiveIdentityId(identityId);
                    try {
                      if (identityId) {
                        window.localStorage.setItem(characterIdentityStorageKey(datasetName), identityId);
                      } else {
                        window.localStorage.setItem(
                          characterIdentityStorageKey(datasetName),
                          CHARACTER_DOP_LEGACY_IDENTITY,
                        );
                      }
                    } catch {
                      // The identity remains selected for this session when storage is unavailable.
                    }
                    setDetection(null);
                    setSelectedCandidateIds([]);
                    setPreview(null);
                  }}
                  className="w-full rounded border border-gray-700 bg-gray-950 px-2.5 py-2 text-gray-200"
                >
                  <option value="">Legacy single-character target</option>
                  {(state?.identities ?? []).map(identity => (
                    <option key={identity.id} value={identity.id}>
                      {identity.display_name} · {identity.trigger_word}
                    </option>
                  ))}
                </select>
                {state?.identity && (
                  <div className="space-y-2 rounded border border-gray-800 bg-gray-950/70 p-2.5">
                    <p className="text-[11px] font-medium uppercase tracking-wide text-gray-500">Edit selected identity</p>
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        value={editIdentityName}
                        onChange={event => setEditIdentityName(event.target.value)}
                        aria-label="Selected identity display name"
                        placeholder="Display name"
                        className="min-w-0 rounded border border-gray-700 bg-gray-900 px-2.5 py-2 text-gray-100"
                      />
                      <input
                        value={editIdentityTrigger}
                        onChange={event => setEditIdentityTrigger(event.target.value)}
                        aria-label="Selected identity trigger word"
                        placeholder="Trigger word"
                        className="min-w-0 rounded border border-gray-700 bg-gray-900 px-2.5 py-2 text-gray-100"
                      />
                      <input
                        value={editIdentityClass}
                        onChange={event => setEditIdentityClass(event.target.value)}
                        aria-label="Selected identity generic class prompt"
                        placeholder="Generic class prompt"
                        className="col-span-2 min-w-0 rounded border border-gray-700 bg-gray-900 px-2.5 py-2 text-gray-100"
                      />
                    </div>
                    <p className="text-[11px] leading-relaxed text-amber-300/80">
                      Changing a trigger does not rewrite caption files. Replace the old trigger in your captions before training.
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        type="button"
                        disabled={Boolean(busy) || !editIdentityName.trim() || !editIdentityTrigger.trim() || !editIdentityClass.trim()}
                        onClick={updateIdentity}
                        className="flex items-center justify-center gap-1.5 rounded border border-violet-700 px-2 py-2 text-xs text-violet-200 hover:bg-violet-950/50 disabled:opacity-40"
                      >
                        {busy === 'identity' ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
                        Update identity
                      </button>
                      <button
                        type="button"
                        disabled={Boolean(busy)}
                        onClick={deleteIdentity}
                        className="flex items-center justify-center gap-1.5 rounded border border-red-800 px-2 py-2 text-xs text-red-300 hover:bg-red-950/40 disabled:opacity-40"
                      >
                        <Trash2 size={14} /> Delete identity
                      </button>
                    </div>
                  </div>
                )}
                <p className="text-[11px] font-medium uppercase tracking-wide text-gray-500">Add a new identity</p>
                <div className="grid grid-cols-2 gap-2">
                  <input
                    value={newIdentityName}
                    onChange={event => setNewIdentityName(event.target.value)}
                    placeholder="Display name"
                    className="min-w-0 rounded border border-gray-700 bg-gray-950 px-2.5 py-2 text-gray-100"
                  />
                  <input
                    value={newIdentityTrigger}
                    onChange={event => setNewIdentityTrigger(event.target.value)}
                    placeholder="Trigger word"
                    className="min-w-0 rounded border border-gray-700 bg-gray-950 px-2.5 py-2 text-gray-100"
                  />
                  <input
                    value={newIdentityClass}
                    onChange={event => setNewIdentityClass(event.target.value)}
                    placeholder="Generic class prompt"
                    className="col-span-2 min-w-0 rounded border border-gray-700 bg-gray-950 px-2.5 py-2 text-gray-100"
                  />
                </div>
                <button
                  type="button"
                  disabled={Boolean(busy) || !newIdentityName.trim() || !newIdentityTrigger.trim() || !newIdentityClass.trim()}
                  onClick={createIdentity}
                  className="flex w-full items-center justify-center gap-2 rounded border border-violet-700 px-3 py-2 text-violet-200 hover:bg-violet-950/50 disabled:opacity-40"
                >
                  {busy === 'identity' ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
                  Add identity
                </button>
                {state?.identity && (
                  <p className="rounded bg-gray-950 p-2 text-[11px] text-gray-400">
                    Active counterfactual: replace <span className="text-violet-300">{state.identity.trigger_word}</span> with{' '}
                    <span className="text-violet-300">{state.identity.class_prompt}</span>, while leaving other named characters in the caption unchanged.
                  </p>
                )}
              </section>

              {hasVisual && (
                <section className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="flex items-center gap-2 font-medium text-gray-100"><MousePointer2 size={17} /> Visual identity</h3>
                    {state?.visual.exists && <span className="flex items-center gap-1 text-xs text-emerald-400"><Check size={14} /> Mask ready</span>}
                  </div>
                  <p className="text-xs leading-relaxed text-gray-400">
                    Auto-mask finds everyone first. Exclude the other people, then track only the training character through the shot.
                  </p>
                  <div className="space-y-3 rounded-lg border border-violet-900/70 bg-violet-950/20 p-3">
                    <div className="flex items-center gap-2 text-xs font-medium text-violet-200">
                      <UserRoundSearch size={16} /> Auto-mask with text
                    </div>
                    <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
                      <input
                        value={concept}
                        onChange={event => setConcept(event.target.value)}
                        placeholder="person"
                        aria-label="Concept to auto-mask"
                        className="min-w-0 rounded border border-gray-700 bg-gray-900 px-2.5 py-2 text-gray-100 outline-none focus:border-violet-500"
                      />
                      <button
                        type="button"
                        onClick={autoMaskPeople}
                        disabled={Boolean(busy) || !concept.trim()}
                        className="flex items-center justify-center gap-1.5 rounded bg-violet-700 px-3 py-2 font-medium text-white hover:bg-violet-600 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        {busy === 'detect' ? <Loader2 size={16} className="animate-spin" /> : <Users size={16} />}
                        Find all
                      </button>
                    </div>
                    <select
                      value={detectorModel}
                      onChange={event => setDetectorModel(event.target.value)}
                      disabled={Boolean(busy)}
                      aria-label="Auto-mask detector model"
                      className="w-full rounded border border-gray-700 bg-gray-900 px-2.5 py-2 text-xs text-gray-200"
                    >
                      {(modelCatalog?.detectors ?? []).map(model => (
                        <option key={model.id} value={model.id}>{model.label}</option>
                      ))}
                    </select>
                    <p className="text-[11px] leading-relaxed text-gray-500">
                      SAM 3 returns separate masks for up to 64 matches per frame. It requires accepted Hugging Face access and an authenticated token in the container.
                    </p>
                  </div>
                  {detection && (
                    <div className="space-y-2 rounded-lg border border-gray-800 bg-gray-900/60 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-medium text-gray-200">
                          {selectedCandidates.length} of {detection.candidates.length} included
                        </span>
                        <button
                          type="button"
                          className="text-[11px] text-gray-400 hover:text-white"
                          onClick={() => {
                            setDetection(null);
                            setSelectedCandidateIds([]);
                          }}
                        >
                          Clear auto-mask
                        </button>
                      </div>
                      <p className="text-[11px] leading-relaxed text-gray-400">
                        All matches start included. Click a numbered person on the image or below to exclude everyone except your character.
                      </p>
                      <div className="grid max-h-36 grid-cols-2 gap-1.5 overflow-y-auto">
                        {detection.candidates.map(candidate => {
                          const selected = selectedCandidateSet.has(candidate.id);
                          return (
                            <button
                              key={candidate.id}
                              type="button"
                              onClick={() => toggleCandidate(candidate.id)}
                              className={`flex items-center gap-2 rounded border px-2 py-1.5 text-left text-xs ${
                                selected
                                  ? 'border-violet-500 bg-violet-950/50 text-violet-100'
                                  : 'border-gray-700 bg-gray-950 text-gray-500'
                              }`}
                            >
                              <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold text-white ${selected ? 'bg-violet-600' : 'bg-red-700'}`}>
                                {candidate.id}
                              </span>
                              <span>{selected ? 'Included' : 'Excluded'}</span>
                              <span className="ml-auto text-[10px] opacity-60">{Math.round(candidate.score * 100)}%</span>
                            </button>
                          );
                        })}
                      </div>
                      {selectedCandidates.length > 1 && (
                        <p className="rounded border border-amber-800 bg-amber-950/40 p-2 text-[11px] text-amber-200">
                          Multiple people are still included. Their identities can be learned together; exclude everyone except the target character.
                        </p>
                      )}
                    </div>
                  )}
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
                  <p className="text-[11px] leading-relaxed text-gray-500">
                    Manual correction: add green points on missed target areas and red points on spill. With an auto-mask, point corrections apply on other video frames; clear it to use points on the seed frame.
                  </p>
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
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-gray-300" htmlFor="character-dop-tracker">Tracking model</label>
                    <select
                      id="character-dop-tracker"
                      value={trackerModel}
                      onChange={event => {
                        const modelId = event.target.value;
                        setTrackerModel(modelId);
                        try {
                          window.localStorage.setItem(CHARACTER_DOP_TRACKER_STORAGE_KEY, modelId);
                        } catch {
                          // The selection still applies to this session when storage is unavailable.
                        }
                      }}
                      disabled={Boolean(busy)}
                      className="w-full rounded border border-gray-700 bg-gray-900 px-2.5 py-2 text-gray-200"
                    >
                      {(modelCatalog?.trackers ?? []).map(model => (
                        <option key={model.id} value={model.id}>{model.label}</option>
                      ))}
                    </select>
                    <p className="text-[11px] text-gray-500">{selectedTracker?.description}</p>
                  </div>
                  <button
                    className="flex w-full items-center justify-center gap-2 rounded bg-violet-600 px-3 py-2 font-medium text-white hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={Boolean(busy) || (detection ? selectedCandidates.length === 0 : prompts.length === 0)}
                    onClick={trackCharacter}
                  >
                    {busy === 'track' ? <Loader2 size={17} className="animate-spin" /> : <ScanSearch size={17} />}
                    {detection ? `Track selected with ${selectedTracker?.label ?? 'SAM 2'}` : `${state?.visual.exists ? 'Regenerate' : 'Generate'} with ${selectedTracker?.label ?? 'SAM 2'}`}
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
