'use client';
import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { Cog, Layers3, ScanSearch, SquareDashed } from 'lucide-react';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import classNames from 'classnames';
import { openConfirm } from './ConfirmModal';
import { apiClient } from '@/utils/api';
import { isVideo, isAudio } from '@/utils/basic';
import AudioPlayer from './AudioPlayer';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import { BoundingBoxEditor, parseBoundingBoxes, extractBoxes } from './BoundingBoxOverlay';
import IdeogramCaptionSidebar, { isIdeogramCaption } from './IdeogramCaptionSidebar';
import datasetTemplates from '@/helpers/datasetTemplates';
import CharacterDOPAnnotator from './CharacterDOPAnnotator';

type IdentityOverlay = {
  identity_id: string;
  display_name: string;
  data_url: string;
  centroid: [number, number];
};

const IDENTITY_COLORS = ['#22d3ee', '#f472b6', '#a3e635', '#fb923c', '#c084fc', '#facc15'];
const IDENTITY_OVERLAY_STORAGE_KEY = 'ai-toolkit.character-dop.show-identity-overlays';

function IdentityOverlayLayer({ overlays }: { overlays: IdentityOverlay[] }) {
  return (
    <div className="pointer-events-none absolute inset-0 z-10 overflow-hidden">
      {overlays.map((overlay, index) => {
        const color = IDENTITY_COLORS[index % IDENTITY_COLORS.length];
        return (
          <div key={overlay.identity_id} className="absolute inset-0">
            <div
              className="absolute inset-0 opacity-45"
              style={{
                backgroundColor: color,
                WebkitMaskImage: `url(${overlay.data_url})`,
                maskImage: `url(${overlay.data_url})`,
                WebkitMaskSize: '100% 100%',
                maskSize: '100% 100%',
              }}
            />
            <span
              className="absolute -translate-x-1/2 -translate-y-1/2 rounded px-1.5 py-0.5 text-[11px] font-semibold text-gray-950 shadow"
              style={{ left: `${overlay.centroid[0] * 100}%`, top: `${overlay.centroid[1] * 100}%`, backgroundColor: color }}
            >
              {overlay.display_name}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function safeParse(text: string): any {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

interface Props {
  datasetName: string;
  imgPath: string | null; // current image path
  imageList: string[]; // all dataset image paths
  onChange: (nextPath: string | null) => void; // parent setter
  refreshImages?: () => void;
  onCaptionSaved?: (imgPath: string, caption: string) => void;
  captionExt?: string;
}

export default function DatasetImageViewer({
  datasetName,
  imgPath,
  imageList,
  onChange,
  refreshImages,
  onCaptionSaved,
  captionExt = 'txt',
}: Props) {
  const [mounted, setMounted] = useState(false);
  const [isOpen, setIsOpen] = useState(Boolean(imgPath));
  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const [isCaptionLoaded, setIsCaptionLoaded] = useState<boolean>(false);
  const [showBoxes, setShowBoxes] = useState<boolean>(false);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [showCharacterAnnotator, setShowCharacterAnnotator] = useState(true);
  const [showIdentityOverlays, setShowIdentityOverlays] = useState(true);
  const [identityOverlays, setIdentityOverlays] = useState<IdentityOverlay[]>([]);
  const [overlayTimeFraction, setOverlayTimeFraction] = useState(0);
  const [overlayRevision, setOverlayRevision] = useState(0);
  const captionRef = useRef<string>('');
  const savedCaptionRef = useRef<string>('');
  const currentImgPathRef = useRef<string | null>(null);
  const captionAbortRef = useRef<AbortController | null>(null);

  const isIdeogram = useMemo(() => isIdeogramCaption(caption), [caption]);

  useEffect(() => {
    setMounted(true);
    setShowIdentityOverlays(window.localStorage.getItem(IDENTITY_OVERLAY_STORAGE_KEY) !== 'false');
  }, []);

  const toggleIdentityOverlays = () => {
    setShowIdentityOverlays(current => {
      const next = !current;
      window.localStorage.setItem(IDENTITY_OVERLAY_STORAGE_KEY, String(next));
      return next;
    });
  };

  // Clear box selection / draw mode whenever the image changes.
  useEffect(() => {
    setSelectedBoxIndex(null);
    setIsDrawing(false);
    setShowCharacterAnnotator(true);
    setOverlayTimeFraction(0);
  }, [imgPath]);

  useEffect(() => {
    if (!imgPath || isAudio(imgPath) || !showIdentityOverlays || !isOpen) {
      setIdentityOverlays([]);
      return;
    }
    const controller = new AbortController();
    apiClient.post('/api/datasets/characterDop', {
      action: 'overlays',
      datasetName,
      mediaPath: imgPath,
      timeFraction: isVideo(imgPath) ? overlayTimeFraction : 0,
    }, { signal: controller.signal }).then(response => {
      if (!controller.signal.aborted) setIdentityOverlays(response.data.overlays ?? []);
    }).catch(() => {
      if (!controller.signal.aborted) setIdentityOverlays([]);
    });
    return () => controller.abort();
  }, [datasetName, imgPath, isOpen, overlayRevision, overlayTimeFraction, showIdentityOverlays]);

  // Default to showing the editable boxes when an Ideogram caption is present.
  useEffect(() => {
    setShowBoxes(isIdeogram);
  }, [isIdeogram, imgPath]);

  // open/close based on external value
  useEffect(() => {
    setIsOpen(Boolean(imgPath));
  }, [imgPath]);

  // after close, clear parent state post-transition
  useEffect(() => {
    if (!isOpen && imgPath) {
      const t = setTimeout(() => onChange(null), 300);
      return () => clearTimeout(t);
    }
  }, [isOpen, imgPath, onChange]);

  // Keep refs in sync so we can save the latest caption on unmount/navigation
  useEffect(() => {
    captionRef.current = caption;
  }, [caption]);
  useEffect(() => {
    savedCaptionRef.current = savedCaption;
  }, [savedCaption]);

  const filename = useMemo(() => {
    if (!imgPath) return '';
    if (imgPath.includes('\\')) {
      const parts = imgPath.split('\\');
      return parts[parts.length - 1];
    }
    return imgPath.split('/').pop() || '';
  }, [imgPath]);

  const currentIndex = useMemo(() => {
    if (!imgPath) return -1;
    return imageList.findIndex(img => img === imgPath);
  }, [imgPath, imageList]);

  const saveCaptionForPath = useCallback(
    (path: string, value: string, prevSaved: string) => {
      const trimmed = value.trim();
      if (trimmed === prevSaved.trim()) return;
      apiClient
        .post('/api/img/caption', { imgPath: path, caption: trimmed, ext: captionExt })
        .then(() => {
          if (currentImgPathRef.current === path) {
            setSavedCaption(trimmed);
          }
          onCaptionSaved?.(path, trimmed);
        })
        .catch(error => {
          console.error('Error saving caption:', error);
        });
    },
    [onCaptionSaved, captionExt],
  );

  // Stable handle to the latest saveCaptionForPath so the fetch effect doesn't
  // re-run (and re-fetch, blanking the caption) every time a save changes its
  // identity via the parent's onCaptionSaved.
  const saveCaptionForPathRef = useRef(saveCaptionForPath);
  useEffect(() => {
    saveCaptionForPathRef.current = saveCaptionForPath;
  }, [saveCaptionForPath]);

  const saveCaption = useCallback(() => {
    if (!imgPath) return;
    saveCaptionForPath(imgPath, caption, savedCaption);
  }, [imgPath, caption, savedCaption, saveCaptionForPath]);

  // Fetch caption whenever the image changes; save any pending edits on the previous image first
  useEffect(() => {
    const previousPath = currentImgPathRef.current;
    if (previousPath && previousPath !== imgPath) {
      saveCaptionForPathRef.current(previousPath, captionRef.current, savedCaptionRef.current);
    }
    currentImgPathRef.current = imgPath;

    if (!imgPath) {
      setCaption('');
      setSavedCaption('');
      setIsCaptionLoaded(false);
      return;
    }

    captionAbortRef.current?.abort();
    const controller = new AbortController();
    captionAbortRef.current = controller;
    setIsCaptionLoaded(false);
    setCaption('');
    setSavedCaption('');

    apiClient
      // transformResponse identity: keep the caption as a raw string. Axios's
      // default parses any JSON-looking body into an object (our bbox captions
      // are JSON), which would render as "[object Object]".
      .post(
        '/api/caption/get',
        { imgPath, ext: captionExt },
        { signal: controller.signal, transformResponse: [d => d] },
      )
      .then(res => res.data)
      .then(data => {
        if (controller.signal.aborted) return;
        const text = typeof data === 'string' ? data : data ? `${data}` : '';
        setCaption(text);
        setSavedCaption(text);
        setIsCaptionLoaded(true);
      })
      .catch(error => {
        if (controller.signal.aborted) return;
        console.error('Error fetching caption:', error);
        setIsCaptionLoaded(true);
      });

    return () => {
      controller.abort();
    };
  }, [imgPath, captionExt]);

  // Save any pending caption when the viewer fully unmounts
  useEffect(() => {
    return () => {
      const path = currentImgPathRef.current;
      if (path) {
        saveCaptionForPath(path, captionRef.current, savedCaptionRef.current);
      }
    };
  }, [saveCaptionForPath]);

  const onCancel = useCallback(() => {
    saveCaption();
    setIsOpen(false);
  }, [saveCaption]);

  const setImageAtIndex = useCallback(
    (idx: number) => {
      if (idx < 0 || idx >= imageList.length) return;
      onChange(imageList[idx]);
    },
    [imageList, onChange],
  );

  const handlePrev = useCallback(() => {
    if (currentIndex <= 0) return;
    setImageAtIndex(currentIndex - 1);
  }, [currentIndex, setImageAtIndex]);

  const handleNext = useCallback(() => {
    if (currentIndex === -1 || currentIndex >= imageList.length - 1) return;
    setImageAtIndex(currentIndex + 1);
  }, [currentIndex, imageList.length, setImageAtIndex]);

  const handleDelete = useCallback(() => {
    if (!imgPath) return;
    openConfirm({
      title: 'Delete Image',
      message: `Are you sure you want to delete this image? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/img/delete', { imgPath })
          .then(() => {
            console.log('Image deleted:', imgPath);
            onChange(null);
            if (refreshImages) refreshImages();
          })
          .catch(error => {
            console.error('Error deleting image:', error);
          });
      },
    });
  }, [imgPath, onChange, refreshImages]);

  // Mutate the caption JSON's element array, updating the textarea ONLY (no
  // network). Persisting happens via the Save button / nav auto-save. Returns
  // whatever the mutator returns.
  const editCaption = useCallback(
    (fn: (elements: any[], data: any) => any): any => {
      let data: any;
      try {
        data = JSON.parse(caption);
      } catch {
        return undefined;
      }
      const elements = data?.compositional_deconstruction?.elements;
      if (!Array.isArray(elements)) return undefined;
      const result = fn(elements, data);
      setCaption(JSON.stringify(data, null, 2));
      return result;
    },
    [caption],
  );

  const handleBoxChange = useCallback(
    (elementIndex: number, box: { y1: number; x1: number; y2: number; x2: number }) => {
      editCaption(els => {
        if (els[elementIndex]) els[elementIndex].bbox = [box.y1, box.x1, box.y2, box.x2];
      });
    },
    [editCaption],
  );

  const handleDeleteBox = useCallback(
    (elementIndex: number) => {
      editCaption(els => {
        els.splice(elementIndex, 1);
      });
      setSelectedBoxIndex(null);
    },
    [editCaption],
  );

  const handleCreateBox = useCallback(
    (box: { y1: number; x1: number; y2: number; x2: number }) => {
      const newIndex = editCaption(els => {
        els.push({ type: 'obj', bbox: [box.y1, box.x1, box.y2, box.x2], desc: '' });
        return els.length - 1;
      });
      setSelectedBoxIndex(typeof newIndex === 'number' ? newIndex : null);
      setIsDrawing(false);
    },
    [editCaption],
  );

  // keyboard events while open — skip nav while caption textarea is focused
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isOpen) return;
      if (showCharacterAnnotator) return;
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      const isTyping = tag === 'TEXTAREA' || tag === 'INPUT' || (target?.isContentEditable ?? false);

      if (event.key === 'Escape') {
        onCancel();
        return;
      }
      if (isTyping) return;

      switch (event.key) {
        case 'ArrowLeft':
        case 'ArrowUp':
          handlePrev();
          break;
        case 'ArrowRight':
        case 'ArrowDown':
          handleNext();
          break;
        case 'Delete':
        case 'Backspace':
          // With boxes shown, Delete removes the selected box (never the image).
          if (showBoxes) {
            if (selectedBoxIndex != null) handleDeleteBox(selectedBoxIndex);
          } else {
            handleDelete();
          }
          break;
        default:
          break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onCancel, handlePrev, handleNext, handleDelete, showBoxes, selectedBoxIndex, handleDeleteBox, showCharacterAnnotator]);

  // Touch swipe navigation
  const touchStartRef = useRef<{ x: number; y: number } | null>(null);
  const multiTouchRef = useRef(false);
  const zoomedRef = useRef(false);
  const SWIPE_THRESHOLD = 40;

  const onTouchStart = useCallback((e: React.TouchEvent) => {
    if (e.touches.length > 1) {
      multiTouchRef.current = true;
      touchStartRef.current = null;
      return;
    }
    multiTouchRef.current = false;
    const t = e.touches[0];
    if (!t) return;
    touchStartRef.current = { x: t.clientX, y: t.clientY };
  }, []);

  const onTouchEnd = useCallback(
    (e: React.TouchEvent) => {
      const start = touchStartRef.current;
      touchStartRef.current = null;
      if (multiTouchRef.current || zoomedRef.current) {
        if (e.touches.length === 0) multiTouchRef.current = false;
        return;
      }
      if (!start) return;
      const t = e.changedTouches[0];
      if (!t) return;
      const dx = t.clientX - start.x;
      const dy = t.clientY - start.y;
      const absX = Math.abs(dx);
      const absY = Math.abs(dy);
      if (absX < SWIPE_THRESHOLD && absY < SWIPE_THRESHOLD) return;
      if (absX > absY) {
        if (dx < 0) handleNext();
        else handlePrev();
      } else {
        if (dy < 0) handleNext();
        else handlePrev();
      }
    },
    [handleNext, handlePrev],
  );

  const handleCaptionKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption();
    }
  };

  const isCaptionCurrent = caption.trim() === savedCaption.trim();
  const boundingBoxes = useMemo(() => parseBoundingBoxes(caption), [caption]);
  const canShowBoxes = Boolean(boundingBoxes && imgPath && !isAudio(imgPath) && !isVideo(imgPath));

  // Boxes are derived from the (locally edited) caption for the image overlay.
  const editBoxes = useMemo(() => extractBoxes(safeParse(caption)), [caption]);

  if (!mounted) return null;

  return createPortal(
    <Dialog
      open={isOpen}
      onClose={onCancel}
      className="relative z-50"
    >
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />
      <div className="fixed inset-0 z-50 w-screen overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-0 sm:p-4 text-center">
          <DialogPanel
            transition
            onTouchStart={onTouchStart}
            onTouchEnd={onTouchEnd}
            className="relative transform rounded-none sm:rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in w-full sm:w-auto sm:max-w-[95vw] sm:max-h-[95vh] data-closed:sm:translate-y-0 data-closed:sm:scale-95 flex flex-col sm:flex-row overflow-hidden touch-pan-y"
          >
            {/* Image / media area */}
            <div className="relative flex-1 min-w-0 flex items-center justify-center bg-gray-900 overflow-hidden">
              {imgPath &&
                (isAudio(imgPath) ? (
                  <div className="w-[500px] h-[500px] max-w-full max-h-[50vh] sm:max-h-[90vh]">
                    <AudioPlayer src={`/api/img/${encodeURIComponent(imgPath)}`} title={filename} autoPlay />
                  </div>
                ) : isVideo(imgPath) ? (
                  <div className="relative max-h-[50vh] max-w-full sm:max-h-[90vh]">
                    <video
                      src={`/api/img/${encodeURIComponent(imgPath)}`}
                      className="block h-auto w-auto max-h-[50vh] max-w-full object-contain sm:max-h-[90vh]"
                      preload="metadata"
                      playsInline
                      loop
                      autoPlay
                      controls={true}
                      onTimeUpdate={event => {
                        const video = event.currentTarget;
                        if (!video.duration) return;
                        const fraction = Math.round((video.currentTime / video.duration) * 20) / 20;
                        setOverlayTimeFraction(current => current === fraction ? current : fraction);
                      }}
                    />
                    {showIdentityOverlays && <IdentityOverlayLayer overlays={identityOverlays} />}
                  </div>
                ) : (
                  <TransformWrapper
                    key={imgPath}
                    initialScale={1}
                    minScale={1}
                    maxScale={6}
                    doubleClick={{ mode: 'toggle', step: 2, disabled: showBoxes }}
                    wheel={{ step: 0.2 }}
                    panning={{ disabled: showBoxes, allowRightClickPan: false }}
                    onTransform={(_ref, state) => {
                      zoomedRef.current = state.scale > 1.01;
                    }}
                  >
                    <TransformComponent>
                      <div className="relative">
                        <img
                          src={`/api/img/${encodeURIComponent(imgPath)}`}
                          alt="Dataset Image"
                          draggable={false}
                          className="w-auto h-auto max-w-full max-h-[50vh] sm:max-h-[90vh] object-contain select-none !pointer-events-auto"
                        />
                        {showIdentityOverlays && <IdentityOverlayLayer overlays={identityOverlays} />}
                        {showBoxes && (
                          <BoundingBoxEditor
                            boxes={editBoxes}
                            selectedIndex={selectedBoxIndex}
                            drawing={isDrawing}
                            onSelect={setSelectedBoxIndex}
                            onChangeBox={handleBoxChange}
                            onCreateBox={handleCreateBox}
                          />
                        )}
                      </div>
                    </TransformComponent>
                  </TransformWrapper>
                ))}

              {/* Controls over the image */}
              <div className="absolute top-2 right-2 flex items-center gap-2 z-20">
                {imgPath && (
                  <button
                    type="button"
                    onClick={toggleIdentityOverlays}
                    title={showIdentityOverlays ? 'Hide saved identity masks' : 'Show saved identity masks'}
                    className={classNames('rounded-full bg-gray-900 p-1 leading-[0px] hover:opacity-100', {
                      'text-cyan-300 opacity-100': showIdentityOverlays,
                      'opacity-50': !showIdentityOverlays,
                    })}
                  >
                    <Layers3 />
                  </button>
                )}
                {canShowBoxes && (
                  <button
                    type="button"
                    onClick={() => {
                      const next = !showBoxes;
                      setShowBoxes(next);
                      if (!next) {
                        setSelectedBoxIndex(null);
                        setIsDrawing(false);
                      }
                    }}
                    title={showBoxes ? 'Hide bounding boxes' : 'Show & edit bounding boxes'}
                    className={classNames('bg-gray-900 rounded-full p-1 leading-[0px] hover:opacity-100', {
                      'opacity-100 text-blue-400': showBoxes,
                      'opacity-50': !showBoxes,
                    })}
                  >
                    <SquareDashed />
                  </button>
                )}
                <div className="bg-gray-900 rounded-full p-1 leading-[0px] opacity-50 hover:opacity-100">
                  <Menu>
                    <MenuButton>
                      <Cog />
                    </MenuButton>
                    <MenuItems
                      anchor="bottom end"
                      className="bg-gray-900 border border-gray-700 rounded shadow-lg w-48 px-2 py-2 mt-1 z-50"
                    >
                      {imgPath && isAudio(imgPath) && (
                        <MenuItem>
                          <a
                            className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded block"
                            href={`/api/img/${encodeURIComponent(imgPath)}`}
                            download={filename}
                          >
                            Download
                          </a>
                        </MenuItem>
                      )}
                      <MenuItem>
                        <div className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded" onClick={handleDelete}>
                          Delete Image
                        </div>
                      </MenuItem>
                    </MenuItems>
                  </Menu>
                </div>
              </div>
            </div>

            {/* Right sidebar: file info + caption + box editor */}
            <div className="bg-gray-950 w-full sm:w-96 shrink-0 flex flex-col gap-2 p-3 overflow-y-auto text-sm">
              <div className="flex items-center justify-between gap-2">
                <div className="text-xs text-gray-400 truncate min-w-0">
                  <span className="text-gray-500 mr-1">File:</span>
                  <span className="text-gray-300">{filename}</span>
                </div>
                <div className="text-xs text-gray-400 whitespace-nowrap">
                  {currentIndex >= 0 ? `${currentIndex + 1} / ${imageList.length}` : ''}
                </div>
              </div>
              {imgPath && (
                <button type="button" className="flex items-center justify-center gap-2 rounded border border-gray-700 px-3 py-2 text-gray-300 hover:bg-gray-800" onClick={() => setShowCharacterAnnotator(true)}>
                  <ScanSearch size={16} /> Character annotation
                </button>
              )}
              {showIdentityOverlays && identityOverlays.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {identityOverlays.map((overlay, index) => (
                    <span key={overlay.identity_id} className="rounded px-2 py-1 text-[11px] font-medium text-gray-950" style={{ backgroundColor: IDENTITY_COLORS[index % IDENTITY_COLORS.length] }}>
                      {overlay.display_name}
                    </span>
                  ))}
                </div>
              )}
              {isCaptionLoaded && caption.trim() === '' && (
                <select
                  className="w-full bg-gray-900 border border-gray-700 text-gray-100 text-sm rounded p-2 outline-none focus:ring-0 focus:outline-none"
                  value=""
                  onChange={e => {
                    const template = datasetTemplates[e.target.value];
                    if (template) setCaption(template.trim());
                  }}
                >
                  <option value="">Templates...</option>
                  {Object.keys(datasetTemplates).map(key => (
                    <option key={key} value={key}>
                      {key}
                    </option>
                  ))}
                </select>
              )}
              {isIdeogram ? (
                <IdeogramCaptionSidebar
                  caption={caption}
                  onChange={setCaption}
                  selectedIndex={selectedBoxIndex}
                  onSelectIndex={i => {
                    setSelectedBoxIndex(i);
                    if (i != null) setShowBoxes(true);
                  }}
                  isDrawing={isDrawing}
                  onToggleDrawing={() => setIsDrawing(d => !d)}
                  onSave={saveCaption}
                  isDirty={!isCaptionCurrent}
                />
              ) : (
                <div
                  className={classNames('flex-1 min-h-[8rem] rounded border-2 bg-gray-900 transition-colors', {
                    'border-blue-500': !isCaptionCurrent,
                    'border-gray-700': isCaptionCurrent,
                  })}
                >
                  <textarea
                    className="w-full h-full bg-transparent text-gray-100 text-sm p-2 resize-none outline-none focus:ring-0 focus:outline-none"
                    placeholder={isCaptionLoaded ? 'Add a caption...' : 'Loading caption...'}
                    value={caption}
                    onChange={e => setCaption(e.target.value)}
                    onKeyDown={handleCaptionKeyDown}
                    onBlur={saveCaption}
                    disabled={!isCaptionLoaded}
                  />
                </div>
              )}
            </div>
            {imgPath && showCharacterAnnotator && (
              <div className="absolute inset-0 z-30">
                <CharacterDOPAnnotator
                  embedded
                  open
                  datasetName={datasetName}
                  mediaPath={imgPath}
                  captionText={caption}
                  onCaptionChange={setCaption}
                  onCaptionSave={saveCaption}
                  captionSaved={isCaptionCurrent}
                  captionLoaded={isCaptionLoaded}
                  onShowStandardDetails={() => {
                    setShowCharacterAnnotator(false);
                    setOverlayRevision(current => current + 1);
                  }}
                  onPrevious={handlePrev}
                  onNext={handleNext}
                  hasPrevious={currentIndex > 0}
                  hasNext={currentIndex >= 0 && currentIndex < imageList.length - 1}
                  mediaPosition={currentIndex >= 0 ? `${currentIndex + 1} / ${imageList.length}` : undefined}
                  onClose={onCancel}
                />
              </div>
            )}
          </DialogPanel>
        </div>
      </div>
    </Dialog>,
    document.body,
  );
}
