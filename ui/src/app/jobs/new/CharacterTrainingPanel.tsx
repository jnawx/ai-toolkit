'use client';

import { AlertTriangle, ChevronDown, ChevronUp, Loader2, RefreshCw, Settings2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import CharacterIdentityManager, { type CharacterIdentity } from '@/components/CharacterIdentityManager';
import type { CharacterTrainingConfig, CharacterTrainingIdentityConfig, DatasetConfig } from '@/types';
import { apiClient } from '@/utils/api';
import type { DatasetInventory } from './datasetBalance';
import { calculateCharacterIdentitySourceShares } from './characterSourceShares.mjs';
import { validateCharacterTrainingCoverage } from './characterTrainingBalance';

type Props = {
  value?: CharacterTrainingConfig;
  onChange: (value: CharacterTrainingConfig) => void;
  datasets: DatasetConfig[];
  inventories: Record<string, DatasetInventory>;
  inventoryStatus: 'idle' | 'loading' | 'success' | 'error';
  refreshInventories: () => void;
};

const inputClass = 'rounded border border-gray-700 bg-gray-950 px-2 py-1.5 text-sm text-gray-100 outline-none focus:border-violet-500';
const percent = (value: number | undefined, fallback: number) => Math.round((value ?? fallback) * 100);
const coverageForDataset = (dataset: DatasetConfig, coverage: NonNullable<DatasetInventory['identities']>[number]) => {
  const audioOnly = Boolean(dataset.do_audio) && (dataset.resolution?.length ?? 1) === 0;
  const video = Boolean(dataset.auto_frame_count) || Number(dataset.num_frames ?? 1) > 1;
  if (audioOnly) return [coverage.audio, coverage.videosAudio];
  if (video) return [coverage.images, dataset.do_audio ? coverage.videos : coverage.videosVisual];
  return [coverage.images];
};

export default function CharacterTrainingPanel({
  value,
  onChange,
  datasets,
  inventories,
  inventoryStatus,
  refreshInventories,
}: Props) {
  const strategy = value ?? { identities: [], joint_training_fraction: 0 };
  const [identities, setIdentities] = useState<CharacterIdentity[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const [manageOpen, setManageOpen] = useState(false);

  const loadIdentities = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.post('/api/datasets/characterDop', { action: 'list-identities' });
      setIdentities(response.data.identities ?? []);
    } catch (reason: any) {
      setError(reason?.response?.data?.error || reason?.message || 'Identities could not be loaded');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadIdentities(); }, [loadIdentities]);

  const trainingDatasets = datasets.filter(
    dataset => !dataset.is_reg && dataset.folder_path.trim() && dataset.character_dop_use_dataset_annotations !== false,
  );
  const sourceControlDatasets = trainingDatasets.filter((dataset, index, items) => {
    const normalized = dataset.folder_path.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase();
    return items.findIndex(item => item.folder_path.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase() === normalized) === index;
  });
  const selected = new Map(strategy.identities.map(identity => [identity.id, identity]));
  const selectedIds = new Set(strategy.identities.map(identity => identity.id));
  const recommendedSoloFraction = (identityId: string) => {
    let solo = 0;
    let group = 0;
    for (const dataset of trainingDatasets) {
      const inventory = inventories[dataset.folder_path]
        ?? Object.values(inventories).find(item => item.path.replace(/\\/g, '/').toLowerCase() === dataset.folder_path.replace(/\\/g, '/').toLowerCase());
      const coverage = inventory?.identities?.find(identity => identity.id === identityId);
      if (!coverage) continue;
      for (const media of coverageForDataset(dataset, coverage)) {
        solo += media.solo;
        group += media.group;
      }
    }
    if (solo > 0 && group === 0) return 1;
    if (group > 0 && solo === 0) return 0;
    return 0.5;
  };
  const updateIdentity = (id: string, patch: Partial<CharacterTrainingIdentityConfig>) => {
    onChange({
      ...strategy,
      identities: strategy.identities.map(identity => identity.id === id ? { ...identity, ...patch } : identity),
    });
  };
  const toggleIdentity = (id: string, enabled: boolean) => {
    onChange({
      ...strategy,
      identities: enabled
        ? [...strategy.identities, { id, weight: 1, solo_fraction: recommendedSoloFraction(id) }]
        : strategy.identities.filter(identity => identity.id !== id),
    });
  };
  const validationErrors = useMemo(
    () => validateCharacterTrainingCoverage(strategy, datasets, inventories),
    [strategy, datasets, inventories],
  );
  const totalWeight = strategy.identities.reduce((sum, identity) => sum + Number(identity.weight || 0), 0);
  const targetSourceShares = useMemo(() => {
    const normalize = (path: string) => path.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase();
    const shares = new Map(sourceControlDatasets.map(dataset => [dataset.folder_path, 0]));
    if (totalWeight <= 0) return shares;
    for (const selectedIdentity of strategy.identities) {
      const identityShare = Number(selectedIdentity.weight) / totalWeight;
      const sourceShares = calculateCharacterIdentitySourceShares(
        selectedIdentity,
        selectedIds,
        strategy.joint_training_fraction,
        trainingDatasets,
        inventories,
      );
      for (const [sourcePath, sourceShare] of sourceShares) {
        const path = trainingDatasets.find(dataset => normalize(dataset.folder_path) === normalize(sourcePath))?.folder_path;
        if (path) shares.set(path, (shares.get(path) ?? 0) + identityShare * sourceShare);
      }
    }
    return shares;
  }, [inventories, sourceControlDatasets, strategy.identities, strategy.joint_training_fraction, totalWeight, trainingDatasets]);

  const coverageFor = (identityId: string) => trainingDatasets.reduce((total, dataset) => {
    const inventory = inventories[dataset.folder_path]
      ?? Object.values(inventories).find(item => item.path.replace(/\\/g, '/').toLowerCase() === dataset.folder_path.replace(/\\/g, '/').toLowerCase());
    const coverage = inventory?.identities?.find(identity => identity.id === identityId);
    if (!coverage) return total;
    return total + coverageForDataset(dataset, coverage).reduce((sum, media) => sum + media.sources, 0);
  }, 0);

  const setSourcePercent = (identity: CharacterTrainingIdentityConfig, datasetPath: string, rawPercent: string) => {
    const explicit = Object.fromEntries(
      Object.entries(identity.source_weights ?? {}).filter(([path]) => path !== '*'),
    );
    if (rawPercent.trim() === '') delete explicit[datasetPath];
    else explicit[datasetPath] = Math.max(0, Math.min(100, Number(rawPercent) || 0)) / 100;
    const explicitTotal = Object.values(explicit).reduce((sum, item) => sum + Number(item), 0);
    if (explicitTotal > 1.000001) return;
    const source_weights = Object.keys(explicit).length
      ? { ...explicit, '*': Math.max(0, 1 - explicitTotal) }
      : undefined;
    updateIdentity(identity.id, { source_weights });
  };

  return (
    <section className="mb-4 overflow-hidden rounded-lg border border-violet-900/70 bg-violet-950/10">
      <div className="flex items-start justify-between gap-3 border-b border-gray-800 px-4 py-3">
        <div>
          <h2 className="font-semibold text-gray-100">Character training curriculum</h2>
          <p className="mt-1 max-w-3xl text-xs leading-relaxed text-gray-400">
            Select exactly who this job trains. Unselected media is ignored. Focus views isolate one identity; joint views retain every selected trigger in a shared scene. Regularization datasets stay in their separate preservation pool.
          </p>
        </div>
        <div className="flex shrink-0 gap-2">
          <button type="button" onClick={() => setManageOpen(true)} className="rounded border border-violet-700 px-2.5 py-1.5 text-xs text-violet-200 hover:bg-violet-950">
            Manage identities
          </button>
          <button type="button" onClick={() => void loadIdentities()} disabled={loading} className="rounded border border-gray-700 p-1.5 text-gray-300 hover:bg-gray-800 disabled:opacity-50" title="Refresh identities">
            {loading ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
          </button>
        </div>
      </div>

      <div className="space-y-3 p-4">
        {error && <p className="rounded border border-red-800 bg-red-950/30 p-2 text-xs text-red-200">{error}</p>}
        {!loading && identities.length === 0 && !error && (
          <p className="rounded border border-dashed border-gray-700 p-3 text-sm text-gray-400">Create a shared identity, then annotate it in any dataset.</p>
        )}
        {identities.map(identity => {
          const config = selected.get(identity.id);
          const share = config && totalWeight > 0 ? Number(config.weight) / totalWeight : 0;
          return (
            <div key={identity.id} className={`rounded-lg border p-3 ${config ? 'border-violet-700 bg-gray-900/80' : 'border-gray-800 bg-gray-950/40'}`}>
              <div className="flex flex-wrap items-center gap-3">
                <input type="checkbox" checked={Boolean(config)} onChange={event => toggleIdentity(identity.id, event.target.checked)} className="h-4 w-4 accent-violet-600" />
                <div className="min-w-48 flex-1">
                  <div className="text-sm font-medium text-gray-100">{identity.display_name}</div>
                  <div className="text-xs text-violet-300">{identity.trigger_word} · {coverageFor(identity.id)} annotated view(s)</div>
                </div>
                {config && (
                  <>
                    <label className="text-xs text-gray-400">Relative weight <input className={`${inputClass} ml-1 w-20`} type="number" min="0.01" step="0.1" value={config.weight} onChange={event => updateIdentity(identity.id, { weight: Math.max(0.01, Number(event.target.value) || 0.01) })} /></label>
                    <span className="w-20 text-right text-xs text-gray-400">{(share * 100).toFixed(1)}% share</span>
                    <label className="text-xs text-gray-400">Solo <input className={`${inputClass} ml-1 w-20`} type="number" min="0" max="100" value={percent(config.solo_fraction, 0.5)} onChange={event => updateIdentity(identity.id, { solo_fraction: Math.max(0, Math.min(100, Number(event.target.value))) / 100 })} />%</label>
                  </>
                )}
              </div>
              {config && advanced && (
                <div className="mt-3 space-y-3 border-t border-gray-800 pt-3">
                  <div className="flex flex-wrap gap-3">
                    {(['image', 'video', 'audio'] as const).map(modality => (
                      <label key={modality} className="text-xs capitalize text-gray-400">{modality === 'audio' ? 'audio-only' : modality} solo
                        <input className={`${inputClass} ml-1 w-20`} type="number" min="0" max="100" value={percent(config.context_fractions?.[modality], config.solo_fraction ?? 0.5)} onChange={event => updateIdentity(identity.id, { context_fractions: { ...config.context_fractions, [modality]: Math.max(0, Math.min(100, Number(event.target.value))) / 100 } })} />%
                      </label>
                    ))}
                  </div>
                  <div>
                    <div className="mb-1 text-xs font-medium text-gray-300">Dataset source mix</div>
                    <p className="mb-2 text-[11px] text-gray-500">Set only the sources you need to control. Blank sources share the automatic remainder according to their available annotations.</p>
                    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                      {sourceControlDatasets.map(dataset => (
                        <label key={dataset.folder_path} className="flex items-center justify-between gap-2 rounded border border-gray-800 px-2 py-1.5 text-xs text-gray-400">
                          <span className="truncate" title={dataset.folder_path}>{dataset.folder_path.split(/[\\/]/).pop()}</span>
                          <span><input className={`${inputClass} w-16`} type="number" min="0" max="100" placeholder="Auto" value={config.source_weights?.[dataset.folder_path] == null ? '' : Math.round(config.source_weights[dataset.folder_path] * 100)} onChange={event => setSourcePercent(config, dataset.folder_path, event.target.value)} />%</span>
                        </label>
                      ))}
                    </div>
                    {config.source_weights && <div className="mt-1 text-[11px] text-gray-500">Automatic remainder: {Math.round((config.source_weights['*'] ?? 0) * 100)}%</div>}
                  </div>
                </div>
              )}
            </div>
          );
        })}

        <div className="flex flex-wrap items-center justify-between gap-3 rounded border border-gray-800 bg-gray-950/50 p-3">
          <label className="text-sm text-gray-300">Joint composition training
            <input className={`${inputClass} ml-2 w-20`} type="number" min="0" max="100" value={percent(strategy.joint_training_fraction, 0)} onChange={event => onChange({ ...strategy, joint_training_fraction: Math.max(0, Math.min(100, Number(event.target.value))) / 100 })} />%
          </label>
          <button type="button" onClick={() => setAdvanced(value => !value)} className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white">
            <Settings2 size={14} /> Per-modality and dataset controls {advanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>

        {strategy.identities.length > 0 && trainingDatasets.length > 0 && (
          <div className="rounded border border-gray-800 bg-gray-950/50 p-3">
            <div className="mb-2 text-xs font-medium text-gray-300">Effective target source share</div>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {sourceControlDatasets.map(dataset => (
                <div key={dataset.folder_path} className="flex items-center justify-between gap-2 text-xs text-gray-400">
                  <span className="truncate" title={dataset.folder_path}>{dataset.folder_path.split(/[\\/]/).pop()}</span>
                  <span className="font-medium text-gray-200">{((targetSourceShares.get(dataset.folder_path) ?? 0) * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
            {datasets.some(dataset => dataset.is_reg) && <p className="mt-2 border-t border-gray-800 pt-2 text-[11px] text-amber-300">Regularization datasets are excluded from these shares and continue to alternate through the existing regularization loader.</p>}
          </div>
        )}

        {inventoryStatus === 'loading' && <p className="text-xs text-gray-500">Checking identity coverage…</p>}
        {validationErrors.length > 0 && inventoryStatus === 'success' && (
          <div className="space-y-1 rounded border border-amber-800 bg-amber-950/30 p-3 text-xs text-amber-200">
            {validationErrors.map(message => <p key={message} className="flex gap-2"><AlertTriangle size={14} className="mt-0.5 shrink-0" />{message}</p>)}
          </div>
        )}
      </div>
      <CharacterIdentityManager open={manageOpen} onClose={() => { setManageOpen(false); void loadIdentities(); refreshInventories(); }} />
    </section>
  );
}
