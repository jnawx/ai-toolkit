export type DatasetMediaInventory = {
  sources: number;
  assignedSources: number;
  characterViews: number;
};

export type DatasetInventory = {
  path: string;
  identityCount: number;
  identities?: import('./characterTrainingBalance').CharacterIdentityCoverage[];
  jointIdentityPairs?: [string, string][];
  images: DatasetMediaInventory;
  videos: DatasetMediaInventory;
  audio: DatasetMediaInventory;
  error?: string;
};

export type DatasetBalanceConfig = {
  folder_path: string;
  resolution: number[];
  do_audio?: boolean;
  num_frames: number;
  auto_frame_count?: boolean;
  character_dop_use_dataset_annotations?: boolean;
  num_repeats?: number;
  flip_x: boolean;
  flip_y: boolean;
  network_weight: number;
  is_reg: boolean;
  trigger_word?: string | null;
};

export type DatasetBalanceOptions = {
  modelGroup?: string;
  characterDop: boolean;
  globalTrigger: boolean;
};

export type DatasetBalanceRow = {
  index: number;
  path: string;
  sourceItems: number;
  assignedSources: number;
  characterViews: number;
  trainingViews: number;
  resolutionFactor: number;
  repeatFactor: number;
  flipFactor: number;
  augmentationFactor: number;
  effectiveItems: number;
  samplingShare: number;
  networkWeight: number;
  isRegularization: boolean;
  identityCount: number;
  usesCharacterViews: boolean;
  unassignedCharacterSources: number;
  error?: string;
};

const emptyInventory = (): DatasetMediaInventory => ({
  sources: 0,
  assignedSources: 0,
  characterViews: 0,
});

const combineInventory = (...groups: DatasetMediaInventory[]): DatasetMediaInventory =>
  groups.reduce(
    (total, group) => ({
      sources: total.sources + group.sources,
      assignedSources: total.assignedSources + group.assignedSources,
      characterViews: total.characterViews + group.characterViews,
    }),
    emptyInventory(),
  );

const normalizedPath = (value: string) => value.replace(/\\/g, '/').replace(/\/+$/, '').toLocaleLowerCase();

const findInventory = (path: string, statsByPath: Record<string, DatasetInventory>) => {
  if (statsByPath[path]) return statsByPath[path];
  const key = normalizedPath(path);
  return Object.values(statsByPath).find(inventory => normalizedPath(inventory.path) === key);
};

const inventoryForConfig = (
  dataset: DatasetBalanceConfig,
  inventory: DatasetInventory,
  modelGroup?: string,
) => {
  const isAudioOnly = Boolean(dataset.do_audio) && dataset.resolution.length === 0;
  if (isAudioOnly) return combineInventory(inventory.audio, inventory.videos);
  if (modelGroup === 'audio') return inventory.audio;
  const isVideo = dataset.num_frames > 1 || Boolean(dataset.auto_frame_count);
  return isVideo
    ? combineInventory(inventory.images, inventory.videos)
    : inventory.images;
};

export function calculateDatasetBalance(
  datasets: DatasetBalanceConfig[],
  statsByPath: Record<string, DatasetInventory>,
  options: DatasetBalanceOptions,
): DatasetBalanceRow[] {
  const rows = datasets.map((dataset, index) => {
    const inventory = findInventory(dataset.folder_path, statsByPath);
    const selected = inventory
      ? inventoryForConfig(dataset, inventory, options.modelGroup)
      : emptyInventory();
    const useCharacterViews =
      options.characterDop &&
      dataset.character_dop_use_dataset_annotations !== false &&
      Boolean(inventory?.identityCount);
    const trainingViews = useCharacterViews
      ? selected.sources - selected.assignedSources + selected.characterViews
      : selected.sources;
    const isAudioOnly = Boolean(dataset.do_audio) && dataset.resolution.length === 0;
    const resolutionFactor = dataset.resolution.length > 0
      ? dataset.resolution.length
      : isAudioOnly
        ? 1
        : 0;
    const configuredRepeats = Number(dataset.num_repeats ?? 1);
    const repeatFactor = Number.isFinite(configuredRepeats) && configuredRepeats > 1
      ? Math.trunc(configuredRepeats)
      : 1;
    const flipFactor = isAudioOnly
      ? 1
      : (dataset.flip_x ? 2 : 1) * (dataset.flip_y ? 2 : 1);
    const augmentationFactor = resolutionFactor * repeatFactor * flipFactor;
    const isRegularization = Boolean(dataset.is_reg);
    const hasFallbackTrigger = options.globalTrigger || Boolean(dataset.trigger_word?.trim());
    const unassignedCharacterSources = useCharacterViews && !hasFallbackTrigger && !isRegularization
      ? selected.sources - selected.assignedSources
      : 0;

    return {
      index,
      path: dataset.folder_path,
      sourceItems: selected.sources,
      assignedSources: selected.assignedSources,
      characterViews: selected.characterViews,
      trainingViews,
      resolutionFactor,
      repeatFactor,
      flipFactor,
      augmentationFactor,
      effectiveItems: trainingViews * augmentationFactor,
      samplingShare: 0,
      networkWeight: Number(dataset.network_weight ?? 1),
      isRegularization,
      identityCount: inventory?.identityCount ?? 0,
      usesCharacterViews: useCharacterViews,
      unassignedCharacterSources,
      error: inventory?.error ?? (inventory ? undefined : 'Inventory unavailable'),
    };
  });
  const trainingItems = rows.reduce(
    (total, row) => total + (row.isRegularization ? 0 : row.effectiveItems),
    0,
  );
  const regularizationItems = rows.reduce(
    (total, row) => total + (row.isRegularization ? row.effectiveItems : 0),
    0,
  );
  const hasBothPools = trainingItems > 0 && regularizationItems > 0;
  return rows.map(row => ({
    ...row,
    samplingShare: row.isRegularization
      ? regularizationItems > 0
        ? (row.effectiveItems / regularizationItems) * (hasBothPools ? 0.5 : 1)
        : 0
      : trainingItems > 0
        ? (row.effectiveItems / trainingItems) * (hasBothPools ? 0.5 : 1)
        : 0,
  }));
}
