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
  characterTraining?: {
    identities: Array<{ id: string; weight: number; source_weights?: Record<string, number> }>;
    joint_training_fraction: number;
  };
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

const identityCoverageForConfig = (
  dataset: DatasetBalanceConfig,
  identity: import('./characterTrainingBalance').CharacterIdentityCoverage,
) => {
  const isAudioOnly = Boolean(dataset.do_audio) && dataset.resolution.length === 0;
  if (isAudioOnly) return [identity.audio, identity.videosAudio];
  const isVideo = dataset.num_frames > 1 || Boolean(dataset.auto_frame_count);
  if (isVideo) return [identity.images, dataset.do_audio ? identity.videos : identity.videosVisual];
  return [identity.images];
};

const selectedIdentityViewCount = (
  dataset: DatasetBalanceConfig,
  inventory: DatasetInventory | undefined,
  options: DatasetBalanceOptions,
) => {
  if (!options.characterTraining || dataset.is_reg) return null;
  if (dataset.character_dop_use_dataset_annotations === false) return 0;
  const selectedIds = new Set(options.characterTraining.identities.map(identity => identity.id));
  return (inventory?.identities ?? []).reduce((total, identity) => {
    if (!selectedIds.has(identity.id)) return total;
    const coverage = identityCoverageForConfig(dataset, identity);
    const focusViews = coverage.reduce((sum, media) => sum + media.sources, 0);
    const jointViews = Number(options.characterTraining?.joint_training_fraction ?? 0) > 0
      ? coverage.reduce((sum, media) => sum + media.group, 0)
      : 0;
    return total + focusViews + jointViews;
  }, 0);
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
      !dataset.is_reg &&
      dataset.character_dop_use_dataset_annotations !== false &&
      Boolean(inventory?.identityCount);
    const curriculumViews = selectedIdentityViewCount(dataset, inventory, options);
    const trainingViews = curriculumViews != null
      ? curriculumViews
      : useCharacterViews
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
    const unassignedCharacterSources = curriculumViews == null && useCharacterViews && !hasFallbackTrigger && !isRegularization
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
  const curriculumShares = new Map<number, number>();
  if (options.characterTraining) {
    const totalIdentityWeight = options.characterTraining.identities.reduce(
      (sum, identity) => sum + Number(identity.weight), 0,
    );
    for (const identity of options.characterTraining.identities) {
      const identityShare = totalIdentityWeight > 0 ? Number(identity.weight) / totalIdentityWeight : 0;
      const eligible = rows.map((row, index) => {
        const dataset = datasets[index];
        const inventory = findInventory(dataset.folder_path, statsByPath);
        const coverage = inventory?.identities?.find(item => item.id === identity.id);
        const count = coverage && !dataset.is_reg && dataset.character_dop_use_dataset_annotations !== false
          ? identityCoverageForConfig(dataset, coverage).reduce((sum, media) => sum + media.sources, 0)
          : 0;
        return { index, path: dataset.folder_path, count };
      }).filter(source => source.count > 0);
      const explicit = new Map(
        Object.entries(identity.source_weights ?? {})
          .filter(([sourcePath]) => sourcePath !== '*')
          .map(([sourcePath, weight]) => [normalizedPath(sourcePath), Number(weight)]),
      );
      const explicitTotal = [...explicit.values()].reduce((sum, weight) => sum + weight, 0);
      const remainder = identity.source_weights?.['*'] ?? Math.max(0, 1 - explicitTotal);
      const automaticCount = eligible.reduce(
        (sum, source) => sum + (explicit.has(normalizedPath(source.path)) ? 0 : source.count), 0,
      );
      const allCount = eligible.reduce((sum, source) => sum + source.count, 0);
      for (const source of eligible) {
        const sourceShare = identity.source_weights
          ? explicit.get(normalizedPath(source.path))
            ?? (automaticCount > 0 ? remainder * source.count / automaticCount : 0)
          : allCount > 0 ? source.count / allCount : 0;
        curriculumShares.set(
          source.index,
          (curriculumShares.get(source.index) ?? 0) + identityShare * sourceShare,
        );
      }
    }
  }
  return rows.map(row => ({
    ...row,
    samplingShare: row.isRegularization
      ? regularizationItems > 0
        ? (row.effectiveItems / regularizationItems) * (hasBothPools ? 0.5 : 1)
        : 0
      : trainingItems > 0
        ? ((options.characterTraining ? curriculumShares.get(row.index) ?? 0 : row.effectiveItems / trainingItems)) * (hasBothPools ? 0.5 : 1)
        : 0,
  }));
}
