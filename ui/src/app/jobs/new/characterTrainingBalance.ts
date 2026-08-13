export type CharacterMediaCoverage = { sources: number; solo: number; group: number };

export type CharacterIdentityCoverage = {
  id: string;
  images: CharacterMediaCoverage;
  videos: CharacterMediaCoverage;
  videosVisual: CharacterMediaCoverage;
  videosAudio: CharacterMediaCoverage;
  audio: CharacterMediaCoverage;
};

export type CharacterTrainingIdentityConfig = {
  id: string;
  weight: number;
  solo_fraction?: number;
  context_fractions?: Partial<Record<'image' | 'video' | 'audio', number>>;
  source_weights?: Record<string, number>;
};

export type CharacterTrainingStrategy = {
  identities: CharacterTrainingIdentityConfig[];
  joint_training_fraction: number;
  epoch_size?: number;
};

type DatasetSelection = {
  folder_path: string;
  is_reg: boolean;
  resolution?: number[];
  do_audio?: boolean;
  num_frames?: number;
  auto_frame_count?: boolean;
  character_dop_use_dataset_annotations?: boolean;
};

type Inventory = {
  path: string;
  identities?: CharacterIdentityCoverage[];
  jointIdentityPairs?: [string, string][];
  jointIdentityPairsByMedia?: {
    images: [string, string][];
    videos: [string, string][];
    videosVisual: [string, string][];
    videosAudio: [string, string][];
    audio: [string, string][];
  };
  error?: string;
};

const normalizedPath = (value: string) => value.replace(/\\/g, '/').replace(/\/+$/, '').toLocaleLowerCase();

const findInventory = (path: string, inventories: Record<string, Inventory>) => {
  if (inventories[path]) return inventories[path];
  const normalized = normalizedPath(path);
  return Object.values(inventories).find(inventory => normalizedPath(inventory.path) === normalized);
};

const relevantCoverage = (dataset: DatasetSelection, identity: CharacterIdentityCoverage) => {
  const audioOnly = Boolean(dataset.do_audio) && (dataset.resolution?.length ?? 1) === 0;
  const video = Boolean(dataset.auto_frame_count) || Number(dataset.num_frames ?? 1) > 1;
  if (audioOnly) return [identity.audio, identity.videosAudio];
  if (video) {
    return [identity.images, dataset.do_audio ? identity.videos : identity.videosVisual];
  }
  return [identity.images];
};

const total = (items: CharacterMediaCoverage[], key: keyof CharacterMediaCoverage) =>
  items.reduce((sum, item) => sum + Number(item[key] ?? 0), 0);

const relevantJointPairs = (dataset: DatasetSelection, inventory: Inventory) => {
  const scoped = inventory.jointIdentityPairsByMedia;
  if (!scoped) return inventory.jointIdentityPairs ?? [];
  const audioOnly = Boolean(dataset.do_audio) && (dataset.resolution?.length ?? 1) === 0;
  const video = Boolean(dataset.auto_frame_count) || Number(dataset.num_frames ?? 1) > 1;
  if (audioOnly) return [...scoped.audio, ...scoped.videosAudio];
  if (video) return [...scoped.images, ...(dataset.do_audio ? scoped.videos : scoped.videosVisual)];
  return scoped.images;
};

/** Strict UI mirror of runtime checks. Regularization never satisfies identity coverage. */
export function validateCharacterTrainingCoverage(
  strategy: CharacterTrainingStrategy | undefined,
  datasets: DatasetSelection[],
  inventories: Record<string, Inventory>,
): string[] {
  if (!strategy?.identities?.length) return ['Select at least one character identity to train.'];
  const errors: string[] = [];
  const trainingDatasets = datasets.filter(
    dataset => !dataset.is_reg && dataset.character_dop_use_dataset_annotations !== false,
  );
  const selectedIds = new Set(strategy.identities.map(identity => identity.id));

  for (const selected of strategy.identities) {
    const coverageBySource = trainingDatasets.map(dataset => {
      const inventory = findInventory(dataset.folder_path, inventories);
      const identity = inventory?.identities?.find(item => item.id === selected.id);
      return { dataset, inventory, coverage: identity ? relevantCoverage(dataset, identity) : [] };
    });
    const allCoverage = coverageBySource.flatMap(item => item.coverage);
    if (total(allCoverage, 'sources') === 0) {
      errors.push(`Selected identity ${selected.id} has no annotated representation in a training dataset.`);
      continue;
    }

    const modalityCoverage: Record<'image' | 'video' | 'audio', CharacterMediaCoverage[]> = {
      image: [], video: [], audio: [],
    };
    for (const { dataset, inventory } of coverageBySource) {
      const identity = inventory?.identities?.find(item => item.id === selected.id);
      if (!identity) continue;
      const audioOnly = Boolean(dataset.do_audio) && (dataset.resolution?.length ?? 1) === 0;
      const video = Boolean(dataset.auto_frame_count) || Number(dataset.num_frames ?? 1) > 1;
      if (audioOnly) modalityCoverage.audio.push(identity.audio, identity.videosAudio);
      else if (video) {
        modalityCoverage.video.push(dataset.do_audio ? identity.videos : identity.videosVisual);
        modalityCoverage.image.push(identity.images);
      } else modalityCoverage.image.push(identity.images);
    }

    for (const modality of ['image', 'video', 'audio'] as const) {
      const coverage = modalityCoverage[modality];
      if (!coverage.length || total(coverage, 'sources') === 0) continue;
      const soloFraction = selected.context_fractions?.[modality] ?? selected.solo_fraction;
      if (soloFraction != null && soloFraction > 0 && total(coverage, 'solo') === 0) {
        errors.push(`${selected.id} requests solo ${modality} training but has no eligible solo ${modality} annotation.`);
      }
      if (soloFraction != null && soloFraction < 1 && total(coverage, 'group') === 0) {
        errors.push(`${selected.id} requests group ${modality} training but has no eligible group ${modality} annotation.`);
      }
    }

    for (const [sourcePath, weight] of Object.entries(selected.source_weights ?? {})) {
      if (sourcePath === '*' || Number(weight) <= 0) continue;
      const source = coverageBySource.find(item => normalizedPath(item.dataset.folder_path) === normalizedPath(sourcePath));
      if (!source || total(source.coverage, 'sources') === 0) {
        errors.push(`${selected.id} requests ${Math.round(Number(weight) * 100)}% from ${sourcePath}, but that dataset has no eligible annotation.`);
      }
    }
  }

  if (Number(strategy.joint_training_fraction ?? 0) > 0) {
    if (selectedIds.size < 2) {
      errors.push('Joint training requires at least two selected identities.');
    } else {
      for (const selectedId of selectedIds) {
        const hasJointCoverage = trainingDatasets.some(dataset => {
          const inventory = findInventory(dataset.folder_path, inventories);
          return inventory != null && relevantJointPairs(dataset, inventory).some(
            pair => pair.includes(selectedId) && pair.every(identityId => selectedIds.has(identityId)),
          );
        });
        if (!hasJointCoverage) {
          errors.push(`Joint training is requested, but ${selectedId} has no eligible shared source with another selected identity.`);
        }
      }
    }
  }
  return [...new Set(errors)];
}
