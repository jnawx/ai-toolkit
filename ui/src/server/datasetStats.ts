import fs from 'node:fs/promises';
import path from 'node:path';

type MediaInventory = {
  sources: number;
  assignedSources: number;
  characterViews: number;
};

type IdentityAssignments = {
  id: string;
  visualImages: Set<string>;
  visualTemporal: Set<string>;
  audio: Set<string>;
  error?: string;
};

type IdentityMediaCoverage = { sources: number; solo: number; group: number };
export type IdentityCoverage = {
  id: string;
  images: IdentityMediaCoverage;
  videosVisual: IdentityMediaCoverage;
  videosAudio: IdentityMediaCoverage;
  audio: IdentityMediaCoverage;
};

export type DatasetInventory = {
  path: string;
  identityCount: number;
  identities: IdentityCoverage[];
  jointIdentityPairs: [string, string][];
  images: MediaInventory;
  videos: MediaInventory;
  audio: MediaInventory;
  error?: string;
};

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp']);
const VIDEO_EXTENSIONS = new Set(['.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv']);
const AUDIO_EXTENSIONS = new Set(['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']);
const IDENTITY_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,63}$/;
const MAX_CATALOG_BYTES = 128 * 1024;
const MAX_IDENTITIES = 64;
const MAX_SELECTED_DATASETS = 32;
const MAX_SCAN_ENTRIES_PER_DATASET = 250_000;
const SHARED_IDENTITY_CATALOG = '_character_dop_identities.json';
const IDENTITY_TEXT_LIMITS: Record<string, [number, number]> = {
  display_name: [128, 512],
  trigger_word: [128, 512],
  class_prompt: [256, 1024],
  caption_description: [1024, 4096],
};
const MEDIA_EXTENSIONS = new Set([
  ...IMAGE_EXTENSIONS,
  ...VIDEO_EXTENSIONS,
  ...AUDIO_EXTENSIONS,
]);

const emptyMediaInventory = (): MediaInventory => ({
  sources: 0,
  assignedSources: 0,
  characterViews: 0,
});

const normalizedStem = (relativePath: string) => {
  const parsed = path.parse(relativePath);
  return path.join(parsed.dir, parsed.name).split(path.sep).join('/');
};

const isWithin = (root: string, target: string) => {
  const relative = path.relative(root, target);
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
};

async function containedStoragePath(datasetRoot: string, target: string): Promise<string> {
  const lexicalTarget = path.resolve(target);
  if (!isWithin(datasetRoot, lexicalTarget)) throw new Error('Character DOP path escapes its dataset');
  let existingAncestor = lexicalTarget;
  while (true) {
    try {
      const realAncestor = await fs.realpath(existingAncestor);
      const resolvedTarget = path.resolve(realAncestor, path.relative(existingAncestor, lexicalTarget));
      if (!isWithin(datasetRoot, resolvedTarget)) throw new Error('Character DOP path escapes its dataset');
      return resolvedTarget;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      const parent = path.dirname(existingAncestor);
      if (parent === existingAncestor) throw error;
      existingAncestor = parent;
    }
  }
}

async function walkFiles(
  root: string,
  skipDatasetPrivateDirectories: boolean,
  allowedExtensions: Set<string>,
  budget: { remaining: number },
): Promise<string[]> {
  const files: string[] = [];
  const pendingDirectories = [root];
  while (pendingDirectories.length) {
    const currentDirectory = pendingDirectories.pop()!;
    let entries;
    try {
      entries = await fs.readdir(currentDirectory, { withFileTypes: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue;
      throw error;
    }
    budget.remaining -= entries.length;
    if (budget.remaining < 0) {
      throw new Error(`Dataset inventory exceeds ${MAX_SCAN_ENTRIES_PER_DATASET.toLocaleString()} filesystem entries`);
    }
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const entryPath = path.join(currentDirectory, entry.name);
      if (entry.isDirectory()) {
        if (skipDatasetPrivateDirectories && (entry.name === '_controls' || entry.name === '_character_dop')) {
          continue;
        }
        pendingDirectories.push(entryPath);
      } else if (entry.isFile() && allowedExtensions.has(path.extname(entry.name).toLowerCase())) {
        files.push(entryPath);
      }
    }
  }
  return files;
}

async function readIdentityIds(
  containmentRoot: string,
  rawCatalogPath: string,
): Promise<{ ids: string[]; exists: boolean; error?: string }> {
  try {
    const catalogPath = await containedStoragePath(
      containmentRoot,
      rawCatalogPath,
    );
    const stat = await fs.stat(catalogPath);
    if (stat.size > MAX_CATALOG_BYTES) throw new Error('Character identity catalog is too large');
    const payload = JSON.parse(await fs.readFile(catalogPath, 'utf8'));
    if (payload?.version !== 1 || !Array.isArray(payload.identities)) {
      throw new Error('Character identity catalog is invalid');
    }
    if (payload.identities.length > MAX_IDENTITIES) {
      throw new Error(`Character identity catalog exceeds ${MAX_IDENTITIES} identities`);
    }
    const ids: string[] = [];
    const triggers: string[] = [];
    for (const rawIdentity of payload.identities) {
      if (!rawIdentity || typeof rawIdentity !== 'object' || Array.isArray(rawIdentity)) {
        throw new Error('Character identity catalog entries must be objects');
      }
      const id = String(rawIdentity.id ?? '').trim();
      if (!IDENTITY_ID_PATTERN.test(id)) {
        throw new Error('Character identity catalog contains an invalid identity id');
      }
      for (const [field, [maxCharacters, maxBytes]] of Object.entries(IDENTITY_TEXT_LIMITS)) {
        const value = String(
          field === 'caption_description'
            ? rawIdentity[field] ?? rawIdentity.class_prompt ?? ''
            : rawIdentity[field] ?? '',
        ).trim();
        if (!value) throw new Error(`Character identity ${field.replace('_', ' ')} cannot be blank`);
        if (Array.from(value).length > maxCharacters || Buffer.byteLength(value, 'utf8') > maxBytes) {
          throw new Error(`Character identity ${field.replace('_', ' ')} exceeds its size limit`);
        }
      }
      const trigger = String(rawIdentity.trigger_word).trim().toLocaleLowerCase('en-US');
      if (triggers.some(existing => existing.includes(trigger) || trigger.includes(existing))) {
        throw new Error('Character identity trigger words must be unique and cannot contain one another');
      }
      ids.push(id);
      triggers.push(trigger);
    }
    if (new Set(ids).size !== ids.length) throw new Error('Character identity catalog contains duplicate ids');
    return { ids, exists: true };
  } catch (error: any) {
    if (error?.code === 'ENOENT') return { ids: [], exists: false };
    return { ids: [], exists: true, error: error?.message || 'Character identity catalog could not be read' };
  }
}

function validateAudioIntervals(payload: any, annotationPath: string) {
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    if (!Object.prototype.hasOwnProperty.call(payload, 'character_intervals')) {
      throw new Error(`Character DOP audio sidecar is missing character_intervals: ${annotationPath}`);
    }
    payload = payload.character_intervals;
  }
  if (!Array.isArray(payload)) {
    throw new Error(`Character DOP audio intervals must be a JSON list: ${annotationPath}`);
  }
  for (const interval of payload) {
    if (
      !Array.isArray(interval) ||
      interval.length !== 2 ||
      interval[0] == null ||
      interval[1] == null ||
      (typeof interval[0] === 'string' && !interval[0].trim()) ||
      (typeof interval[1] === 'string' && !interval[1].trim())
    ) {
      throw new Error(`Character DOP audio interval must be [start, end]: ${annotationPath}`);
    }
    const start = Number(interval[0]);
    const end = Number(interval[1]);
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) {
      throw new Error(`Character DOP audio interval must satisfy 0 <= start < end: ${annotationPath}`);
    }
  }
}

async function annotatedStemsForIdentity(
  datasetPath: string,
  identityId: string,
  budget: { remaining: number },
): Promise<IdentityAssignments> {
  const identityRoot = path.join(datasetPath, '_character_dop', 'identities', identityId);
  const visualRoot = await containedStoragePath(datasetPath, path.join(identityRoot, 'visual'));
  const audioRoot = await containedStoragePath(datasetPath, path.join(identityRoot, 'audio'));
  const visualFiles = await walkFiles(visualRoot, false, new Set(['.png', '.npy']), budget);
  const audioFiles = await walkFiles(audioRoot, false, new Set(['.json']), budget);
  const assignments: IdentityAssignments = {
    id: identityId,
    visualImages: new Set(),
    visualTemporal: new Set(),
    audio: new Set(),
  };
  for (const annotationPath of visualFiles) {
    const extension = path.extname(annotationPath).toLowerCase();
    const stem = normalizedStem(path.relative(visualRoot, annotationPath));
    if (extension === '.png') assignments.visualImages.add(stem);
    else if (extension === '.npy') assignments.visualTemporal.add(stem);
  }
  for (const annotationPath of audioFiles) {
    if (path.extname(annotationPath).toLowerCase() !== '.json') continue;
    try {
      const payload = JSON.parse(await fs.readFile(annotationPath, 'utf8'));
      validateAudioIntervals(payload, annotationPath);
      assignments.audio.add(normalizedStem(path.relative(audioRoot, annotationPath)));
    } catch (error) {
      assignments.error = error instanceof Error
        ? error.message
        : `Character DOP audio sidecar is invalid: ${annotationPath}`;
      break;
    }
  }
  return assignments;
}

export async function collectDatasetInventory(
  datasetPath: string,
  sharedIdentityIds?: Set<string>,
  sharedCatalogError?: string,
): Promise<DatasetInventory> {
  const resolvedDatasetPath = await fs.realpath(path.resolve(datasetPath));
  const budget = { remaining: MAX_SCAN_ENTRIES_PER_DATASET };
  const sourceFiles = await walkFiles(resolvedDatasetPath, true, MEDIA_EXTENSIONS, budget);
  const catalog = await readIdentityIds(
    resolvedDatasetPath,
    path.join(resolvedDatasetPath, '_character_dop', 'identities.json'),
  );
  const activeIdentityIds = sharedIdentityIds
    ? catalog.ids.filter(identityId => sharedIdentityIds.has(identityId))
    : catalog.ids;
  const identityAssignments: IdentityAssignments[] = [];
  for (const identityId of activeIdentityIds) {
    identityAssignments.push(await annotatedStemsForIdentity(resolvedDatasetPath, identityId, budget));
  }
  const inventory: DatasetInventory = {
    path: resolvedDatasetPath,
    identityCount: activeIdentityIds.length,
    identities: activeIdentityIds.map(id => ({
      id,
      images: { sources: 0, solo: 0, group: 0 },
      videosVisual: { sources: 0, solo: 0, group: 0 },
      videosAudio: { sources: 0, solo: 0, group: 0 },
      audio: { sources: 0, solo: 0, group: 0 },
    })),
    jointIdentityPairs: [],
    images: emptyMediaInventory(),
    videos: emptyMediaInventory(),
    audio: emptyMediaInventory(),
    error: sharedCatalogError ?? catalog.error ?? identityAssignments.find(assignment => assignment.error)?.error,
  };
  const jointPairKeys = new Set<string>();

  for (const sourcePath of sourceFiles) {
    const extension = path.extname(sourcePath).toLowerCase();
    const category = IMAGE_EXTENSIONS.has(extension)
      ? inventory.images
      : VIDEO_EXTENSIONS.has(extension)
        ? inventory.videos
        : AUDIO_EXTENSIONS.has(extension)
          ? inventory.audio
          : null;
    if (!category) continue;
    category.sources += 1;
    const sourceStem = normalizedStem(path.relative(resolvedDatasetPath, sourcePath));
    const viewCount = identityAssignments.reduce((count, assignments) => {
      const hasExpectedVisualMask = IMAGE_EXTENSIONS.has(extension)
        ? assignments.visualImages.has(sourceStem)
        : assignments.visualTemporal.has(sourceStem);
      return count + (assignments.audio.has(sourceStem) || hasExpectedVisualMask ? 1 : 0);
    }, 0);
    if (viewCount > 0) category.assignedSources += 1;
    category.characterViews += viewCount;

    const sourceIdentities = identityAssignments.filter(assignment => {
      const hasVisual = IMAGE_EXTENSIONS.has(extension)
        ? assignment.visualImages.has(sourceStem)
        : assignment.visualTemporal.has(sourceStem);
      return hasVisual || assignment.audio.has(sourceStem);
    });
    const sourceIdentityCount = sourceIdentities.length;
    for (let left = 0; left < sourceIdentities.length; left++) {
      for (let right = left + 1; right < sourceIdentities.length; right++) {
        const pair = [sourceIdentities[left].id, sourceIdentities[right].id].sort() as [string, string];
        const pairKey = `${pair[0]}\0${pair[1]}`;
        if (!jointPairKeys.has(pairKey)) {
          jointPairKeys.add(pairKey);
          inventory.jointIdentityPairs.push(pair);
        }
      }
    }
    const updateCoverage = (matching: IdentityAssignments[], field: keyof Omit<IdentityCoverage, 'id'>) => {
      for (const assignment of matching) {
        const coverage = inventory.identities.find(identity => identity.id === assignment.id)?.[field];
        if (!coverage) continue;
        coverage.sources += 1;
        if (sourceIdentityCount > 1) coverage.group += 1;
        else coverage.solo += 1;
      }
    };
    if (IMAGE_EXTENSIONS.has(extension)) {
      updateCoverage(identityAssignments.filter(assignment => assignment.visualImages.has(sourceStem)), 'images');
    } else if (VIDEO_EXTENSIONS.has(extension)) {
      updateCoverage(identityAssignments.filter(assignment => assignment.visualTemporal.has(sourceStem)), 'videosVisual');
      updateCoverage(identityAssignments.filter(assignment => assignment.audio.has(sourceStem)), 'videosAudio');
    } else if (AUDIO_EXTENSIONS.has(extension)) {
      updateCoverage(identityAssignments.filter(assignment => assignment.audio.has(sourceStem)), 'audio');
    }
  }
  return inventory;
}

export async function collectDatasetInventories(
  datasetsRoot: string,
  requestedDatasetPaths: string[],
): Promise<Record<string, DatasetInventory>> {
  if (requestedDatasetPaths.length > MAX_SELECTED_DATASETS) {
    throw new Error(`Dataset inventory supports at most ${MAX_SELECTED_DATASETS} selected datasets`);
  }
  const lexicalRoot = path.resolve(datasetsRoot);
  let resolvedRoot: string;
  try {
    resolvedRoot = await fs.realpath(lexicalRoot);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return {};
    throw error;
  }
  const datasetSelections: Array<{ requestedPath: string; realPath: string }> = [];
  for (const requestedPath of requestedDatasetPaths) {
    const lexicalPath = path.resolve(requestedPath);
    if (!isWithin(lexicalRoot, lexicalPath)) {
      throw new Error('Selected dataset path is outside the configured datasets root');
    }
    const realPath = await fs.realpath(lexicalPath);
    if (!isWithin(resolvedRoot, realPath)) {
      throw new Error('Selected dataset path escapes the configured datasets root');
    }
    const stat = await fs.stat(realPath);
    if (!stat.isDirectory()) throw new Error('Selected dataset path must be a directory');
    if (!datasetSelections.some(selection => selection.realPath === realPath)) {
      datasetSelections.push({ requestedPath: lexicalPath, realPath });
    }
  }
  const sharedCatalog = await readIdentityIds(
    resolvedRoot,
    path.join(resolvedRoot, SHARED_IDENTITY_CATALOG),
  );
  const sharedIdentityIds = sharedCatalog.exists && !sharedCatalog.error
    ? new Set(sharedCatalog.ids)
    : undefined;
  const inventories: DatasetInventory[] = [];
  for (const selection of datasetSelections) {
    try {
      inventories.push({
        ...await collectDatasetInventory(
          selection.realPath,
          sharedIdentityIds,
          sharedCatalog.error,
        ),
        path: selection.requestedPath,
      });
    } catch (error) {
      inventories.push({
        path: selection.requestedPath,
        identityCount: 0,
        identities: [],
        jointIdentityPairs: [],
        images: emptyMediaInventory(),
        videos: emptyMediaInventory(),
        audio: emptyMediaInventory(),
        error: error instanceof Error ? error.message : 'Dataset inventory could not be collected',
      });
    }
  }
  return Object.fromEntries(inventories.map(inventory => [inventory.path, inventory]));
}
