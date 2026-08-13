const normalizedPath = value => value.replace(/\\/g, '/').replace(/\/+$/, '').toLocaleLowerCase();

const findInventory = (path, inventories) => {
  if (inventories[path]) return inventories[path];
  const normalized = normalizedPath(path);
  return Object.values(inventories).find(inventory => normalizedPath(inventory.path) === normalized);
};

const augmentationFactor = dataset => {
  const audioOnly = Boolean(dataset.do_audio) && (dataset.resolution?.length ?? 1) === 0;
  const resolutionCount = dataset.resolution?.length;
  const resolutions = resolutionCount == null ? 1 : resolutionCount || (audioOnly ? 1 : 0);
  const repeats = Number(dataset.num_repeats ?? 1);
  const repeatFactor = Number.isFinite(repeats) && repeats > 1 ? Math.trunc(repeats) : 1;
  const flips = audioOnly ? 1 : (dataset.flip_x ? 2 : 1) * (dataset.flip_y ? 2 : 1);
  return resolutions * repeatFactor * flips;
};

const coverageByModality = (dataset, identity) => {
  const audioOnly = Boolean(dataset.do_audio) && (dataset.resolution?.length ?? 1) === 0;
  const video = Boolean(dataset.auto_frame_count) || Number(dataset.num_frames ?? 1) > 1;
  if (audioOnly) return [
    { modality: 'audio', coverage: identity.audio },
    { modality: 'audio', coverage: identity.videosAudio },
  ];
  if (video) return [
    { modality: 'image', coverage: identity.images },
    { modality: 'video', coverage: dataset.do_audio ? identity.videos : identity.videosVisual },
  ];
  return [{ modality: 'image', coverage: identity.images }];
};

const relevantIdentityGroups = (dataset, inventory) => {
  const groups = inventory.identityGroupsByMedia;
  if (!groups) return [];
  const audioOnly = Boolean(dataset.do_audio) && (dataset.resolution?.length ?? 1) === 0;
  const video = Boolean(dataset.auto_frame_count) || Number(dataset.num_frames ?? 1) > 1;
  if (audioOnly) return [...groups.audio, ...groups.videosAudio];
  if (video) return [...groups.images, ...(dataset.do_audio ? groups.videos : groups.videosVisual)];
  return groups.images;
};

const jointCount = (dataset, inventory, identityId, selectedIds) =>
  relevantIdentityGroups(dataset, inventory).reduce((sum, group) => {
    const selectedOnSource = group.identityIds.filter(id => selectedIds.has(id));
    return sum + (selectedOnSource.length > 1 && selectedOnSource.includes(identityId) ? group.sources : 0);
  }, 0);

const buildSources = (identity, selectedIds, datasets, inventories) => datasets
  .map((dataset, index) => ({ dataset, index }))
  .filter(({ dataset }) => !dataset.is_reg && dataset.character_dop_use_dataset_annotations !== false)
  .map(({ dataset, index }) => {
    const inventory = findInventory(dataset.folder_path, inventories);
    const coverage = inventory?.identities?.find(item => item.id === identity.id);
    const factor = augmentationFactor(dataset);
    const modalities = coverage ? coverageByModality(dataset, coverage) : [];
    const focusCount = modalities.reduce((sum, item) => sum + item.coverage.sources, 0) * factor;
    const sharedCount = inventory ? jointCount(dataset, inventory, identity.id, selectedIds) * factor : 0;
    return {
      dataset,
      index,
      sourceKey: normalizedPath(dataset.folder_path),
      modalities,
      factor,
      focusCount,
      jointCount: sharedCount,
    };
  });

const buildRows = (identity, jointFraction, sources) => {
  const rows = [];
  if (jointFraction > 0) {
    rows.push({ target: jointFraction, counts: sources.map(source => source.jointCount) });
  }
  const modalityTotals = new Map();
  for (const source of sources) {
    for (const item of source.modalities) {
      modalityTotals.set(item.modality, (modalityTotals.get(item.modality) ?? 0) + item.coverage.sources * source.factor);
    }
  }
  const focusTotal = [...modalityTotals.values()].reduce((sum, count) => sum + count, 0);
  for (const [modality, modalityCount] of modalityTotals) {
    if (modalityCount <= 0 || focusTotal <= 0) continue;
    const modalityTarget = (1 - jointFraction) * modalityCount / focusTotal;
    const contextCounts = context => sources.map(source => source.modalities
      .filter(item => item.modality === modality)
      .reduce((sum, item) => sum + item.coverage[context] * source.factor, 0));
    const soloCounts = contextCounts('solo');
    const groupCounts = contextCounts('group');
    const soloTotal = soloCounts.reduce((sum, count) => sum + count, 0);
    const groupTotal = groupCounts.reduce((sum, count) => sum + count, 0);
    const soloFraction = identity.context_fractions?.[modality]
      ?? identity.solo_fraction
      ?? (soloTotal + groupTotal > 0 ? soloTotal / (soloTotal + groupTotal) : 0);
    if (soloFraction > 0) rows.push({ target: modalityTarget * soloFraction, counts: soloCounts });
    if (soloFraction < 1) rows.push({ target: modalityTarget * (1 - soloFraction), counts: groupCounts });
  }
  return rows;
};

const buildSourceTargets = (identity, sources) => {
  const sourceCandidates = new Map();
  for (const source of sources) {
    const count = source.focusCount + source.jointCount;
    if (count > 0) sourceCandidates.set(source.sourceKey, (sourceCandidates.get(source.sourceKey) ?? 0) + count);
  }
  if (!identity.source_weights) {
    const total = [...sourceCandidates.values()].reduce((sum, count) => sum + count, 0);
    return new Map([...sourceCandidates].map(([key, count]) => [key, total > 0 ? count / total : 0]));
  }
  const explicit = new Map(
    Object.entries(identity.source_weights)
      .filter(([path]) => path !== '*')
      .map(([path, weight]) => [normalizedPath(path), Number(weight)]),
  );
  const explicitTotal = [...explicit.values()].reduce((sum, weight) => sum + weight, 0);
  const remainder = Number(identity.source_weights['*'] ?? Math.max(0, 1 - explicitTotal));
  const automaticCount = [...sourceCandidates].reduce(
    (sum, [key, count]) => sum + (explicit.has(key) ? 0 : count), 0,
  );
  return new Map([...sourceCandidates].map(([key, count]) => [
    key,
    explicit.get(key) ?? (automaticCount > 0 ? remainder * count / automaticCount : 0),
  ]));
};

const solveCharacterMix = (identity, selectedIds, jointFraction, datasets, inventories) => {
  const sources = buildSources(identity, selectedIds, datasets, inventories);
  const rows = buildRows(identity, jointFraction, sources);
  const cells = rows.map(row => [...row.counts]);

  if (!identity.source_weights) {
    for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
      const total = cells[rowIndex].reduce((sum, mass) => sum + mass, 0);
      if (rows[rowIndex].target > 0 && total <= 0) return { feasible: false, shares: new Map() };
      cells[rowIndex] = cells[rowIndex].map(mass => total > 0 ? rows[rowIndex].target * mass / total : 0);
    }
  } else {
    const sourceTargets = buildSourceTargets(identity, sources);
    const eligibleKeys = new Set(sources
      .filter(source => source.focusCount + source.jointCount > 0)
      .map(source => source.sourceKey));
    for (const [rawPath, rawWeight] of Object.entries(identity.source_weights)) {
      if (rawPath !== '*' && Number(rawWeight) > 0 && !eligibleKeys.has(normalizedPath(rawPath))) {
        return { feasible: false, shares: new Map() };
      }
    }
    for (let iteration = 0; iteration < 500; iteration++) {
      for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
        const current = cells[rowIndex].reduce((sum, mass) => sum + mass, 0);
        if (rows[rowIndex].target > 0 && current <= 0) return { feasible: false, shares: new Map() };
        const scale = current > 0 ? rows[rowIndex].target / current : 0;
        cells[rowIndex] = cells[rowIndex].map(mass => mass * scale);
      }
      for (const [sourceKey, target] of sourceTargets) {
        let current = 0;
        for (const row of cells) {
          for (let column = 0; column < sources.length; column++) {
            if (sources[column].sourceKey === sourceKey) current += row[column];
          }
        }
        if (target > 0 && current <= 0) return { feasible: false, shares: new Map() };
        const scale = current > 0 ? target / current : 0;
        for (const row of cells) {
          for (let column = 0; column < sources.length; column++) {
            if (sources[column].sourceKey === sourceKey) row[column] *= scale;
          }
        }
      }
      const rowError = Math.max(0, ...rows.map((row, index) =>
        Math.abs(cells[index].reduce((sum, mass) => sum + mass, 0) - row.target)));
      const sourceError = Math.max(0, ...[...sourceTargets].map(([sourceKey, target]) => {
        let current = 0;
        for (const row of cells) {
          for (let column = 0; column < sources.length; column++) {
            if (sources[column].sourceKey === sourceKey) current += row[column];
          }
        }
        return Math.abs(current - target);
      }));
      if (Math.max(rowError, sourceError) <= 1e-7) break;
      if (iteration === 499) return { feasible: false, shares: new Map() };
    }
  }

  const shares = new Map(sources.map(source => [source.index, 0]));
  for (const row of cells) {
    for (let column = 0; column < sources.length; column++) {
      shares.set(sources[column].index, (shares.get(sources[column].index) ?? 0) + row[column]);
    }
  }
  return { feasible: true, shares };
};

/** Exact per-config marginal implied by runtime mode/context/source sampling. */
export function calculateCharacterIdentityDatasetShares(
  identity,
  selectedIds,
  jointFraction,
  datasets,
  inventories,
) {
  return solveCharacterMix(identity, selectedIds, jointFraction, datasets, inventories).shares;
}

/** Exact path-level source marginal; duplicate configs remain one source control. */
export function calculateCharacterIdentitySourceShares(
  identity,
  selectedIds,
  jointFraction,
  datasets,
  inventories,
) {
  const datasetShares = calculateCharacterIdentityDatasetShares(
    identity, selectedIds, jointFraction, datasets, inventories,
  );
  const shares = new Map();
  for (const [index, share] of datasetShares) {
    const path = datasets[index].folder_path;
    const existing = [...shares.keys()].find(key => normalizedPath(key) === normalizedPath(path));
    const key = existing ?? path;
    shares.set(key, (shares.get(key) ?? 0) + share);
  }
  return shares;
}

/** Whether requested source and mode/context marginals have a joint solution. */
export function characterSourceMixIsFeasible(
  identity,
  selectedIds,
  jointFraction,
  datasets,
  inventories,
) {
  if (!identity.source_weights) return true;
  return solveCharacterMix(identity, selectedIds, jointFraction, datasets, inventories).feasible;
}
