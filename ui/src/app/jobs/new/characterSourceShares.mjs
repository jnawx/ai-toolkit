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
  .filter(dataset => !dataset.is_reg && dataset.character_dop_use_dataset_annotations !== false)
  .map(dataset => {
    const inventory = findInventory(dataset.folder_path, inventories);
    const coverage = inventory?.identities?.find(item => item.id === identity.id);
    const factor = augmentationFactor(dataset);
    const modalities = coverage ? coverageByModality(dataset, coverage) : [];
    const focusCount = modalities.reduce((sum, item) => sum + item.coverage.sources, 0) * factor;
    const sharedCount = inventory ? jointCount(dataset, inventory, identity.id, selectedIds) * factor : 0;
    return { dataset, modalities, factor, focusCount, jointCount: sharedCount };
  });

/** Exact source marginal implied by runtime mode/context sampling for one identity. */
export function calculateCharacterIdentitySourceShares(
  identity,
  selectedIds,
  jointFraction,
  datasets,
  inventories,
) {
  const sources = buildSources(identity, selectedIds, datasets, inventories);
  const shares = new Map(sources.map(source => [source.dataset.folder_path, 0]));
  const candidateCount = source => source.focusCount + (jointFraction > 0 ? source.jointCount : 0);
  const eligible = sources.filter(source => candidateCount(source) > 0);
  if (identity.source_weights) {
    const explicit = new Map(
      Object.entries(identity.source_weights)
        .filter(([path]) => path !== '*')
        .map(([path, weight]) => [normalizedPath(path), Number(weight)]),
    );
    const explicitTotal = [...explicit.values()].reduce((sum, weight) => sum + weight, 0);
    const remainder = Number(identity.source_weights['*'] ?? Math.max(0, 1 - explicitTotal));
    const automaticCount = eligible.reduce(
      (sum, source) => sum + (explicit.has(normalizedPath(source.dataset.folder_path)) ? 0 : candidateCount(source)), 0,
    );
    for (const source of eligible) {
      const value = explicit.get(normalizedPath(source.dataset.folder_path))
        ?? (automaticCount > 0 ? remainder * candidateCount(source) / automaticCount : 0);
      shares.set(source.dataset.folder_path, value);
    }
    return shares;
  }

  const jointTotal = eligible.reduce((sum, source) => sum + source.jointCount, 0);
  if (jointFraction > 0 && jointTotal > 0) {
    for (const source of eligible) {
      shares.set(source.dataset.folder_path, jointFraction * source.jointCount / jointTotal);
    }
  }
  const modalityTotals = new Map();
  for (const source of eligible) {
    for (const item of source.modalities) {
      modalityTotals.set(item.modality, (modalityTotals.get(item.modality) ?? 0) + item.coverage.sources * source.factor);
    }
  }
  const focusTotal = [...modalityTotals.values()].reduce((sum, count) => sum + count, 0);
  for (const [modality, modalityCount] of modalityTotals) {
    if (modalityCount <= 0 || focusTotal <= 0) continue;
    const modalityShare = (1 - jointFraction) * modalityCount / focusTotal;
    const soloCount = eligible.reduce((sum, source) => sum + source.modalities
      .filter(item => item.modality === modality)
      .reduce((subtotal, item) => subtotal + item.coverage.solo * source.factor, 0), 0);
    const groupCount = eligible.reduce((sum, source) => sum + source.modalities
      .filter(item => item.modality === modality)
      .reduce((subtotal, item) => subtotal + item.coverage.group * source.factor, 0), 0);
    const soloFraction = identity.context_fractions?.[modality]
      ?? identity.solo_fraction
      ?? (soloCount + groupCount > 0 ? soloCount / (soloCount + groupCount) : 0);
    for (const source of eligible) {
      const scoped = source.modalities.filter(item => item.modality === modality);
      const sourceSolo = scoped.reduce((sum, item) => sum + item.coverage.solo * source.factor, 0);
      const sourceGroup = scoped.reduce((sum, item) => sum + item.coverage.group * source.factor, 0);
      const contribution = modalityShare * (
        (soloCount > 0 ? soloFraction * sourceSolo / soloCount : 0)
        + (groupCount > 0 ? (1 - soloFraction) * sourceGroup / groupCount : 0)
      );
      shares.set(source.dataset.folder_path, (shares.get(source.dataset.folder_path) ?? 0) + contribution);
    }
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
  const sources = buildSources(identity, selectedIds, datasets, inventories);
  const sourceTargets = calculateCharacterIdentitySourceShares(
    identity, selectedIds, jointFraction, datasets, inventories,
  );
  const rows = [];
  if (jointFraction > 0) {
    rows.push({
      target: jointFraction,
      counts: new Map(sources.map(source => [source.dataset.folder_path, source.jointCount])),
    });
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
    const contextCounts = context => new Map(sources.map(source => [
      source.dataset.folder_path,
      source.modalities
        .filter(item => item.modality === modality)
        .reduce((sum, item) => sum + item.coverage[context] * source.factor, 0),
    ]));
    const soloCounts = contextCounts('solo');
    const groupCounts = contextCounts('group');
    const soloTotal = [...soloCounts.values()].reduce((sum, count) => sum + count, 0);
    const groupTotal = [...groupCounts.values()].reduce((sum, count) => sum + count, 0);
    const soloFraction = identity.context_fractions?.[modality]
      ?? identity.solo_fraction
      ?? (soloTotal + groupTotal > 0 ? soloTotal / (soloTotal + groupTotal) : 0);
    if (soloFraction > 0) rows.push({ target: modalityTarget * soloFraction, counts: soloCounts });
    if (soloFraction < 1) rows.push({ target: modalityTarget * (1 - soloFraction), counts: groupCounts });
  }
  const sourcePaths = sources
    .map(source => source.dataset.folder_path)
    .filter(path => (sourceTargets.get(path) ?? 0) > 0);
  const cells = rows.map(row => sourcePaths.map(path => Number(row.counts.get(path) ?? 0)));
  for (let iteration = 0; iteration < 500; iteration++) {
    for (let row = 0; row < rows.length; row++) {
      const current = cells[row].reduce((sum, mass) => sum + mass, 0);
      if (rows[row].target > 0 && current <= 0) return false;
      const scale = current > 0 ? rows[row].target / current : 0;
      cells[row] = cells[row].map(mass => mass * scale);
    }
    for (let column = 0; column < sourcePaths.length; column++) {
      const target = Number(sourceTargets.get(sourcePaths[column]) ?? 0);
      const current = cells.reduce((sum, row) => sum + row[column], 0);
      if (target > 0 && current <= 0) return false;
      const scale = current > 0 ? target / current : 0;
      for (const row of cells) row[column] *= scale;
    }
    const rowError = Math.max(...rows.map((row, index) =>
      Math.abs(cells[index].reduce((sum, mass) => sum + mass, 0) - row.target)));
    const sourceError = Math.max(...sourcePaths.map((path, column) =>
      Math.abs(cells.reduce((sum, row) => sum + row[column], 0) - Number(sourceTargets.get(path) ?? 0))));
    if (Math.max(rowError, sourceError) <= 1e-7) return true;
  }
  return false;
}
