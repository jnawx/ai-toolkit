'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import type { DatasetInventory } from '@/app/jobs/new/datasetBalance';
import { apiClient } from '@/utils/api';

export default function useDatasetStats(datasetPaths: string[]) {
  const [stats, setStats] = useState<Record<string, DatasetInventory>>({});
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const requestGeneration = useRef(0);

  const datasetPathKey = JSON.stringify(datasetPaths);
  const refresh = useCallback(() => {
    const generation = ++requestGeneration.current;
    setStatus('loading');
    const query = new URLSearchParams();
    (JSON.parse(datasetPathKey) as string[]).forEach(datasetPath => query.append('path', datasetPath));
    apiClient
      .get(`/api/datasets/stats?${query.toString()}`)
      .then(response => response.data)
      .then(data => {
        if (generation !== requestGeneration.current) return;
        setStats(data);
        setStatus('success');
      })
      .catch(error => {
        if (generation !== requestGeneration.current) return;
        console.error('Failed to load dataset inventory:', error);
        setStatus('error');
      });
  }, [datasetPathKey]);

  useEffect(() => {
    refresh();
    return () => {
      requestGeneration.current += 1;
    };
  }, [refresh]);

  return { stats, status, refresh };
}
