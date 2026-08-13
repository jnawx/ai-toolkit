import { NextResponse } from 'next/server';

import { getDatasetsRoot } from '@/server/settings';
import { collectDatasetInventories } from '@/server/datasetStats';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  try {
    const datasetPaths = new URL(request.url).searchParams.getAll('path');
    const inventories = await collectDatasetInventories(await getDatasetsRoot(), datasetPaths);
    return NextResponse.json(inventories);
  } catch (error) {
    console.error('Failed to collect dataset inventory:', error);
    return NextResponse.json({ error: 'Failed to collect dataset inventory' }, { status: 500 });
  }
}
