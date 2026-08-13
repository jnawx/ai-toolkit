'use client';

import { Loader2, Pencil, Plus, Save, Trash2, X } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

import { Modal } from './Modal';
import { apiClient } from '@/utils/api';

export type CharacterIdentity = {
  id: string;
  display_name: string;
  trigger_word: string;
  class_prompt: string;
  caption_description: string;
};

type Props = { open: boolean; onClose: () => void };
type Draft = { displayName: string; triggerWord: string; classPrompt: string; captionDescription: string };

const emptyDraft = (): Draft => ({
  displayName: '',
  triggerWord: '',
  classPrompt: 'a person',
  captionDescription: '',
});

const identityDraft = (identity: CharacterIdentity): Draft => ({
  displayName: identity.display_name,
  triggerWord: identity.trigger_word,
  classPrompt: identity.class_prompt,
  captionDescription: identity.caption_description,
});

export default function CharacterIdentityManager({ open, onClose }: Props) {
  const [identities, setIdentities] = useState<CharacterIdentity[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft>(emptyDraft);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const request = useCallback(async (action: string, extra: Record<string, unknown> = {}) => {
    const response = await apiClient.post('/api/datasets/characterDop', { action, ...extra });
    return response.data;
  }, []);

  const reload = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await request('list-identities');
      setIdentities(result.identities ?? []);
    } catch (reason: any) {
      setError(reason?.response?.data?.error || reason.message || 'Identities could not be loaded');
    } finally {
      setBusy(false);
    }
  }, [request]);

  useEffect(() => {
    if (open) void reload();
  }, [open, reload]);

  const save = async () => {
    if (!draft.displayName.trim() || !draft.triggerWord.trim() || !draft.classPrompt.trim() || !draft.captionDescription.trim()) {
      setError('Display name, trigger word, DOP class, and caption description are required.');
      return;
    }
    const identityId = editingId ?? `${draft.triggerWord.toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 48)}-${crypto.randomUUID().slice(0, 8)}`;
    setBusy(true);
    setError(null);
    try {
      const result = await request(editingId ? 'update-global-identity' : 'create-global-identity', {
        identityId,
        displayName: draft.displayName.trim(),
        triggerWord: draft.triggerWord.trim(),
        classPrompt: draft.classPrompt.trim(),
        captionDescription: draft.captionDescription.trim(),
      });
      setIdentities(result.identities ?? []);
      setEditingId(null);
      setDraft(emptyDraft());
    } catch (reason: any) {
      setError(reason?.response?.data?.error || reason.message || 'Identity could not be saved');
    } finally {
      setBusy(false);
    }
  };

  const remove = async (identity: CharacterIdentity) => {
    if (!window.confirm(`Delete ${identity.display_name} everywhere, including all masks and speaking intervals?`)) return;
    setBusy(true);
    setError(null);
    try {
      const result = await request('delete-global-identity', { identityId: identity.id });
      setIdentities(result.identities ?? []);
      if (editingId === identity.id) {
        setEditingId(null);
        setDraft(emptyDraft());
      }
    } catch (reason: any) {
      setError(reason?.response?.data?.error || reason.message || 'Identity could not be deleted');
    } finally {
      setBusy(false);
    }
  };

  const fieldClass = 'w-full rounded border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none focus:border-violet-500';
  return (
    <Modal isOpen={open} onClose={onClose} title="Identity Management" size="xl">
      <div className="grid max-h-[75vh] gap-5 overflow-y-auto lg:grid-cols-[minmax(0,1fr)_minmax(20rem,0.8fr)]">
        <section className="space-y-2">
          <p className="text-xs leading-relaxed text-gray-400">
            Identities are shared across every dataset. Their DOP class stays broad; their caption description replaces the trigger whenever that person is visible but is not the active training identity.
          </p>
          {busy && identities.length === 0 && <Loader2 className="animate-spin text-violet-400" />}
          {identities.map(identity => (
            <div key={identity.id} className="rounded-lg border border-gray-700 bg-gray-950 p-3">
              <div className="flex items-start gap-2">
                <div className="min-w-0 flex-1">
                  <div className="font-medium text-gray-100">{identity.display_name}</div>
                  <div className="text-xs text-violet-300">{identity.trigger_word}</div>
                </div>
                <button type="button" className="rounded p-1.5 text-gray-400 hover:bg-gray-800 hover:text-white" onClick={() => { setEditingId(identity.id); setDraft(identityDraft(identity)); }} aria-label={`Edit ${identity.display_name}`}><Pencil size={15} /></button>
                <button type="button" className="rounded p-1.5 text-gray-400 hover:bg-red-950 hover:text-red-300" onClick={() => void remove(identity)} aria-label={`Delete ${identity.display_name}`}><Trash2 size={15} /></button>
              </div>
              <div className="mt-2 text-[11px] text-gray-500">DOP: {identity.class_prompt}</div>
              <div className="mt-1 text-xs leading-relaxed text-gray-300">{identity.caption_description}</div>
            </div>
          ))}
          {!busy && identities.length === 0 && <p className="rounded border border-dashed border-gray-700 p-4 text-center text-sm text-gray-500">No shared identities yet.</p>}
        </section>
        <section className="space-y-3 rounded-lg border border-gray-700 bg-gray-800/60 p-4">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-gray-100">{editingId ? 'Edit identity' : 'Add identity'}</h3>
            {editingId && <button type="button" className="text-gray-400 hover:text-white" onClick={() => { setEditingId(null); setDraft(emptyDraft()); }}><X size={17} /></button>}
          </div>
          <input className={fieldClass} value={draft.displayName} onChange={event => setDraft(value => ({ ...value, displayName: event.target.value }))} placeholder="Display name" />
          <input className={fieldClass} value={draft.triggerWord} onChange={event => setDraft(value => ({ ...value, triggerWord: event.target.value }))} placeholder="Unique trigger word" />
          <input className={fieldClass} value={draft.classPrompt} onChange={event => setDraft(value => ({ ...value, classPrompt: event.target.value }))} placeholder="Broad DOP class, e.g. a woman" />
          <textarea className={`${fieldClass} min-h-32 resize-y`} value={draft.captionDescription} onChange={event => setDraft(value => ({ ...value, captionDescription: event.target.value }))} placeholder="Rich global description used when this identity is not active" />
          <button type="button" disabled={busy} onClick={() => void save()} className="flex w-full items-center justify-center gap-2 rounded bg-violet-700 px-3 py-2 text-sm font-medium text-white hover:bg-violet-600 disabled:opacity-50">
            {busy ? <Loader2 size={16} className="animate-spin" /> : editingId ? <Save size={16} /> : <Plus size={16} />}
            {editingId ? 'Update identity' : 'Add identity'}
          </button>
          {error && <p className="rounded border border-red-800 bg-red-950/40 p-2 text-xs text-red-200">{error}</p>}
        </section>
      </div>
    </Modal>
  );
}
