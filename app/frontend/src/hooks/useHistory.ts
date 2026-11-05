import { useEffect, useState } from 'react'
import type { HistoryEntry } from '../types'

const STORAGE_KEY = 'peso-bovino-historico'
const MAX_ENTRIES = 20

// Formato anterior salvava só `animalId`. Migra pra `lote`/`brinco` ao carregar,
// pra registros antigos no localStorage do usuário não quebrarem o agrupamento.
interface StoredHistoryEntry {
  id?: string
  animalId?: string
  lote?: string
  brinco?: string
  timestamp?: number
  pesoKg?: number
  faixa?: string
  thumbnail?: string
}

function normalizeEntry(raw: StoredHistoryEntry): HistoryEntry | null {
  if (!raw.id || !raw.timestamp || typeof raw.pesoKg !== 'number' || !raw.faixa || !raw.thumbnail) {
    return null
  }
  return {
    id: raw.id,
    lote: raw.lote ?? '',
    brinco: raw.brinco ?? raw.animalId ?? 'não identificado',
    timestamp: raw.timestamp,
    pesoKg: raw.pesoKg,
    faixa: raw.faixa,
    thumbnail: raw.thumbnail,
  }
}

function loadEntries(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.map(normalizeEntry).filter((e): e is HistoryEntry => e !== null)
  } catch {
    return []
  }
}

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>(() => loadEntries())

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
    } catch {
      // localStorage unavailable or full — history just won't persist across reloads.
    }
  }, [entries])

  const addEntry = (entry: Omit<HistoryEntry, 'id'>) => {
    const withId: HistoryEntry = { ...entry, id: crypto.randomUUID() }
    setEntries((prev) => [withId, ...prev].slice(0, MAX_ENTRIES))
  }

  const clear = () => setEntries([])

  return { entries, addEntry, clear }
}
