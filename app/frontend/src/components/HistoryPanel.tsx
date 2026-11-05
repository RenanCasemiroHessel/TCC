import { useMemo, useState } from 'react'
import type { HistoryEntry } from '../types'
import { pillClass } from '../utils/scale'

interface HistoryPanelProps {
  entries: HistoryEntry[]
  onClear: () => void
}

interface AnimalGroup {
  key: string
  lote: string
  brinco: string
  entries: HistoryEntry[] // ordenadas da mais antiga para a mais recente
  latest: HistoryEntry
}

function groupKey(lote: string, brinco: string): string {
  return `${lote.trim().toLowerCase()}::${brinco.trim().toLowerCase()}`
}

function formatTimestamp(ts: number): string {
  return new Date(ts).toLocaleString('pt-BR', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatDelta(delta: number): string {
  const sign = delta > 0 ? '+' : ''
  return `${sign}${delta.toFixed(1)} kg`
}

function groupByAnimal(entries: HistoryEntry[]): AnimalGroup[] {
  const map = new Map<string, HistoryEntry[]>()
  for (const entry of entries) {
    const key = groupKey(entry.lote, entry.brinco)
    const list = map.get(key)
    if (list) list.push(entry)
    else map.set(key, [entry])
  }
  const groups: AnimalGroup[] = []
  for (const [key, list] of map) {
    const sorted = [...list].sort((a, b) => a.timestamp - b.timestamp)
    const latest = sorted[sorted.length - 1]
    groups.push({ key, lote: latest.lote, brinco: latest.brinco, entries: sorted, latest })
  }
  return groups.sort((a, b) => b.latest.timestamp - a.latest.timestamp)
}

export function HistoryPanel({ entries, onClear }: HistoryPanelProps) {
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<string | null>(null)

  const groups = useMemo(() => groupByAnimal(entries), [entries])
  const selectedGroup = selected ? groups.find((g) => g.key === selected) ?? null : null

  const filteredGroups = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return groups
    return groups.filter(
      (g) => g.brinco.toLowerCase().includes(q) || g.lote.toLowerCase().includes(q),
    )
  }, [groups, query])

  const handleClear = () => {
    if (window.confirm(`Apagar as ${entries.length} estimativas salvas? Essa ação não pode ser desfeita.`)) {
      onClear()
      setSelected(null)
    }
  }

  if (entries.length === 0) {
    return (
      <div className="history-view reveal">
        <div className="head">
          <h1>Histórico de estimativas</h1>
          <p>Fotos em que o animal foi identificado ficam salvas aqui, neste navegador.</p>
        </div>
        <p className="history-empty">Nenhuma estimativa salva ainda.</p>
      </div>
    )
  }

  if (selectedGroup) {
    const totalGanho =
      selectedGroup.entries.length > 1
        ? selectedGroup.latest.pesoKg - selectedGroup.entries[0].pesoKg
        : null

    return (
      <div className="history-view reveal">
        <button type="button" className="history-back" onClick={() => setSelected(null)}>
          ← Todos os animais
        </button>

        <div className="head">
          <h1>{selectedGroup.brinco}</h1>
          <p>
            {selectedGroup.lote && <>Lote {selectedGroup.lote} · </>}
            {selectedGroup.entries.length}{' '}
            {selectedGroup.entries.length === 1 ? 'pesagem estimada' : 'pesagens estimadas'}
            {totalGanho !== null && (
              <>
                {' '}
                · ganho total:{' '}
                <b className={totalGanho >= 0 ? 'gain-pos' : 'gain-neg'}>
                  {formatDelta(totalGanho)}
                </b>
              </>
            )}
          </p>
        </div>

        <ul className="animal-timeline">
          {[...selectedGroup.entries].reverse().map((entry, idx, reversed) => {
            const older = reversed[idx + 1] // entrada anterior em ordem cronológica
            const delta = older ? entry.pesoKg - older.pesoKg : null
            return (
              <li key={entry.id} className="timeline-item">
                <img src={entry.thumbnail} alt="" className="history-thumb" />
                <div className="timeline-main">
                  <div className="timeline-row">
                    <span className="history-peso">{entry.pesoKg.toFixed(1)} kg</span>
                    <span className={pillClass(entry.faixa)}>{entry.faixa}</span>
                  </div>
                  {delta !== null && (
                    <span className={`timeline-delta ${delta >= 0 ? 'gain-pos' : 'gain-neg'}`}>
                      {formatDelta(delta)} desde a pesagem anterior
                    </span>
                  )}
                </div>
                <span className="history-time">{formatTimestamp(entry.timestamp)}</span>
              </li>
            )
          })}
        </ul>
      </div>
    )
  }

  return (
    <div className="history-view reveal">
      <div className="head">
        <h1>Histórico de estimativas</h1>
        <p>Agrupado por animal — toque em um para ver a evolução do peso.</p>
      </div>

      <input
        type="search"
        className="history-search"
        placeholder="Buscar por lote, nome ou nº do brinco…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        aria-label="Buscar no histórico por lote ou identificador do animal"
      />

      {filteredGroups.length === 0 ? (
        <p className="history-empty">Nenhum animal encontrado para "{query}".</p>
      ) : (
        <ul className="history-list">
          {filteredGroups.map((g) => (
            <li key={g.key}>
              <button
                type="button"
                className="history-item history-item-btn"
                onClick={() => setSelected(g.key)}
              >
                <img src={g.latest.thumbnail} alt="" className="history-thumb" />
                <span className="history-animal-wrap">
                  <span className="history-animal">{g.brinco}</span>
                  {g.lote && <span className="history-lote">Lote {g.lote}</span>}
                </span>
                <span className="history-peso">{g.latest.pesoKg.toFixed(1)} kg</span>
                <span className={pillClass(g.latest.faixa)}>{g.latest.faixa}</span>
                {g.entries.length > 1 && <span className="history-count">{g.entries.length}×</span>}
                <span className="history-time">{formatTimestamp(g.latest.timestamp)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}

      <button type="button" className="history-clear" onClick={handleClear}>
        Limpar histórico
      </button>
    </div>
  )
}
