import type { KeyboardEvent } from 'react'

export type Tab = 'estimar' | 'historico'

const TABS: Tab[] = ['estimar', 'historico']

const LABEL: Record<Tab, string> = {
  estimar: 'Estimar',
  historico: 'Histórico',
}

interface TabNavProps {
  active: Tab
  historyCount: number
  onChange: (tab: Tab) => void
}

export function TabNav({ active, historyCount, onChange }: TabNavProps) {
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = TABS.indexOf(active)
    let next: Tab | null = null
    if (e.key === 'ArrowRight') next = TABS[(i + 1) % TABS.length]
    else if (e.key === 'ArrowLeft') next = TABS[(i - 1 + TABS.length) % TABS.length]
    else if (e.key === 'Home') next = TABS[0]
    else if (e.key === 'End') next = TABS[TABS.length - 1]

    if (next) {
      e.preventDefault()
      onChange(next)
      document.getElementById(`tab-${next}`)?.focus()
    }
  }

  return (
    <div className="tabnav" role="tablist" aria-label="Seções" onKeyDown={onKeyDown}>
      {TABS.map((tab) => (
        <button
          key={tab}
          id={`tab-${tab}`}
          type="button"
          role="tab"
          aria-selected={active === tab}
          aria-controls={`panel-${tab}`}
          tabIndex={active === tab ? 0 : -1}
          className={`tab-btn${active === tab ? ' active' : ''}`}
          onClick={() => onChange(tab)}
        >
          {LABEL[tab]}
          {tab === 'historico' && historyCount > 0 ? ` (${historyCount})` : ''}
        </button>
      ))}
    </div>
  )
}
