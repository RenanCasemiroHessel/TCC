import { useState } from 'react'
import type { FormEvent } from 'react'

interface IdentifyAnimalProps {
  onSave: (data: { lote: string; brinco: string }) => void
  saving: boolean
}

export function IdentifyAnimal({ onSave, saving }: IdentifyAnimalProps) {
  const [lote, setLote] = useState('')
  const [brinco, setBrinco] = useState('')

  const trimmedBrinco = brinco.trim()

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (!trimmedBrinco || saving) return
    onSave({ lote: lote.trim(), brinco: trimmedBrinco })
  }

  return (
    <form className="identify reveal stagger-2" onSubmit={handleSubmit}>
      <span className="identify-label">Qual animal é esse?</span>

      <div className="identify-grid">
        <div className="identify-field">
          <label htmlFor="animal-lote" className="identify-sublabel">
            Lote (opcional)
          </label>
          <input
            id="animal-lote"
            type="text"
            className="identify-input"
            placeholder="ex.: B4"
            value={lote}
            onChange={(e) => setLote(e.target.value)}
            disabled={saving}
            autoComplete="off"
          />
        </div>
        <div className="identify-field identify-field-wide">
          <label htmlFor="animal-brinco" className="identify-sublabel">
            Nome ou nº do brinco
          </label>
          <input
            id="animal-brinco"
            type="text"
            className="identify-input"
            placeholder="ex.: 231 ou Mimosa"
            value={brinco}
            onChange={(e) => setBrinco(e.target.value)}
            disabled={saving}
            autoComplete="off"
          />
        </div>
      </div>

      <button type="submit" className="btn-save" disabled={!trimmedBrinco || saving}>
        {saving ? 'Salvando…' : 'Salvar no histórico'}
      </button>

      <p className="identify-hint">
        Sem identificação, essa estimativa não entra no histórico do animal.
      </p>
    </form>
  )
}
