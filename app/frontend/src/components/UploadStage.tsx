import { useRef } from 'react'
import type { ChangeEvent } from 'react'
import { Dropzone } from './Dropzone'
import { PreviewPanel } from './PreviewPanel'
import type { Sexo } from '../types'

interface UploadStageProps {
  previewUrl: string | null
  predicting: boolean
  showWaitHint: boolean
  predictDisabled: boolean
  sexo: Sexo | null
  onSexo: (sexo: Sexo) => void
  onFile: (file: File) => void
  onPredict: () => void
}

export function UploadStage({
  previewUrl,
  predicting,
  showWaitHint,
  predictDisabled,
  sexo,
  onSexo,
  onFile,
  onPredict,
}: UploadStageProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  const openPicker = () => inputRef.current?.click()

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onFile(file)
    e.target.value = ''
  }

  return (
    <div className="stage">
      <input
        ref={inputRef}
        type="file"
        className="file-input"
        accept="image/*"
        onChange={onChange}
        tabIndex={-1}
        aria-hidden="true"
      />

      {previewUrl ? (
        <PreviewPanel previewUrl={previewUrl} onTrocar={openPicker} />
      ) : (
        <Dropzone onOpen={openPicker} onFile={onFile} />
      )}

      <fieldset className="sexo" disabled={predicting}>
        <legend className="sexo-label">Sexo do animal</legend>
        <div className="sexo-options">
          {(
            [
              ['F', 'Fêmea'],
              ['M', 'Macho'],
            ] as const
          ).map(([valor, rotulo]) => (
            <label key={valor} className="sexo-option">
              <input
                type="radio"
                name="sexo"
                value={valor}
                checked={sexo === valor}
                onChange={() => onSexo(valor)}
              />
              <span>{rotulo}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <button
        className={`btn-predict${predicting ? ' is-loading' : ''}`}
        disabled={predictDisabled}
        aria-busy={predicting}
        onClick={onPredict}
      >
        {predicting ? (
          <>
            <span className="spin" aria-hidden="true" /> Analisando…
          </>
        ) : (
          'Estimar peso'
        )}
      </button>

      {showWaitHint && (
        <p className="predict-hint reveal">
          Isso pode levar até 1 minuto — a análise roda no servidor.
        </p>
      )}
    </div>
  )
}
