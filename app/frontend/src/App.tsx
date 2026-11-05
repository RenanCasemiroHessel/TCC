import { useEffect, useState } from 'react'
import { TopBar } from './components/TopBar'
import { TabNav } from './components/TabNav'
import type { Tab } from './components/TabNav'
import { UploadStage } from './components/UploadStage'
import { ResultCard } from './components/ResultCard'
import { ErrorBanner } from './components/ErrorBanner'
import { HistoryPanel } from './components/HistoryPanel'
import { IdentifyAnimal } from './components/IdentifyAnimal'
import { predict } from './api'
import { useReducedMotion } from './hooks/useReducedMotion'
import { useHistory } from './hooks/useHistory'
import { validateImageFile } from './utils/validateFile'
import { downscaleImage } from './utils/thumbnail'
import type { PredictResponse, Sexo } from './types'

const HISTORY_THUMB_SIZE = 96

const MAX_MB = 25
const WAIT_HINT_DELAY_MS = 7000

function statusForResult(data: PredictResponse): string {
  const extrapola = data.peso_kg < data.treino_min || data.peso_kg > data.treino_max
  let msg = `Peso estimado: ${data.peso_kg.toFixed(1)} quilos. Faixa ${data.faixa}.`
  if (extrapola) msg += ' Fora da faixa que o modelo conhece bem, estimativa menos precisa.'
  return msg
}

function App() {
  const reduceMotion = useReducedMotion()
  const history = useHistory()

  const [activeTab, setActiveTab] = useState<Tab>('estimar')
  const [file, setFile] = useState<File | null>(null)
  const [sexo, setSexo] = useState<Sexo | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [showWaitHint, setShowWaitHint] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState('')
  const [saving, setSaving] = useState(false)
  const [savedAnimal, setSavedAnimal] = useState<{ lote: string; brinco: string } | null>(null)

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  useEffect(() => {
    if (!predicting) {
      setShowWaitHint(false)
      return
    }
    const timer = setTimeout(() => {
      setShowWaitHint(true)
      setStatus((s) => `${s} Isso pode levar até 1 minuto.`)
    }, WAIT_HINT_DELAY_MS)
    return () => clearTimeout(timer)
  }, [predicting])

  const handleFile = (f: File) => {
    const validationError = validateImageFile(f, MAX_MB)
    if (validationError) {
      setError(validationError)
      setStatus(validationError)
      return
    }

    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old)
      return URL.createObjectURL(f)
    })
    setFile(f)
    setSexo(null) // foto nova pode ser de outro animal
    setResult(null)
    setError(null)
    setSavedAnimal(null)
  }

  const handleSexo = (s: Sexo) => {
    setSexo(s)
    setResult(null) // o sexo entra no cálculo: o resultado anterior deixa de valer
    setError(null)
    setSavedAnimal(null)
  }

  const handlePredict = async () => {
    if (!file || !sexo || predicting) return
    setPredicting(true)
    setError(null)
    setResult(null)
    setSavedAnimal(null)
    setStatus('Analisando a imagem…')

    try {
      const data = await predict(file, sexo)
      setResult(data)
      setStatus(statusForResult(data))
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setError(message)
      setStatus(message)
    } finally {
      setPredicting(false)
    }
  }

  const handleSaveToHistory = async ({ lote, brinco }: { lote: string; brinco: string }) => {
    if (!result || saving) return
    setSaving(true)
    try {
      const thumbnail = await downscaleImage(
        `data:image/jpeg;base64,${result.img_segmentada}`,
        HISTORY_THUMB_SIZE,
        HISTORY_THUMB_SIZE,
      )
      history.addEntry({
        lote,
        brinco,
        timestamp: Date.now(),
        pesoKg: result.peso_kg,
        faixa: result.faixa,
        thumbnail,
      })
      setSavedAnimal({ lote, brinco })
      setStatus(`Peso salvo no histórico do animal ${brinco}${lote ? ` (lote ${lote})` : ''}.`)
    } catch {
      setError('Não foi possível salvar no histórico. Tente de novo.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <>
      <TopBar />
      <TabNav active={activeTab} historyCount={history.entries.length} onChange={setActiveTab} />
      <main>
        {activeTab === 'estimar' ? (
          <div id="panel-estimar" role="tabpanel" aria-labelledby="tab-estimar" tabIndex={0}>
            <div className="head">
              <h1>Estimativa de peso por imagem</h1>
              <p>
                Envie uma foto do animal e informe o sexo. O modelo segmenta o boi e estima
                o peso corporal.
              </p>
            </div>

            <UploadStage
              previewUrl={previewUrl}
              predicting={predicting}
              showWaitHint={showWaitHint}
              predictDisabled={!file || !sexo}
              sexo={sexo}
              onSexo={handleSexo}
              onFile={handleFile}
              onPredict={handlePredict}
            />

            {(error || result) && (
              <div className="result">
                {error && <ErrorBanner message={error} />}
                {result && <ResultCard data={result} reduceMotion={reduceMotion} />}

                {result && (
                  savedAnimal ? (
                    <p className="identify-saved reveal">
                      ✓ Salvo no histórico do animal <b>{savedAnimal.brinco}</b>
                      {savedAnimal.lote && <> (lote {savedAnimal.lote})</>}.
                    </p>
                  ) : (
                    <IdentifyAnimal onSave={handleSaveToHistory} saving={saving} />
                  )
                )}
              </div>
            )}

            <p className="sr-only" role="status" aria-live="polite">
              {status}
            </p>
          </div>
        ) : (
          <div id="panel-historico" role="tabpanel" aria-labelledby="tab-historico" tabIndex={0}>
            <HistoryPanel entries={history.entries} onClear={history.clear} />
          </div>
        )}

      </main>
    </>
  )
}

export default App
