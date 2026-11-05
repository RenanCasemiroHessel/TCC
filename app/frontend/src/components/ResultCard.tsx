import type { PredictResponse } from '../types'
import { pillClass } from '../utils/scale'
import { CountUpNumber } from './CountUpNumber'
import { ScaleTrack } from './ScaleTrack'

interface ResultCardProps {
  data: PredictResponse
  reduceMotion: boolean
}

export function ResultCard({ data, reduceMotion }: ResultCardProps) {
  const tMin = data.treino_min
  const tMax = data.treino_max
  const extrapola = data.peso_kg < tMin || data.peso_kg > tMax

  return (
    <div className="card reveal">
      <div className="card-top">
        <div className="readout-label">
          <span>Peso estimado</span>
          <span className={pillClass(data.faixa)}>{data.faixa}</span>
        </div>
        <div className="readout">
          <CountUpNumber target={data.peso_kg} durationMs={750} reduceMotion={reduceMotion} />
          <span className="readout-unit">kg</span>
        </div>
        <div className="readout-margin">
          Em testes, 8 de cada 10 estimativas erraram até <b>{data.margem_kg.toFixed(0)} kg</b>
        </div>
      </div>

      <ScaleTrack
        pesoKg={data.peso_kg}
        treinoMin={tMin}
        treinoMax={tMax}
        reduceMotion={reduceMotion}
      />

      {extrapola && (
        <div className="extrap">
          ⚠ Esse peso está fora da faixa que o modelo conhece bem — a estimativa pode ser menos
          precisa.
        </div>
      )}

      <div className="evidence reveal stagger-2">
        <div className="evidence-head">Evidência da segmentação</div>
        <div className="evidence-grid">
          <figure className="ev">
            <figcaption>
              <span className="ev-dot" />
              Original
            </figcaption>
            <img src={`data:image/jpeg;base64,${data.img_original}`} alt="Foto original enviada" />
          </figure>
          <figure className="ev">
            <figcaption>
              <span className="ev-dot on" />
              Segmentada
            </figcaption>
            <img
              src={`data:image/jpeg;base64,${data.img_segmentada}`}
              alt="Foto com o boi segmentado automaticamente"
            />
          </figure>
        </div>
      </div>
    </div>
  )
}
