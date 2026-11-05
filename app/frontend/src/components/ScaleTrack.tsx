import { useEffect, useState } from 'react'
import { clampPct } from '../utils/scale'

interface ScaleTrackProps {
  pesoKg: number
  treinoMin: number
  treinoMax: number
  reduceMotion: boolean
}

export function ScaleTrack({ pesoKg, treinoMin, treinoMax, reduceMotion }: ScaleTrackProps) {
  const target = clampPct(pesoKg)
  const [markerPct, setMarkerPct] = useState(reduceMotion ? target : 0)

  useEffect(() => {
    if (reduceMotion) {
      setMarkerPct(target)
      return
    }
    const raf1 = requestAnimationFrame(() => {
      const raf2 = requestAnimationFrame(() => setMarkerPct(target))
      return () => cancelAnimationFrame(raf2)
    })
    return () => cancelAnimationFrame(raf1)
  }, [target, reduceMotion])

  const bandL = clampPct(treinoMin)
  const bandR = clampPct(treinoMax)

  return (
    <div className="scale">
      <div className="scale-track">
        <div className="scale-band" style={{ left: `${bandL}%`, right: `${100 - bandR}%` }} />
        <div className="scale-marker" style={{ left: `${markerPct}%` }} />
      </div>
      <div className="scale-ticks">
        <span>0</span>
        <span>60</span>
        <span>120</span>
        <span>180</span>
        <span>240</span>
        <span>300</span>
      </div>
      <div className="scale-legend">
        <span className="legend-swatch" />
        faixa que o modelo conhece bem ({treinoMin.toFixed(0)}–{treinoMax.toFixed(0)} kg)
      </div>
    </div>
  )
}
