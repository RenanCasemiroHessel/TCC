import { useEffect, useRef, useState } from 'react'

interface CountUpNumberProps {
  target: number
  durationMs: number
  reduceMotion: boolean
}

export function CountUpNumber({ target, durationMs, reduceMotion }: CountUpNumberProps) {
  const [display, setDisplay] = useState(() => (reduceMotion ? target : target * 0.4))
  const frameRef = useRef<number>(0)

  useEffect(() => {
    if (reduceMotion) {
      setDisplay(target)
      return
    }

    const from = target * 0.4
    const t0 = performance.now()

    const step = (now: number) => {
      const p = Math.min((now - t0) / durationMs, 1)
      const eased = 1 - Math.pow(1 - p, 3)
      setDisplay(from + (target - from) * eased)
      if (p < 1) {
        frameRef.current = requestAnimationFrame(step)
      } else {
        setDisplay(target)
      }
    }
    frameRef.current = requestAnimationFrame(step)

    return () => cancelAnimationFrame(frameRef.current)
  }, [target, durationMs, reduceMotion])

  return <span className="readout-num">{display.toFixed(1)}</span>
}
