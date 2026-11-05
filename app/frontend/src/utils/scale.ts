const SCALE_MIN = 0
const SCALE_MAX = 300

export function clampPct(kg: number): number {
  return Math.max(2, Math.min(98, ((kg - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * 100))
}

export function pillClass(faixa: string): string {
  if (faixa.startsWith('Leve')) return 'pill pill-leve'
  if (faixa.startsWith('Médio')) return 'pill pill-medio'
  return 'pill pill-pesado'
}
