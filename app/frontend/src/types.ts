export type Sexo = 'F' | 'M'

export interface PredictResponse {
  peso_kg: number
  faixa: string
  margem_kg: number
  treino_min: number
  treino_max: number
  img_original: string
  img_segmentada: string
}

export interface PredictError {
  erro: string
}

export interface HistoryEntry {
  id: string
  lote: string
  brinco: string
  timestamp: number
  pesoKg: number
  faixa: string
  thumbnail: string
}
