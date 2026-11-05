import type { PredictError, PredictResponse, Sexo } from './types'

export async function predict(file: File, sexo: Sexo): Promise<PredictResponse> {
  const form = new FormData()
  form.append('imagem', file)
  form.append('sexo', sexo)

  let resp: Response
  try {
    resp = await fetch('/predict', { method: 'POST', body: form })
  } catch {
    throw new Error(
      'Não foi possível conectar ao servidor. Verifique sua conexão e tente novamente.',
    )
  }

  let data: PredictResponse | PredictError
  try {
    data = (await resp.json()) as PredictResponse | PredictError
  } catch {
    throw new Error('O servidor respondeu de forma inesperada. Tente novamente em instantes.')
  }

  if ('erro' in data) {
    throw new Error(data.erro)
  }
  if (!resp.ok) {
    throw new Error('Algo deu errado no servidor. Tente novamente em instantes.')
  }
  return data
}
