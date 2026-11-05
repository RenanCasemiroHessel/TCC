export function validateImageFile(file: File, maxMb: number): string | null {
  if (!file.type.startsWith('image/')) {
    return 'Esse arquivo não é uma imagem. Envie um JPG ou PNG.'
  }
  if (file.size > maxMb * 1024 * 1024) {
    const sizeMb = (file.size / 1048576).toFixed(0)
    return `Imagem muito grande (${sizeMb} MB). O limite é ${maxMb} MB.`
  }
  return null
}
