import { useState } from 'react'
import type { DragEvent, KeyboardEvent } from 'react'

interface DropzoneProps {
  onOpen: () => void
  onFile: (file: File) => void
}

export function Dropzone({ onOpen, onFile }: DropzoneProps) {
  const [over, setOver] = useState(false)

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onOpen()
    }
  }

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setOver(true)
  }
  const onDragLeave = () => setOver(false)
  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) onFile(file)
  }

  return (
    <div
      className={`dropzone${over ? ' over' : ''}`}
      role="button"
      tabIndex={0}
      aria-label="Selecionar foto do animal — clique, pressione Enter ou arraste a imagem"
      onClick={onOpen}
      onKeyDown={onKeyDown}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <svg
        className="dz-icon"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M12 16.5V9.75m0 0 3 3m-3-3-3 3M6.75 19.5a4.5 4.5 0 0 1-1.41-8.775 5.25 5.25 0 0 1 10.338-2.32 5.75 5.75 0 0 1 .857 11.096" />
      </svg>
      <strong>Clique ou arraste a foto</strong>
      <span>Vista lateral do animal · JPG ou PNG</span>
    </div>
  )
}
