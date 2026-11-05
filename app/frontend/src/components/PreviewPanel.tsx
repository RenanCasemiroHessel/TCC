interface PreviewPanelProps {
  previewUrl: string
  onTrocar: () => void
}

export function PreviewPanel({ previewUrl, onTrocar }: PreviewPanelProps) {
  return (
    <div className="preview-wrap">
      <img className="preview-img" src={previewUrl} alt="Pré-visualização da foto enviada" />
      <button type="button" className="chip-troca" onClick={onTrocar}>
        Trocar foto
      </button>
    </div>
  )
}
