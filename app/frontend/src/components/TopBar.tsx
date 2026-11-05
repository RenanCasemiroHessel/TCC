export function TopBar() {
  return (
    <header className="bar">
      <div className="bar-id">
        <span className="bar-mark">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M3 6h18M6 6l1.5 12h9L18 6M12 6v-2" />
          </svg>
        </span>
        Peso Bovino
      </div>
    </header>
  )
}
