import React from 'react'

export default function PdfDownloadButton({ blob, filename, onClear }) {
  if (!blob) return null
  const url = URL.createObjectURL(blob)

  return (
    <div className="card" style={{ textAlign: 'center' }}>
      <h2>Report Ready</h2>
      <p className="subtle">Your PDF report has been generated. You can download it now.</p>
      <div className="btn-row" style={{ justifyContent: 'center', marginTop: 8 }}>
        <a className="btn-primary" href={url} download={filename}>Download PDF</a>
        <button className="btn-ghost" onClick={onClear}>Start new</button>
      </div>
    </div>
  )
}
