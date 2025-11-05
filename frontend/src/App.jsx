import React, { useEffect, useMemo, useState } from 'react'
import UploadCard from './components/UploadCard'
import ResultsPanel from './components/ResultsPanel'
import PdfDownloadButton from './components/PdfDownloadButton'
import { health, predictPDF, predictJSON, downloadReport } from './api'

export default function App() {
  const [backendStatus, setBackendStatus] = useState('Checking backend…')
  const [file, setFile] = useState(null)
  const [threshold, setThreshold] = useState(0.5)
  const [modelPath, setModelPath] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [pdf, setPdf] = useState(null)
  const [json, setJson] = useState(null)

  useEffect(() => {
    let mounted = true
    health()
      .then((d) => mounted && setBackendStatus(`Online: ${new Date(d.timestamp).toLocaleString()}`))
      .catch(() => mounted && setBackendStatus('Backend unreachable'))
    return () => { mounted = false }
  }, [])

  const canSubmit = useMemo(() => !!file && !busy, [file, busy])

  async function handleGetPDF() {
    if (!file) return
    setError('')
    setBusy(true)
    setPdf(null)
    setJson(null)
    try {
      const { blob, filename } = await predictPDF({ file, modelPath, threshold })
      setPdf({ blob, filename })
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleGetJSON() {
    if (!file) return
    setError('')
    setBusy(true)
    setPdf(null)
    try {
      const data = await predictJSON({ file, modelPath, threshold })
      setJson(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  function clearAll() {
    setFile(null)
    setPdf(null)
    setJson(null)
    setError('')
  }

  return (
    <div className="container">
      <header className="header">
        <div className="logo">
          <div className="logo-badge">🩻</div>
          <div>
            Chest X-ray AI
            <div className="subtle" style={{ fontSize: 12 }}>{backendStatus}</div>
          </div>
        </div>
        <div className="header-cta">
          <a href="/" onClick={(e) => e.preventDefault()}>Docs</a>
          <a
            href="/doctor"
            target="_blank"
            rel="noopener noreferrer"
            style={{ marginLeft: 12 }}
          >
            Doctor Connect
          </a>
        </div>
      </header>

      <section className="hero">
        <div className="card">
          <h2>AI-assisted radiology report</h2>
          <p className="subtle">Upload a chest X-ray to get predictions for 14 thoracic diseases, Grad-CAM localization, and a downloadable PDF report.</p>

          <div className="grid-2" style={{ marginTop: 16 }}>
            <div>
              <UploadCard onFileSelected={setFile} />
            </div>
            <div className="card">
              <h2>Options</h2>
              <div style={{ marginTop: 8 }}>
                <label className="subtle">Probability threshold: {threshold}
                  <input type="range" min="0" max="1" step="0.05" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} style={{ width: '100%' }} />
                </label>
              </div>
              <div style={{ marginTop: 12 }}>
                <label className="subtle">Model path (optional)
                  <input value={modelPath} onChange={(e) => setModelPath(e.target.value)} placeholder="Model/final_model.pth" style={{ width: '100%', marginTop: 6, padding: 10, borderRadius: 10, border: '1px solid rgba(255,255,255,0.2)', background: 'rgba(0,0,0,0.2)', color: 'inherit' }} />
                </label>
              </div>

              {error ? (
                <div style={{ color: '#ffb4b4', marginTop: 10 }}>{error}</div>
              ) : null}

              <div className="btn-row" style={{ marginTop: 14 }}>
                <button className="btn-primary" disabled={!canSubmit} onClick={handleGetPDF}>
                  {busy ? 'Generating…' : 'Generate PDF report'}
                </button>
                <button className="btn-ghost" disabled={!canSubmit} onClick={handleGetJSON}>
                  {busy ? 'Please wait…' : 'Preview predictions (JSON)'}
                </button>
              </div>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gap: 16 }}>
          {pdf && (
            <PdfDownloadButton blob={pdf.blob} filename={pdf.filename} onClear={clearAll} />
          )}
          {json && (
            <>
              <ResultsPanel results={json} />
              {json.report_filename ? (
                <div className="card" style={{ display: 'grid', gap: 10 }}>
                  <h2>Report from preview</h2>
                  <p className="subtle">You can download the PDF generated during the JSON preview.</p>
                  <div>
                    <button
                      className="btn-primary"
                      onClick={async () => {
                        try {
                          setBusy(true)
                          const { blob, filename } = await downloadReport(json.report_filename)
                          const url = URL.createObjectURL(blob)
                          const a = document.createElement('a')
                          a.href = url
                          a.download = filename
                          document.body.appendChild(a)
                          a.click()
                          a.remove()
                          URL.revokeObjectURL(url)
                        } catch (e) {
                          setError(e.message)
                        } finally {
                          setBusy(false)
                        }
                      }}
                    >
                      Download PDF
                    </button>
                  </div>
                </div>
              ) : null}
            </>
          )}
        </div>
      </section>

      <footer className="footer">
        Built with React + Vite. For clinical decision support only; not a diagnostic device.
      </footer>
    </div>
  )
}
