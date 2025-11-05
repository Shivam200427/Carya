import React from 'react'

export default function ResultsPanel({ results }) {
  if (!results) return null
  const { predictions, probabilities, detected_diseases, keywords, report_filename, report_text } = results

  const sorted = Object.entries(probabilities || {})
    .sort((a, b) => b[1] - a[1])

  return (
    <div className="card">
      <h2>Predictions</h2>
      <p className="subtle">Model outputs with probability scores</p>

      <div className="list" style={{ marginTop: 10 }}>
        {sorted.map(([label, p]) => (
          <div className="list-item" key={label}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontWeight: 600 }}>{label}</span>
            </div>
            <span className={`badge ${p >= 0.5 ? 'success' : 'warn'}`}>{(p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      {detected_diseases?.length ? (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ marginTop: 0 }}>Detected diseases</h3>
          <div className="list">
            {detected_diseases.map((d) => (
              <div className="list-item" key={d}>{d}</div>
            ))}
          </div>
        </div>
      ) : null}

      {keywords?.length ? (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ marginTop: 0 }}>Report keywords</h3>
          <div className="list" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))' }}>
            {keywords.map((k) => (
              <div className="list-item" key={k}>{k}</div>
            ))}
          </div>
        </div>
      ) : null}

      {report_text ? (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ marginTop: 0 }}>Impression</h3>
          <p className="subtle" style={{ whiteSpace: 'pre-wrap' }}>{report_text}</p>
        </div>
      ) : null}

    </div>
  )
}
