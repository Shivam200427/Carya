const base = import.meta.env.VITE_API_BASE_URL || '/api'

export async function health() {
  const r = await fetch(`${base}/health`)
  return r.json()
}

export async function predictPDF({ file, modelPath, threshold }) {
  const form = new FormData()
  form.append('file', file)
  if (modelPath) form.append('model_path', modelPath)
  if (threshold !== undefined && threshold !== null) form.append('threshold', String(threshold))

  const resp = await fetch(`${base}/predict`, {
    method: 'POST',
    body: form,
  })

  if (!resp.ok) {
    const maybeJson = await safeJson(resp)
    throw new Error(maybeJson?.error || `Server error (${resp.status})`)
  }
  const blob = await resp.blob()
  const filename = getFilenameFromDisposition(resp.headers.get('Content-Disposition')) || 'Chest_Report.pdf'
  return { blob, filename }
}

export async function predictJSON({ file, modelPath, threshold }) {
  const form = new FormData()
  form.append('file', file)
  if (modelPath) form.append('model_path', modelPath)
  if (threshold !== undefined && threshold !== null) form.append('threshold', String(threshold))

  const resp = await fetch(`${base}/predict_json`, {
    method: 'POST',
    body: form,
  })
  const data = await resp.json()
  if (!resp.ok || data.success === false) {
    throw new Error(data.error || `Server error (${resp.status})`)
  }
  return data
}

export async function downloadReport(filename) {
  const url = `${base}/download_report/${encodeURIComponent(filename)}`
  const resp = await fetch(url)
  if (!resp.ok) throw new Error('Report not found')
  const blob = await resp.blob()
  return { blob, filename }
}

function getFilenameFromDisposition(disposition) {
  if (!disposition) return null
  const match = /filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i.exec(disposition)
  return decodeURIComponent(match?.[1] || match?.[2] || '')
}

async function safeJson(resp) {
  try { return await resp.json() } catch { return null }
}
