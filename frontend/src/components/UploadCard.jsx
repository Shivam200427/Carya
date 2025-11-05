import React, { useCallback, useMemo, useState } from 'react'
import { useDropzone } from 'react-dropzone'

export default function UploadCard({ onFileSelected }) {
  const [error, setError] = useState('')
  const [preview, setPreview] = useState('')

  const onDrop = useCallback((accepted, rejected) => {
    setError('')
    if (rejected?.length) {
      setError('Unsupported file type or too large. Please upload a PNG/JPG/TIFF under 16MB.')
      return
    }
    const f = accepted[0]
    if (f) {
      setPreview(URL.createObjectURL(f))
      onFileSelected?.(f)
    }
  }, [onFileSelected])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    },
    multiple: false,
    maxSize: 16 * 1024 * 1024,
    onDrop
  })

  const dzClass = useMemo(() => `dropzone ${isDragActive ? 'active' : ''}`, [isDragActive])

  return (
    <div className="card">
      <h2>Upload Chest X-ray</h2>
      <p className="subtle">PNG, JPG, BMP, or TIFF. Max 16MB.</p>

      <div {...getRootProps({ className: dzClass })}>
        <input {...getInputProps()} />
        <div>
          <h3 style={{ margin: 0 }}>Drag & drop here</h3>
          <p className="subtle">or click to browse your files</p>
        </div>
      </div>

      {error && (
        <div style={{ color: '#ffb4b4', marginTop: 10 }}>{error}</div>
      )}

      {preview && (
        <div style={{ marginTop: 16 }}>
          <img className="preview" src={preview} alt="preview" />
        </div>
      )}
    </div>
  )
}
