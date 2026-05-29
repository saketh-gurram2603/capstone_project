import { useState, useCallback, useRef } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, FileSpreadsheet, CheckCircle, X, AlertTriangle,
  Database, Clock, SkipForward, RefreshCw, Info,
} from 'lucide-react'
import { uploadFile, getIngestStatus } from '../api/ingestionApi'

export default function IngestionPage() {
  const [dragOver, setDragOver]       = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const upload = useMutation({
    mutationFn: (file: File) => uploadFile(file),
    onSuccess: () => {
      setSelectedFile(null)
      refetchStatus()
    },
  })

  const { data: statusData, refetch: refetchStatus } = useQuery({
    queryKey: ['ingest-status'],
    queryFn:  getIngestStatus,
    staleTime: 10_000,
  })

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && /\.(xlsx|xls)$/i.test(file.name)) {
      setSelectedFile(file)
      upload.reset()
    }
  }, [upload])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) { setSelectedFile(file); upload.reset() }
    e.target.value = ''  // allow re-selecting the same file
  }

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation()
    setSelectedFile(null)
    upload.reset()
  }

  return (
    <div className="flex flex-col gap-5" style={{ maxWidth: 680 }}>

      {/* ── Header card ──────────────────────────────────────── */}
      <div className="panel" style={{ padding: '18px 24px' }}>
        <div className="flex items-start gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
               style={{ background: 'rgba(79,140,255,0.12)', border: '1px solid rgba(79,140,255,0.2)' }}>
            <Database className="w-4 h-4" style={{ color: '#4F8CFF' }} />
          </div>
          <div>
            <h2 className="font-semibold text-[14px]" style={{ color: 'var(--text-primary)' }}>
              Ingest Incident Dataset
            </h2>
            <p className="text-[12px] mt-0.5" style={{ color: 'var(--text-secondary)' }}>
              Upload an XLSX file to populate the Qdrant vector store and rebuild the BM25 keyword index.
              New records are immediately searchable and triageable.
            </p>
          </div>
        </div>

        {/* Format requirements */}
        <div className="mt-4 pt-4 grid grid-cols-3 gap-4 text-[11px]"
             style={{ borderTop: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
          {[
            { label: 'Required columns',  value: 'incident_id · title · description' },
            { label: 'Accepted formats',  value: '.xlsx  ·  .xls'                    },
            { label: 'Max file size',     value: '50 MB'                             },
          ].map(({ label, value }) => (
            <div key={label}>
              <p className="section-label mb-0.5">{label}</p>
              <p className="mono">{value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── Drop zone ────────────────────────────────────────── */}
      <div
        onDrop={handleDrop}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => !upload.isPending && fileInputRef.current?.click()}
        className="rounded-2xl border-2 border-dashed transition-all duration-200 flex flex-col items-center justify-center py-14 px-8"
        style={{
          borderColor: dragOver
            ? '#4F8CFF'
            : selectedFile
              ? 'rgba(35,198,168,0.5)'
              : 'var(--border-strong)',
          background: dragOver
            ? 'rgba(79,140,255,0.06)'
            : selectedFile
              ? 'rgba(35,198,168,0.04)'
              : 'var(--surface)',
          cursor: upload.isPending ? 'default' : 'pointer',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".xlsx,.xls"
          className="hidden"
          onChange={handleFileChange}
          disabled={upload.isPending}
        />

        <AnimatePresence mode="wait">
          {selectedFile ? (
            <motion.div
              key="selected"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="flex flex-col items-center gap-3 text-center"
            >
              <FileSpreadsheet className="w-10 h-10" style={{ color: '#23C6A8' }} />
              <div>
                <p className="font-semibold text-[14px]" style={{ color: 'var(--text-primary)' }}>
                  {selectedFile.name}
                </p>
                <p className="text-[12px] mt-0.5" style={{ color: 'var(--text-secondary)' }}>
                  {selectedFile.size > 1024 * 1024
                    ? `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB`
                    : `${(selectedFile.size / 1024).toFixed(1)} KB`}
                </p>
              </div>
              {!upload.isPending && (
                <button className="btn-ghost text-[11px] mt-1" style={{ color: '#F05A5A' }}
                        onClick={clearFile}>
                  <X className="w-3 h-3" /> Remove
                </button>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="flex flex-col items-center gap-3 text-center"
            >
              <Upload className="w-10 h-10" style={{ color: 'var(--text-secondary)' }} />
              <div>
                <p className="font-semibold text-[14px]" style={{ color: 'var(--text-primary)' }}>
                  Drop your XLSX file here
                </p>
                <p className="text-[12px] mt-1" style={{ color: 'var(--text-secondary)' }}>
                  or click to browse · accepts .xlsx and .xls
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Upload button ─────────────────────────────────────── */}
      <AnimatePresence>
        {selectedFile && (
          <motion.button
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="btn-primary py-2.5 justify-center"
            disabled={upload.isPending}
            onClick={() => upload.mutate(selectedFile)}
          >
            {upload.isPending ? (
              <>
                <span className="w-4 h-4 rounded-full border-2 animate-spin"
                      style={{ borderColor: 'rgba(255,255,255,0.25)', borderTopColor: '#fff' }} />
                Processing dataset…
              </>
            ) : (
              <><Database className="w-4 h-4" /> Ingest Dataset</>
            )}
          </motion.button>
        )}
      </AnimatePresence>

      {/* ── Error ─────────────────────────────────────────────── */}
      <AnimatePresence>
        {upload.isError && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="card-sm flex items-start gap-2.5 text-[13px]"
            style={{ borderColor: 'rgba(240,90,90,0.25)', background: 'rgba(240,90,90,0.06)', color: '#F05A5A' }}
          >
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>{(upload.error as Error).message}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Success result ────────────────────────────────────── */}
      <AnimatePresence>
        {upload.data && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="card space-y-4"
            style={{ borderColor: 'rgba(35,198,168,0.2)', background: 'rgba(35,198,168,0.03)' }}
          >
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5" style={{ color: '#23C6A8' }} />
              <span className="font-semibold text-[14px]" style={{ color: 'var(--text-primary)' }}>
                Ingestion Complete
              </span>
              <span className="badge badge-teal ml-auto">success</span>
            </div>
            <div className="divider" />
            <div className="grid grid-cols-3 gap-4">
              {[
                { label: 'Records Ingested', value: upload.data.ingested,                       icon: Database,    color: '#23C6A8' },
                { label: 'Records Skipped',  value: upload.data.skipped,                        icon: SkipForward, color: '#F4B740' },
                { label: 'Duration',         value: `${upload.data.duration_ms.toFixed(0)} ms`, icon: Clock,       color: '#4F8CFF' },
              ].map(({ label, value, icon: Icon, color }) => (
                <div key={label} className="card-sm text-center space-y-1.5">
                  <Icon className="w-4 h-4 mx-auto" style={{ color }} />
                  <p className="text-[22px] font-bold tabular-nums leading-none"
                     style={{ color: 'var(--text-primary)' }}>
                    {value}
                  </p>
                  <p className="text-[10px] uppercase tracking-wide"
                     style={{ color: 'var(--text-secondary)' }}>
                    {label}
                  </p>
                </div>
              ))}
            </div>
            <div className="flex items-start gap-2 text-[12px]" style={{ color: 'var(--text-secondary)' }}>
              <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" style={{ color: '#4F8CFF' }} />
              <span>
                Qdrant collection and BM25 index updated. New records are immediately searchable.
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Previous ingestion status ─────────────────────────── */}
      {statusData && statusData.status !== 'idle' && !upload.data && (
        <div className="card-sm space-y-2">
          <div className="flex items-center justify-between">
            <p className="section-label">Previous Ingestion Run</p>
            <button className="btn-ghost p-1" onClick={() => refetchStatus()}>
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="flex items-center gap-3 flex-wrap text-[12px]"
               style={{ color: 'var(--text-secondary)' }}>
            <span className={`badge ${
              statusData.status === 'completed' ? 'badge-teal'     :
              statusData.status === 'failed'    ? 'badge-critical' :
              'badge-blue'
            }`}>
              {statusData.status}
            </span>
            <span>
              <strong style={{ color: 'var(--text-primary)' }}>{statusData.ingested}</strong> ingested
            </span>
            <span>
              <strong style={{ color: 'var(--text-primary)' }}>{statusData.skipped}</strong> skipped
            </span>
            {statusData.duration_ms != null && (
              <span>{statusData.duration_ms.toFixed(0)} ms</span>
            )}
          </div>
          {statusData.error && (
            <p className="text-[11px] mt-1" style={{ color: '#F05A5A' }}>{statusData.error}</p>
          )}
          {statusData.completed_at && (
            <p className="text-[10px]" style={{ color: '#374151' }}>
              {new Date(statusData.completed_at).toLocaleString()}
            </p>
          )}
        </div>
      )}

    </div>
  )
}
