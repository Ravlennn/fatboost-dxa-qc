import { useRef, useState, type DragEvent } from 'react'
import { filesFromDrop, type PickedFile } from '../api'
import { CRITERIA, plural } from '../labels'
import { IconAlert, IconFolder, IconLock, IconUpload } from './Icons'

interface Props {
  onFiles: (files: PickedFile[]) => void
  uploading: { loaded: number; total: number; files: number } | null
  error: string
  onRetry: () => void
}

const mb = (b: number) => (b / 1024 / 1024).toLocaleString('ru-RU', { maximumFractionDigits: 1 })

export default function DropStage({ onFiles, uploading, error, onRetry }: Props) {
  const [lit, setLit] = useState(false)
  const depth = useRef(0)
  const filesInput = useRef<HTMLInputElement>(null)
  const folderInput = useRef<HTMLInputElement>(null)

  const onDrop = async (e: DragEvent) => {
    e.preventDefault()
    depth.current = 0
    setLit(false)
    if (uploading) return
    onFiles(await filesFromDrop(e.dataTransfer))
  }
  const pick = (list: FileList | null) => {
    if (!list) return
    onFiles(Array.from(list).map((file) => ({ file, path: file.webkitRelativePath || file.name })))
  }
  const pct = uploading && uploading.total ? Math.min(100, (uploading.loaded / uploading.total) * 100) : 0

  return (
    <div
      className="stage"
      onDragEnter={(e) => { e.preventDefault(); depth.current += 1; setLit(true) }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => { depth.current -= 1; if (depth.current <= 0) setLit(false) }}
      onDrop={onDrop}
    >
      <header className="stage-top">
        <span className="wordmark">DXA<span>QC</span></span>
        <span className="stage-local"><IconLock width={15} height={15} />Снимки не покидают этот компьютер</span>
      </header>

      <section className="lightbox" data-lit={lit || undefined} data-busy={uploading ? true : undefined} aria-label="Загрузка исследований">
        <span className="clip clip-l" aria-hidden />
        <span className="clip clip-r" aria-hidden />
        {uploading ? (
          <div className="lightbox-copy" role="status" aria-live="polite">
            <h1>Передаём {uploading.files} {plural(uploading.files, 'файл', 'файла', 'файлов')}</h1>
            <p className="muted tabular">{mb(uploading.loaded)} из {mb(uploading.total)} МБ</p>
            <div className="meter" aria-hidden><span style={{ transform: `scaleX(${pct / 100})` }} /></div>
          </div>
        ) : error ? (
          <div className="lightbox-copy" role="alert">
            <IconAlert width={28} height={28} className="tone-bad" />
            <h1>Пакет не загрузился</h1>
            <p className="muted">{error}</p>
            <button className="btn" onClick={onRetry}>Попробовать снова</button>
          </div>
        ) : (
          <div className="lightbox-copy">
            <h1>{lit ? 'Отпустите — начнём проверку' : 'Перетащите исследования сюда'}</h1>
            <p className="muted">ZIP-архив, папка или отдельные файлы DICOM — позвоночник и бедро</p>
            <div className="actions">
              <button className="btn btn-primary" onClick={() => filesInput.current?.click()}>
                <IconUpload />Выбрать файлы
              </button>
              <button className="btn" onClick={() => folderInput.current?.click()}>
                <IconFolder />Выбрать папку
              </button>
            </div>
            <input ref={filesInput} type="file" multiple hidden accept=".zip,.dcm,application/dicom,application/zip,*/*"
                   onChange={(e) => pick(e.target.files)} />
            <input ref={folderInput} type="file" hidden onChange={(e) => pick(e.target.files)}
                   {...({ webkitdirectory: '', directory: '' } as Record<string, string>)} />
          </div>
        )}
      </section>

      <footer className="criteria" aria-label="Что проверяется">
        {(['spine', 'hip'] as const).map((f) => (
          <div key={f} className="criteria-group">
            <h2>{f === 'spine' ? 'Поясничный отдел' : 'Проксимальный отдел бедра'}</h2>
            <ul>
              {CRITERIA[f].map((c) => (
                <li key={c.code}><strong>{c.title}</strong><span>{c.rule}</span></li>
              ))}
            </ul>
          </div>
        ))}
      </footer>
    </div>
  )
}
