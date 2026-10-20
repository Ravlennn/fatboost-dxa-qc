import { useCallback, useRef, useState, type DragEvent } from 'react'
import { filesFromDrop, type PickedFile } from '../api'
import DeskDrawing from '../hero/DeskDrawing'
import Densitometer, { type Checks } from '../hero/Densitometer'
import { LIMIT_TILT } from '../hero/model'
import { CRITERIA, fmt, plural } from '../labels'
import { IconAlert, IconFolder, IconLock, IconUpload } from './Icons'
import Validation from './Validation'

interface Props {
  onFiles: (files: PickedFile[]) => void
  uploading: { loaded: number; total: number; files: number } | null
  error: string
  onRetry: () => void
}

const mb = (b: number) => (b / 1024 / 1024).toLocaleString('ru-RU', { maximumFractionDigits: 1 })

type State = 'wait' | 'ok' | 'bad'
interface LogRow { code: string; title: string; limit: string; value: string; state: State }

/** The log's spine rows, filled in from what the phantom scan has measured so far. */
function spineRows(c: Checks | null): LogRow[] {
  const r = c?.reading
  const tiltKnown = r && !isNaN(r.tilt)
  const done = r?.top != null
  return [
    { code: 'spine_scan_range', title: 'Границы сканирования', limit: '½ Th12 … гребни',
      value: done ? (r!.th12 && r!.crests ? 'на снимке' : 'не захвачены') : r?.crests ? 'гребни на снимке' : '—',
      state: done ? (r!.th12 && r!.crests ? 'ok' : 'bad') : 'wait' },
    { code: 'spine_axis_tilt', title: 'Ось позвоночника', limit: `≤ ${LIMIT_TILT}°`,
      value: tiltKnown ? `${fmt(Math.abs(r!.tilt), 1)}°` : '—',
      state: tiltKnown ? (Math.abs(r!.tilt) > LIMIT_TILT ? 'bad' : 'ok') : 'wait' },
    { code: 'spine_artifact', title: 'Посторонние предметы', limit: 'нет', value: done ? 'не найдены' : '—', state: done ? 'ok' : 'wait' },
  ]
}

export default function DropStage({ onFiles, uploading, error, onRetry }: Props) {
  const [lit, setLit] = useState(false)
  const [checks, setChecks] = useState<Checks | null>(null)
  const depth = useRef(0)
  const filesInput = useRef<HTMLInputElement>(null)
  const folderInput = useRef<HTMLInputElement>(null)
  const onChecks = useCallback((c: Checks) => setChecks(c), [])

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
  const rows = spineRows(checks)

  return (
    <div
      className="stage"
      onDragEnter={(e) => { e.preventDefault(); depth.current += 1; setLit(true) }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => { depth.current -= 1; if (depth.current <= 0) setLit(false) }}
      onDrop={onDrop}
    >
      <DeskDrawing />
      <header className="masthead">
        <span className="wordmark">DXA<span>QC</span></span>
        <span className="masthead-sub">Журнал контроля качества снимков денситометрии</span>
        <span className="masthead-local"><IconLock width={15} height={15} />Снимки не покидают этот компьютер</span>
      </header>

      <main className="hero">
        <div className="hero-fig">
          <Densitometer onChecks={onChecks} />
          <p className="fig-note">Рис. 1 — модель фантома, не снимок пациента. Система измеряет его в реальном времени.</p>
            <table className="log" aria-label="Критерии ТЗ">
              <thead>
                <tr><th scope="col">Критерий ТЗ</th><th scope="col">Предел</th><th scope="col" className="num">Фантом, рис. 1</th></tr>
              </thead>
              <tbody>
                <tr className="log-group"><th colSpan={3} scope="rowgroup">Поясничный отдел L1–L4</th></tr>
                {rows.map((r) => (
                  <tr key={r.code} data-state={r.state}>
                    <th scope="row">{r.title}</th>
                    <td className="limit tabular">{r.limit}</td>
                    <td className="num tabular"><span className="value">{r.value}</span></td>
                  </tr>
                ))}
                <tr className="log-group"><th colSpan={3} scope="rowgroup">Проксимальный отдел бедра</th></tr>
                {CRITERIA.hip.map((c) => (
                  <tr key={c.code} data-state="hip">
                    <th scope="row">{c.title}</th>
                    <td className="limit" colSpan={2}>{c.code === 'hip_roi_margins' ? '3 см сверху и снизу, 2 см сбоку' : 'малый вертел слегка выступает'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
        </div>

        <section className="sheet" aria-labelledby="hero-title">
          <h1 id="hero-title">Каждый критерий ТЗ измерен прямо на&nbsp;снимке</h1>
          <p className="lead">
            Система находит позвонки и малый вертел, меряет наклон в градусах и поля в миллиметрах, отмечает брак
            с причиной и сводит пакет в контрольный листок.
          </p>

          <div className="insert" data-lit={lit || undefined} data-busy={uploading ? true : undefined}>
            <span className="corner c-tl" aria-hidden /><span className="corner c-tr" aria-hidden />
            <span className="corner c-bl" aria-hidden /><span className="corner c-br" aria-hidden />
            {uploading ? (
              <div className="insert-copy" role="status" aria-live="polite">
                <h2>Передаём {uploading.files} {plural(uploading.files, 'файл', 'файла', 'файлов')}</h2>
                <p className="muted tabular">{mb(uploading.loaded)} из {mb(uploading.total)} МБ</p>
                <div className="meter" aria-hidden><span style={{ transform: `scaleX(${pct / 100})` }} /></div>
              </div>
            ) : error ? (
              <div className="insert-copy" role="alert">
                <h2 className="tone-bad"><IconAlert width={20} height={20} />Пакет не загрузился</h2>
                <p className="muted">{error}</p>
                <div className="actions"><button className="btn" onClick={onRetry}>Попробовать снова</button></div>
              </div>
            ) : (
              <div className="insert-copy">
                <h2>{lit ? 'Отпустите — начнём проверку' : 'Перетащите исследования сюда'}</h2>
                <p className="muted">ZIP-архив, папка или файлы DICOM — позвоночник и бедро в одном пакете</p>
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
          </div>

          <ol className="next" aria-label="Что будет после загрузки">
            <li><span><strong>Вердикт и причина</strong> для каждого снимка: норма или брак, какой критерий нарушен и на сколько</span></li>
            <li><span><strong>Разметка на снимке:</strong> уровни L1–L4, ось, поля ROI, малый вертел — врач двигает точку, система пересчитывает</span></li>
            <li><span><strong>Отчёт по ТЗ</strong> в CSV и XLSX, протокол DICOM SR и файл исправлений врача</span></li>
          </ol>

          <a className="vlink" href="#validation"
             onClick={(e) => { e.preventDefault(); document.getElementById('validation')?.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' }) }}>
            <span className="vlink-num tabular">56 из 73</span>
            <span className="vlink-text">бракованных снимков найдено при проверке на 249 размеченных снимках — подробности ниже</span>
          </a>
        </section>
      </main>

      <Validation />

      <footer className="colophon">
        <span>Работает без сети</span>
        <span>Пять кодов нарушений по ТЗ</span>
        <span>Отчёт: CSV, XLSX, DICOM SR</span>
      </footer>
    </div>
  )
}
