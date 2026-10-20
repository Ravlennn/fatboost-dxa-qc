import { useEffect, useRef, useState } from 'react'
import { correctionsUrl, reportUrl, srUrl } from '../api'
import type { Batch, Phase, Row } from '../App'
import { plural } from '../labels'
import { IconAlert, IconDownload, IconReviewed, IconUpload } from './Icons'
import type { Filter } from './ResultList'

interface Props {
  batch: Batch
  rows: Row[]
  phase: Phase
  error: string
  filter: Filter
  onFilter: (f: Filter) => void
  onNew: () => void
}

const clock = (s: number) => {
  const m = Math.floor(s / 60)
  return m ? `${m} мин ${Math.round(s % 60)} с` : `${Math.round(s)} с`
}

export default function Header({ batch, rows, phase, error, filter, onFilter, onNew }: Props) {
  const done = batch.items.size
  const total = batch.total ?? rows.length
  const items = Array.from(batch.items.values())
  const ok = items.filter((i) => i.status === 'Success' && i.quality === 0).length
  const bad = items.filter((i) => i.status === 'Success' && i.quality === 1).length
  const fail = items.filter((i) => i.status === 'Failure').length
  const decidable = ok + bad
  const reviewed = items.filter((i) => i.review).length
  const ready = phase === 'done'
  const pct = total ? done / total : 0
  // the tally is the verdict filter: one set of numbers, one place to click
  const tally: { key: Filter; label: string; n: number; tone?: string }[] = [
    { key: 'all', label: 'Все', n: rows.length },
    { key: 'ok', label: 'Норма', n: ok, tone: 'tone-ok' },
    { key: 'bad', label: 'Брак', n: bad, tone: 'tone-bad' },
    { key: 'fail', label: 'Ошибки', n: fail, tone: 'tone-fail' },
  ]

  return (
    <header className="topbar">
      <div className="topbar-row">
        <span className="wordmark">DXA<span>QC</span></span>
        <div className="batch">
          <span className="batch-name" title={batch.name}>{batch.name || 'Пакет'}</span>
          <span className="batch-state tabular" role="status" aria-live="polite">
            {phase === 'error' ? <><IconAlert width={15} height={15} className="tone-bad" />{error}</>
              : ready ? `${total} ${plural(total, 'снимок', 'снимка', 'снимков')} · ${clock(batch.elapsed)}`
              : total ? `Анализ ${Math.min(done + 1, total)} из ${total}`
              : 'Распаковка…'}
          </span>
        </div>
        <div className="tally tabular" role="tablist" aria-label="Вердикт">
          {tally.map((t) => (
            <button key={t.key} role="tab" aria-selected={filter === t.key} onClick={() => onFilter(t.key)}>
              <span className="tally-label">{t.label}</span>
              <span className={`tally-n ${t.tone ?? ''}`}>{t.n}</span>
            </button>
          ))}
          <div className="tally-reviewed">
            <span className="tally-label">Подтверждено врачом</span>
            <span className="tally-n">{reviewed} <small>из {decidable}</small></span>
          </div>
        </div>
        <div className="topbar-actions">
          <ReportMenu batchId={batch.id} ready={ready} />
          {reviewed > 0 && (
            <a className="btn" href={correctionsUrl(batch.id)} title="Исправления и решения врача — для дообучения модели">
              <IconReviewed />Исправления
            </a>
          )}
          <button className="btn btn-quiet" onClick={onNew} title="Загрузить другой пакет">
            <IconUpload />Новый пакет
          </button>
        </div>
      </div>
      <div className="progress" data-done={ready || undefined} aria-hidden>
        <span style={{ transform: `scaleX(${ready ? 1 : pct})` }} />
      </div>
    </header>
  )
}

const EXPORTS = (id: string) => [
  { href: reportUrl(id, 'csv'), title: 'CSV', note: 'Отчёт по ТЗ, 8 колонок + quality_prob' },
  { href: reportUrl(id, 'xlsx'), title: 'XLSX', note: 'Тот же отчёт для Excel' },
  { href: srUrl(id), title: 'DICOM SR', note: 'Вердикт и измерения на каждый снимок' },
]

/** One «Отчёт ▾» button instead of three: the formats, each with what it is for. */
function ReportMenu({ batchId, ready }: { batchId: string; ready: boolean }) {
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const away = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false) }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); setOpen(false) } }
    document.addEventListener('pointerdown', away)
    window.addEventListener('keydown', esc, true)
    return () => { document.removeEventListener('pointerdown', away); window.removeEventListener('keydown', esc, true) }
  }, [open])

  return (
    <div className="menu-anchor" ref={box}>
      <button className="btn" aria-haspopup="menu" aria-expanded={open} disabled={!ready}
              title={ready ? undefined : 'Станет доступно после обработки'} onClick={() => setOpen((o) => !o)}>
        <IconDownload />Отчёт<span className="caret" aria-hidden>▾</span>
      </button>
      {open && (
        <div className="menu" role="menu">
          {EXPORTS(batchId).map((e) => (
            <a key={e.title} role="menuitem" href={e.href} onClick={() => setOpen(false)}>
              <span className="menu-title">{e.title}</span>
              <span className="menu-note">{e.note}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  )
}
