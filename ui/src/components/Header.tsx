import { correctionsUrl, reportUrl, srUrl } from '../api'
import type { Batch, Phase, Row } from '../App'
import { plural } from '../labels'
import { IconAlert, IconDownload, IconReviewed, IconUpload } from './Icons'

interface Props { batch: Batch; rows: Row[]; phase: Phase; error: string; onNew: () => void }

const clock = (s: number) => {
  const m = Math.floor(s / 60)
  return m ? `${m} мин ${Math.round(s % 60)} с` : `${Math.round(s)} с`
}

export default function Header({ batch, rows, phase, error, onNew }: Props) {
  const done = batch.items.size
  const total = batch.total ?? rows.length
  const items = Array.from(batch.items.values())
  const ok = items.filter((i) => i.status === 'Success' && i.quality === 0).length
  const bad = items.filter((i) => i.status === 'Success' && i.quality === 1).length
  const fail = items.filter((i) => i.status === 'Failure').length
  const reviewed = items.filter((i) => i.review).length
  const ready = phase === 'done'
  const pct = total ? done / total : 0

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
        <dl className="tally tabular" aria-label="Итог">
          <div><dt>Норма</dt><dd className="tone-ok">{ok}</dd></div>
          <div><dt>Брак</dt><dd className="tone-bad">{bad}</dd></div>
          <div><dt>Ошибки</dt><dd className="tone-fail">{fail}</dd></div>
          <div><dt>Проверено</dt><dd>{reviewed}</dd></div>
        </dl>
        <div className="topbar-actions">
          <a className="btn" aria-disabled={!ready || undefined} href={ready ? reportUrl(batch.id, 'csv') : undefined}
             title={ready ? 'Отчёт по ТЗ: 8 колонок + quality_prob' : 'Станет доступно после обработки'}>
            <IconDownload />CSV
          </a>
          <a className="btn" aria-disabled={!ready || undefined} href={ready ? reportUrl(batch.id, 'xlsx') : undefined}
             title={ready ? 'Отчёт по ТЗ: 8 колонок + quality_prob' : 'Станет доступно после обработки'}>
            <IconDownload />XLSX
          </a>
          <a className="btn" aria-disabled={!ready || undefined} href={ready ? srUrl(batch.id) : undefined}
             title={ready ? 'DICOM SR: текстовый отчёт с вердиктом и измерениями на каждый снимок' : 'Станет доступно после обработки'}>
            <IconDownload />SR
          </a>
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
