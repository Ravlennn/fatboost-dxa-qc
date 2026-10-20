import { memo, useMemo } from 'react'
import { imageUrl, type Item } from '../api'
import type { Row } from '../App'
import { REGION_SHORT, VIOLATION_LABEL, fmt, type Region } from '../labels'
import { IconAlert, IconCheck, IconCross, IconReviewed } from './Icons'

export type Filter = 'all' | 'bad' | 'ok' | 'fail'

interface Props {
  rows: Row[]
  all: Row[]
  selected: number | null
  onSelect: (i: number, via: 'pointer' | 'key') => void
  filter: Filter
  onFilter: (f: Filter) => void
  region: string
  onRegion: (r: string) => void
  arrived: Set<number>
  onDeveloped: (i: number) => void
  jobId: string
  running: boolean
}

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all', label: 'Все' }, { key: 'bad', label: 'Брак' }, { key: 'ok', label: 'Норма' }, { key: 'fail', label: 'Ошибки' },
]

export function Verdict({ item }: { item: Item }) {
  if (item.status === 'Failure') return <span className="verdict verdict-fail"><IconAlert width={14} height={14} />Ошибка</span>
  return item.quality === 1
    ? <span className="verdict verdict-bad"><IconCross width={14} height={14} />Брак</span>
    : <span className="verdict verdict-ok"><IconCheck width={14} height={14} />Норма</span>
}

const RowView = memo(function RowView({ row, active, next, jobId, develop, onSelect, onDeveloped }: {
  row: Row; active: boolean; next: boolean; jobId: string; develop: boolean
  onSelect: Props['onSelect']; onDeveloped: Props['onDeveloped']
}) {
  const { item } = row
  const name = row.path.split('/').pop() || row.path
  const dir = row.path.includes('/') ? row.path.slice(0, row.path.lastIndexOf('/')) : ''
  return (
    <li id={`row-${row.i}`} className="row" data-active={active || undefined} data-pending={!item || undefined}>
      <button className="row-hit" onClick={() => onSelect(row.i, 'pointer')} aria-current={active || undefined}
              disabled={!item}>
        <span className="thumb">
          {item?.has_image
            ? <img src={imageUrl(jobId, row.i)} alt="" loading="lazy" decoding="async"
                   className={develop ? 'develop' : undefined}
                   onAnimationEnd={() => onDeveloped(row.i)} />
            : item ? <IconAlert width={16} height={16} /> : null}
        </span>
        <span className="row-file">
          <span className="row-name" title={row.path}>{name}</span>
          {dir && <span className="row-dir" title={dir}>{dir}</span>}
        </span>
        <span className="row-region">{item?.region ? REGION_SHORT[item.region as Region] ?? item.region : ''}</span>
        <span className="row-verdict">
          {item ? <Verdict item={item} /> : <span className="pending-label">{next ? 'анализ' : 'в очереди'}</span>}
          {item?.review && <span className="row-reviewed" title="Проверено врачом"><IconReviewed width={15} height={15} /></span>}
        </span>
        <span className="row-reasons">
          {item?.status === 'Failure'
            ? <span className="reason-text">{item.error.replace(/^\w+Error:\s*/, '') || 'Не удалось прочитать файл'}</span>
            : item?.violations.map((v) => <span key={v} className="tag">{VIOLATION_LABEL[v] ?? v}</span>)}
        </span>
        <span className="row-time tabular"
              title={item?.status === 'Success' && (item.seconds ?? 1) < 0.05 ? 'Те же пиксели уже встречались в пакете — результат переиспользован' : undefined}>
          {item?.seconds == null ? '' : item.status === 'Success' && item.seconds < 0.05 ? 'повтор' : `${fmt(item.seconds, 1)} с`}
        </span>
      </button>
    </li>
  )
})

export default function ResultList({ rows, all, selected, onSelect, filter, onFilter, region, onRegion,
                                      arrived, onDeveloped, jobId, running }: Props) {
  const counts = useMemo(() => {
    const items = all.map((r) => r.item).filter(Boolean) as Item[]
    return {
      all: all.length,
      bad: items.filter((i) => i.status === 'Success' && i.quality === 1).length,
      ok: items.filter((i) => i.status === 'Success' && i.quality === 0).length,
      fail: items.filter((i) => i.status === 'Failure').length,
    }
  }, [all])
  const regions = useMemo(() => Array.from(new Set(all.map((r) => r.item?.region).filter(Boolean) as string[])).sort(), [all])
  const nextIndex = running ? all.findIndex((r) => !r.item) : -1

  return (
    <section className="results" aria-label="Результаты">
      <div className="filters">
        <div className="segmented" role="tablist" aria-label="Вердикт">
          {FILTERS.map((f) => (
            <button key={f.key} role="tab" aria-selected={filter === f.key} onClick={() => onFilter(f.key)}>
              {f.label}<span className="count tabular">{counts[f.key]}</span>
            </button>
          ))}
        </div>
        {regions.length > 1 && (
          <div className="segmented segmented-quiet" role="tablist" aria-label="Область">
            <button role="tab" aria-selected={region === 'all'} onClick={() => onRegion('all')}>Все области</button>
            {regions.map((r) => (
              <button key={r} role="tab" aria-selected={region === r} onClick={() => onRegion(r)}>
                {REGION_SHORT[r as Region] ?? r}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="list-head" aria-hidden>
        <span />
        <span>Файл</span>
        <span>Область</span>
        <span>Вердикт</span>
        <span>Нарушения</span>
        <span className="right">Время</span>
      </div>
      {rows.length ? (
        <ol className="list">
          {rows.map((row) => (
            <RowView key={row.i} row={row} active={row.i === selected} next={row.i === nextIndex} jobId={jobId}
                     develop={arrived.has(row.i)} onSelect={onSelect} onDeveloped={onDeveloped} />
          ))}
        </ol>
      ) : (
        <p className="empty">В этом срезе снимков нет. <button className="link" onClick={() => { onFilter('all'); onRegion('all') }}>Показать все</button></p>
      )}
    </section>
  )
}
