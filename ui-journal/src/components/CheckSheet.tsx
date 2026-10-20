import { memo, useEffect, useMemo, useRef, useState } from 'react'
import type { Item } from '../api'
import type { Row } from '../App'
import { CRITERIA, family, fmt } from '../labels'

// The batch's check sheet: one row per TZ criterion, one column per image in batch order. A dot in
// ink where the criterion holds, a red cross on a hatched cell where it is broken, nothing where it
// does not apply. Marks land as the images are analysed; a click opens that image.

const ALL_ROWS = [...CRITERIA.spine, ...CRITERIA.hip]
// a region with no images in the finished batch gets one explaining line instead of empty rows
const ABSENT: Record<'spine' | 'hip', string> = {
  spine: 'Позвоночник — в пакете нет снимков, критерии границ, оси и посторонних предметов скрыты',
  hip: 'Бедро — в пакете нет снимков, критерии ротации и полей ROI скрыты',
}
const SHORT: Record<string, string> = {
  spine_scan_range: 'Границы', spine_axis_tilt: 'Ось', spine_artifact: 'Предметы',
  hip_positioning: 'Ротация', hip_roi_margins: 'Поля ROI',
}
const ROW_H = 22
const HEAD = 16
const LABEL_W = 92
const TOTAL_W = 56
const MIN_COL = 9
const MAX_COL = 34

type Cell = 'ok' | 'bad' | 'fail' | 'wait' | null

function cellOf(item: Item | undefined, code: string): Cell {
  if (!item) return 'wait'
  if (item.status === 'Failure') return 'fail'
  const fam = family(item.region)
  if (!fam || !CRITERIA[fam].some((c) => c.code === code)) return null
  return item.violations.includes(code) ? 'bad' : 'ok'
}

function titleOf(row: Row): string {
  const name = row.path.split('/').pop() || row.path
  const it = row.item
  if (!it) return `${name}: ещё не проанализирован`
  if (it.status === 'Failure') return `${name}: файл не обработан`
  const m = it.measurements ?? {}
  const tilt = m.spine_tilt_deg_calibrated != null ? `, наклон ${fmt(Math.abs(m.spine_tilt_deg_calibrated), 1)}°` : ''
  const bad = ALL_ROWS.filter((c) => it.violations.includes(c.code)).map((c) => c.title.toLowerCase())
  return `${name}: ${bad.length ? 'нарушено — ' + bad.join(', ') : 'норма'}${tilt}${it.review ? ' · проверено врачом' : ''}`
}

interface Props {
  rows: Row[]
  selected: number | null
  onSelect: (i: number, via: 'pointer' | 'key') => void
}

export default memo(function CheckSheet({ rows, selected, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(600)
  const [hover, setHover] = useState<number | null>(null)
  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  const n = rows.length
  const complete = n > 0 && rows.every((r) => r.item)
  const present = new Set(rows.map((r) => family(r.item?.region)).filter(Boolean))
  const absent = complete && present.size ? (['spine', 'hip'] as const).filter((f) => !present.has(f)) : []
  const absentKey = absent.join()
  const ROWS = useMemo(() => [...(absentKey.includes('spine') ? [] : CRITERIA.spine), ...(absentKey.includes('hip') ? [] : CRITERIA.hip)],
                       [absentKey])
  const splitAfter = absent.length ? -1 : CRITERIA.spine.length - 1
  const col = Math.max(MIN_COL, Math.min(MAX_COL, (width - LABEL_W - TOTAL_W) / Math.max(1, n)))
  const plotW = col * n
  const svgW = Math.max(width, LABEL_W + plotW + TOTAL_W)
  const H = HEAD + ROW_H * ROWS.length + 12
  const cells = useMemo(() => ROWS.map((c) => rows.map((r) => cellOf(r.item, c.code))), [rows, ROWS])
  const totals = cells.map((line) => ({ bad: line.filter((x) => x === 'bad').length, seen: line.filter((x) => x === 'ok' || x === 'bad').length }))
  const r = Math.min(3, col * 0.22)
  const tick = Math.min(4.2, col * 0.3)

  return (
    <section className="sheetcard" aria-label="Контрольный листок пакета">
      <div className="sheetcard-head">
        <h2>Контрольный листок</h2>
        <span className="sheetcard-key">
          <span className="key-ok" aria-hidden />норма<span className="key-bad" aria-hidden />нарушение
        </span>
      </div>
      <div ref={box} className="checksheet">
        <svg width={svgW} height={H} viewBox={`0 0 ${svgW} ${H}`} role="img"
             aria-label={ROWS.map((c, k) => `${c.title}: нарушений ${totals[k].bad} из ${totals[k].seen}`).join('; ')}
             onPointerLeave={() => setHover(null)}>
          {/* column under the pointer and the open image */}
          {selected != null && selected < n && <rect className="cs-sel" x={LABEL_W + selected * col} y={HEAD - 4} width={col} height={H - HEAD + 4} />}
          {hover != null && hover !== selected && <rect className="cs-hover" x={LABEL_W + hover * col} y={HEAD - 4} width={col} height={H - HEAD + 4} />}
          {/* ruled lines: one per criterion, and a stronger rule between spine and hip */}
          {ROWS.map((c, k) => (
            <g key={c.code}>
              <text x={LABEL_W - 10} y={HEAD + k * ROW_H + ROW_H / 2 + 4} className="cs-label" textAnchor="end">{SHORT[c.code]}</text>
              <line x1={0} x2={svgW} y1={HEAD + (k + 1) * ROW_H} y2={HEAD + (k + 1) * ROW_H}
                    className={k === splitAfter ? 'cs-rule cs-rule-strong' : 'cs-rule'} />
              <text x={svgW - 6} y={HEAD + k * ROW_H + ROW_H / 2 + 4} className="cs-total" data-bad={totals[k].bad > 0 || undefined} textAnchor="end">
                {totals[k].seen ? `${totals[k].bad}/${totals[k].seen}` : '—'}
              </text>
            </g>
          ))}
          <line x1={LABEL_W - 0.5} x2={LABEL_W - 0.5} y1={HEAD - 4} y2={H} className="cs-rule cs-rule-strong" />
          <line x1={LABEL_W + plotW + 0.5} x2={LABEL_W + plotW + 0.5} y1={HEAD - 4} y2={H} className="cs-rule" />
          {/* column numbers every fifth image */}
          {rows.map((row, i) => (i % 5 === 4 || n <= 12) && (
            <text key={`n${row.i}`} x={LABEL_W + i * col + col / 2} y={HEAD - 6} className="cs-num" textAnchor="middle">{i + 1}</text>
          ))}
          {cells.map((line, k) => line.map((cell, i) => {
            if (!cell) return null
            const cx = LABEL_W + i * col + col / 2, cy = HEAD + k * ROW_H + ROW_H / 2
            return (
              <g key={`${k}-${i}`} className={`cs-mark cs-${cell}`}>
                {cell === 'bad' && <rect x={LABEL_W + i * col + 0.5} y={HEAD + k * ROW_H + 1} width={col - 1} height={ROW_H - 2} className="cs-hatch" />}
                {cell === 'ok' && <circle cx={cx} cy={cy} r={r} />}
                {cell === 'bad' && <path d={`M${cx - tick} ${cy - tick}L${cx + tick} ${cy + tick}M${cx + tick} ${cy - tick}L${cx - tick} ${cy + tick}`} />}
                {cell === 'fail' && <path d={`M${cx - tick} ${cy}L${cx + tick} ${cy}`} />}
                {cell === 'wait' && <circle cx={cx} cy={cy} r={Math.max(1, r * 0.55)} />}
              </g>
            )
          }))}
          {/* the doctor's confirmation, under each reviewed column */}
          {rows.map((row, i) => row.item?.review && (
            <path key={`rv${row.i}`} className="cs-review" d={`M${LABEL_W + i * col + col / 2 - 3} ${H - 6}l2.2 2.2 4-4.6`} />
          ))}
          {/* hit areas: whole columns */}
          {rows.map((row, i) => (
            <rect key={`hit${row.i}`} x={LABEL_W + i * col} y={0} width={col} height={H} className="cs-hit"
                  onPointerEnter={() => setHover(i)}
                  onClick={() => row.item && onSelect(row.i, 'pointer')}>
              <title>{titleOf(row)}</title>
            </rect>
          ))}
        </svg>
      </div>
      {absent.map((f) => <p key={f} className="sheetcard-absent">{ABSENT[f]}</p>)}
    </section>
  )
})
