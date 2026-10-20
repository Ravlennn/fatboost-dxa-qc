// The grease-pencil layer: the model's measurements drawn on the film in technical-drawing
// grammar (dimension lines with end ticks, a reference axis, the angle). Lines live in image
// coordinates with non-scaling strokes; labels are HTML so their size never follows the zoom.
import type { CSSProperties, ReactNode } from 'react'
import type { Item } from '../api'
import { fmt } from '../labels'

type P = [number, number]

interface Props {
  item: Item
  highlight: string | null
  onHover: (code: string | null) => void
  draw: boolean
}

interface Label { x: number; y: number; text: string; code: string; align?: 'left' | 'right' | 'center'; tone?: 'quiet' }

export default function Annotation({ item, highlight, onHover, draw }: Props) {
  const W = item.width ?? 1, H = item.height ?? 1
  const pts = item.points ?? {}
  const m = item.measurements ?? {}
  const bad = new Set(item.violations)
  const shapes: ReactNode[] = []
  const labels: Label[] = []
  const tick = Math.max(W, H) * 0.018
  let seq = 0

  const g = (code: string, children: ReactNode, key: string) => (
    <g key={key} data-code={code} data-bad={bad.has(code) || undefined}
       data-dim={(highlight && highlight !== code) || undefined} data-hl={highlight === code || undefined}
       onPointerEnter={() => onHover(code)} onPointerLeave={() => onHover(null)}
       style={{ '--i': seq++ } as CSSProperties}>
      {children}
    </g>
  )
  const line = (a: P, b: P, cls = '') => (
    <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className={cls} pathLength={1} vectorEffect="non-scaling-stroke" />
  )
  // vertical dimension line between y1 and y2 at x, with perpendicular end ticks
  const dimV = (x: number, y1: number, y2: number) => (
    <>
      {line([x, y1], [x, y2])}
      {line([x - tick, y1], [x + tick, y1], 'cap')}
      {line([x - tick, y2], [x + tick, y2], 'cap')}
    </>
  )
  const dimH = (y: number, x1: number, x2: number) => (
    <>
      {line([x1, y], [x2, y])}
      {line([x1, y - tick], [x1, y + tick], 'cap')}
      {line([x2, y - tick], [x2, y + tick], 'cap')}
    </>
  )

  if (item.region === 'lumbar_spine' && pts.L1_top && pts.L4_bottom) {
    const seqPts = ['L1_top', 'L1L2', 'L2L3', 'L3L4', 'L4_bottom'].map((k) => pts[k]).filter(Boolean) as P[]
    const half = W * 0.13
    shapes.push(g('levels', <>{seqPts.map((p, k) => <g key={k}>{line([p[0] - half, p[1]], [p[0] + half, p[1]], 'level')}</g>)}</>, 'levels'))
    seqPts.slice(0, -1).forEach((p, k) => {
      const q = seqPts[k + 1]
      labels.push({ x: Math.min(p[0], q[0]) - half - W * 0.02, y: (p[1] + q[1]) / 2, text: `L${k + 1}`, code: 'levels', align: 'right', tone: 'quiet' })
    })
    const top = pts.L1_top, bottom = pts.L4_bottom
    const tilt = m.spine_tilt_deg_calibrated ?? m.spine_tilt_deg
    shapes.push(g('spine_axis_tilt', <>
      {line([bottom[0], bottom[1]], [bottom[0], top[1]], 'ref')}
      {line(top, bottom, 'axis')}
    </>, 'axis'))
    labels.push({ x: Math.max(top[0], bottom[0]) + W * 0.03, y: top[1] + (bottom[1] - top[1]) * 0.18,
                  text: `${fmt(Math.abs(tilt ?? NaN), 1)}°`, code: 'spine_axis_tilt', align: 'left' })
    const xr = W * 0.9
    shapes.push(g('spine_scan_range', <>{dimV(xr, 0, top[1])}{dimV(xr, bottom[1], H)}</>, 'range'))
    labels.push({ x: xr - W * 0.02, y: top[1] / 2, text: `${fmt(m.top_margin_mm, 0)} мм`, code: 'spine_scan_range', align: 'right' })
    labels.push({ x: xr - W * 0.02, y: (bottom[1] + H) / 2, text: `${fmt(m.bottom_margin_mm, 0)} мм`, code: 'spine_scan_range', align: 'right' })
  }

  // what the artifact detector found: a box around each object, its kind on a plate beside it
  const found = item.region === 'lumbar_spine' ? item.artifacts ?? [] : []
  if (found.length) {
    shapes.push(g('spine_artifact', <>{found.map((a, k) => (
      <g key={k}>
        <rect x={a.x} y={a.y} width={a.w} height={a.h} className="artifact-halo" vectorEffect="non-scaling-stroke" />
        <rect x={a.x} y={a.y} width={a.w} height={a.h} className="artifact" vectorEffect="non-scaling-stroke" />
      </g>
    ))}</>, 'artifacts'))
    found.forEach((a) => {
      const right = a.x + a.w / 2 < W / 2
      labels.push({ x: right ? a.x + a.w + W * 0.015 : a.x - W * 0.015, y: a.y + a.h / 2,
                    text: a.type === 'metal' ? 'Металл' : 'Проволока', code: 'spine_artifact', align: right ? 'left' : 'right' })
    })
  }

  if (item.region?.startsWith('hip') && pts.roi_top && pts.roi_bottom && pts.roi_left && pts.roi_right) {
    const x0 = Math.min(pts.roi_left[0], pts.roi_right[0]), x1 = Math.max(pts.roi_left[0], pts.roi_right[0])
    const y0 = pts.roi_top[1], y1 = pts.roi_bottom[1]
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2
    shapes.push(g('hip_roi_margins', <>
      <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} className="roi" vectorEffect="non-scaling-stroke" />
      {dimV(cx, 0, y0)}{dimV(cx, y1, H)}{dimH(cy, 0, x0)}{dimH(cy, x1, W)}
    </>, 'roi'))
    const side = (x: number) => (pts.roi_left[0] <= pts.roi_right[0]) === (x < cx) ? m.roi_left_mm : m.roi_right_mm
    labels.push({ x: cx + W * 0.025, y: y0 / 2, text: `${fmt(m.roi_top_mm, 0)} мм`, code: 'hip_roi_margins', align: 'left' })
    labels.push({ x: cx + W * 0.025, y: (y1 + H) / 2, text: `${fmt(m.roi_bottom_mm, 0)} мм`, code: 'hip_roi_margins', align: 'left' })
    labels.push({ x: x0 / 2, y: cy - H * 0.035, text: `${fmt(side(x0 / 2), 0)} мм`, code: 'hip_roi_margins', align: 'center' })
    labels.push({ x: (x1 + W) / 2, y: cy - H * 0.035, text: `${fmt(side((x1 + W) / 2), 0)} мм`, code: 'hip_roi_margins', align: 'center' })

    const lt = item.lt
    const head = pts.neck_head_side, troch = pts.neck_troch_side
    shapes.push(g('hip_positioning', <>
      {head && troch && (() => {
        const dx = head[0] - troch[0], dy = head[1] - troch[1]
        return line([troch[0] - dx * 0.55, troch[1] - dy * 0.55], [head[0] + dx * 0.55, head[1] + dy * 0.55], 'ref')
      })()}
      {lt?.edge && lt.edge.length > 1 && (
        <polyline points={lt.edge.map((p) => p.join(',')).join(' ')} className="edge" pathLength={1} vectorEffect="non-scaling-stroke" />
      )}
      {lt?.apex && <circle cx={lt.apex[0]} cy={lt.apex[1]} r={Math.max(W, H) * 0.012} className="apex" vectorEffect="non-scaling-stroke" />}
    </>, 'lt'))
    if (lt?.apex) {
      const medialLeft = head && troch ? head[0] < troch[0] : false
      labels.push({ x: lt.apex[0] + (medialLeft ? -1 : 1) * W * 0.035, y: lt.apex[1],
                    text: lt.visible ? `выступ ${fmt(lt.prominence_mm, 1)} мм` : 'вертел не виден',
                    code: 'hip_positioning', align: medialLeft ? 'right' : 'left' })
    }
    if (head && m.neck_axis_deg != null) {
      labels.push({ x: head[0], y: head[1] - H * 0.06, text: `ось шейки ${fmt(m.neck_axis_deg, 0)}°`, code: 'hip_positioning', align: 'center', tone: 'quiet' })
    }
  }

  return (
    <div className="annotation" data-draw={draw || undefined}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>{shapes}</svg>
      <div className="labels">
        {labels.map((l, k) => (
          <span key={k} className="mlabel" data-align={l.align ?? 'left'} data-tone={l.tone}
                data-bad={bad.has(l.code) || undefined} data-dim={(highlight && highlight !== l.code) || undefined}
                style={{ left: `${(l.x / W) * 100}%`, top: `${(l.y / H) * 100}%` }}
                onPointerEnter={() => onHover(l.code)} onPointerLeave={() => onHover(null)}>
            {l.text}
          </span>
        ))}
      </div>
    </div>
  )
}
