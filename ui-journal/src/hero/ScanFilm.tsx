import { forwardRef, useImperativeHandle, useMemo, useRef } from 'react'
import { ringOf, type Plan } from '../iso/iso-kit'
import { fmt } from '../labels'
import { CRESTS, LIMIT_TILT, SCAN, SPINE, bodyCentre, landmarks, rotate, type Reading, type V2 } from './model'
import { bodyRing } from './scene'

// The film the sweep produces: the phantom seen from above, as the detector sees it (bone light on
// dark), built row by row behind the arm. The system's markup is drawn on it in the product's own
// grammar as each measurement becomes possible. Film coordinates: horizontal = world y, vertical =
// world x (head end at the top), in phantom millimetres.

const HALF = 40
const VIEW = `${-HALF} ${SCAN.to} ${HALF * 2} ${SCAN.from - SCAN.to}`
const swap = ([x, y]: V2) => `${y.toFixed(2)},${x.toFixed(2)}`
const poly = (pts: V2[]) => pts.map(swap).join(' ')

function bones(tilt: number) {
  const r = (pts: [number, number][]) => pts.map((p) => rotate(p as V2, tilt))
  const plan = (p: Plan, steps = 5) => poly(r(ringOf(p, steps)))
  const bodies = Array.from({ length: 6 }, (_, i) => poly(r(bodyRing(i))))
  const eyes = Array.from({ length: 6 }, (_, i) => [-1, 1].map((s) => { const [x, y] = rotate([bodyCentre(i) - 3, s * 5.6], tilt); return { cx: y, cy: x } })).flat()
  const procs = Array.from({ length: 5 }, (_, i) => [-1, 1].map((s) => plan({ x: bodyCentre(i) - 3.5, y: s > 0 ? 10 : -17.5, w: 7, d: 7.5, r: 2 }, 3))).flat()
  const wing = (s: number) => poly(r([[CRESTS.x0, 12.5], [56, 21], [64, 26], [74, 27], [82, 24], [CRESTS.x1, 17], [80, 12.5]].map(([x, y]) => [x, s * y] as [number, number])))
  const bar = plan({ x: bodyCentre(0) - 12, y: -5, w: bodyCentre(5) - bodyCentre(0) + 24, d: 10, r: 2 }, 3)
  const tray = plan({ x: -84, y: -31, w: 186, d: 62, r: 5 }, 6)
  return { bodies, eyes, procs, wings: [wing(-1), wing(1)], bar, tray }
}

/** Endplate lines across the spine at L1 top … L4 bottom, perpendicular to the (tilted) axis. */
function levels(tilt: number) {
  const half = SPINE.body.len / 2, w = 15
  const xs = [bodyCentre(1) - half, ...[1, 2, 3].map((i) => bodyCentre(i) + half + SPINE.gap / 2), bodyCentre(4) + half]
  return xs.map((x) => [rotate([x, -w], tilt), rotate([x, w], tilt)] as [V2, V2])
}

interface Props { tilt: number; reading: Reading }
export interface FilmHandle { set: (reach: number, x: number, live: boolean) => void }

const ScanFilm = forwardRef<FilmHandle, Props>(function ScanFilm({ tilt, reading }, ref) {
  const reveal = useRef<SVGRectElement>(null)
  const line = useRef<SVGLineElement>(null)
  const b = useMemo(() => bones(tilt), [tilt])
  const lv = useMemo(() => levels(tilt), [tilt])
  const p = useMemo(() => landmarks(tilt), [tilt])

  useImperativeHandle(ref, () => ({
    set(reach, x, live) {
      const top = Math.max(SCAN.to, Math.min(SCAN.from, reach))
      reveal.current?.setAttribute('y', top.toFixed(2))
      reveal.current?.setAttribute('height', (SCAN.from - top).toFixed(2))
      if (line.current) {
        line.current.setAttribute('y1', x.toFixed(2)); line.current.setAttribute('y2', x.toFixed(2))
        line.current.style.opacity = live && x < SCAN.from - 0.5 && x > SCAN.to + 0.5 ? '1' : '0'
      }
    },
  }), [])

  const tiltShown = !isNaN(reading.tilt)
  const bad = tiltShown && Math.abs(reading.tilt) > LIMIT_TILT
  const dimX = HALF - 7
  return (
    <div className="scanfilm">
      <svg viewBox={VIEW} preserveAspectRatio="xMidYMid meet" aria-hidden>
        <defs>
          <clipPath id="scan-reveal"><rect ref={reveal} x={-HALF} y={SCAN.from} width={HALF * 2} height={0} /></clipPath>
          <filter id="scan-soft" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="0.7" /></filter>
          <linearGradient id="scan-bone" x1="0" x2="1">
            <stop offset="0" stopColor="#8d9095" /><stop offset="0.5" stopColor="#e8e7e2" /><stop offset="1" stopColor="#8d9095" />
          </linearGradient>
        </defs>
        <g clipPath="url(#scan-reveal)">
          <g filter="url(#scan-soft)">
            <polygon points={b.tray} className="sf-tray" />
            <polygon points={b.bar} className="sf-soft" />
            {b.procs.map((d, i) => <polygon key={`p${i}`} points={d} className="sf-proc" />)}
            {b.wings.map((d, i) => <polygon key={`w${i}`} points={d} className="sf-wing" />)}
            {b.bodies.map((d, i) => <polygon key={`b${i}`} points={d} fill="url(#scan-bone)" className="sf-body" />)}
            {b.eyes.map((e, i) => <ellipse key={`e${i}`} cx={e.cx} cy={e.cy} rx={1.9} ry={2.6} className="sf-eye" />)}
          </g>
          {/* the system's markup, drawn as each value becomes measurable */}
          {tiltShown && (
            <g className="sf-mark" data-bad={bad || undefined}>
              {lv.map(([a, c], i) => <line key={i} x1={a[1]} y1={a[0]} x2={c[1]} y2={c[0]} className="sf-level" />)}
              <line x1={p.L4_bottom[1]} y1={p.L4_bottom[0]} x2={p.L4_bottom[1]} y2={p.L1_top[0]} className="sf-ref" />
              <line x1={p.L1_top[1]} y1={p.L1_top[0]} x2={p.L4_bottom[1]} y2={p.L4_bottom[0]} className="sf-axis" />
              <text x={Math.max(p.L1_top[1], p.L4_bottom[1]) + 4} y={p.L1_top[0] + 14} className="sf-label">{fmt(Math.abs(reading.tilt), 1)}°</text>
            </g>
          )}
          {reading.bottom != null && (
            <g className="sf-mark">
              <line x1={dimX} y1={p.L4_bottom[0]} x2={dimX} y2={SCAN.from} className="sf-dim" />
              <line x1={dimX - 2.5} y1={p.L4_bottom[0]} x2={dimX + 2.5} y2={p.L4_bottom[0]} className="sf-dim" />
              <text x={dimX - 2.5} y={(p.L4_bottom[0] + SCAN.from) / 2 + 2} className="sf-label" textAnchor="end">{fmt(reading.bottom, 0)} мм</text>
            </g>
          )}
          {reading.top != null && (
            <g className="sf-mark">
              <line x1={dimX} y1={SCAN.to} x2={dimX} y2={p.L1_top[0]} className="sf-dim" />
              <line x1={dimX - 2.5} y1={p.L1_top[0]} x2={dimX + 2.5} y2={p.L1_top[0]} className="sf-dim" />
              <text x={dimX - 2.5} y={(SCAN.to + p.L1_top[0]) / 2 + 2} className="sf-label" textAnchor="end">{fmt(reading.top, 0)} мм</text>
            </g>
          )}
        </g>
        <line ref={line} x1={-HALF} x2={HALF} y1={SCAN.from} y2={SCAN.from} className="sf-scanline" />
      </svg>
      <span className="scanfilm-cap">Снимок строится за дугой</span>
    </div>
  )
})

export default ScanFilm
