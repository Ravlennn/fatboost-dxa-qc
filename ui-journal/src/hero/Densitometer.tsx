import { useEffect, useMemo, useRef, useState } from 'react'
import { fmt } from '../labels'
import ScanFilm from './ScanFilm'
import { LIMIT_TILT, SCAN, read, type Reading } from './model'
import { ARM, CARRIAGE, H, HEAD_Z, STATIC, W, iso, phantom, step, surfaceAt } from './scene'

// Fig. 1: the densitometer scans a spine phantom. The arm runs a tour (park, a constant-speed
// sweep, a pause on the finished film, an eased return); the pointer scrubs it; ←/→ turn the
// phantom on its turntable. Positions are written to the DOM per frame, React only hears about
// the readings when they change.

const PARK = SCAN.from + 16
const T = { park: 1.1, sweep: 6.4, hold: 3.2, back: 1.5 }
const CYCLE = T.park + T.sweep + T.hold + T.back
const TILT_RANGE = 10

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)
function tourAt(t: number): { x: number; scanning: boolean } {
  const c = t % CYCLE
  if (c < T.park) return { x: PARK, scanning: false }
  if (c < T.park + T.sweep) {
    const u = (c - T.park) / T.sweep
    return { x: PARK + (SCAN.to - PARK) * Math.min(1, u * 1.04), scanning: true }
  }
  if (c < T.park + T.sweep + T.hold) return { x: SCAN.to, scanning: false }
  const u = (c - T.park - T.sweep - T.hold) / T.back
  return { x: SCAN.to + (PARK - SCAN.to) * ease(u), scanning: false }
}

export interface Checks { reading: Reading; tilt: number }

interface Props { onChecks?: (c: Checks) => void }

export default function Densitometer({ onChecks }: Props) {
  const [tilt, setTilt] = useState(2)
  const [reading, setReading] = useState<Reading>(() => read(2, PARK))
  const stage = useRef<HTMLDivElement>(null)
  const car = useRef<SVGGElement>(null)
  const arm = useRef<SVGGElement>(null)
  const beam = useRef<SVGPathElement>(null)
  const spot = useRef<SVGCircleElement>(null)
  const film = useRef<{ set: (reach: number, x: number, live: boolean) => void }>(null)
  const live = useRef({ x: PARK, target: PARK, reach: PARK, clock: 0, touring: true, last: 0, frame: 0, visible: true, idle: 0, tilt: 2 })
  const still = useMemo(() => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches, [])
  const phantomSvg = useMemo(() => phantom(tilt), [tilt])

  useEffect(() => { live.current.tilt = tilt }, [tilt])
  useEffect(() => { onChecks?.({ reading, tilt }) }, [reading, tilt, onChecks])

  useEffect(() => {
    const s = live.current
    if (still) { s.touring = false; s.x = s.target = SCAN.to; s.reach = SCAN.to }
    let lastKey = ''
    const draw = () => {
      const dx = s.x
      const t = step(dx)
      car.current?.setAttribute('transform', t)
      arm.current?.setAttribute('transform', t)
      const zs = surfaceAt(s.x, s.tilt)
      const a = iso([s.x, 0, HEAD_Z]), b = iso([s.x, 0, zs])
      beam.current?.setAttribute('d', `M${a[0].toFixed(1)} ${a[1].toFixed(1)}L${b[0].toFixed(1)} ${b[1].toFixed(1)}`)
      spot.current?.setAttribute('cx', b[0].toFixed(1)); spot.current?.setAttribute('cy', b[1].toFixed(1))
      const scanning = s.x <= SCAN.from + 0.5 && s.x >= SCAN.to - 0.5
      stage.current?.toggleAttribute('data-scanning', scanning && s.reach > SCAN.to + 0.5)
      film.current?.set(s.reach, s.x, scanning)
      const r = read(s.tilt, s.reach)
      const key = [isNaN(r.tilt) ? '' : r.tilt.toFixed(1), r.top, r.bottom, r.th12, r.crests].join('|')
      if (key !== lastKey) { lastKey = key; setReading(r) }
    }
    const tick = (now: number) => {
      const dt = s.last ? Math.min((now - s.last) / 1000, 1 / 30) : 1 / 60
      s.last = now
      if (s.touring) {
        s.clock += dt
        const tour = tourAt(s.clock)
        s.target = tour.x
        if (!tour.scanning && tour.x > SCAN.from) s.reach = PARK      // a new sweep starts a new film
      }
      s.x = s.touring ? s.target : s.x + (s.target - s.x) * (1 - Math.exp(-dt / 0.22))
      if (Math.abs(s.target - s.x) < 0.02) s.x = s.target
      if (s.x < s.reach) s.reach = Math.max(SCAN.to, s.x)
      draw()
      s.frame = s.visible && (s.touring || s.x !== s.target) ? requestAnimationFrame(tick) : 0
    }
    const run = () => { if (!s.frame && s.visible && !still) { s.last = 0; s.frame = requestAnimationFrame(tick) } }
    const el = stage.current!
    const io = new IntersectionObserver((e) => { s.visible = e[e.length - 1].isIntersecting; if (s.visible) run() })
    io.observe(el)
    const resume = () => {
      clearTimeout(s.idle)
      s.idle = window.setTimeout(() => {
        if (still) return
        s.touring = true; s.clock = T.park + T.sweep + T.hold          // glide home, then a fresh sweep
        s.reach = Math.min(s.reach, s.x); run()
      }, 3200)
    }
    const onMove = (e: PointerEvent) => {
      if (e.pointerType === 'touch') return
      const box = el.getBoundingClientRect()
      const share = Math.min(1, Math.max(0, ((e.clientX - box.left) / box.width - 0.12) / 0.76))
      if (s.touring) { s.touring = false; s.reach = s.x > SCAN.from ? PARK : s.reach }
      s.target = SCAN.to + share * (PARK - SCAN.to)
      if (still) { s.x = s.target; if (s.x < s.reach) s.reach = Math.max(SCAN.to, s.x); draw() } else run()
      resume()
    }
    el.addEventListener('pointermove', onMove)
    draw(); run()
    ;(el as unknown as { redraw: () => void }).redraw = draw
    return () => { cancelAnimationFrame(s.frame); s.frame = 0; io.disconnect(); clearTimeout(s.idle); el.removeEventListener('pointermove', onMove) }
  }, [still])

  // a turn of the phantom re-reads what is already on the film
  useEffect(() => { (stage.current as unknown as { redraw?: () => void })?.redraw?.() }, [tilt])

  const turn = (by: number) => setTilt((t) => Math.max(-TILT_RANGE, Math.min(TILT_RANGE, t + by)))
  const over = Math.abs(reading.tilt) > LIMIT_TILT
  const readout = isNaN(reading.tilt)
    ? reading.progress > 0 ? `Скан ${Math.round(reading.progress * 100)}%` : 'Сканер в исходной позиции'
    : [`наклон ${fmt(Math.abs(reading.tilt), 1)}°`, reading.top != null && `над L1 ${fmt(reading.top, 0)} мм`,
       reading.bottom != null && `под L4 ${fmt(reading.bottom, 0)} мм`].filter(Boolean).join(' · ')

  return (
    <figure className="plate" aria-label="Рисунок 1: денситометр сканирует фантом позвоночника">
      <div className="plate-head">
        <span className="plate-fig">Рис. 1</span>
        <span className="plate-title">Денситометр сканирует фантом позвоночника</span>
      </div>
      <div className="plate-body">
        <div ref={stage} className="plate-stage iso" data-theme="light" tabIndex={0} role="slider"
             aria-label="Наклон фантома на поворотном столе: стрелки влево и вправо"
             aria-valuemin={-TILT_RANGE} aria-valuemax={TILT_RANGE} aria-valuenow={tilt}
             aria-valuetext={`Фантом повёрнут на ${tilt}°. ${readout}`}
             onKeyDown={(e) => {
               if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') { e.preventDefault(); turn(-1) }
               if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { e.preventDefault(); turn(1) }
             }}>
          <svg className="iso-svg" viewBox={`0 0 ${W} ${H}`} role="img"
               aria-label="Стол денситометра, рельс с кареткой и С-дугой, на столе — фантом позвоночника на поворотном круге">
            <g dangerouslySetInnerHTML={{ __html: STATIC }} />
            <g ref={car} dangerouslySetInnerHTML={{ __html: CARRIAGE }} />
            <g dangerouslySetInnerHTML={{ __html: phantomSvg }} />
            <path ref={beam} className="beam" />
            <circle ref={spot} className="beam-spot" r={1.6} />
            <g ref={arm} dangerouslySetInnerHTML={{ __html: ARM }} />
          </svg>
        </div>
        <ScanFilm ref={film} tilt={tilt} reading={reading} />
      </div>
      <div className="plate-foot">
        <span className="plate-hint">
          <span className="hint-pointer">Ведите курсором — сканер следует.</span> Наклон фантома:
          <button className="nudge" onClick={() => turn(-1)} aria-label="Повернуть фантом на −1°">−</button>
          <span className="nudge-value tabular" data-over={Math.abs(tilt) > LIMIT_TILT || undefined}>{tilt > 0 ? '+' : ''}{tilt}°</span>
          <button className="nudge" onClick={() => turn(1)} aria-label="Повернуть фантом на +1°">+</button>
        </span>
        <span className="plate-readout tabular" data-over={over || undefined} aria-live="off">{readout}</span>
      </div>
    </figure>
  )
}
