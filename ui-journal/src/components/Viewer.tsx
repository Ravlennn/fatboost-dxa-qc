import { useEffect, useRef, useState, type PointerEvent as RPointerEvent } from 'react'
import { imageUrl, rescore, review, type Item } from '../api'
import { liveMeasurements, POINT_LABEL, type Points } from '../geometry'
import { CRITERIA, REGION_LABEL, family, fmt, type Region } from '../labels'
import Annotation from './Annotation'
import { IconAlert, IconBack, IconCheck, IconCross, IconLayers, IconPencil, IconReviewed } from './Icons'
import { Verdict } from './ResultList'

interface Props {
  jobId: string
  item?: Item
  waiting: boolean
  path: string
  onClose: () => void
  onUpdate: (item: Item) => void
}

// Drawing the pencil lines is the viewer's one authored moment: it plays the first time a film is
// opened, and is skipped while someone walks the list quickly (repeated actions stay instant).
const seen = new Set<string>()
let lastDraw = 0

function valueFor(code: string, item: Item): string {
  const m = item.measurements ?? {}
  switch (code) {
    case 'spine_axis_tilt': return `Наклон ${fmt(Math.abs(m.spine_tilt_deg_calibrated ?? m.spine_tilt_deg ?? NaN), 1)}°`
    case 'spine_scan_range': return `Над L1 ${fmt(m.top_margin_mm, 0)} мм · под L4 ${fmt(m.bottom_margin_mm, 0)} мм`
    case 'spine_artifact': {
      if (!item.violations.includes(code)) return 'Не найдено'
      // say what was actually found: the detector's objects, or that the network decided on its own
      const found = item.artifacts ?? []
      const wires = found.filter((a) => a.type === 'wire').length, metal = found.length - wires
      if (!found.length) return 'Предмет не локализован детектором — нарушение по общей оценке сети, проверьте снимок'
      return [wires && `проволока ×${wires}`, metal && `металл ×${metal}`].filter(Boolean).join(', ').replace(/^./, (c) => c.toUpperCase())
    }
    case 'hip_positioning': {
      const lt = item.lt
      if (!lt) return 'Малый вертел: пересчёт после отпускания точки'
      return lt.visible ? `Малый вертел выступает на ${fmt(lt.prominence_mm, 1)} мм` : 'Малый вертел за контуром не виден'
    }
    case 'hip_roi_margins':
      return `Сверху ${fmt(m.roi_top_mm, 0)} · снизу ${fmt(m.roi_bottom_mm, 0)} · слева ${fmt(m.roi_left_mm, 0)} · справа ${fmt(m.roi_right_mm, 0)} мм`
    default: return ''
  }
}

type Busy = null | 'rescore' | 'review'

export default function Viewer({ jobId, item, waiting, path, onClose, onUpdate }: Props) {
  const [wipe, setWipe] = useState(100)          // % of the film covered by the markup, from the left
  const [animateWipe, setAnimateWipe] = useState(false)
  const [highlight, setHighlight] = useState<string | null>(null)
  const [pinned, setPinned] = useState<string | null>(null)
  // which film has loaded: a cached image (the list thumbnail shares its URL) can fire onLoad before the
  // reset effect runs, so the flag is tied to the key instead of being reset to false on every change
  const [loadedKey, setLoadedKey] = useState('')
  const [draw, setDraw] = useState(false)
  const [box, setBox] = useState<{ w: number; h: number } | null>(null)
  // specialist review
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState<Points | null>(null)
  const [dragging, setDragging] = useState<string | null>(null)
  const [preview, setPreview] = useState<Item | null>(null)
  const [overrides, setOverrides] = useState<Record<string, boolean>>({})
  const [busy, setBusy] = useState<Busy>(null)
  const [error, setError] = useState('')
  const stage = useRef<HTMLDivElement>(null)
  const area = useRef<HTMLDivElement>(null)
  const seq = useRef(0)
  const key = item ? `${jobId}:${item.i}` : ''
  const loaded = loadedKey !== '' && loadedKey === key
  const ratio = item?.width && item?.height ? item.width / item.height : 1

  const leaveEdit = () => {
    seq.current += 1   // answers still in flight belong to the edit being left
    setEditing(false); setDraft(null); setPreview(null); setOverrides({}); setError(''); setBusy(null); setDragging(null)
  }

  useEffect(() => {
    setHighlight(null); setPinned(null); leaveEdit()
    if (!key) return
    const now = performance.now()
    const fresh = !seen.has(key) && now - lastDraw > 700
    seen.add(key)
    if (fresh) lastDraw = now
    setDraw(fresh)
  }, [key])

  // the film is sized in px from the space it has, so the markup layer and the image share one box
  useEffect(() => {
    const el = area.current
    if (!el) return
    const fit = () => {
      const cs = getComputedStyle(el)
      const W = el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)
      const H = el.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom)
      if (W <= 0 || H <= 0) return
      const w = Math.min(W, H * ratio)
      setBox({ w: Math.floor(w), h: Math.floor(w / ratio) })
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(el)
    return () => ro.disconnect()
  }, [ratio, item?.has_image])

  // M toggles the markup, E starts correcting the points, Enter confirms, Esc leaves editing or the
  // sheet — instantly (keyboard). Keys pressed on a focused control belong to that control.
  const keys = useRef({ startEdit: () => {}, confirm: () => {}, canConfirm: false })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const onControl = !!t?.closest('button, a, [role="button"], [role="radio"], [role="slider"]')
      if ((e.key === 'm' || e.key === 'ь') && !editing) { setAnimateWipe(false); setWipe((w) => (w > 0 ? 0 : 100)) }
      if ((e.key === 'e' || e.key === 'у') && !editing) keys.current.startEdit()
      if (e.key === 'Enter' && !onControl && keys.current.canConfirm) { e.preventDefault(); keys.current.confirm() }
      if (e.key === 'Escape') { if (editing) leaveEdit(); else onClose() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, editing])

  const wipeDrag = (e: RPointerEvent<HTMLDivElement>) => {
    const b = stage.current?.getBoundingClientRect()
    if (!b) return
    setAnimateWipe(false)
    setWipe(Math.min(100, Math.max(0, ((e.clientX - b.left) / b.width) * 100)))
  }

  const fam = family(item?.region)
  // broken criteria first, the rest in TZ order; ordered by the model's verdict so rows stay put
  // while the specialist toggles decisions
  const criteria = fam ? [...CRITERIA[fam]].sort((a, b) =>
    Number(item?.violations.includes(b.code) ?? false) - Number(item?.violations.includes(a.code) ?? false)) : []
  const W = item?.width ?? 1, H = item?.height ?? 1
  const base = preview ?? item
  // what the film shows while editing: the draft points and their live measurements
  const shown: Item | undefined = item && editing && draft && base ? {
    ...base,
    points: draft,
    measurements: { ...(base.measurements ?? {}), ...liveMeasurements(item.region ?? '', draft, W, H, item.mm_per_px ?? 0.6) },
    lt: dragging ? null : base.lt,
  } : item
  const modelBad = (code: string) => (base?.violations ?? []).includes(code)
  const decided = (code: string) => (code in overrides ? overrides[code] : modelBad(code))

  const startEdit = () => {
    if (!item?.points) return
    setEditing(true); setWipe(100); setDraft({ ...(item.points as Points) }); setPreview(null); setOverrides({})
  }

  const runRescore = async (points: Points) => {
    if (!item) return
    const n = ++seq.current
    setBusy('rescore'); setError('')
    try {
      const res = await rescore(jobId, item.i, points)
      if (n === seq.current) setPreview(res)
    } catch (e) {
      if (n === seq.current) setError((e as Error).message)
    } finally {
      if (n === seq.current) setBusy(null)
    }
  }

  const confirm = async () => {
    if (!item) return
    setBusy('review'); setError('')
    try {
      const moved = editing && draft && JSON.stringify(draft) !== JSON.stringify(item.points)
      const res = await review(jobId, item.i, moved ? draft : null, editing ? overrides : {})
      onUpdate(res)
      leaveEdit()
    } catch (e) {
      setError((e as Error).message); setBusy(null)
    }
  }

  // handle drag in image coordinates
  const handleMove = (name: string, e: RPointerEvent<HTMLButtonElement>) => {
    const b = stage.current?.getBoundingClientRect()
    if (!b || !draft) return
    const x = Math.min(W, Math.max(0, ((e.clientX - b.left) / b.width) * W))
    const y = Math.min(H, Math.max(0, ((e.clientY - b.top) / b.height) * H))
    setDraft({ ...draft, [name]: [x, y] })
  }

  const reviewed = item?.review?.state === 'confirmed'
  const overrideCount = item?.review ? Object.keys(item.review.overrides).length : 0
  // the button names what is being confirmed: the verdict as it stands now
  const verdictBad = editing ? criteria.some((c) => decided(c.code)) : item?.quality === 1
  const confirmText = verdictBad ? 'Подтвердить брак' : 'Подтвердить норму'
  useEffect(() => {
    keys.current = { startEdit, confirm, canConfirm: !!item?.has_image && busy === null && (editing || !reviewed) }
  })

  return (
    <aside className="viewer" aria-label="Снимок" data-editing={editing || undefined}>
      <div className="viewer-head">
        <button className="btn btn-icon viewer-back" onClick={onClose} aria-label="К списку"><IconBack /></button>
        <div className="viewer-title">
          <span className="viewer-name" title={path}>{path.split('/').pop() || 'Снимок'}</span>
          <span className="viewer-sub">
            {item?.region ? REGION_LABEL[item.region as Region] ?? item.region : item ? 'Область не определена' : ''}
            {reviewed && <span className="reviewed"><IconReviewed width={14} height={14} />проверено врачом</span>}
          </span>
        </div>
        {item && <Verdict item={item} />}
      </div>

      {!item ? (
        <div className="viewer-empty">
          <p>{waiting ? 'Первый снимок появится здесь, как только будет проанализирован.' : 'Снимок ещё в очереди.'}</p>
        </div>
      ) : !item.has_image ? (
        <div className="viewer-empty" role="alert">
          <IconAlert width={26} height={26} className="tone-fail" />
          <p><strong>Файл не обработан.</strong> {item.error || 'Это не DICOM-снимок или он повреждён.'}</p>
          <p className="muted">Строка попадёт в отчёт со статусом Failure — остальной пакет не затронут.</p>
        </div>
      ) : (
        <>
          <div ref={area} className="lightbox-film" data-loaded={loaded || undefined}>
            <span className="corner c-tl" aria-hidden /><span className="corner c-tr" aria-hidden />
            <span className="corner c-bl" aria-hidden /><span className="corner c-br" aria-hidden />
            <div ref={stage} className="film" style={box ? { width: box.w, height: box.h } : { visibility: 'hidden' }}>
              <img key={key} src={imageUrl(jobId, item.i)} alt={`Снимок ${path}`} onLoad={() => setLoadedKey(key)} draggable={false} />
              {shown?.points && Object.keys(shown.points).length > 0 && (
                <div className="markup" data-animate={animateWipe || undefined}
                     style={{ clipPath: wipe >= 100 ? 'inset(-48px -240px -48px -240px)' : `inset(-48px ${100 - wipe}% -48px -240px)` }}>
                  <Annotation key={key}
                              item={{ ...shown, violations: editing ? criteria.map((c) => c.code).filter(decided) : shown.violations }}
                              highlight={highlight ?? pinned} onHover={setHighlight} draw={draw && loaded && !editing} />
                </div>
              )}
              {editing && draft && Object.entries(draft).map(([name, [x, y]]) => (
                <button key={name} className="handle" data-active={dragging === name || undefined}
                        style={{ left: `${(x / W) * 100}%`, top: `${(y / H) * 100}%` }}
                        aria-label={`Точка «${POINT_LABEL[name] ?? name}»: перетащите или сдвиньте стрелками`}
                        title={POINT_LABEL[name] ?? name}
                        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setDragging(name) }}
                        onPointerMove={(e) => { if (dragging === name) handleMove(name, e) }}
                        onPointerUp={() => { if (dragging) { setDragging(null); if (draft) runRescore(draft) } }}
                        onKeyDown={(e) => {
                          const step = e.shiftKey ? 5 : 1
                          const d = ({ ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] } as Record<string, number[]>)[e.key]
                          if (!d || !draft) return
                          e.preventDefault(); e.stopPropagation()
                          const next: Points = { ...draft, [name]: [draft[name][0] + d[0], draft[name][1] + d[1]] }
                          setDraft(next); runRescore(next)
                        }} />
              ))}
              {!editing && (
                <div className="wipe" style={{ left: `${wipe}%` }} data-hidden={wipe >= 100 || wipe <= 0 || undefined}
                     onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); wipeDrag(e) }}
                     onPointerMove={(e) => { if (e.currentTarget.hasPointerCapture(e.pointerId)) wipeDrag(e) }}
                     onDoubleClick={() => { setAnimateWipe(true); setWipe(100) }}
                     role="slider" aria-label="Шторка разметки" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(wipe)}
                     tabIndex={0}
                     onKeyDown={(e) => {
                       if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                         e.preventDefault(); e.stopPropagation(); setAnimateWipe(false)
                         setWipe((w) => Math.min(100, Math.max(0, w + (e.key === 'ArrowLeft' ? -5 : 5))))
                       }
                     }}>
                  <span className="wipe-grip" />
                </div>
              )}
            </div>
          </div>

          {editing ? (
            <div className="film-tools film-tools-edit" role="toolbar" aria-label="Коррекция разметки">
              <span className="edit-status" role="status" aria-live="polite">
                {busy === 'rescore' ? 'Пересчёт…' : error ? <span className="tone-bad">{error}</span> : 'Перетащите точки'}
              </span>
              <button className="btn btn-quiet" onClick={() => { seq.current += 1; setDraft({ ...(item.points as Points) }); setPreview(null); setBusy(null) }}>Сбросить</button>
              <button className="btn btn-quiet" onClick={leaveEdit}>Отмена</button>
              <button className="btn btn-primary" onClick={confirm} disabled={busy !== null}>
                <IconCheck />{busy === 'review' ? 'Сохраняем…' : confirmText}<kbd>Enter</kbd>
              </button>
            </div>
          ) : (
            <div className="film-tools">
              <button className="btn btn-quiet" aria-pressed={wipe > 0} aria-label="Разметка"
                      onClick={() => { setAnimateWipe(true); setWipe((w) => (w > 0 ? 0 : 100)) }}>
                <IconLayers /><span className="tool-label">Разметка</span><kbd>M</kbd>
              </button>
              <button className="btn btn-quiet" onClick={() => { setAnimateWipe(true); setWipe((w) => (w === 50 ? 100 : 50)) }}>
                Сравнить
              </button>
              {item.points && Object.keys(item.points).length > 0 && (
                <button className="btn btn-quiet" onClick={startEdit} aria-label="Исправить разметку"><IconPencil /><span className="tool-label">Исправить</span><kbd>E</kbd></button>
              )}
              {reviewed ? (
                <button className="btn btn-done" disabled><IconReviewed />Подтверждено</button>
              ) : (
                <button className="btn btn-primary" onClick={confirm} disabled={busy !== null}
                        title="Согласен с решением модели — записать в отчёт как проверенное">
                  <IconCheck />{busy === 'review' ? 'Сохраняем…' : confirmText}<kbd>Enter</kbd>
                </button>
              )}
            </div>
          )}

          <div className="checks" role="group" aria-label="Критерии ТЗ">
            {criteria.map((c) => {
              const bad = editing ? decided(c.code) : item.violations.includes(c.code)
              const changed = editing && c.code in overrides
              return (
                <div key={c.code} className="check" data-bad={bad || undefined} data-busy={(editing && busy === 'rescore') || undefined}
                    data-hl={(highlight ?? pinned) === c.code || undefined} aria-pressed={pinned === c.code}
                    role="button" tabIndex={0} title={c.rule}
                    onClick={() => setPinned((p) => (p === c.code ? null : c.code))}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setPinned((p) => (p === c.code ? null : c.code)) } }}
                    onPointerEnter={() => setHighlight(c.code)} onPointerLeave={() => setHighlight(null)}>
                  <span className="check-mark">{bad ? <IconCross width={16} height={16} /> : <IconCheck width={16} height={16} />}</span>
                  <span className="check-body">
                    <span className="check-title">{c.title}</span>
                    <span className="check-value tabular">{valueFor(c.code, editing && shown ? shown : item)}</span>
                    {editing ? (
                      <span className="decide" role="radiogroup" aria-label={`Решение врача: ${c.title}`}
                            onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
                        {[true, false].map((v) => (
                          <button key={String(v)} role="radio" aria-checked={decided(c.code) === v}
                                  onClick={() => setOverrides((o) => {
                                    const next = { ...o }
                                    if (v === modelBad(c.code)) delete next[c.code]; else next[c.code] = v
                                    return next
                                  })}>
                            {v ? 'Нарушение' : 'Норма'}
                          </button>
                        ))}
                        {changed && <span className="decide-note">не как у модели</span>}
                      </span>
                    ) : (
                      <>
                        <span className="check-rule"><span className="check-rule-label">По ТЗ:</span> {c.rule[0].toLowerCase() + c.rule.slice(1)}</span>
                        <span className="check-decides">{c.decides}</span>
                      </>
                    )}
                  </span>
                </div>
              )
            })}
          </div>
          {!editing && reviewed && item.review && (item.review.corrected_points || overrideCount > 0) && (
            <p className="note"><IconReviewed width={15} height={15} />
              {[item.review.corrected_points && 'точки исправлены врачом', overrideCount > 0 && 'вердикт изменён врачом'].filter(Boolean).join(', ')}
              {' · модель: '}{item.review.model.violations.length ? 'брак' : 'норма'}
            </p>
          )}
          {!editing && item.warnings?.some((w) => w.includes('closest one')) && (
            <p className="note"><IconAlert width={15} height={15} />Причина выведена: брак найден, но ни один критерий не превысил порог — показан ближайший.</p>
          )}
          {!editing && item.warnings?.some((w) => w.includes('anatomy')) && (
            <p className="note"><IconAlert width={15} height={15} />Область определена неуверенно — проверьте, что отдел и сторона указаны верно. Вердикт по критериям посчитан для указанной области.</p>
          )}
        </>
      )}
    </aside>
  )
}
