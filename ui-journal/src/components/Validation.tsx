import { useEffect, useRef, useState, type CSSProperties } from 'react'

// How the system did on the 249 labelled images, one mark per image, in the check sheet's own
// grammar. Counts are the fully isolated validation with the release logic since 18.10 (hip ROI edge
// rule + closest cause): docs/METRICS_BY_TYPE.md, «По снимкам: логика релиза».

type Kind = 'hit' | 'miss' | 'ok' | 'alarm'
const run = (kind: Kind, n: number): Kind[] => Array.from({ length: n }, () => kind)

const ALL = { tp: 56, fn: 17, fp: 33, tn: 143 }
// headline metrics of the same run, 95% CI by bootstrap over 100 studies (README «Метрики»)
const AUC = { v: '0,837', ci: '0,77–0,90' }
const F1 = { v: '0,691', ci: '0,58–0,78' }
const REGIONS = [
  { name: 'Позвоночник', tp: 23, fn: 9, tn: 57, fp: 10 },
  { name: 'Бедро', tp: 33, fn: 8, tn: 86, fp: 23 },
]
const CAUSES = [
  { name: 'Посторонние предметы', tp: 12, fn: 5, fp: 2 },
  { name: 'Ротация бедра', tp: 27, fn: 9, fp: 26 },
  { name: 'Ось позвоночника', tp: 6, fn: 4, fp: 9 },
  { name: 'Границы сканирования', tp: 3, fn: 3, fp: 5 },
  { name: 'Поля ROI', tp: 4, fn: 3, fp: 0 },
]

function Marks({ kinds, cols, cell = 15, label }: { kinds: Kind[]; cols: number; cell?: number; label: string }) {
  const rows = Math.ceil(kinds.length / cols)
  const r = cell * 0.2, t = cell * 0.24
  return (
    <svg className="marks" width={cols * cell} height={rows * cell} viewBox={`0 0 ${cols * cell} ${rows * cell}`} role="img" aria-label={label}>
      {kinds.map((k, i) => {
        const x = (i % cols) * cell + cell / 2, y = Math.floor(i / cols) * cell + cell / 2
        const d = { '--d': `${Math.min(i, 160) * 6}ms` } as CSSProperties
        if (k === 'ok') return <circle key={i} className="mk mk-ok" style={d} cx={x} cy={y} r={r} />
        if (k === 'alarm') return <circle key={i} className="mk mk-alarm" style={d} cx={x} cy={y} r={r * 1.35} />
        return <path key={i} className={`mk mk-${k}`} style={d} d={`M${x - t} ${y - t}L${x + t} ${y + t}M${x + t} ${y - t}L${x - t} ${y + t}`} />
      })}
    </svg>
  )
}

const pct = (a: number, b: number) => `${Math.round((a / b) * 100)}%`

export default function Validation() {
  const box = useRef<HTMLElement>(null)
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    const el = box.current
    if (!el) return
    const io = new IntersectionObserver((e) => { if (e.some((x) => x.isIntersecting)) { setSeen(true); io.disconnect() } }, { threshold: 0.15 })
    io.observe(el)
    return () => io.disconnect()
  }, [])

  return (
    <section ref={box} id="validation" className="validation" data-seen={seen || undefined} aria-labelledby="validation-title">
      <div className="validation-sheet">
        <header className="validation-head">
          <h2 id="validation-title">Проверка на 249 снимках с разметкой экспертов</h2>
          <p>Одна отметка — один снимок. Каждый оценён моделью, которая его никогда не видела при обучении.</p>
        </header>

        <dl className="headline tabular">
          <div><dt>ROC-AUC</dt><dd><b>{AUC.v}</b><span>[{AUC.ci}]</span></dd></div>
          <div><dt>F1</dt><dd><b>{F1.v}</b><span>[{F1.ci}]</span></dd></div>
          <div><dt>брака найдено</dt><dd><b className="tone-bad">{ALL.tp} из {ALL.tp + ALL.fn}</b></dd></div>
          <div><dt>нормы без замечаний</dt><dd><b className="tone-ok">{ALL.tn} из {ALL.tn + ALL.fp}</b></dd></div>
        </dl>

        <div className="outcomes">
          <figure className="outcome">
            <figcaption>
              <span className="big tabular"><b>{ALL.tp}</b> из {ALL.tp + ALL.fn}</span>
              <span className="what">бракованных снимков система нашла</span>
            </figcaption>
            <Marks kinds={[...run('hit', ALL.tp), ...run('miss', ALL.fn)]} cols={19} label={`Брак: найдено ${ALL.tp}, пропущено ${ALL.fn}`} />
            <p className="split tabular">{REGIONS.map((g) => `${g.name.toLowerCase()} ${g.tp} из ${g.tp + g.fn}`).join(' · ')}</p>
          </figure>
          <figure className="outcome">
            <figcaption>
              <span className="big tabular"><b>{ALL.tn}</b> из {ALL.tn + ALL.fp}</span>
              <span className="what">нормальных снимков пропустила без замечаний</span>
            </figcaption>
            <Marks kinds={[...run('ok', ALL.tn), ...run('alarm', ALL.fp)]} cols={20} label={`Норма: подтверждено ${ALL.tn}, ложных тревог ${ALL.fp}`} />
            <p className="split tabular">{REGIONS.map((g) => `${g.name.toLowerCase()} ${g.tn} из ${g.tn + g.fp}`).join(' · ')}</p>
          </figure>
        </div>

        <ul className="legend" aria-label="Обозначения">
          <li><svg viewBox="0 0 12 12" aria-hidden><path className="mk mk-hit" d="M3 3L9 9M9 3L3 9" /></svg>брак найден</li>
          <li><svg viewBox="0 0 12 12" aria-hidden><path className="mk mk-miss" d="M3 3L9 9M9 3L3 9" /></svg>брак пропущен</li>
          <li><svg viewBox="0 0 12 12" aria-hidden><circle className="mk mk-ok" cx={6} cy={6} r={2.6} /></svg>норма подтверждена</li>
          <li><svg viewBox="0 0 12 12" aria-hidden><circle className="mk mk-alarm" cx={6} cy={6} r={3.4} /></svg>ложная тревога</li>
        </ul>

        <h3 className="causes-title">По каждой причине брака из ТЗ</h3>
        <table className="causes">
          <tbody>
            {CAUSES.map((c) => (
              <tr key={c.name}>
                <th scope="row">{c.name}</th>
                <td className="causes-marks">
                  <Marks kinds={[...run('hit', c.tp), ...run('miss', c.fn)]} cols={Math.min(c.tp + c.fn, 18)} cell={12}
                         label={`${c.name}: найдено ${c.tp} из ${c.tp + c.fn}`} />
                </td>
                <td className="causes-num tabular"><b>{c.tp}</b> из {c.tp + c.fn}<span>{pct(c.tp, c.tp + c.fn)}</span></td>
                <td className="causes-alarm tabular">ложных тревог {c.fp}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <p className="tech">
          <span className="tabular"><b>ROC-AUC {AUC.v}</b> [{AUC.ci}]</span> — в 84% пар «бракованный + нормальный снимок» бракованный
          получает более высокую оценку. <span className="tabular"><b>F1 {F1.v}</b> [{F1.ci}]</span> — баланс найденного брака и ложных
          тревог. В скобках 95% интервал, бутстрэп по 100 исследованиям. Правило полей ROI (рамка за краем снимка) найдено на этой же выборке; без него F1 0,675, ROC-AUC 0,828. Это выборка разработки; независимый тест проводит организатор.
        </p>
      </div>

      <dl className="vfacts">
        <div><dt>499 из 499</dt><dd>DICOM обучающего архива организатора обработаны со статусом Success</dd></div>
        <div><dt>≈ 9 с</dt><dd>на исследование на CPU в 2 потока — при лимите ТЗ 3 минуты</dd></div>
        <div><dt>0,9°</dt><dd>наибольший дрейф наклона оси при поворотах, сдвигах и смене контраста</dd></div>
        <div><dt>1 в 1</dt><dd>два прогона на 305 снимках дают одинаковый отчёт; образ для linux/amd64 — те же решения</dd></div>
      </dl>
    </section>
  )
}
