// The drawing lying on the desk under the log: the lumbar spine and pelvis in AP projection, drawn
// as a technical sheet, with the TZ limits pencilled on it (the ±5° fan on the spine axis, the ROI
// frame round the left femoral neck). Pure geometry, faint, behind everything; decorative.

const BODY = { w: 118, h: 68, gap: 17, top: 96, x: 300 }
const LEVELS = ['L1', 'L2', 'L3', 'L4', 'L5']

function vertebra(i: number) {
  const y = BODY.top + i * (BODY.h + BODY.gap), h = BODY.h, x0 = BODY.x - BODY.w / 2, x1 = BODY.x + BODY.w / 2
  const w = 7   // the waist of the lateral sides
  const body = `M${x0 + 8} ${y}H${x1 - 8}Q${x1} ${y} ${x1} ${y + 8}Q${x1 - w} ${y + h / 2} ${x1} ${y + h - 8}Q${x1} ${y + h} ${x1 - 8} ${y + h}`
    + `H${x0 + 8}Q${x0} ${y + h} ${x0} ${y + h - 8}Q${x0 + w} ${y + h / 2} ${x0} ${y + 8}Q${x0} ${y} ${x0 + 8} ${y}Z`
  const tp = (s: number) => {
    const a = BODY.x + s * (BODY.w / 2 - 4), b = BODY.x + s * (BODY.w / 2 + 40 - i * 4)
    return `M${a} ${y + 20}L${b} ${y + 16 + i * 2}Q${b + s * 6} ${y + 24} ${b} ${y + 32}L${a} ${y + 34}`
  }
  return (
    <g key={i}>
      <path d={body} className="dd-bone" />
      <path d={tp(-1) + tp(1)} className="dd-bone" />
      <ellipse cx={BODY.x - 32} cy={y + 24} rx={10} ry={14} className="dd-fine" />
      <ellipse cx={BODY.x + 32} cy={y + 24} rx={10} ry={14} className="dd-fine" />
      <path d={`M${BODY.x} ${y + 22}Q${BODY.x + 9} ${y + 40} ${BODY.x} ${y + 58}Q${BODY.x - 9} ${y + 40} ${BODY.x} ${y + 22}Z`} className="dd-fine" />
      <text x={x0 - 14} y={y + h / 2 + 5} className="dd-label" textAnchor="end">{LEVELS[i]}</text>
    </g>
  )
}

/** One half of the pelvis and its femur; s = −1 is the patient's right (drawn on the left). */
function half(s: number) {
  const X = (x: number) => 300 + s * x
  const wing = `M${X(62)} 470C${X(110)} 432 ${X(190)} 398 ${X(242)} 430C${X(272)} 450 ${X(270)} 520 ${X(232)} 590`
    + `C${X(214)} 622 ${X(186)} 640 ${X(170)} 664`
  const acetab = `M${X(184)} 652A52 52 0 0 ${s > 0 ? 1 : 0} ${X(92)} 690`
  const obtur = <ellipse cx={X(76)} cy={770} rx={34} ry={44} className="dd-fine" />
  const ramus = `M${X(20)} 742C${X(50)} 716 ${X(110)} 712 ${X(140)} 730C${X(150)} 760 ${X(130)} 812 ${X(90)} 820C${X(60)} 826 ${X(30)} 818 ${X(14)} 800`
  const head = <circle cx={X(146)} cy={706} r={42} className="dd-bone" />
  const femur = `M${X(176)} 724C${X(196)} 736 ${X(214)} 742 ${X(236)} 740C${X(262)} 736 ${X(272)} 760 ${X(266)} 790`
    + `C${X(262)} 830 ${X(252)} 880 ${X(250)} 940M${X(170)} 744C${X(184)} 772 ${X(192)} 800 ${X(196)} 822`
    + `C${X(204)} 832 ${X(206)} 846 ${X(198)} 856C${X(196)} 880 ${X(196)} 910 ${X(198)} 940`
  return (
    <g key={s}>
      <path d={wing} className="dd-bone" />
      <path d={acetab} className="dd-fine" />
      <path d={ramus} className="dd-bone" />
      {obtur}
      {head}
      <path d={femur} className="dd-bone" />
    </g>
  )
}

export default function DeskDrawing() {
  const top = BODY.top, bottom = BODY.top + 4 * (BODY.h + BODY.gap) - BODY.gap   // L1 top … L4 bottom
  const fan = Math.tan((5 * Math.PI) / 180) * (bottom - top)
  return (
    <svg className="desk-drawing" viewBox="0 0 600 960" aria-hidden focusable="false">
      {[0, 1, 2, 3, 4].map(vertebra)}
      <path d="M262 524L338 524L318 660L300 690L282 660Z" className="dd-bone" />
      {[560, 600, 636].map((y) => <g key={y}><circle cx={282 + (y - 560) * 0.08} cy={y} r={6} className="dd-fine" /><circle cx={318 - (y - 560) * 0.08} cy={y} r={6} className="dd-fine" /></g>)}
      {half(-1)}{half(1)}
      {/* the system's axis and the TZ limit fan on it */}
      <path d={`M300 ${top - 30}V${bottom + 24}`} className="dd-axis" />
      <path d={`M300 ${bottom}L${300 - fan} ${top}M300 ${bottom}L${300 + fan} ${top}`} className="dd-limit" />
      <path d={`M${300 - fan * 0.32} ${bottom - (bottom - top) * 0.32}A${(bottom - top) * 0.32} ${(bottom - top) * 0.32} 0 0 1 ${300 + fan * 0.32} ${bottom - (bottom - top) * 0.32}`} className="dd-limit" />
      <text x={300 + fan + 10} y={top + 18} className="dd-label dd-red">±5°</text>
      {/* margins above L1 and below L4 */}
      <path d={`M470 40V${top}M462 40H478M462 ${top}H478`} className="dd-dim" />
      <text x={482} y={(40 + top) / 2 + 4} className="dd-label">над L1</text>
      {/* the ROI frame round the femoral neck, with its TZ margins */}
      <rect x={378} y={610} width={176} height={200} className="dd-roi" />
      <path d="M466 572V610M458 572H474M466 810V848M458 848H474M340 712H378M340 704V720M554 712H592" className="dd-limit" />
      <text x={474} y={596} className="dd-label dd-red">3 см</text>
      <text x={474} y={836} className="dd-label dd-red">3 см</text>
      <text x={344} y={700} className="dd-label dd-red">2 см</text>
    </svg>
  )
}
