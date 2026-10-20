// Live measurements while a point is being dragged. Same formulas as dxaqc/landmarks.py
// (measurements()); the server's values replace these as soon as the point is released.

export type Points = Record<string, [number, number]>

const TILT_SLOPE = 0.921   // tilt calibration against known rotations (landmarks.py)
const deg = (r: number) => (r * 180) / Math.PI

export function liveMeasurements(region: string, p: Points, W: number, H: number, mm: number): Record<string, number> {
  if (region === 'lumbar_spine' && p.L1_top && p.L4_bottom) {
    const a = p.L1_top, b = p.L4_bottom
    const tilt = deg(Math.atan2(b[0] - a[0], b[1] - a[1]))
    return { spine_tilt_deg: tilt, spine_tilt_deg_calibrated: tilt / TILT_SLOPE,
             top_margin_mm: a[1] * mm, bottom_margin_mm: (H - b[1]) * mm }
  }
  if (region.startsWith('hip') && p.roi_top && p.roi_bottom && p.roi_left && p.roi_right) {
    const out: Record<string, number> = {
      roi_top_mm: p.roi_top[1] * mm, roi_bottom_mm: (H - p.roi_bottom[1]) * mm,
      roi_left_mm: p.roi_left[0] * mm, roi_right_mm: (W - p.roi_right[0]) * mm,
    }
    if (p.neck_head_side && p.neck_troch_side) {
      const dx = p.neck_head_side[0] - p.neck_troch_side[0], dy = p.neck_head_side[1] - p.neck_troch_side[1]
      out.neck_axis_deg = Math.abs(deg(Math.atan2(dy, Math.abs(dx))))
    }
    return out
  }
  return {}
}

export const POINT_LABEL: Record<string, string> = {
  L1_top: 'верх L1', L1L2: 'L1/L2', L2L3: 'L2/L3', L3L4: 'L3/L4', L4_bottom: 'низ L4',
  roi_top: 'верх ROI', roi_bottom: 'низ ROI', roi_left: 'левый край ROI', roi_right: 'правый край ROI',
  neck_center: 'центр шейки', neck_head_side: 'ось шейки: к головке', neck_troch_side: 'ось шейки: к вертелу',
}
