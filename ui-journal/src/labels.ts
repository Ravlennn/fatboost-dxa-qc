// Product language: the TZ's own criteria, in the words a radiologist uses.

export type Region = 'lumbar_spine' | 'hip_left' | 'hip_right'

export const REGION_LABEL: Record<Region, string> = {
  lumbar_spine: 'Поясничный отдел',
  hip_left: 'Бедро, левое',
  hip_right: 'Бедро, правое',
}

export const REGION_SHORT: Record<Region, string> = {
  lumbar_spine: 'L1–L4',
  hip_left: 'Бедро Л',
  hip_right: 'Бедро П',
}

export interface Criterion {
  code: string
  title: string
  rule: string       // the TZ requirement, as written
  decides: string    // how the system actually decides this criterion (the TZ numbers alone are not the rule)
}

export const CRITERIA: Record<'spine' | 'hip', Criterion[]> = {
  spine: [
    { code: 'spine_scan_range', title: 'Границы сканирования', rule: 'Снизу видны верхние края подвздошных костей, сверху — половина Th12', decides: 'Решает модель по ориентирам L1–L4 и гребням, обученная на разметке экспертов' },
    { code: 'spine_axis_tilt', title: 'Ось позвоночника', rule: 'Наклон не больше 5°', decides: 'Решает модель по наклону оси, обученная на разметке экспертов' },
    { code: 'spine_artifact', title: 'Посторонние предметы', rule: 'Нет металла, косточек белья и наложений', decides: 'Решает детектор проволоки и металла вместе с моделью' },
  ],
  hip: [
    { code: 'hip_positioning', title: 'Ротация бедра', rule: 'Малый вертел слегка выступает за контур кости', decides: 'Решает нейросеть вместе с замером выступа малого вертела' },
    { code: 'hip_roi_margins', title: 'Поля области интереса', rule: '3 см сверху и снизу, 2 см сбоку', decides: 'Брак — если рамка ROI выходит за край снимка. Поля меньше ТЗ эксперты браком не считают' },
  ],
}

export const VIOLATION_LABEL: Record<string, string> = {
  spine_scan_range: 'Границы сканирования',
  spine_axis_tilt: 'Наклон оси',
  spine_artifact: 'Посторонние предметы',
  hip_positioning: 'Ротация',
  hip_roi_margins: 'Поля ROI',
  quality_unspecified: 'Причина не уточнена',
}

export const family = (region?: string | null): 'spine' | 'hip' | null =>
  region === 'lumbar_spine' ? 'spine' : region?.startsWith('hip') ? 'hip' : null

export const plural = (n: number, one: string, few: string, many: string) => {
  const m10 = n % 10, m100 = n % 100
  if (m10 === 1 && m100 !== 11) return one
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few
  return many
}

export const fmt = (v: number | null | undefined, digits = 1) =>
  v == null || !Number.isFinite(v) ? '—' : v.toLocaleString('ru-RU', { minimumFractionDigits: digits, maximumFractionDigits: digits })
