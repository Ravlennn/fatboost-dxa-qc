// The demonstration's one model: a spine phantom on a turntable, scanned by the densitometer arm.
// Everything the hero shows (arm position, the film being built, the readout, the ticked log rows)
// is computed here, in phantom millimetres, with the same rules the product applies to a real film:
// tilt between the top of L1 and the bottom of L4, margins from the film edges to those points.

export type V2 = [number, number]

/** Spine axis runs along world x; the head end is at −x. One world unit is one phantom millimetre. */
export const SPINE = {
  centre: 6,                        // turntable axis, x
  body: { len: 20, half: 9.5 },     // vertebral body along the axis / half-width
  gap: 5,                           // disc
  levels: ['Th12', 'L1', 'L2', 'L3', 'L4', 'L5'] as const,
  first: -62,                       // centre of Th12 along x
}

export const bodyCentre = (index: number) => SPINE.first + index * (SPINE.body.len + SPINE.gap)

/** Iliac crests: the lateral wings beside L5, their top edge is what the scan must reach. */
export const CRESTS = { x0: 52, x1: 84, y0: 12.5, y1: 27, top: 52 }

/** The arm sweeps the film from the pelvis (+x) to the head end (−x). */
export const SCAN = { from: 96, to: -78 }
export const LIMIT_TILT = 5

const rot = ([x, y]: V2, deg: number): V2 => {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a)
  const dx = x - SPINE.centre
  return [SPINE.centre + dx * c - y * s, dx * s + y * c]
}
export const rotate = rot

export interface Landmarks { L1_top: V2; L4_bottom: V2; Th12_mid: V2; crest: V2 }

export function landmarks(tilt: number): Landmarks {
  const half = SPINE.body.len / 2
  return {
    L1_top: rot([bodyCentre(1) - half, 0], tilt),
    L4_bottom: rot([bodyCentre(4) + half, 0], tilt),
    Th12_mid: rot([bodyCentre(0), 0], tilt),
    crest: rot([CRESTS.top, 0], tilt),
  }
}

export interface Reading {
  tilt: number            // degrees, measured on the film from L1 top to L4 bottom
  top: number | null      // mm from the film's top edge to L1 top, once the arm has passed it
  bottom: number | null   // mm from L4 bottom to the film's bottom edge
  th12: boolean | null    // half of Th12 is on the film
  crests: boolean | null  // the iliac crests are on the film
  progress: number        // 0..1 of the sweep
}

/** What the system can say when the arm stands at `x` (it has scanned from SCAN.from down to x). */
export function read(tilt: number, x: number): Reading {
  const p = landmarks(tilt)
  const progress = Math.min(1, Math.max(0, (SCAN.from - x) / (SCAN.from - SCAN.to)))
  const seen = (pt: V2) => x <= pt[0]
  const done = progress >= 1
  const measuredTilt = (Math.atan2(p.L4_bottom[1] - p.L1_top[1], p.L4_bottom[0] - p.L1_top[0]) * 180) / Math.PI
  return {
    tilt: seen(p.L1_top) ? measuredTilt : NaN,
    top: done ? p.L1_top[0] - SCAN.to : null,
    bottom: seen(p.L4_bottom) ? SCAN.from - p.L4_bottom[0] : null,
    th12: done ? SCAN.to <= p.Th12_mid[0] : null,
    crests: seen(p.crest) || done ? SCAN.from >= p.crest[0] : null,
    progress,
  }
}
