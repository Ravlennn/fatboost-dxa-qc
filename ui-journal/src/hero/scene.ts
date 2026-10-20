// The densitometer, drawn with the isometric kit: a scanning table on a cabinet, a rail along its
// back carrying the C-arm, and a spine phantom on a turntable. Static parts are built once; the arm
// is built at x = 0 and translated; the phantom is rebuilt when its tilt changes (a turn about z
// can't be a screen transform). World units are phantom millimetres; z is up.
import * as k from '../iso/iso-kit'
import type { Plan, Plane, Point3, SolidPaths } from '../iso/iso-kit'
import { CRESTS, SCAN, SPINE, bodyCentre, rotate, type V2 } from './model'

export const W = 700
export const H = 430

const CAB: Plan = { x: -158, y: -38, w: 316, d: 76, r: 5 }
const PLINTH: Plan = { x: -148, y: -29, w: 296, d: 58, r: 4 }
const BOARD: Plan = { x: -170, y: -44, w: 340, d: 88, r: 6 }
const PAD: Plan = { x: -165, y: -40, w: 330, d: 80, r: 9 }
const RAIL: Plan = { x: -164, y: -58, w: 328, d: 13, r: 2 }
const Z = { foot: -4, plinth: 2, cab: 8, board: 54, pad: 58, padTop: 66, ring: 67.5, plate: 70, floor: 72, wall: 79,
            railLo: 24, railHi: 84, carTop: 94, colTop: 146, beamTop: 158, head: 122 }
const TRAY: Plan = { x: -84, y: -31, w: 186, d: 62, r: 5 }
const TT = { x: SPINE.centre, y: 0, base: 47, plate: 40 }

export const P = k.fitProjection([
  ...k.boxCorners(BOARD, Z.foot, Z.padTop),
  ...k.boxCorners(RAIL, Z.railLo, Z.railHi + 2),
  ...k.boxCorners({ x: SCAN.to - 17, y: -61, w: SCAN.from - SCAN.to + 34, d: 89, r: 0 }, Z.head, Z.beamTop),
], W, H, { pad: 12, azimuth: 62, elevation: 30 })

const at = (p: Point3) => k.iso(p, P)
const S = (paths: SolidPaths, style?: k.SolidStyle) => k.solidSvg(paths, style)
const L = (d: string, style?: k.LineStyle) => k.lineSvg(d, style)
const D = (pts: Array<[number, number] | [number, number, number]>, style?: { size?: number; tone?: k.Tone; pulse?: boolean }) => k.dotsSvg(pts, style)
const depth = (p: Point3) => k.depthOf(p, P)
const byDepth = (pts: Plane[], z = 0) => [...pts].sort((a, b) => depth([a[0], a[1], z]) - depth([b[0], b[1], z]))

/** A closed outline in the vertical plane y = face (u along x, v up). */
const onFaceY = (face: number, pts: [number, number][]) => k.pathOf(pts.map(([u, v]) => at([u, face, v])), true)
const slotY = (face: number, x: number, z0: number, w: number, h: number) => {
  const r = w / 2, cx = x + r, pts: [number, number][] = []
  for (let i = 0; i <= 6; i++) { const a = Math.PI - (Math.PI * i) / 6; pts.push([cx + r * Math.cos(a), z0 + h - r + r * Math.sin(a)]) }
  for (let i = 0; i <= 6; i++) { const a = -(Math.PI * i) / 6; pts.push([cx + r * Math.cos(a), z0 + r + r * Math.sin(a)]) }
  return onFaceY(face, pts)
}

// ---------------------------------------------------------------- static: table, cabinet, rail
function table(): string {
  const out: string[] = []
  out.push(`<path class="iso-halo" d="${k.haloOf(BOARD, Z.foot, Z.board - Z.foot, P)}"/>`)
  // rail along the back, on three brackets bolted to the cabinet
  out.push(S(k.slabOf({ x: -169, y: -59.5, w: 5, d: 16, r: 1.2 }, Z.railLo - 2, Z.railHi - Z.railLo + 4, P, 3, 0.5), { tone: 'mid' }))
  const railInner = L(k.sideSeam(RAIL, 32, P, 2), { tone: 'faint' })
  out.push(S(k.slabOf(RAIL, Z.railLo, Z.railHi - Z.railLo, P, 2, 0.8), { tone: 'mid', inner: railInner }))
  out.push(L(k.planOutline({ x: -158, y: -53.5, w: 316, d: 4, r: 2 }, Z.railHi, P, 4), { tone: 'lo' }))
  const ticks = k.sideTicks(-156, 156, 4, 5, -45, Z.railHi, [2.2, 4.4], P, 'x')
  out.push(L(ticks.minor, { tone: 'lo' }), L(ticks.major, { tone: 'mid' }))
  out.push(S(k.slabOf({ x: 164, y: -59.5, w: 5, d: 16, r: 1.2 }, Z.railLo - 2, Z.railHi - Z.railLo + 4, P, 3, 0.5), { tone: 'mid' }))
  out.push(D([at([166.5, -51.5, Z.railHi + 2]), at([-166.5, -51.5, Z.railHi + 2])], { size: 0.5 }))
  for (const x of [-120, 0, 120]) out.push(S(k.slabOf({ x: x - 7, y: -45, w: 14, d: 7, r: 1 }, 30, 12, P, 2), { tone: 'lo' }))
  // feet and plinth
  for (const [x, y] of byDepth(k.corners(PLINTH, 12), Z.foot)) {
    out.push(S(k.cylinder(x, y, 6.5, Z.foot, 1.4, P, 28), { tone: 'lo' }))
    out.push(S(k.cylinder(x, y, 4.8, Z.foot + 1.4, Z.plinth - Z.foot - 1.4, P, 24), { tone: 'mid' }))
  }
  out.push(S(k.slabOf(PLINTH, Z.plinth, Z.cab - Z.plinth, P, 6), { tone: 'lo' }))
  // cabinet: two trim bands, vents and a label plate on the long face, a power key on the end
  const cabInner = L(k.sideSeam(CAB, 16, P, 8), { tone: 'faint' }) + L(k.sideSeam(CAB, 47, P, 8), { tone: 'faint' })
  out.push(S(k.slabOf(CAB, Z.cab, Z.board - Z.cab, P, 8, 1.2), { tone: 'mid', inner: cabInner }))
  const face = CAB.y + CAB.d
  const vents: string[] = []
  for (let i = 0; i < 18; i++) vents.push(slotY(face, -136 + i * 6.2, 22, 2.6, 18))
  out.push(L(vents.join(''), { tone: 'lo' }))
  out.push(L(onFaceY(face, [[86, 21], [134, 21], [134, 41], [86, 41]]), { tone: 'mid' }))
  out.push(L(onFaceY(face, [[88, 23], [132, 23], [132, 39], [88, 39]]), { tone: 'faint' }))
  for (const [z, len] of [[35.5, 34], [31, 26], [26.5, 18]] as const) out.push(L(k.segment([92, face, z], [92 + len, face, z], P), { tone: 'lo', free: true }))
  out.push(D([at([90, face, 37]), at([130, face, 37])], { size: 0.45 }))
  const end = CAB.x + CAB.w
  out.push(L(k.sideRing([end, 18, 32], 3.4, 'x', P, 28) + k.sideRing([end, 18, 32], 2.1, 'x', P, 24), { tone: 'mid' }))
  out.push(D([at([end, 18, 32])], { size: 0.6, tone: 'hi' }))
  // table board with a ruler strip along the front, then the padded top with its piping
  out.push(S(k.slabOf(BOARD, Z.board, Z.pad - Z.board, P, 8, 0.8), { tone: 'mid' }))
  const strip: Plan = { x: -160, y: 40.6, w: 320, d: 3, r: 0.6 }
  out.push(S(k.slabOf(strip, Z.pad, 0.3, P, 2), { tone: 'lo', crease: 'none' }))
  const rt = k.topTicks(strip.x + 2, strip.x + strip.w - 2, 4, 5, strip.y + 0.1, Z.pad + 0.3, [1.1, 2.3], P)
  out.push(L(rt.minor, { tone: 'lo' }), L(rt.major, { tone: 'mid' }))
  out.push(S(k.slabOf(PAD, Z.pad, Z.padTop - Z.pad, P, 10, 1.4), { tone: 'mid', inner: L(k.sideSeam(PAD, 62, P, 10), { tone: 'faint' }) }))
  out.push(L(k.planOutline(k.insetPlan(PAD, 5), Z.padTop, P, 10), { tone: 'faint' }))
  // turntable base with a degree arc facing the viewer; ±5° (the TZ limit) in red pencil
  out.push(S(k.cylinder(TT.x, TT.y, TT.base, Z.padTop, Z.ring - Z.padTop, P, 72, 0.6), { tone: 'mid' }))
  const arc = k.radialTicks(TT.x, TT.y, TT.base - 0.6, 30, 5, Z.ring, [1.6, 3.2], P, Math.PI - (15 * Math.PI) / 180, (30 * Math.PI) / 180)
  out.push(L(arc.minor, { tone: 'lo' }), L(arc.major, { tone: 'mid' }))
  for (const s of [-1, 1]) {
    const a = Math.PI + (s * 5 * Math.PI) / 180, r0 = TT.base - 4.4, r1 = TT.base - 0.6
    out.push(L(k.lineOnTop([TT.x + Math.sin(a) * r0, TT.y - Math.cos(a) * r0], [TT.x + Math.sin(a) * r1, TT.y - Math.cos(a) * r1], Z.ring, P), { tone: 'red' }))
  }
  return out.join('')
}

// ---------------------------------------------------------------- moving: carriage + column (behind the table)
function carriage(): string {
  const out: string[] = []
  out.push(S(k.slabOf({ x: -18, y: -61, w: 36, d: 2.6, r: 0.6 }, 74, Z.railHi - 74, P, 2, 0.3), { tone: 'mid' }))
  const front = -45 + 2.6
  out.push(S(k.slabOf({ x: -18, y: -45, w: 36, d: 2.6, r: 0.6 }, 74, Z.railHi - 74, P, 2, 0.3), { tone: 'mid' }))
  out.push(L([-10, 10].map((x) => k.sideRing([x, front, 78.5], 2.6, 'y', P, 24)).join(''), { tone: 'mid' }))
  out.push(D([-10, 10].map((x) => at([x, front, 78.5])), { size: 0.55 }))
  out.push(S(k.slabOf({ x: -20, y: -61.5, w: 40, d: 19.5, r: 3 }, Z.railHi, Z.carTop - Z.railHi, P, 4, 0.8), { tone: 'mid' }))
  out.push(D(k.corners({ x: -20, y: -61.5, w: 40, d: 19.5, r: 3 }, 3.4).map(([x, y]) => at([x, y, Z.carTop])), { size: 0.5 }))
  const col: Plan = { x: -11, y: -58, w: 22, d: 13, r: 3 }
  const inner = L(k.sideSeam(col, 104, P, 4) + k.sideSeam(col, 136, P, 4), { tone: 'faint' })
  out.push(S(k.slabOf(col, Z.carTop, Z.colTop - Z.carTop, P, 4, 0.8), { tone: 'mid', inner }))
  return out.join('')
}

// ---------------------------------------------------------------- moving: beam + detector head (above everything)
function arm(): string {
  const out: string[] = []
  const beam: Plan = { x: -15, y: -61, w: 30, d: 89, r: 5 }
  out.push(S(k.slabOf(beam, Z.colTop, Z.beamTop - Z.colTop, P, 6, 1.2), { tone: 'mid', inner: L(k.sideSeam(beam, 150, P, 6), { tone: 'faint' }) }))
  const panel: Plan = { x: -10, y: -56, w: 20, d: 16, r: 2 }
  out.push(L(k.planOutline(panel, Z.beamTop, P, 4), { tone: 'lo' }))
  for (const x of [-5, 0, 5]) out.push(S(k.cylinder(x, -50, 1.7, Z.beamTop, 0.9, P, 18), { tone: 'mid' }))
  out.push(D([at([6.5, -43, Z.beamTop])], { size: 0.6, tone: 'mid' }))
  const head: Plan = { x: -17, y: -19, w: 34, d: 38, r: 4 }
  const hInner = L(k.sideSeam(head, 128, P, 6) + k.sideSeam(head, 141, P, 6), { tone: 'faint' })
  out.push(S(k.slabOf(head, Z.head, Z.colTop - Z.head, P, 6, 0.8), { tone: 'hi', lit: true, inner: hInner }))
  out.push(L(onFaceY(19, [[-8, 131.5], [8, 131.5], [8, 137.5], [-8, 137.5]]), { tone: 'mid' }))
  return out.join('')
}

// ---------------------------------------------------------------- the phantom, rebuilt per tilt
interface Item { at: Point3; svg: string }
const rotRing = (ring: Plane[], tilt: number) => ring.map((p) => rotate(p as V2, tilt) as Plane)
const rslab = (plan: Plan, z: number, h: number, tilt: number, steps = 4, bevel = 0) =>
  k.extrude(rotRing(k.ringOf(plan, steps), tilt), z, h, P, { bevel })
const rpt = (x: number, y: number, z: number, tilt: number): Point3 => { const [a, b] = rotate([x, y], tilt); return [a, b, z] }
const centre = (plan: Plan, z: number, tilt: number) => rpt(plan.x + plan.w / 2, plan.y + plan.d / 2, z, tilt)

const BAR: Plan = { x: bodyCentre(0) - 12, y: -5, w: bodyCentre(5) - bodyCentre(0) + 24, d: 10, r: 2 }
const bodyPlan = (i: number): Plan => ({ x: bodyCentre(i) - SPINE.body.len / 2, y: -SPINE.body.half, w: SPINE.body.len, d: SPINE.body.half * 2, r: 4.5 })
/** A vertebral body seen from the front: straight endplates, lateral sides drawn in at the waist. */
export function bodyRing(i: number): Plane[] {
  const c = bodyCentre(i), h = SPINE.body.len / 2, w = SPINE.body.half, waist = 1.8, r = 3.2, out: Plane[] = []
  const corner = (cx: number, cy: number, a0: number) => { for (let s = 0; s <= 4; s++) { const a = ((a0 + (90 * s) / 4) * Math.PI) / 180; out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]) } }
  const side = (y: number, dir: number) => { for (let s = 1; s < 8; s++) { const t = s / 8, x = c + dir * (h - r) * (2 * t - 1); out.push([x, y - Math.sign(y) * waist * Math.sin(Math.PI * t)]) } }
  corner(c + h - r, -w + r, -90); corner(c + h - r, w - r, 0); side(w, -1); corner(c - h + r, w - r, 90); corner(c - h + r, -w + r, 180); side(-w, 1)
  return out
}
const discPlan = (i: number): Plan => ({ x: bodyCentre(i) + SPINE.body.len / 2, y: -7.5, w: SPINE.gap, d: 15, r: 2 })
const BODY_Z = [76, 88] as const
const wing = (s: number): Plane[] => {
  const pts: Plane[] = [[CRESTS.x0, 12.5], [56, 21], [64, 26], [74, 27], [82, 24], [CRESTS.x1, 17], [80, 12.5]]
  const out = pts.map(([x, y]) => [x, s * y] as Plane)
  return s > 0 ? out : out.reverse()
}

export function phantom(tilt: number): string {
  const out: string[] = []
  // turntable plate and its pointer (turns with the phantom)
  out.push(S(k.cylinder(TT.x, TT.y, TT.plate, Z.ring, Z.plate - Z.ring, P, 72, 0.5), { tone: 'mid' }))
  const tip = rotate([TT.x, TT.base - 1.2], tilt), base = rotate([TT.x, TT.plate - 6], tilt)
  const side = (d: number) => { const p = rotate([TT.x + d, TT.plate - 0.5], tilt); return p as Plane }
  out.push(L(k.onTop([side(-2.2), tip as Plane, side(2.2)], Z.ring, P), { tone: 'hi' }))
  out.push(L(k.lineOnTop(base as Plane, rotate([TT.x, TT.plate], tilt) as Plane, Z.plate, P), { tone: 'hi' }))
  // tray: floor, the two far walls, the contents by depth, the two near walls
  out.push(S(rslab(TRAY, Z.plate, Z.floor - Z.plate, tilt, 6, 0.5), { tone: 'mid' }))
  const t = 2.4
  const walls = {
    back: { x: TRAY.x, y: TRAY.y, w: TRAY.w, d: t, r: 1 },
    left: { x: TRAY.x, y: TRAY.y + t, w: t, d: TRAY.d - 2 * t, r: 0.5 },
    front: { x: TRAY.x, y: TRAY.y + TRAY.d - t, w: TRAY.w, d: t, r: 1 },
    right: { x: TRAY.x + TRAY.w - t, y: TRAY.y + t, w: t, d: TRAY.d - 2 * t, r: 0.5 },
  }
  const wall = (p: Plan) => S(rslab(p, Z.floor, Z.wall - Z.floor, tilt, 2, 0.4), { tone: 'mid' })
  out.push(wall(walls.back), wall(walls.left))

  const items: Item[] = []
  items.push({ at: centre(BAR, 74, tilt), svg: S(rslab(BAR, Z.floor, BODY_Z[0] - Z.floor, tilt, 3), { tone: 'mid' }) })
  for (let i = 0; i < 6; i++) {
    const b = bodyPlan(i), ring = rotRing(k.ringOf(b, 5), tilt), c = bodyCentre(i)
    const inner = L(k.ringSeam(ring, 82, P), { tone: 'faint' })
    const eyes = [-1, 1].map((s) => k.onTop(Array.from({ length: 20 }, (_, n) => {
      const a = (n / 20) * Math.PI * 2
      return rotate([c - 2.5 + 2.4 * Math.cos(a), s * 5.4 + 1.7 * Math.sin(a)], tilt) as Plane
    }), BODY_Z[1], P, true)).join('')
    const rim = k.onTop(rotRing(k.ringOf(k.insetPlan(b, 1.6), 5), tilt), BODY_Z[1], P, true)
    const spinous = k.onTop([rotate([c + 2, 0], tilt), rotate([c + 6.5, -1.3], tilt), rotate([c + 7.6, 0], tilt), rotate([c + 6.5, 1.3], tilt)] as Plane[], BODY_Z[1], P, true)
    items.push({ at: centre(b, 82, tilt), svg: S(k.extrude(ring, BODY_Z[0], BODY_Z[1] - BODY_Z[0], P, { bevel: 0.5 }), { tone: 'hi', inner })
      + L(rim, { tone: 'faint' }) + L(eyes + spinous, { tone: 'mid' }) })
    if (i < 5) { const d = discPlan(i); items.push({ at: centre(d, 81, tilt), svg: S(rslab(d, BODY_Z[0], 11, tilt, 2), { tone: 'lo' }) }) }
    if (i < 5) for (const s of [-1, 1]) {
      const pr: Plan = { x: bodyCentre(i) - 3.5, y: s > 0 ? 10 : -17.5, w: 7, d: 7.5, r: 2 }
      items.push({ at: centre(pr, 74, tilt), svg: S(rslab(pr, Z.floor, 4, tilt, 3), { tone: 'mid' }) })
    }
  }
  for (const s of [-1, 1]) {
    const ring = rotRing(wing(s), tilt)
    const cx = ring.reduce((a, p) => a + p[0], 0) / ring.length, cy = ring.reduce((a, p) => a + p[1], 0) / ring.length
    items.push({ at: [cx, cy, 77], svg: S(k.extrude(ring, Z.floor, 10, P, { bevel: 0.6 }), { tone: 'hi' }) })
  }
  items.sort((a, b) => depth(a.at) - depth(b.at))
  out.push(...items.map((i) => i.svg))
  out.push(wall(walls.right), wall(walls.front))
  // tray corner screws and its label plate on the front wall's top
  const corners = k.corners(TRAY, 1.2).map(([x, y]) => at(rpt(x, y, Z.wall, tilt)))
  out.push(D(corners, { size: 0.5 }))
  return out.join('')
}

/** Height of the phantom's top surface under the arm's centre line (y = 0), for the beam's landing point. */
export function surfaceAt(x: number, tilt: number): number {
  const [qx, qy] = rotate([x, 0], -tilt)
  const inside = (p: Plan) => qx >= p.x && qx <= p.x + p.w && qy >= p.y && qy <= p.y + p.d
  for (let i = 0; i < 6; i++) if (inside(bodyPlan(i))) return BODY_Z[1]
  for (let i = 0; i < 5; i++) if (inside(discPlan(i))) return BODY_Z[0] + 11
  if (inside(BAR)) return BODY_Z[0]
  if (inside(TRAY)) return Z.floor
  return Z.padTop
}

export const HEAD_Z = Z.head
export const STATIC = table()
export const CARRIAGE = carriage()
export const ARM = arm()
export const step = (dx: number) => k.translateAlong(P, [dx, 0, 0])
export const iso = at
