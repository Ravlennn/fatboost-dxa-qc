// Talks only to the local server that served this page. Nothing leaves the machine.

export interface Item {
  i: number
  path: string
  status: 'Success' | 'Failure'
  region: string | null
  quality: number | null
  violations: string[]
  seconds: number | null
  error: string
  has_image: boolean
  width?: number
  height?: number
  mm_per_px?: number | null
  scores?: Record<string, number>
  points?: Record<string, [number, number]>
  measurements?: Record<string, number>
  lt?: { visible: boolean | null; prominence_mm: number | null; apex: [number, number] | null; edge: [number, number][] | null } | null
  warnings?: string[]
  review?: {
    state: 'confirmed'
    at: string
    corrected_points: boolean
    overrides: Record<string, boolean>
    model: { quality: number | null; violations: string[] }
  }
}

export interface JobView {
  id: string
  state: 'receiving' | 'queued' | 'running' | 'done' | 'error'
  name: string
  total: number | null
  done: number
  paths: string[] | null
  items: Item[]
  summary: Record<string, number> | null
  error: string
  elapsed: number
}

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error((body as { error?: string }).error || `Сервер ответил ${res.status}`)
  return body as T
}

export const createJob = async () => (await json<{ id: string }>(await fetch('/api/jobs', { method: 'POST' }))).id

export const startJob = async (id: string) => json<JobView>(await fetch(`/api/jobs/${id}/start`, { method: 'POST' }))

export const pollJob = async (id: string, since: number) => json<JobView>(await fetch(`/api/jobs/${id}?since=${since}`))

export const imageUrl = (id: string, i: number) => `/api/jobs/${id}/image/${i}.png`

export const reportUrl = (id: string, kind: 'csv' | 'xlsx') => `/api/jobs/${id}/report.${kind}`

export const correctionsUrl = (id: string) => `/api/jobs/${id}/corrections.json`

export const srUrl = (id: string) => `/api/jobs/${id}/sr.zip`

type Points = Record<string, [number, number]>

const post = async <T,>(url: string, body: unknown) =>
  json<T>(await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }))

/** The model again, on the specialist's points. A preview: nothing is stored. */
export const rescore = (id: string, i: number, points: Points) => post<Item>(`/api/jobs/${id}/items/${i}/rescore`, { points })

/** The specialist's confirmed decision; it replaces the model's in the report. */
export const review = (id: string, i: number, points: Points | null, overrides: Record<string, boolean>) =>
  post<Item>(`/api/jobs/${id}/items/${i}/review`, { points, overrides })

/** PUT one file; XHR because fetch cannot report upload progress. */
export function uploadFile(id: string, file: File, name: string, onProgress: (loaded: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', `/api/jobs/${id}/files?name=${encodeURIComponent(name)}`)
    xhr.upload.onprogress = (e) => onProgress(e.loaded)
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve()
      let msg = `Сервер ответил ${xhr.status}`
      try { msg = JSON.parse(xhr.responseText).error || msg } catch { /* keep the status text */ }
      reject(new Error(msg))
    }
    xhr.onerror = () => reject(new Error('Сервер недоступен — проверьте, что dxaqc serve запущен'))
    xhr.send(file)
  })
}

export interface PickedFile { file: File; path: string }

/** Files from a drop: folders are walked so their relative paths reach the report. */
export async function filesFromDrop(dt: DataTransfer): Promise<PickedFile[]> {
  const entries = Array.from(dt.items).map((it) => it.webkitGetAsEntry?.()).filter(Boolean) as FileSystemEntry[]
  if (!entries.length) return Array.from(dt.files).map((file) => ({ file, path: file.name }))
  const out: PickedFile[] = []
  const walk = async (entry: FileSystemEntry, prefix: string): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej))
      out.push({ file, path: prefix + entry.name })
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader()
      let batch: FileSystemEntry[]
      do {
        batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej))
        for (const child of batch) await walk(child, `${prefix}${entry.name}/`)
      } while (batch.length)
    }
  }
  for (const entry of entries) await walk(entry, '')
  return out.filter(({ file }) => !file.name.startsWith('.'))
}
