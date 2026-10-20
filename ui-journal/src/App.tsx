import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createJob, pollJob, startJob, uploadFile, type Item, type JobView, type PickedFile } from './api'
import DropStage from './components/DropStage'
import Header from './components/Header'
import ResultList, { type Filter } from './components/ResultList'
import Viewer from './components/Viewer'

export type Phase = 'idle' | 'uploading' | 'processing' | 'done' | 'error'

export interface Batch {
  id: string
  name: string
  paths: string[]
  items: Map<number, Item>
  total: number | null
  state: JobView['state']
  elapsed: number
  summary: JobView['summary']
}

export interface Row { i: number; path: string; item?: Item }

const POLL_MS = 500

export default function App() {
  const [phase, setPhase] = useState<Phase>('idle')
  const [batch, setBatch] = useState<Batch | null>(null)
  const [upload, setUpload] = useState({ loaded: 0, total: 0, files: 0 })
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<number | null>(null)
  const [filter, setFilter] = useState<Filter>('all')
  const [region, setRegion] = useState<string>('all')
  const [mobileViewer, setMobileViewer] = useState(false)
  // items that arrived while the page watched: they "develop" once, like film on the light box
  const arrived = useRef(new Set<number>())
  const [, bump] = useState(0)

  const reset = useCallback(() => {
    setPhase('idle'); setBatch(null); setError(''); setSelected(null)
    setFilter('all'); setRegion('all'); setMobileViewer(false); arrived.current = new Set()
    history.replaceState(null, '', location.pathname)
  }, [])

  // #job=<id> keeps the batch across a reload (the server still holds it)
  useEffect(() => {
    const id = new URLSearchParams(location.hash.slice(1)).get('job')
    if (!id) return
    setBatch({ id, name: '', paths: [], items: new Map(), total: null, state: 'running', elapsed: 0, summary: null })
    setPhase('processing')
  }, [])

  const run = useCallback(async (files: PickedFile[]) => {
    if (!files.length) return
    setError('')
    const total = files.reduce((s, f) => s + f.file.size, 0)
    setUpload({ loaded: 0, total, files: files.length })
    setPhase('uploading')
    try {
      const id = await createJob()
      let done = 0
      for (const { file, path } of files) {
        await uploadFile(id, file, path, (loaded) => setUpload((u) => ({ ...u, loaded: done + loaded })))
        done += file.size
      }
      const view = await startJob(id)
      history.replaceState(null, '', `#job=${id}`)
      setBatch({ id, name: view.name || files[0].path.split('/')[0], paths: [], items: new Map(), total: null,
                 state: view.state, elapsed: 0, summary: null })
      setPhase('processing')
    } catch (e) {
      setError((e as Error).message); setPhase('error')
    }
  }, [])

  // live progress: fetch only what is new since the last poll
  const itemCount = batch?.items.size ?? 0
  useEffect(() => {
    if (phase !== 'processing' || !batch) return
    let stop = false
    const timer = window.setTimeout(async () => {
      try {
        const view = await pollJob(batch.id, itemCount)
        if (stop) return
        setBatch((b) => {
          if (!b) return b
          const items = new Map(b.items)
          // develop only what arrives live (one or two per poll); a reload that brings the whole
          // batch back at once should not replay the effect on every row
          const live = view.items.length <= 3
          for (const it of view.items) {
            if (live && !items.has(it.i)) arrived.current.add(it.i)
            items.set(it.i, it)
          }
          return { ...b, name: b.name || view.name, items, paths: view.paths ?? b.paths, total: view.total ?? b.total, state: view.state,
                   elapsed: view.elapsed, summary: view.summary }
        })
        if (view.state === 'done') setPhase('done')
        else if (view.state === 'error') { setError(view.error || 'Обработка прервалась'); setPhase('error') }
        else if (!view.items.length) bump((x) => x + 1)          // nothing new yet: poll again
      } catch (e) {
        if (!stop) { setError((e as Error).message); setPhase('error') }
      }
    }, itemCount ? POLL_MS : 200)
    return () => { stop = true; window.clearTimeout(timer) }
  })

  const rows: Row[] = useMemo(() => {
    if (!batch) return []
    const n = Math.max(batch.total ?? 0, batch.paths.length, batch.items.size)
    return Array.from({ length: n }, (_, i) => ({ i, path: batch.items.get(i)?.path ?? batch.paths[i] ?? '', item: batch.items.get(i) }))
  }, [batch])

  const visible = useMemo(() => rows.filter(({ item }) => {
    if (region !== 'all' && item?.region !== region) return false
    if (filter === 'all') return true
    if (!item) return false
    if (filter === 'fail') return item.status === 'Failure'
    if (filter === 'bad') return item.status === 'Success' && item.quality === 1
    return item.status === 'Success' && item.quality === 0
  }), [rows, filter, region])

  // the first finished image opens by itself; a selection the user made is never taken away
  useEffect(() => {
    if (selected != null || !batch) return
    const first = rows.find((r) => r.item?.has_image)
    if (first) setSelected(first.i)
  }, [rows, selected, batch])

  const select = useCallback((i: number, via: 'pointer' | 'key') => {
    setSelected(i)
    if (via === 'pointer') setMobileViewer(true)
  }, [])

  // ↑/↓ (or j/k) walk the visible list; keyboard navigation is never animated — it repeats too often
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!visible.length || e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return
      if (!['ArrowDown', 'ArrowUp', 'j', 'k'].includes(e.key)) return
      e.preventDefault()
      const pos = visible.findIndex((r) => r.i === selected)
      const step = e.key === 'ArrowDown' || e.key === 'j' ? 1 : -1
      const next = visible[Math.min(visible.length - 1, Math.max(0, pos + step))]
      if (next) {
        select(next.i, 'key')
        document.getElementById(`row-${next.i}`)?.scrollIntoView({ block: 'nearest' })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [visible, selected, select])

  if (phase === 'idle' || phase === 'uploading' || (phase === 'error' && !batch)) {
    return <DropStage onFiles={run} uploading={phase === 'uploading' ? upload : null} error={error} onRetry={reset} />
  }

  const current = selected != null ? batch!.items.get(selected) : undefined
  return (
    <div className="app" data-mobile-viewer={mobileViewer || undefined}>
      <Header batch={batch!} rows={rows} phase={phase} error={error} filter={filter} onFilter={setFilter} onNew={reset} />
      <main className="workspace">
        <ResultList
          rows={visible} all={rows} selected={selected} onSelect={select}
          filter={filter} onFilter={setFilter} region={region} onRegion={setRegion}
          arrived={arrived.current} onDeveloped={(i) => arrived.current.delete(i)}
          jobId={batch!.id} running={phase === 'processing'}
        />
        <Viewer jobId={batch!.id} item={current} waiting={selected == null}
                path={selected != null ? rows[selected]?.path : ''} onClose={() => setMobileViewer(false)}
                onUpdate={(it) => setBatch((b) => {
                  if (!b) return b
                  const items = new Map(b.items)
                  items.set(it.i, it)
                  return { ...b, items }
                })} />
      </main>
    </div>
  )
}
