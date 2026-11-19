import { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, Box, CheckCircle2, ChevronRight, Clock3, Factory, Network as NetworkIcon,
  Orbit, Pause, Play, RotateCcw, ScanLine,
} from 'lucide-react'
import '@xyflow/react/dist/style.css'
import './style.css'
import { Network } from './Network'
import { stations } from './topology'
import { loadTrace } from './trace'
import type { NodeId, StationId, Trace, TraceEvent } from './trace'

function clock(minutes: number) {
  const seconds = Math.floor(minutes * 60)
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60]
    .map((value) => String(value).padStart(2, '0')).join(':')
}

function eventTitle(event: TraceEvent) {
  return event.event.startsWith('arrival')
    ? stations[event.origin].name
    : `${stations[event.origin].name} completed`
}

function countEvents(trace: Trace) {
  let batches = 0
  let corrections = 0
  let scrap = 0
  return [{ batches, corrections, scrap }, ...trace.events.map((event) => {
    for (const [, to] of event.transfers) {
      if (to === 'packed') batches += 1
      if (to === 'COR') corrections += 1
      if (to === 'scrap') scrap += 1
    }
    return { batches, corrections, scrap }
  })]
}

function EventChart({ trace, cursor, target, title, description, unit, icon: Icon }: {
  trace: Trace; cursor: number; target: NodeId; title: string;
  description: string; unit: string; icon: typeof Activity;
}) {
  const interval = 120
  const counts = Array.from({ length: Math.ceil(trace.horizon / interval) }, () => 0)
  for (let index = 0; index < cursor; index += 1) {
    const event = trace.events[index]
    if (event.transfers.some(([, to]) => to === target)) {
      counts[Math.min(counts.length - 1, Math.floor(event.time / interval))] += 1
    }
  }
  const max = Math.max(...counts, 1)
  return (
    <section className={`panel chart ${target === 'COR' ? 'correction-chart' : ''}`}>
      <div className="panel-heading"><div><h2>{title}</h2><p>{description}</p></div><Icon size={18} aria-hidden="true" /></div>
      <div className="bars" style={{ gridTemplateColumns: `repeat(${counts.length}, minmax(0, 1fr))` }} role="img" aria-label={counts.map((count, index) => `Hour ${index * 2} to ${index * 2 + 2}: ${count} ${unit}`).join('; ')}>
        {counts.map((count, index) => (
          <div className="bar-column" key={index} title={`Hour ${index * 2}–${index * 2 + 2} · ${count} ${unit}`}>
            <div className="bar-track"><div className="bar" style={{ height: `${count / max * 100}%` }}>{count > 0 && <span>{count}</span>}</div></div>
            <span>{String(index * 2).padStart(2, '0')}h</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function Dashboard({ trace }: { trace: Trace }) {
  const [cursor, setCursor] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [animationStart, setAnimationStart] = useState(0)
  const [selected, setSelected] = useState<NodeId | null>(null)
  const total = trace.events.length
  const event = trace.events[cursor - 1]
  const time = event?.time ?? 0
  const snapshot = event?.state ?? trace.initial
  const { batches, corrections, scrap } = useMemo(() => countEvents(trace), [trace])[cursor]
  const recent = trace.events.slice(Math.max(0, cursor - 5), cursor).reverse()
  const destinations = [...new Set(event?.transfers.map(([, to]) => stations[to].name))]
  const eventInterval = 220 / speed

  useEffect(() => {
    if (!playing || cursor >= total) return
    const timer = window.setTimeout(() => setCursor(cursor + 1), eventInterval)
    return () => window.clearTimeout(timer)
  }, [playing, eventInterval, cursor, total])

  const isPlaying = playing && cursor < total

  function seek(next: number) {
    setPlaying(false)
    setCursor(next)
  }

  return (
    <>
      <div className="page-intro">
        <div>
          <h1>Garment Factory <span>Digital Twin</span></h1>
          <p>Replay the Python simulation from incoming pieces to packed shirts.</p>
        </div>
        <div className="trace-tag"><span className="trace-dot" /> SOURCE TRACE <b>SEED {trace.seed}</b></div>
      </div>

      <section className="metrics" aria-label="Playback summary">
        <Metric icon={Clock3} label="Simulation clock" value={clock(time)} detail={cursor === 0 ? 'Before first dispatch' : time <= trace.warmUp ? 'Warm-up period' : 'After warm-up'} />
        <Metric icon={Activity} label="Events processed" value={cursor.toLocaleString()} detail={`of ${total.toLocaleString()} recorded events`} />
        <Metric icon={Box} label="Shirts packed" value={String(batches * 12)} detail={`${batches} completed ${batches === 1 ? 'batch' : 'batches'} · 12 shirts each`} />
        <Metric icon={ScanLine} label="Sent to correction" value={String(corrections)} detail={`${scrap} scrapped after reinspection`} />
      </section>

      <div className="workspace">
        <section className="panel topology">
          <div className="panel-heading network-heading">
            <div className="heading-icon"><NetworkIcon size={22} /><div><h2>Production network</h2><p>Select a station to inspect it. Drag the map to pan.</p></div></div>
            <span className="panel-meta">{Object.keys(trace.capacities).length} STATIONS</span>
          </div>
          <section className="playback" aria-label="Playback timeline">
            <div className="transport">
              <button className="play" type="button" onClick={() => {
                if (cursor === total) setCursor(0)
                if (!isPlaying) setAnimationStart(cursor === total ? 0 : cursor)
                setPlaying(!isPlaying)
              }}>{isPlaying ? <Pause size={17} /> : <Play size={17} />}{isPlaying ? 'Pause' : cursor === total ? 'Replay' : 'Play'}</button>
              <button className="icon-button" type="button" title="Reset to start" aria-label="Reset to start" onClick={() => seek(0)}><RotateCcw size={17} /></button>
              <label className="speed">Speed <select value={speed} onChange={(e) => {
                setAnimationStart(cursor)
                setSpeed(Number(e.target.value))
              }}>
                {[0.5, 1, 2, 4, 8].map((value) => <option key={value} value={value}>{value}×</option>)}
              </select></label>
              <span className="play-state">{isPlaying ? 'Playing' : cursor === total ? 'Complete' : 'Paused'}</span>
            </div>
            <input className="timeline" type="range" min="0" max={total} value={cursor}
              aria-label="Seek by event" aria-valuetext={`Event ${cursor} of ${total}, ${clock(time)}`}
              onChange={(e) => seek(Number(e.target.value))} />
            <div className="timeline-labels"><span>EVENT {cursor.toLocaleString()} / {total.toLocaleString()}</span><span>HORIZON {clock(trace.horizon)}</span></div>
          </section>
          <div className="current-event">
            <span className={`dispatch-indicator ${event ? 'recorded' : ''}`} aria-hidden="true" />
            <div>
              <small>Current dispatch</small>
              <strong>{event ? eventTitle(event) : 'Ready to replay'}</strong>
            </div>
            {event && <p>{destinations.length ? `Transferred to ${destinations.join(' and ')}` : 'No transfer recorded for this event'}.</p>}
          </div>
          <Network trace={trace} snapshot={snapshot} cursor={cursor} selected={selected} onSelect={setSelected}
            animationStart={isPlaying ? animationStart : null} eventInterval={eventInterval} />
        </section>

        <aside className="sidebar">
          <section className="panel inspector">
            <div className="panel-heading"><h2>Station inspector</h2><Factory size={17} aria-hidden="true" /></div>
            {selected ? (() => {
              const station = stations[selected]
              const reading = snapshot.stations[selected as StationId]
              return <div className="inspector-body">
                <span className={`inspect-icon ${station.group}`}><station.icon size={20} /></span>
                <h3>{station.name}</h3><p>{station.detail}</p>
                {reading && <dl><div><dt>Busy servers</dt><dd>{reading[0]} / {trace.capacities[selected as StationId]}</dd></div>
                  <div><dt>Idle servers</dt><dd>{trace.capacities[selected as StationId] - reading[0]}</dd></div>
                  <div><dt>Waiting {selected === 'PA' ? 'batches' : 'jobs'}</dt><dd>{reading[1]}</dd></div>
                  {selected === 'F' && <div><dt>Folded shirts awaiting a batch</dt><dd>{snapshot.buffers.folded} / 12</dd></div>}
                </dl>}
              </div>
            })() : <div className="inspector-body">
              <span className="inspect-icon"><Factory size={20} /></span>
              <h3>Component buffers</h3><p>Parts waiting to be matched and folded shirts waiting for a full batch.</p>
              <dl>{Object.entries(snapshot.buffers).map(([name, count]) => <div key={name}><dt>{name}</dt><dd>{count}</dd></div>)}</dl>
            </div>}
          </section>
          <section className="panel event-panel">
            <div className="panel-heading"><h2>Recent events</h2><Activity size={17} aria-hidden="true" /></div>
            {recent.length ? <ol>{recent.map((item, index) => (
              <li key={cursor - index}>
                <button type="button" onClick={() => { seek(cursor - index); setSelected(item.origin) }}>
                  <span><strong>{eventTitle(item)}</strong><time>{clock(item.time)}</time></span><ChevronRight size={15} />
                </button>
              </li>
            ))}</ol> : <p className="event-empty">Press Play or scrub the timeline to see recorded events.</p>}
          </section>
        </aside>
      </div>
      <div className="charts" aria-label="Recorded activity through the current event">
        <EventChart trace={trace} cursor={cursor} target="packed" title="Packing activity" description="Completed batches of 12 shirts per two-hour window" unit="batches" icon={Box} />
        <EventChart trace={trace} cursor={cursor} target="COR" title="Correction activity" description="Shirts sent to correction per two-hour window" unit="shirts" icon={ScanLine} />
      </div>
      <footer><CheckCircle2 size={15} /> Counters reflect this trace from time zero. Queue counts and busy servers are instantaneous snapshots.</footer>
    </>
  )
}

function Metric({ icon: Icon, label, value, detail }: {
  icon: typeof Activity; label: string; value: string; detail: string;
}) {
  return <article className="metric"><div><span>{label}</span><Icon size={17} aria-hidden="true" /></div><strong>{value}</strong><p>{detail}</p></article>
}

function App() {
  const [trace, setTrace] = useState<Trace>()
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    loadTrace(controller.signal).then(setTrace).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(String(reason))
    })
    return () => controller.abort()
  }, [])
  return (
    <>
      <header className="site-header"><div className="header-inner">
        <div className="brand"><span><Orbit size={22} aria-hidden="true" /></span> ORBIT</div>
        <div className="header-context">FACTORY SIMULATION</div>
        <a href="https://github.com/mohammad-jamalzehi/Simulation-of-Cloth-Factory-Production">Source on GitHub</a>
      </div></header>
      <main>{error ? <p role="alert">Could not load the source trace. {error}</p> : trace ? <Dashboard trace={trace} /> : <p role="status">Loading source trace…</p>}</main>
    </>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
